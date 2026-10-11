"""Reddit hyperlink graph (SNAP soc-redditHyperlinks-body), BFS bands and splits.

Graph construction ports `notebooks/pilot/Reddit_Network_Data_EDA.ipynb`: nodes are
subreddits (sources and targets of hyperlink posts), directed edges are the
286,561 body hyperlinks (duplicates kept, as in the pilot), and each node's 86
features are the post properties averaged over the posts it *sends*. Subreddits
that only receive links (7,913) have no posts, so their features are all zero;
`is_source` marks the nodes that do. They stay in the graph for message passing
but are never supervised or evaluated.

Splits port `notebooks/pilot/GNN.ipynb`: BFS over the undirected view from the
highest out-degree node (the anchor). The distance split trains on 0-1 hops, validates
on 2 and tests on 3+; the random control split draws the same number of nodes per
split uniformly from the same pool (reachable source nodes).

Data sources, in order: the cached graph `<root>/reddit_graph.pt`; a pilot
`reddit_base_graph.pt` from Drive (`cached_pt`, a PyG Data with x and edge_index);
the TSV at `<root>/soc-redditHyperlinks-body.tsv`; download from SNAP_URL. The SNAP
host is blocked from the build sandbox, so on Colab either download works or point
`cached_pt` at the Drive copy.
"""
from __future__ import annotations

import os
import os.path as osp
import urllib.request
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data

SNAP_URL = "https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv"
TSV_NAME = "soc-redditHyperlinks-body.tsv"
CACHE_NAME = "reddit_graph.pt"

# The 86 PROPERTIES of each hyperlink post, in order, from the SNAP dataset page
# (snap.stanford.edu/data/soc-RedditHyperlinks.html): 18 text statistics, 3 VADER
# sentiment scores and 65 LIWC categories. Not all features are LIWC.
FEATURE_NAMES: list[str] = [
    "n_chars", "n_chars_no_space", "frac_alpha", "frac_digits", "frac_upper", "frac_white",
    "frac_special", "n_words", "n_unique_words", "n_long_words", "avg_word_len",
    "n_unique_stopwords", "frac_stopwords", "n_sentences", "n_long_sentences",
    "avg_chars_per_sentence", "avg_words_per_sentence", "readability_index",
    "vader_pos", "vader_neg", "vader_compound",
    "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We", "LIWC_You", "LIWC_SheHe",
    "LIWC_They", "LIWC_Ipron", "LIWC_Article", "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past",
    "LIWC_Present", "LIWC_Future", "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate",
    "LIWC_Quant", "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family", "LIWC_Friends",
    "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx", "LIWC_Anger",
    "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight", "LIWC_Cause", "LIWC_Discrep", "LIWC_Tentat",
    "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl", "LIWC_Excl", "LIWC_Percept", "LIWC_See",
    "LIWC_Hear", "LIWC_Feel", "LIWC_Bio", "LIWC_Body", "LIWC_Health", "LIWC_Sexual",
    "LIWC_Ingest", "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time", "LIWC_Work",
    "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig", "LIWC_Death",
    "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler",
]
assert len(FEATURE_NAMES) == 86
SENTIMENT_COLS = [FEATURE_NAMES.index(c) for c in ("vader_pos", "vader_neg", "vader_compound")]
LIWC_COLS = [i for i, c in enumerate(FEATURE_NAMES) if c.startswith("LIWC_")]


def build_graph(tsv_path: str) -> Data:
    """Graph from the SNAP TSV: x [N, 86] mean post properties per source subreddit."""
    import pandas as pd

    df = pd.read_csv(tsv_path, sep="\t", usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "PROPERTIES"])
    names = pd.unique(df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].values.ravel())
    index = {n: i for i, n in enumerate(names)}
    src = df["SOURCE_SUBREDDIT"].map(index).to_numpy(np.int64)
    dst = df["TARGET_SUBREDDIT"].map(index).to_numpy(np.int64)
    props = np.array([np.fromstring(p, sep=",") for p in df["PROPERTIES"]], dtype=np.float64)
    if props.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} properties per post, got {props.shape[1]}")
    n = len(names)
    sums = np.zeros((n, props.shape[1]))
    np.add.at(sums, src, props)
    counts = np.bincount(src, minlength=n).astype(float)
    x = sums / np.maximum(counts, 1.0)[:, None]
    data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=torch.from_numpy(np.stack([src, dst])),
                num_nodes=n)
    data.is_source = torch.from_numpy(counts > 0)
    data.subreddits = list(names)
    return data


def _with_source_mask(data: Data) -> Data:
    if getattr(data, "is_source", None) is None:
        is_source = torch.zeros(data.num_nodes, dtype=torch.bool)
        is_source[data.edge_index[0].unique()] = True
        data.is_source = is_source
    return data


def load_reddit_graph(root: str = "./data/reddit", cached_pt: Optional[str] = None,
                      url: str = SNAP_URL) -> Data:
    """Load (or build and cache) the full graph with all 86 features and `is_source`."""
    cache = osp.join(root, CACHE_NAME)
    if osp.exists(cache):
        return torch.load(cache, weights_only=False)
    if cached_pt and osp.exists(cached_pt):
        data = _with_source_mask(torch.load(cached_pt, weights_only=False))
    else:
        tsv = osp.join(root, TSV_NAME)
        if not osp.exists(tsv):
            os.makedirs(root, exist_ok=True)
            print(f"downloading {url} -> {tsv}")
            urllib.request.urlretrieve(url, tsv + ".part")
            os.replace(tsv + ".part", tsv)
        data = build_graph(tsv)
    os.makedirs(root, exist_ok=True)
    torch.save(data, cache)
    return data


def out_degree(edge_index: torch.Tensor, num_nodes: int) -> np.ndarray:
    return np.bincount(np.asarray(edge_index[0]), minlength=num_nodes)


def pick_anchor(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Highest out-degree node (lowest id on ties), as in the pilot."""
    return int(np.argmax(out_degree(edge_index, num_nodes)))


def undirected_adjacency(edge_index, num_nodes: int) -> sp.csr_matrix:
    """Symmetric 0/1 adjacency without self loops or duplicate edges."""
    ei = np.asarray(edge_index)
    keep = ei[0] != ei[1]
    a = sp.coo_matrix((np.ones(keep.sum()), (ei[0][keep], ei[1][keep])), shape=(num_nodes, num_nodes)).tocsr()
    a = a + a.T
    a.data[:] = 1.0
    return a.tocsr()


def bfs_hops(edge_index, num_nodes: int, anchor: int) -> np.ndarray:
    """Hop distance from `anchor` over the undirected view; -1 for unreachable nodes."""
    d = shortest_path(undirected_adjacency(edge_index, num_nodes), method="D", unweighted=True,
                      directed=False, indices=anchor)
    return np.where(np.isfinite(d), d, -1).astype(np.int64)


def eligible_nodes(dist: np.ndarray, is_source) -> np.ndarray:
    """Nodes that can be supervised: reachable source nodes."""
    return (np.asarray(dist) >= 0) & np.asarray(is_source, dtype=bool)


def distance_split(dist: np.ndarray, is_source, train_max: int = 1, val_hops: int = 2) -> dict[str, np.ndarray]:
    """train: 0..train_max hops, val: val_hops, test: > val_hops; reachable sources only."""
    ok = eligible_nodes(dist, is_source)
    dist = np.asarray(dist)
    return {"train": ok & (dist <= train_max), "val": ok & (dist == val_hops), "test": ok & (dist > val_hops)}


def random_split(dist: np.ndarray, is_source, sizes: dict[str, int], seed: int) -> dict[str, np.ndarray]:
    """Uniformly random split of the same pool with the given split sizes (default: the distance split's)."""
    ok = np.flatnonzero(eligible_nodes(dist, is_source))
    if sum(sizes.values()) != ok.size:
        raise ValueError(f"split sizes {sizes} do not add up to the {ok.size} eligible nodes")
    perm = np.random.default_rng(seed).permutation(ok)
    out, start = {}, 0
    for name in ("train", "val", "test"):
        m = np.zeros(len(dist), dtype=bool)
        m[perm[start:start + sizes[name]]] = True
        out[name] = m
        start += sizes[name]
    return out


def make_split(kind: str, dist: np.ndarray, is_source, seed: int) -> dict[str, np.ndarray]:
    """"distance" (fixed) or "random" (seeded, same sizes as the distance split)."""
    dsplit = distance_split(dist, is_source)
    if kind == "distance":
        return dsplit
    if kind == "random":
        return random_split(dist, is_source, {k: int(v.sum()) for k, v in dsplit.items()}, seed)
    raise ValueError(f"split must be 'distance' or 'random', got {kind!r}")


# --------------------------------------------------------------------- synthetic


def synthetic_graph(n: int = 400, n_feat: int = 10, seed: int = 0, frac_sink: float = 0.15) -> Data:
    """Small Reddit-like graph for tests and CPU smoke runs (no download).

    A random recursive tree (each node links to a uniformly chosen earlier one, so BFS
    bands 0..5+ are all populated) plus a
    few random edges, directed both ways at random. `frac_sink` of the nodes send no
    links (zero features, is_source False). Features are standard normal except:
        col 0: col 7 + mean over 1-hop neighbours of col 2 + mean over 2-hop of col 3
               (planted neighbour- and multi-hop-dependent target)
        col 1: col 4 + col 6 + 0.1 noise (a function of the node's own inputs)
        col 5: near-duplicate of col 1 (|r| > 0.9 with it)
    """
    rng = np.random.default_rng(seed)
    parents = [int(rng.integers(0, v)) for v in range(1, n)]   # random recursive tree: depth ~ log n
    edges = [(v, p) for v, p in zip(range(1, n), parents)]
    edges += [tuple(int(a) for a in rng.choice(n, 2, replace=False)) for _ in range(n // 20)]
    sink = rng.random(n) < frac_sink
    sink[0] = False
    src, dst = [], []
    for a, b in edges:
        if sink[a] or (not sink[b] and rng.random() < 0.5):
            a, b = b, a                    # sinks only receive links
        src.append(a)
        dst.append(b)
    edge_index = np.array([src, dst], dtype=np.int64)
    x = rng.normal(size=(n, n_feat))
    a = undirected_adjacency(edge_index, n)
    deg = np.asarray(a.sum(1)).ravel()
    mean1 = (a @ x) / np.maximum(deg, 1)[:, None]
    a2 = ((a @ a) > 0).astype(float).tolil()
    a2.setdiag(0)
    a2 = a2.tocsr() - a.multiply(a2)       # exactly-2-hop (approximately; tree-like graph)
    a2.data = np.maximum(a2.data, 0)
    a2.eliminate_zeros()
    deg2 = np.asarray(a2.sum(1)).ravel()
    mean2 = (a2 @ x) / np.maximum(deg2, 1)[:, None]
    x[:, 0] = x[:, 7] + mean1[:, 2] + mean2[:, 3] + 0.05 * rng.normal(size=n)
    x[:, 1] = x[:, 4] + x[:, 6] + 0.1 * rng.normal(size=n)
    x[:, 5] = x[:, 1] + 0.05 * rng.normal(size=n)
    x[sink] = 0.0
    data = Data(x=torch.tensor(x, dtype=torch.float), edge_index=torch.from_numpy(edge_index), num_nodes=n)
    data.is_source = torch.from_numpy(~sink)
    return data
