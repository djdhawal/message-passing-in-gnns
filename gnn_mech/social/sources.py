"""Download and parse SNAP social graphs into one cached full-graph `Data` per source.

    python -m gnn_mech.social.sources twitch-gamers reddit-body higgs-follow --root ./data

Layout (all under `<root>/social/`; every step is skipped when its file exists, and every
file is written to a temp name first and moved into place with `os.replace`, so an
interrupted Colab session never leaves a truncated file behind):

    twitch-gamers/raw/twitch_gamers.zip                   28 MB   -> twitch-gamers/graph.pt
    twitch-<lang>/raw/twitch.zip (all 6 languages, small)   3 MB   -> twitch-<lang>/graph.pt
    reddit-body/raw/soc-redditHyperlinks-body.tsv         319 MB  -> reddit-body/graph.pt
    higgs/raw/higgs-*.gz  (4 layers + activity times)      62 MB  -> higgs/graph.pt (all layers)

The cached graph (`load_graph`) is a `torch_geometric.data.Data` with
    edge_index   [2, E] long, undirected (both directions), deduplicated, no self loops
    num_nodes
    attrs        dict[str, np.ndarray]: the raw node attribute table (one entry per column)
    source, dataset
plus, for Higgs, `layers: dict[str, LongTensor [2, E]]` (all four message-passing layers,
same node set) and `cutoff` (unix seconds). Inputs and targets are *derived* from `attrs` by
`node_inputs` / `node_target`, so changing the target never requires a re-download or re-parse.

Sources (snap.stanford.edu/data/...):
    twitch_gamers.html            large_twitch_features.csv: views, mature, life_time, created_at,
                                  updated_at, numeric_id, dead_account, language, affiliate;
                                  large_twitch_edges.csv: numeric_id_1, numeric_id_2 (mutual follows).
    twitch-social-networks.html   per language (DE, ENGB, ES, FR, PTBR, RU): musae_<L>_edges.csv
                                  (from,to), musae_<L>_target.csv (id, days, mature, views, partner,
                                  new_id), musae_<L>_features.json (new_id -> list of game ids; the
                                  DE file is called musae_DE.json).
    soc-RedditHyperlinks.html     parsed by `gnn_mech.reddit.data.build_graph` (reused unchanged).
    higgs-twitter.html            higgs-social_network.edgelist.gz (follow, directed), retweet /
                                  reply / mention edgelists, higgs-activity_time.txt.gz
                                  (userA userB unix_time RT|MT|RE).

Higgs: target and leakage
-------------------------
Target `active_after`: the user is the *actor* (userA) of at least one RT/MT/RE interaction at or
after the cutoff (default 2012-07-04 00:00 UTC, a few hours before the CERN announcement). Nothing
derived from interactions at or after the cutoff may enter the node set, the graph or the inputs:

1. Node set. SNAP's user set is "users involved in the activities" over 1-7 July, so a user that
   appears in the files but has no interaction before the cutoff was included *because* of
   post-cutoff activity (or that of a neighbour). Keeping such users would leak through the node set
   itself and through neighbour counts. We therefore keep only users involved (as actor or receiver)
   in at least one interaction strictly before the cutoff, and re-index them 0..N-1.
2. Graph. `higgs-follow` uses the follower graph induced on that node set (the follow layer has no
   timestamps; it is treated as fixed structure, see the caveat below). The retweet / mention /
   reply layers are **rebuilt from the timestamped activity file using interactions before the
   cutoff only**; the SNAP layer edgelists (which aggregate all of 1-7 July) are downloaded for
   reference but never used, because they encode the post-cutoff activity that defines the target.
3. Inputs. In/out follow degree, and per activity type (RT, MT, RE) the pre-cutoff out-count,
   in-count and number of distinct partners. All counts filter on `time < cutoff`.
4. Verification (`check_higgs_leakage`, run at build time): every activity row used for inputs or
   layers has time < cutoff, and the rebuilt layers are subsets of the SNAP layers.

Caveat: SNAP does not say when the follower graph was crawled, so follow edges formed after the
cutoff cannot be excluded. This is the only residual channel, and it carries no timestamps.

Twitch-gamers / musae / Reddit targets are documented in `DEFAULT_TARGETS` and in the configs.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import os.path as osp
import shutil
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

SNAP = "https://snap.stanford.edu/data/"
MUSAE_LANGS = {"de": "DE", "en": "ENGB", "es": "ES", "fr": "FR", "pt": "PTBR", "ru": "RU"}
HIGGS_LAYERS = ("follow", "retweet", "mention", "reply")
HIGGS_KIND = {"retweet": "RT", "mention": "MT", "reply": "RE"}
HIGGS_CUTOFF = int(datetime(2012, 7, 4, tzinfo=timezone.utc).timestamp())   # 1341360000
HIGGS_FILES = ["higgs-social_network.edgelist.gz", "higgs-retweet_network.edgelist.gz",
               "higgs-reply_network.edgelist.gz", "higgs-mention_network.edgelist.gz",
               "higgs-activity_time.txt.gz"]
MUSAE_TOP_K = 128           # game columns kept for twitch-<lang> (most frequent in that graph)
GRAPH_VERSION = 1           # bump to force a re-parse of graph.pt

DATASETS: tuple[str, ...] = (("twitch-gamers",) + tuple(f"twitch-{l}" for l in MUSAE_LANGS)
                             + ("reddit-body",) + tuple(f"higgs-{l}" for l in HIGGS_LAYERS))

# Provisional default targets (docs/social_plan.md, build contract). reddit-body: see REDDIT_TARGET.
# Chosen by gnn_mech.reddit.target_select on the real body graph (ridge, random split, k_max 4):
# LIWC_Money ranks first by multi-hop gain (own R2 0.108 -> hop4 0.185; multi-hop gain +0.047).
REDDIT_TARGET = "LIWC_Money"
DEFAULT_TARGETS = {"twitch-gamers": "mature", "reddit-body": REDDIT_TARGET, "social-tiny": "label",
                   **{f"twitch-{l}": "mature" for l in MUSAE_LANGS},
                   **{f"higgs-{l}": "active_after" for l in HIGGS_LAYERS}}


# ------------------------------------------------------------------ paths & download


def source_of(name: str) -> str:
    """Directory under <root>/social holding graph.pt (higgs layers share one graph file)."""
    name = name.lower()
    return "higgs" if name.startswith("higgs-") else name


def source_dir(name: str, root: str) -> str:
    """Absolute path of <root>/social/<source_of(name)> (raw/, graph.pt and processed/ live here)."""
    return osp.abspath(osp.join(root, "social", source_of(name)))


def raw_dir(name: str, root: str) -> str:
    return osp.join(source_dir(name, root), "raw")


def graph_path(name: str, root: str) -> str:
    return osp.join(source_dir(name, root), "graph.pt")


def download(url: str, path: str, retries: int = 3) -> str:
    """Download `url` to `path` unless it exists. Resumes a partial `<path>.part` (HTTP Range)."""
    if osp.exists(path):
        return path
    os.makedirs(osp.dirname(path), exist_ok=True)
    part = path + ".part"
    for attempt in range(retries):
        try:
            have = osp.getsize(part) if osp.exists(part) else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:          # server ignored the range: start over
                    have = 0
                with open(part, "ab" if have else "wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
            os.replace(part, path)
            return path
        except Exception as e:  # network hiccup: retry (resuming)
            if attempt == retries - 1:
                raise
            print(f"download of {url} failed ({e!r}); retrying")
            time.sleep(2 * (attempt + 1))
    return path


def _save(obj, path: str) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def undirected(src: np.ndarray, dst: np.ndarray, n: int) -> torch.Tensor:
    """[2, E] long edge_index: both directions, deduplicated, self loops removed, sorted by row."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keep = lo != hi
    key = np.unique(lo[keep] * n + hi[keep])
    lo, hi = key // n, key % n
    row = np.concatenate([lo, hi])
    col = np.concatenate([hi, lo])
    order = np.lexsort((col, row))
    return torch.from_numpy(np.stack([row[order], col[order]]))


# ------------------------------------------------------------------------ parsers


def _build_twitch_gamers(raw: str) -> Data:
    import pandas as pd

    zpath = download(SNAP + "twitch_gamers.zip", osp.join(raw, "twitch_gamers.zip"))
    with zipfile.ZipFile(zpath) as z:
        feats = pd.read_csv(z.open("large_twitch_features.csv"))
        edges = pd.read_csv(z.open("large_twitch_edges.csv"))
    feats = feats.sort_values("numeric_id").reset_index(drop=True)
    n = len(feats)
    if not np.array_equal(feats["numeric_id"].to_numpy(), np.arange(n)):
        raise ValueError("twitch-gamers numeric_id is not 0..N-1")
    epoch = pd.Timestamp("1970-01-01")
    attrs = {
        "views": feats["views"].to_numpy(np.float64),
        "mature": feats["mature"].to_numpy(np.float64),
        "life_time": feats["life_time"].to_numpy(np.float64),
        "created_at": ((pd.to_datetime(feats["created_at"]) - epoch).dt.days).to_numpy(np.float64),
        "updated_at": ((pd.to_datetime(feats["updated_at"]) - epoch).dt.days).to_numpy(np.float64),
        "dead_account": feats["dead_account"].to_numpy(np.float64),
        "affiliate": feats["affiliate"].to_numpy(np.float64),
        "language": feats["language"].astype(str).to_numpy(),
    }
    ei = undirected(edges["numeric_id_1"].to_numpy(), edges["numeric_id_2"].to_numpy(), n)
    return Data(edge_index=ei, num_nodes=n, attrs=attrs)


def _build_musae(raw: str, lang: str) -> Data:
    import pandas as pd

    code = MUSAE_LANGS[lang]
    zpath = download(SNAP + "twitch.zip", osp.join(raw, "twitch.zip"))
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        target = pd.read_csv(z.open(f"twitch/{code}/musae_{code}_target.csv"))
        edges = pd.read_csv(z.open(f"twitch/{code}/musae_{code}_edges.csv"))
        fname = next(f for f in names if f.startswith(f"twitch/{code}/") and f.endswith(".json"))
        games = json.load(z.open(fname))
    target = target.sort_values("new_id").reset_index(drop=True)
    n = len(target)
    if not np.array_equal(target["new_id"].to_numpy(), np.arange(n)):
        raise ValueError(f"twitch-{lang}: new_id is not 0..N-1")
    # Multi-hot game features as CSR (indptr, indices); ids are shared across languages.
    lists = [sorted(set(int(g) for g in games.get(str(i), []))) for i in range(n)]
    indptr = np.cumsum([0] + [len(l) for l in lists]).astype(np.int64)
    indices = np.array([g for l in lists for g in l], dtype=np.int64)
    attrs = {
        "days": target["days"].to_numpy(np.float64),
        "mature": target["mature"].astype(float).to_numpy(np.float64),
        "views": target["views"].to_numpy(np.float64),
        "partner": target["partner"].astype(float).to_numpy(np.float64),
        "twitch_id": target["id"].to_numpy(np.int64),
        "games_indptr": indptr, "games_indices": indices,
    }
    ei = undirected(edges["from"].to_numpy(), edges["to"].to_numpy(), n)
    return Data(edge_index=ei, num_nodes=n, attrs=attrs)


def _build_reddit(raw: str) -> Data:
    from ..reddit.data import FEATURE_NAMES, SNAP_URL, TSV_NAME, build_graph

    tsv = download(SNAP_URL, osp.join(raw, TSV_NAME))
    g = build_graph(tsv)
    n = int(g.num_nodes)
    x = g.x.numpy().astype(np.float64)
    attrs = {c: x[:, j] for j, c in enumerate(FEATURE_NAMES)}
    attrs["is_source"] = g.is_source.numpy().astype(bool)
    attrs["out_posts"] = np.bincount(g.edge_index[0].numpy(), minlength=n).astype(np.float64)
    attrs["in_posts"] = np.bincount(g.edge_index[1].numpy(), minlength=n).astype(np.float64)
    attrs["subreddit"] = np.array(g.subreddits, dtype=object)
    ei = undirected(g.edge_index[0].numpy(), g.edge_index[1].numpy(), n)
    return Data(edge_index=ei, num_nodes=n, attrs=attrs)


def _read_edgelist_gz(path: str) -> np.ndarray:
    import pandas as pd

    df = pd.read_csv(path, sep=" ", header=None, usecols=[0, 1], dtype=np.int64)
    return df.to_numpy().T


def _build_higgs(raw: str, cutoff: int = HIGGS_CUTOFF) -> Data:
    import pandas as pd

    paths = {f: download(SNAP + f, osp.join(raw, f)) for f in HIGGS_FILES}
    act = pd.read_csv(paths["higgs-activity_time.txt.gz"], sep=" ", header=None,
                      names=["a", "b", "t", "kind"], dtype={"a": np.int64, "b": np.int64, "t": np.int64})
    pre = act[act["t"] < cutoff]
    post = act[act["t"] >= cutoff]
    # 1. node set: users involved in any interaction strictly before the cutoff (see module docstring)
    keep_ids = np.unique(np.concatenate([pre["a"].to_numpy(), pre["b"].to_numpy()]))
    n = keep_ids.size
    max_id = int(max(act["a"].max(), act["b"].max()))
    follow = _read_edgelist_gz(paths["higgs-social_network.edgelist.gz"])
    max_id = max(max_id, int(follow.max()))
    remap = np.full(max_id + 1, -1, dtype=np.int64)
    remap[keep_ids] = np.arange(n)

    def local(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        la, lb = remap[a], remap[b]
        ok = (la >= 0) & (lb >= 0)
        return la[ok], lb[ok]

    # 2. graph layers: follow induced on the node set; activity layers from pre-cutoff rows only
    fa, fb = local(follow[0], follow[1])
    layers = {"follow": undirected(fa, fb, n)}
    attrs: dict[str, np.ndarray] = {"user_id": keep_ids,
                                    "follow_out": np.bincount(fa, minlength=n).astype(np.float64),
                                    "follow_in": np.bincount(fb, minlength=n).astype(np.float64)}
    for layer, kind in HIGGS_KIND.items():
        rows = pre[pre["kind"] == kind]
        a, b = local(rows["a"].to_numpy(), rows["b"].to_numpy())   # all ids are in the node set
        attrs[f"{kind.lower()}_out_pre"] = np.bincount(a, minlength=n).astype(np.float64)
        attrs[f"{kind.lower()}_in_pre"] = np.bincount(b, minlength=n).astype(np.float64)
        layers[layer] = undirected(a, b, n)
        attrs[f"{kind.lower()}_deg_pre"] = np.bincount(layers[layer][0].numpy(), minlength=n).astype(np.float64)
    # 3. target: actor of any interaction at/after the cutoff (never used for inputs or graph)
    post_actor = remap[post["a"].to_numpy()]
    active = np.zeros(n, dtype=np.float64)
    active[post_actor[post_actor >= 0]] = 1.0
    attrs["active_after"] = active
    data = Data(edge_index=layers["follow"], num_nodes=n, attrs=attrs)
    data.layers = layers
    data.cutoff = int(cutoff)
    check_higgs_leakage(data, act, paths)
    return data


def check_higgs_leakage(data: Data, act, paths: Optional[dict] = None) -> None:
    """Assert the Higgs inputs and layers only use pre-cutoff activity (see module docstring)."""
    cutoff = data.cutoff
    ids = data.attrs["user_id"]
    pre = act[act["t"] < cutoff]
    for kind in ("RT", "MT", "RE"):
        rows = pre[pre["kind"] == kind]
        assert data.attrs[f"{kind.lower()}_out_pre"].sum() == len(rows), f"{kind} out-count uses post-cutoff rows"
        assert data.attrs[f"{kind.lower()}_in_pre"].sum() == len(rows), f"{kind} in-count uses post-cutoff rows"
    involved = np.unique(np.concatenate([pre["a"].to_numpy(), pre["b"].to_numpy()]))
    assert np.array_equal(np.sort(ids), involved), "node set is not the pre-cutoff involved users"
    if paths is not None:   # rebuilt layers must be subsets of SNAP's all-time layers
        for layer, kind in HIGGS_KIND.items():
            snap = _read_edgelist_gz(paths[f"higgs-{layer}_network.edgelist.gz"])
            m = int(max(snap.max(), ids.max())) + 1
            ref = np.unique(np.minimum(snap[0], snap[1]) * m + np.maximum(snap[0], snap[1]))
            ei = data.layers[layer].numpy()
            u, v = ids[ei[0]], ids[ei[1]]
            mine = np.unique(np.minimum(u, v) * m + np.maximum(u, v))
            assert np.isin(mine, ref).all(), f"rebuilt {layer} layer is not a subset of the SNAP layer"


# ---------------------------------------------------------------------------- API


def fetch(name: str, root: str = "./data") -> str:
    """Download and parse dataset `name` into its cached graph file.

    Returns the absolute path of the source folder `<root>/social/<source_of(name)>` (it holds
    raw/, graph.pt and, after `load_social`, processed/).

    Idempotent: raw files and graph.pt are reused when present. `name` is any `data.name` in
    `gnn_mech.social.SOCIAL_DATASETS` except social-tiny (which needs no download).
    """
    name = name.lower()
    if name not in DATASETS:
        raise ValueError(f"unknown social dataset {name!r}; expected one of {DATASETS}")
    path = graph_path(name, root)
    if osp.exists(path):         # delete graph.pt to force a re-parse (e.g. after a GRAPH_VERSION bump)
        return source_dir(name, root)
    raw = raw_dir(name, root)
    t0 = time.time()
    if name == "twitch-gamers":
        data = _build_twitch_gamers(raw)
    elif name.startswith("twitch-"):
        data = _build_musae(raw, name.split("-", 1)[1])
    elif name == "reddit-body":
        data = _build_reddit(raw)
    else:
        data = _build_higgs(raw)
    data.source = source_of(name)
    data.version = GRAPH_VERSION
    _save(data, path)
    print(f"{name}: {data.num_nodes} nodes, {data.edge_index.size(1) // 2} undirected edges, "
          f"parsed in {time.time() - t0:.1f}s -> {path}")
    return source_dir(name, root)


def load_graph(name: str, root: str = "./data") -> Data:
    """Full cached graph for `name` (fetching it if needed). Higgs: edge_index = the named layer."""
    name = name.lower()
    if name == "social-tiny":
        from .dataset import tiny_graph
        return tiny_graph()
    fetch(name, root)
    data = torch.load(graph_path(name, root), weights_only=False)
    if name.startswith("higgs-"):
        data.edge_index = data.layers[name.split("-", 1)[1]]
    data.dataset = name
    return data


# ----------------------------------------------------------- inputs and targets


def _log1p(v: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(v, 0.0))


def _one_hot(values: np.ndarray, min_count: int = 1) -> tuple[np.ndarray, list[str]]:
    cats, inv, counts = np.unique(values.astype(str), return_inverse=True, return_counts=True)
    keep = np.flatnonzero(counts >= min_count)
    out = np.zeros((values.size, keep.size))
    col = {c: j for j, c in enumerate(keep)}
    for c, j in col.items():
        out[inv == c, j] = 1.0
    return out, [str(cats[c]) for c in keep]


def target_columns(name: str) -> dict[str, str]:
    """{target name: task} available for dataset `name`."""
    name = name.lower()
    if name == "twitch-gamers":
        return {"mature": "binary", "affiliate": "binary", "dead_account": "binary", "views": "regression",
                "life_time": "regression", "language": "multiclass"}
    if name.startswith("twitch-"):
        return {"mature": "binary", "partner": "binary", "views": "regression", "days": "regression"}
    if name == "reddit-body":
        from ..reddit.data import FEATURE_NAMES
        return {c: "regression" for c in FEATURE_NAMES}
    if name.startswith("higgs-"):
        # activate_passive: the active_after label, supervised only on users with no actor activity
        # (RT/MT/RE out-count 0) before the cutoff, so their own history is uninformative and the
        # signal has to come from neighbours (diffusion-style task).
        return {"active_after": "binary", "activate_passive": "binary"}
    if name == "social-tiny":
        return {"label": "binary", "reg": "regression", "block": "multiclass"}
    raise ValueError(f"unknown social dataset {name!r}")


def node_target(name: str, graph: Data, target: str) -> tuple[np.ndarray, str]:
    """(y [N] float64 with nan for unlabelled nodes, or object array for multiclass; task)."""
    name = name.lower()
    tasks = target_columns(name)
    if target not in tasks:
        raise ValueError(f"target {target!r} not available for {name}; expected one of {sorted(tasks)}")
    task = tasks[target]
    if name.startswith("higgs-") and target == "activate_passive":
        a = graph.attrs
        y = np.asarray(a["active_after"], dtype=np.float64).copy()
        y[(a["rt_out_pre"] + a["mt_out_pre"] + a["re_out_pre"]) > 0] = np.nan   # active before -> unlabelled
        return y, task
    y = np.asarray(graph.attrs[target])
    if task == "multiclass":
        return y, task
    y = y.astype(np.float64).copy()
    if name == "reddit-body":
        y[~graph.attrs["is_source"]] = np.nan      # no outgoing posts -> no label
    return y, task


def node_inputs(name: str, graph: Data, target: str,
                fit_mask: Optional[np.ndarray] = None) -> tuple[np.ndarray, list[str]]:
    """Raw (unstandardized) input features [N, F] for predicting `target`, and their names.

    `fit_mask` marks the nodes whose values may be used to *choose* columns (train core
    nodes); only Reddit's correlation filter uses it.
    """
    name = name.lower()
    a = graph.attrs
    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(v: np.ndarray, label: str) -> None:
        cols.append(np.asarray(v, dtype=np.float64).reshape(len(v), -1))
        names.extend([label] if cols[-1].shape[1] == 1 else [f"{label}={i}" for i in range(cols[-1].shape[1])])

    if name == "twitch-gamers":
        # life_time = updated_at - created_at (days), so all three go when any of them is the target.
        drop = {target} | ({"life_time", "created_at", "updated_at"} if target == "life_time" else set())
        if "views" not in drop:
            add(_log1p(a["views"]), "log_views")
        for c in ("life_time", "created_at", "updated_at", "dead_account", "affiliate", "mature"):
            if c not in drop:
                add(a[c], c)
        if "language" not in drop:
            oh, langs = _one_hot(a["language"])
            cols.append(oh)
            names.extend(f"language={l}" for l in langs)
    elif name.startswith("twitch-"):
        n = graph.num_nodes
        indptr, indices = a["games_indptr"], a["games_indices"]
        counts = np.bincount(indices)
        top = np.argsort(-counts, kind="stable")[:MUSAE_TOP_K]
        top = top[counts[top] > 0]
        col_of = np.full(counts.size, -1)
        col_of[top] = np.arange(top.size)
        multi = np.zeros((n, top.size))
        rows = np.repeat(np.arange(n), np.diff(indptr))
        ok = col_of[indices] >= 0
        multi[rows[ok], col_of[indices[ok]]] = 1.0
        cols.append(multi)
        names.extend(f"game={g}" for g in top)
        add(np.diff(indptr).astype(float), "n_games")
        for c in ("days", "partner", "mature"):
            if c != target:
                add(a[c], c)
        if target != "views":
            add(_log1p(a["views"]), "log_views")
    elif name == "reddit-body":
        from ..reddit.data import FEATURE_NAMES
        from ..reddit.target_select import drop_correlated

        X = np.stack([a[c] for c in FEATURE_NAMES], axis=1)
        j = FEATURE_NAMES.index(target)
        others = np.array([k for k in range(len(FEATURE_NAMES)) if k != j])
        m = a["is_source"] if fit_mask is None else (np.asarray(fit_mask, bool) & a["is_source"])
        keep = others[drop_correlated(X[m][:, others], X[m, j], 0.9)]
        cols.append(X[:, keep])
        names.extend(FEATURE_NAMES[k] for k in keep)
        add(a["is_source"].astype(float), "is_source")
        add(_log1p(a["out_posts"]), "log_out_posts")
        add(_log1p(a["in_posts"]), "log_in_posts")
    elif name.startswith("higgs-"):
        for c in ("follow_out", "follow_in", "rt_out_pre", "rt_in_pre", "rt_deg_pre", "mt_out_pre", "mt_in_pre",
                  "mt_deg_pre", "re_out_pre", "re_in_pre", "re_deg_pre"):
            add(_log1p(a[c]), f"log_{c}")
    elif name == "social-tiny":
        X = a["features"]
        cols.append(X)
        names.extend(f"f{i}" for i in range(X.shape[1]))
    else:
        raise ValueError(f"unknown social dataset {name!r}")
    return np.concatenate(cols, axis=1), names


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="download + parse SNAP social graphs")
    p.add_argument("names", nargs="+", help=f"any of {DATASETS} (or 'all')")
    p.add_argument("--root", default="./data")
    args = p.parse_args(argv)
    names = DATASETS if args.names == ["all"] else args.names
    for name in names:
        fetch(name, args.root)


if __name__ == "__main__":
    main()
