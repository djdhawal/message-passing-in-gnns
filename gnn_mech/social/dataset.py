"""Social subgraph datasets: `load_social(cfg) -> (train_ds, val_ds, test_ds, DataInfo)`.

Pipeline (docs/social_plan.md, build contract S0-S2):

1. `sources.load_graph(name, root)`: the cached full graph (downloaded and parsed on first use).
2. `subgraph.make_subgraphs`: disjoint parts of ~`subgraph_size` nodes (METIS / BFS-grown),
   whole parts assigned to train / val / test with `split_seed`, optional halo (`halo_hops`).
3. Inputs `sources.node_inputs` and target `sources.node_target` for `data.target` (default per
   dataset in `sources.DEFAULT_TARGETS`). x is standardized with the mean / std of the **train core
   nodes** only. Regression targets: log1p when the train labels are non-negative with skew > 1, then
   standardized with train-labelled-node stats. Multiclass targets: classes 0..C-1 (sorted labels).
4. Per subgraph `Data`: x [N, F] float32, pe [N, pe_dim] float32 (RWSE, when pe_dim > 0), edge_index
   [2, E] long (local, undirected, no self loops), edge_attr [E, 1] long zeros, y [N] float32 (long for
   multiclass; 0 where unlabelled), y_mask [N] bool (core nodes with a label), node_id [N] long (index
   in the full graph), graph_id [1] long (part id, unique across splits), num_nodes.

The processed result is cached at `<root>/social/<source>/processed/<key>.pt`, where key hashes every
DataConfig field that changes the data (all but root, structure_cache, num_workers) plus
`PROCESS_VERSION`, so repeated loads are a single torch.load.

Full-graph mode (`data.sampler = "full"`, transductive): the METIS parts (`subgraph_size`,
`split_seed`, `val_frac`, `test_frac`, `max_subgraphs`) only decide **which nodes are labelled** in
each split (`subgraph.label_parts`). Each split's dataset is a one-element list holding the **whole
graph** (same schema as above: node_id = arange(N), graph_id [0]) whose `y_mask` marks that split's
labelled nodes, so message passing and attention see every node and edge while only train labels
are trained on. x is standardized with the mean / std of the **train-labelled** nodes. `halo_hops`
is ignored. RWSE: the dense eigh is not used above `RWSE_DENSE_MAX_NODES` nodes; there
`rwse_sampled` gives a Monte-Carlo estimate (`RWSE_WALKS` walks per node, std error <= 0.5 /
sqrt(walks)). The full-graph configs use pe_dim 0, so this is only a fallback.

`social-tiny` is a seeded synthetic SBM (600 nodes, 6 blocks, 8 features) with a binary target
`label` that depends on the node's own features and on its neighbours' (plus `reg`, `block`), used
by tests and CPU smoke runs; no download.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import os.path as osp
import time
from typing import Optional

import numpy as np
import torch
from scipy.stats import skew
from torch_geometric.data import Data, Dataset

from . import sources
from .subgraph import csr_adjacency, label_parts, make_subgraphs

PROCESS_VERSION = 1
FULL_SAMPLER = "full"
RWSE_DENSE_MAX_NODES = 5000
RWSE_WALKS = 256
_NON_DATA_FIELDS = ("root", "structure_cache", "num_workers")


class SubgraphDataset(Dataset):
    """In-memory list of subgraph `Data` objects."""

    def __init__(self, graphs: list[Data]):
        super().__init__()
        self._graphs = graphs

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx].clone()


# ------------------------------------------------------------------------- helpers


def data_key(cfg) -> str:
    """Hash of the data-defining DataConfig fields (+ PROCESS_VERSION); defaults resolved."""
    d = dataclasses.asdict(cfg)
    for k in _NON_DATA_FIELDS:
        d.pop(k, None)
    d["name"] = d["name"].lower()
    d["target"] = d.get("target") or sources.DEFAULT_TARGETS[d["name"]]
    d["_version"] = PROCESS_VERSION
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def processed_path(cfg) -> str:
    return osp.join(sources.source_dir(cfg.name, cfg.root), "processed", f"{data_key(cfg)}.pt")


def rwse(edge_index: np.ndarray, n: int, k: int) -> np.ndarray:
    """Random-walk structural encoding [n, k]: diag(P^1..P^k), P = D^-1 A (as PyG's AddRandomWalkPE).

    Uses the symmetric similar matrix S = D^-1/2 A D^-1/2 (same diagonal powers) and one eigh,
    which is much cheaper than k dense matmuls for subgraphs of a few thousand nodes.
    """
    if k <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    a = np.zeros((n, n))
    a[edge_index[0], edge_index[1]] = 1.0
    deg = a.sum(1)
    dis = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1)), 0.0)
    s = dis[:, None] * a * dis[None, :]
    lam, u = np.linalg.eigh(s)
    u2 = u ** 2
    pows = lam[:, None] ** np.arange(1, k + 1)[None, :]
    pe = u2 @ pows
    pe[deg == 0] = 0.0
    return pe.astype(np.float32)


def rwse_sampled(edge_index: np.ndarray, n: int, k: int, walks: int = RWSE_WALKS, seed: int = 0) -> np.ndarray:
    """Monte-Carlo RWSE [n, k] for large graphs: the fraction of `walks` simple random walks from
    each node that are back at their start after t = 1..k steps (an unbiased estimate of
    diag(P^t)). Isolated nodes get 0. Cost O(n * walks * k), memory O(n)."""
    if k <= 0:
        return np.zeros((n, 0), dtype=np.float32)
    adj = csr_adjacency(edge_index, n)
    indptr, indices = adj.indptr, adj.indices
    deg = np.diff(indptr)
    rng = np.random.default_rng(seed)
    start = np.arange(n)
    hits = np.zeros((n, k))
    has = deg > 0
    for _ in range(walks):
        pos = start.copy()
        for t in range(k):
            d = deg[pos]
            step = indptr[pos] + (rng.random(n) * np.maximum(d, 1)).astype(np.int64)
            pos = np.where(d > 0, indices[np.minimum(step, len(indices) - 1)], pos)
            hits[:, t] += pos == start
    pe = hits / walks
    pe[~has] = 0.0
    return pe.astype(np.float32)


def tiny_graph(n_blocks: int = 6, block: int = 100, n_feat: int = 8, seed: int = 0) -> Data:
    """Synthetic SBM for tests: label = 1[own f1 + mean of neighbours' f0 > 0]."""
    rng = np.random.default_rng(seed)
    n = n_blocks * block
    b = np.repeat(np.arange(n_blocks), block)
    p = np.where(b[:, None] == b[None, :], 0.06, 0.002)
    up = np.triu(rng.random((n, n)) < p, 1)
    src, dst = np.nonzero(up)
    ei = sources.undirected(src, dst, n)
    x = rng.normal(size=(n, n_feat))
    x[:, 4] += 0.5 * (b - n_blocks / 2)          # block signal
    a = np.zeros((n, n))
    a[ei[0].numpy(), ei[1].numpy()] = 1.0
    deg = np.maximum(a.sum(1), 1)
    nb = (a @ x) / deg[:, None]
    label = ((x[:, 1] + 2.0 * nb[:, 0]) > 0).astype(np.float64)
    reg = np.exp(0.5 * (x[:, 3] + 2.0 * nb[:, 2]))      # skewed, positive -> log1p path
    attrs = {"features": x, "label": label, "reg": reg, "block": b.astype(str)}
    g = Data(edge_index=ei, num_nodes=n, attrs=attrs)
    g.source = g.dataset = "social-tiny"
    return g


def _standardize_stats(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = v.mean(0)
    sd = v.std(0)
    return mu, np.where(sd > 1e-12, sd, 1.0)


def _save(obj, path: str) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _data_info(out_dim: int, pe_dim: int, task: str, feat_dim: int):
    from ..data import DataInfo

    try:
        return DataInfo(node_vocab=[], edge_vocab=[1], out_dim=out_dim, pe_dim=pe_dim, task=task,
                        task_level="node", feat_dim=feat_dim)
    except TypeError:     # DataInfo without the social fields (older pipeline): attach them
        info = DataInfo([], [1], out_dim, pe_dim, task)
        info.task_level, info.feat_dim = "node", feat_dim
        return info


# ---------------------------------------------------------------------------- build


def build_social(cfg) -> dict:
    """Build the split subgraphs (no caching); returns {"splits": {split: [Data]}, "info": {...}, "meta": {...}}."""
    t0 = time.time()
    name = cfg.name.lower()
    target = getattr(cfg, "target", "") or sources.DEFAULT_TARGETS[name]
    graph = sources.load_graph(name, cfg.root)
    n = int(graph.num_nodes)
    sampler = getattr(cfg, "sampler", "partition")
    if sampler == FULL_SAMPLER:
        return _build_full(cfg, name, target, graph, t0)
    subs = make_subgraphs(graph.edge_index.numpy(), n, size=getattr(cfg, "subgraph_size", 2000),
                          seed=getattr(cfg, "split_seed", 0), sampler=sampler,
                          val_frac=getattr(cfg, "val_frac", 0.15), test_frac=getattr(cfg, "test_frac", 0.15),
                          halo_hops=getattr(cfg, "halo_hops", 0), max_subgraphs=getattr(cfg, "max_subgraphs", 0))
    train_core = np.zeros(n, dtype=bool)
    for s in subs["train"]:
        train_core[s["nodes"][s["core"]]] = True

    y_raw, task = sources.node_target(name, graph, target)
    x_raw, feat_names = sources.node_inputs(name, graph, target, fit_mask=train_core)
    mu, sd = _standardize_stats(x_raw[train_core])
    x = ((x_raw - mu) / sd).astype(np.float32)

    meta: dict = {"dataset": name, "target": target, "task": task, "feat_names": feat_names,
                  "n_nodes_full": n, "n_edges_full": int(graph.edge_index.size(1) // 2)}
    y, has, out_dim = _encode_target(y_raw, task, train_core, meta)
    return _finish_subgraphs(cfg, subs, x, y, has, out_dim, task, meta, t0)


def _encode_target(y_raw, task: str, train_core: np.ndarray, meta: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """(y [N] float32 / int64, has-label [N] bool, out_dim); regression stats from train-labelled nodes."""
    n = len(train_core)
    if task == "multiclass":
        labels = np.asarray(y_raw).astype(str)
        classes = np.unique(labels)
        y = np.searchsorted(classes, labels).astype(np.int64)
        has = np.ones(n, dtype=bool)
        meta["classes"] = classes.tolist()
        out_dim = int(classes.size)
    else:
        y = np.asarray(y_raw, dtype=np.float64)
        has = np.isfinite(y)
        out_dim = 1
        if task == "regression":
            tr = y[train_core & has]
            log = bool(tr.min() >= 0 and skew(tr) > 1)
            if log:
                y = np.log1p(np.where(has, y, 0.0))
            ymu, ysd = _standardize_stats(y[train_core & has])
            y = (y - ymu) / ysd
            meta.update(y_log1p=log, y_mean=float(ymu), y_std=float(ysd))
        y = np.where(has, y, 0.0).astype(np.float32)
    return y, has, out_dim


def _finish_subgraphs(cfg, subs, x, y, has, out_dim, task, meta, t0) -> dict:
    pe_dim = int(cfg.pe_dim)
    splits: dict[str, list[Data]] = {}
    for split, rows in subs.items():
        out = []
        for s in rows:
            nodes, ei = s["nodes"], s["edge_index"]
            d = Data(x=torch.from_numpy(x[nodes]), edge_index=torch.from_numpy(ei),
                     edge_attr=torch.zeros((ei.shape[1], 1), dtype=torch.long),
                     y=torch.from_numpy(y[nodes]),
                     y_mask=torch.from_numpy(s["core"] & has[nodes]),
                     node_id=torch.from_numpy(nodes.astype(np.int64)),
                     graph_id=torch.tensor([s["part"]], dtype=torch.long), num_nodes=int(nodes.size))
            if pe_dim > 0:
                d.pe = torch.from_numpy(rwse(ei, int(nodes.size), pe_dim))
            out.append(d)
        splits[split] = out

    for split, graphs in splits.items():
        m = torch.cat([g.y_mask for g in graphs]) if graphs else torch.zeros(0, dtype=torch.bool)
        yy = torch.cat([g.y for g in graphs])[m] if graphs else torch.zeros(0)
        meta[split] = {"n_subgraphs": len(graphs), "n_nodes": int(sum(g.num_nodes for g in graphs)),
                       "n_labelled": int(m.sum()), "n_edges": int(sum(g.edge_index.size(1) for g in graphs) // 2)}
        if task == "binary" and yy.numel():
            meta[split]["pos_rate"] = float(yy.float().mean())
    meta["seconds"] = round(time.time() - t0, 1)
    info = {"out_dim": out_dim, "pe_dim": pe_dim, "task": task, "feat_dim": int(x.shape[1])}
    return {"splits": splits, "info": info, "meta": meta}


def _build_full(cfg, name: str, target: str, graph, t0: float) -> dict:
    """Full-graph (transductive) splits: one whole-graph `Data` per split, y_mask = that split's labelled nodes."""
    n = int(graph.num_nodes)
    ei = graph.edge_index.numpy()
    lab = label_parts(ei, n, size=getattr(cfg, "subgraph_size", 2000), seed=getattr(cfg, "split_seed", 0),
                      val_frac=getattr(cfg, "val_frac", 0.15), test_frac=getattr(cfg, "test_frac", 0.15),
                      max_subgraphs=getattr(cfg, "max_subgraphs", 0))
    train_core = lab["mask"]["train"]
    y_raw, task = sources.node_target(name, graph, target)
    x_raw, feat_names = sources.node_inputs(name, graph, target, fit_mask=train_core)
    meta: dict = {"dataset": name, "target": target, "task": task, "feat_names": feat_names, "sampler": FULL_SAMPLER,
                  "n_nodes_full": n, "n_edges_full": int(ei.shape[1] // 2), "n_parts": int(lab["part"].max()) + 1}
    y, has, out_dim = _encode_target(y_raw, task, train_core, meta)
    fit = train_core & has
    mu, sd = _standardize_stats(x_raw[fit])
    x = torch.from_numpy(((x_raw - mu) / sd).astype(np.float32))

    pe_dim = int(cfg.pe_dim)
    pe = None
    if pe_dim > 0:
        pe = rwse(ei, n, pe_dim) if n <= RWSE_DENSE_MAX_NODES else rwse_sampled(ei, n, pe_dim)
        pe = torch.from_numpy(pe)
        meta["rwse"] = "dense" if n <= RWSE_DENSE_MAX_NODES else f"sampled ({RWSE_WALKS} walks)"
    edge_index = torch.from_numpy(np.ascontiguousarray(ei, dtype=np.int64))
    edge_attr = torch.zeros((ei.shape[1], 1), dtype=torch.long)
    y_t = torch.from_numpy(y)
    node_id = torch.arange(n, dtype=torch.long)
    splits: dict[str, list[Data]] = {}
    for split in ("train", "val", "test"):
        mask = lab["mask"][split] & has
        d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_t, y_mask=torch.from_numpy(mask),
                 node_id=node_id, graph_id=torch.zeros(1, dtype=torch.long), num_nodes=n)
        if pe is not None:
            d.pe = pe
        splits[split] = [d]
        meta[split] = {"n_subgraphs": 1, "n_parts": int(len(lab["assign"][split])), "n_nodes": n,
                       "n_labelled": int(mask.sum()), "n_edges": int(ei.shape[1] // 2)}
        if task == "binary" and mask.any():
            meta[split]["pos_rate"] = float(y[mask].mean())
    meta["seconds"] = round(time.time() - t0, 1)
    info = {"out_dim": out_dim, "pe_dim": pe_dim, "task": task, "feat_dim": int(x.shape[1])}
    return {"splits": splits, "info": info, "meta": meta}


def load_social(cfg):
    """(train_ds, val_ds, test_ds, DataInfo) for a social `DataConfig` (cached; see module docstring)."""
    name = cfg.name.lower()
    if name not in sources.DEFAULT_TARGETS:
        raise ValueError(f"unknown social dataset {cfg.name!r}")
    path = processed_path(cfg)
    if osp.exists(path):
        blob = torch.load(path, weights_only=False)
    else:
        blob = build_social(cfg)
        _save(blob, path)
        with open(osp.splitext(path)[0] + ".json", "w") as f:   # human-readable record of what was built
            json.dump({"config": dataclasses.asdict(cfg), **{k: v for k, v in blob["meta"].items()
                                                              if k != "feat_names"},
                       "feat_names": blob["meta"]["feat_names"]}, f, indent=1, default=str)
    i = blob["info"]
    info = _data_info(i["out_dim"], i["pe_dim"], i["task"], i["feat_dim"])
    tr, va, te = (SubgraphDataset(blob["splits"][s]) for s in ("train", "val", "test"))
    return tr, va, te, info


def load_meta(cfg) -> dict:
    """Build summary (sizes, label balance, timing, transforms) of the cached processed file."""
    load_social(cfg)
    return torch.load(processed_path(cfg), weights_only=False)["meta"]
