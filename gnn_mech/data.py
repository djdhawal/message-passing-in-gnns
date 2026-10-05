"""Dataset loading, RWSE positional encodings and cached per-graph structure stats.

Every graph is a PyG `Data` with Long categorical `x` / `edge_attr`, float `y`
of shape [1, out_dim], RWSE `pe` (when `pe_dim > 0`) and `graph_id` (LongTensor [1],
index of the graph within its split). See docs/phase1_plan.md for the contract.

Datasets:
    "Peptides-func" / "Peptides-struct": LRGB via `torch_geometric.datasets.LRGBDataset`.
        Processed files go to `<root>/<name>/processed_pe<pe_dim>` so caches built with
        different RWSE settings never collide (PyG only warns on a pre_transform mismatch).
        Raw downloads are shared across PE settings.
    "synthetic-tiny": small seeded random molecule-like graphs, no download.
"""
from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Optional

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import LRGBDataset
from torch_geometric.transforms import AddRandomWalkPE

from .config import DataConfig
from .measure.resistance import (
    avg_effective_resistance,
    hop_distance_matrix,
    lcc_nodes,
)

# OGB `get_atom_feature_dims()` / `get_bond_feature_dims()` (hardcoded; ogb not required).
OGB_ATOM_VOCAB: list[int] = [119, 5, 12, 12, 10, 6, 6, 2, 2]
OGB_BOND_VOCAB: list[int] = [5, 6, 2]

_LRGB = {
    "peptides-func": (10, "multilabel"),
    "peptides-struct": (11, "regression"),
}
_SYNTHETIC_SIZES = {"train": 128, "val": 32, "test": 32}
SPLITS = ("train", "val", "test")


@dataclass
class DataInfo:
    node_vocab: list[int]     # vocabulary size of each categorical node column
    edge_vocab: list[int]     # vocabulary size of each categorical edge column
    out_dim: int
    pe_dim: int
    task: str                 # "multilabel" (AP metric) or "regression" (MAE metric)


# --------------------------------------------------------------------------- LRGB


class PeptidesDataset(LRGBDataset):
    """LRGBDataset with a PE-specific processed dir and `graph_id` attached on access."""

    def __init__(self, root: str, name: str, split: str, pe_dim: int):
        self.pe_dim = pe_dim
        pre = AddRandomWalkPE(walk_length=pe_dim, attr_name="pe") if pe_dim > 0 else None
        super().__init__(root, name=name, split=split, pre_transform=pre)

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, f"processed_pe{self.pe_dim}")

    def process(self) -> None:
        # Per-graph RWSE is many tiny dense matmuls; intra-op threading only adds overhead
        # (and thrashes badly on a busy machine), so process single-threaded.
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            super().process()
        finally:
            torch.set_num_threads(n_threads)

    def get(self, idx: int) -> Data:
        data = super().get(idx)
        data.graph_id = torch.tensor([idx], dtype=torch.long)
        return data


# --------------------------------------------------------------------- synthetic


class GraphListDataset(Dataset):
    """In-memory list of `Data` objects (used for synthetic-tiny)."""

    def __init__(self, graphs: list[Data]):
        super().__init__()
        self._graphs = graphs

    def len(self) -> int:
        return len(self._graphs)

    def get(self, idx: int) -> Data:
        return self._graphs[idx].clone()


# OGB atomic-number index (atomic_num - 1): C, N, O, S
_ATOMS = np.array([5, 6, 7, 15])
_ATOM_P = np.array([0.65, 0.17, 0.15, 0.03])


def _random_molecule(rng: np.random.Generator) -> tuple[int, list[tuple[int, int]], bool]:
    """Backbone chain, optional ring closures, random side branches. Always connected."""
    n = int(rng.integers(8, 31))
    n_back = int(rng.integers(max(4, n // 2), n + 1))
    edges = [(i, i + 1) for i in range(n_back - 1)]
    has_ring = False
    for _ in range(int(rng.integers(0, 3))):
        size = int(rng.integers(5, 7))
        if n_back > size:
            s = int(rng.integers(0, n_back - size + 1))
            edges.append((s, s + size - 1))
            has_ring = True
    for v in range(n_back, n):
        edges.append((int(rng.integers(0, v)), v))
    return n, edges, has_ring


def _synthetic_graph(rng: np.random.Generator, transform: Optional[AddRandomWalkPE], gid: int) -> Data:
    n, edges, has_ring = _random_molecule(rng)
    e = np.array(edges, dtype=np.int64).T
    edge_index = np.concatenate([e, e[::-1]], axis=1)

    x = np.zeros((n, len(OGB_ATOM_VOCAB)), dtype=np.int64)
    x[:, 0] = rng.choice(_ATOMS, size=n, p=_ATOM_P)
    for c in range(1, len(OGB_ATOM_VOCAB)):
        x[:, c] = rng.integers(0, min(OGB_ATOM_VOCAB[c], 4), size=n)
    m = e.shape[1]
    ea = np.stack([rng.integers(0, 4, m), np.zeros(m, dtype=np.int64), rng.integers(0, 2, m)], axis=1)
    edge_attr = np.concatenate([ea, ea], axis=0)

    deg = np.bincount(edge_index[0], minlength=n)
    diam = hop_distance_matrix(e, n, np.arange(n)).max()
    atoms = x[:, 0]
    y = np.array([
        has_ring,
        n > 18,
        (atoms == 6).sum() >= 3,
        (atoms == 15).any(),
        diam > 12,
        (deg >= 3).sum() >= 3,
        (atoms == 7).sum() >= 3,
        (ea[:, 0] == 1).sum() >= 4,
        diam > 8 and has_ring,
        rng.random() < 0.3,                       # pure noise label
    ], dtype=np.float32)[None, :]

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(np.ascontiguousarray(edge_index)),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(y),
        num_nodes=n,
    )
    if transform is not None:
        data = transform(data)
    data.graph_id = torch.tensor([gid], dtype=torch.long)
    return data


def _synthetic_split(split: str, pe_dim: int, seed: int = 0) -> GraphListDataset:
    rng = np.random.default_rng([seed, SPLITS.index(split)])
    pe = AddRandomWalkPE(walk_length=pe_dim, attr_name="pe") if pe_dim > 0 else None
    return GraphListDataset([_synthetic_graph(rng, pe, i) for i in range(_SYNTHETIC_SIZES[split])])


# -------------------------------------------------------------------------- API


def load_dataset(cfg: DataConfig) -> tuple[Dataset, Dataset, Dataset, DataInfo]:
    """Return (train, val, test, info) for `cfg.name`."""
    key = cfg.name.lower()
    if key == "synthetic-tiny":
        tr, va, te = (_synthetic_split(s, cfg.pe_dim) for s in SPLITS)
        info = DataInfo(list(OGB_ATOM_VOCAB), list(OGB_BOND_VOCAB), 10, cfg.pe_dim, "multilabel")
        return tr, va, te, info
    if key in _LRGB:
        out_dim, task = _LRGB[key]
        tr, va, te = (PeptidesDataset(cfg.root, cfg.name, s, cfg.pe_dim) for s in SPLITS)
        info = DataInfo(list(OGB_ATOM_VOCAB), list(OGB_BOND_VOCAB), out_dim, cfg.pe_dim, task)
        return tr, va, te, info
    raise ValueError(f"unknown dataset {cfg.name!r}; expected Peptides-func, Peptides-struct or synthetic-tiny")


STAT_KEYS = ("n_nodes", "n_edges", "avg_resistance", "diameter", "lcc_size")


def _graph_stats(args: tuple[np.ndarray, int]) -> tuple[float, ...]:
    ei, n = args
    lo, hi = np.minimum(ei[0], ei[1]), np.maximum(ei[0], ei[1])
    n_edges = np.unique((lo * n + hi)[lo != hi]).size
    lcc = lcc_nodes(ei, n)
    diam = float(hop_distance_matrix(ei, n, lcc).max()) if lcc.size > 0 else float("nan")
    return float(n), float(n_edges), avg_effective_resistance(ei, n), diam, float(lcc.size)


def _single_thread() -> None:
    threadpool_limits(limits=1)


def structure_stats(dataset: Dataset, split: str, cfg: DataConfig) -> dict[str, np.ndarray]:
    """Per-graph structure stats indexed by graph_id, cached as .npz.

    Keys: n_nodes, n_edges (undirected), avg_resistance (mean over LCC pairs),
    diameter (of the LCC), lcc_size. Arrays have length max(graph_id) + 1; graphs
    missing from `dataset` (e.g. a subset) are nan.
    """
    cache_dir = cfg.structure_cache or osp.join(cfg.root, "structure")
    path = osp.join(cache_dir, f"{cfg.name}_{split}.npz")

    items, gids = [], []
    for i in range(len(dataset)):
        d = dataset[i]
        gids.append(int(d.graph_id.view(-1)[0]) if hasattr(d, "graph_id") else i)
        items.append((d.edge_index.numpy(), int(d.num_nodes)))
    size = max(gids) + 1 if gids else 0

    if osp.exists(path):
        with np.load(path) as z:
            cached = {k: z[k] for k in STAT_KEYS if k in z}
        if len(cached) == len(STAT_KEYS) and all(len(v) == size for v in cached.values()):
            if not np.isnan(cached["n_nodes"][gids]).any():
                return cached

    # Small per-graph matrices: multithreaded BLAS is far slower than one thread per process.
    with threadpool_limits(limits=1):
        if len(items) >= 256:
            with Pool(min(4, os.cpu_count() or 1), initializer=_single_thread) as pool:
                rows = pool.map(_graph_stats, items, chunksize=32)
        else:
            rows = [_graph_stats(it) for it in items]

    out = {k: np.full(size, np.nan) for k in STAT_KEYS}
    for gid, row in zip(gids, rows):
        for k, v in zip(STAT_KEYS, row):
            out[k][gid] = v
    os.makedirs(cache_dir, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)
    return out
