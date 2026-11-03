"""Disjoint partitions of a full graph into subgraphs, split assignment and halos.

Subgraphs are the unit of training and evaluation. A full graph with N nodes is cut into
P = max(3, round(N / subgraph_size)) disjoint parts:

    sampler="partition"  METIS (pymetis, k-way, seeded) when importable, otherwise BFS-grown parts
    sampler="bfs"        BFS-grown parts: from a random unassigned seed, add whole BFS layers of
                         unassigned nodes until the part has `size` nodes (the last layer is cut in
                         node order); a part whose component runs out restarts from a new seed

Whole parts are then assigned to train / val / test (`split_parts`), so the core node sets of
the three splits are disjoint and neither labels nor features of other splits are used...
unless `halo_hops > 0`: then each part is extended by the nodes within that many hops (from any
split, features only, y_mask False). Everything is deterministic given `seed`.

Full-graph (transductive) mode uses `label_parts`: the same METIS parts and split assignment, but
only as per-split label masks over the whole graph (no subgraphs are cut).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp


def csr_adjacency(edge_index, num_nodes: int) -> sp.csr_matrix:
    """Symmetric 0/1 CSR adjacency (edge_index is assumed undirected)."""
    ei = np.asarray(edge_index)
    a = sp.csr_matrix((np.ones(ei.shape[1], dtype=np.int8), (ei[0], ei[1])), shape=(num_nodes, num_nodes))
    a.sum_duplicates()
    a.data[:] = 1
    return a


def neighbors(indptr: np.ndarray, indices: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Concatenated neighbour lists of `nodes` (with repeats)."""
    starts, ends = indptr[nodes], indptr[nodes + 1]
    lens = ends - starts
    if lens.sum() == 0:
        return np.zeros(0, dtype=indices.dtype)
    offs = np.repeat(starts - np.cumsum(np.concatenate([[0], lens[:-1]])), lens)
    return indices[np.arange(lens.sum()) + offs]


def metis_available() -> bool:
    try:
        import pymetis  # noqa: F401
        return True
    except ImportError:
        return False


def metis_parts(adj: sp.csr_matrix, n_parts: int, seed: int) -> np.ndarray:
    import pymetis

    opts = pymetis.Options(seed=int(seed))
    _, parts = pymetis.part_graph(int(n_parts), xadj=adj.indptr.astype(np.int64), adjncy=adj.indices.astype(np.int64),
                                  options=opts)
    return np.asarray(parts, dtype=np.int64)


def bfs_parts(adj: sp.csr_matrix, size: int, seed: int) -> np.ndarray:
    """Disjoint BFS-grown parts of about `size` nodes each (pure numpy fallback for METIS)."""
    n = adj.shape[0]
    indptr, indices = adj.indptr, adj.indices
    rng = np.random.default_rng(seed)
    part = np.full(n, -1, dtype=np.int64)
    order = rng.permutation(n)        # candidate seeds, in random order
    ptr = 0
    pid = 0
    n_left = n
    while n_left > 0:
        target = min(size, n_left)
        if n_left - target < size // 2:   # absorb a small remainder into the last part
            target = n_left
        members = 0
        frontier = np.zeros(0, dtype=np.int64)
        while members < target:
            if frontier.size == 0:
                while part[order[ptr]] >= 0:
                    ptr += 1
                frontier = order[ptr:ptr + 1]
                part[frontier] = pid
                members += 1
                continue
            nb = np.unique(neighbors(indptr, indices, frontier))
            nb = nb[part[nb] < 0]
            nb = nb[:target - members]
            part[nb] = pid
            members += nb.size
            frontier = nb
        n_left -= members
        pid += 1
    return part


def partition(edge_index, num_nodes: int, size: int, seed: int = 0, sampler: str = "partition",
              use_metis: Optional[bool] = None) -> np.ndarray:
    """Part id per node (0..P-1), parts of about `size` nodes, every part non-empty."""
    adj = csr_adjacency(edge_index, num_nodes)
    n_parts = max(3, int(round(num_nodes / max(size, 1))))
    if sampler == "partition":
        if use_metis is None:
            use_metis = metis_available()
        parts = metis_parts(adj, n_parts, seed) if use_metis else bfs_parts(adj, int(np.ceil(num_nodes / n_parts)), seed)
    elif sampler == "bfs":
        parts = bfs_parts(adj, int(np.ceil(num_nodes / n_parts)), seed)
    else:
        raise ValueError(f"sampler must be 'partition' or 'bfs', got {sampler!r}")
    # relabel to 0..P-1 (METIS may leave a part empty)
    _, parts = np.unique(parts, return_inverse=True)
    return parts.astype(np.int64)


def split_parts(n_parts: int, val_frac: float, test_frac: float, seed: int,
                max_subgraphs: int = 0) -> dict[str, np.ndarray]:
    """Assign whole parts to splits (seeded); each split gets >= 1 part; optional cap per split."""
    if n_parts < 3:
        raise ValueError(f"need at least 3 parts for train/val/test, got {n_parts}")
    perm = np.random.default_rng([seed, 1]).permutation(n_parts)
    n_val = min(max(1, int(round(val_frac * n_parts))), n_parts - 2)
    n_test = min(max(1, int(round(test_frac * n_parts))), n_parts - 1 - n_val)
    out = {"val": perm[:n_val], "test": perm[n_val:n_val + n_test], "train": perm[n_val + n_test:]}
    if max_subgraphs > 0:
        out = {k: v[:max_subgraphs] for k, v in out.items()}
    return out


def halo(adj: sp.csr_matrix, core: np.ndarray, hops: int) -> np.ndarray:
    """Nodes within `hops` of `core`, excluding the core itself (sorted)."""
    if hops <= 0:
        return np.zeros(0, dtype=np.int64)
    seen = np.zeros(adj.shape[0], dtype=bool)
    seen[core] = True
    frontier, out = core, []
    for _ in range(hops):
        nb = np.unique(neighbors(adj.indptr, adj.indices, frontier))
        nb = nb[~seen[nb]]
        seen[nb] = True
        out.append(nb)
        frontier = nb
    return np.sort(np.concatenate(out)).astype(np.int64)


def bfs_ball(adj: sp.csr_matrix, center: int, max_nodes: int, allowed: Optional[np.ndarray] = None) -> np.ndarray:
    """BFS ball around `center` (whole layers until `max_nodes`, last layer cut), within `allowed` nodes."""
    ok = np.ones(adj.shape[0], dtype=bool) if allowed is None else np.zeros(adj.shape[0], dtype=bool)
    if allowed is not None:
        ok[allowed] = True
    seen = np.zeros(adj.shape[0], dtype=bool)
    seen[center] = True
    out, frontier, total = [np.array([center])], np.array([center]), 1
    while frontier.size and total < max_nodes:
        nb = np.unique(neighbors(adj.indptr, adj.indices, frontier))
        nb = nb[~seen[nb] & ok[nb]][:max_nodes - total]
        seen[nb] = True
        out.append(nb)
        total += nb.size
        frontier = nb
    return np.concatenate(out).astype(np.int64)


def induced_edges(adj: sp.csr_matrix, nodes: np.ndarray) -> np.ndarray:
    """[2, E] local edge_index of the subgraph induced on `nodes` (both directions, sorted)."""
    sub = adj[nodes][:, nodes].tocoo()
    order = np.lexsort((sub.col, sub.row))
    return np.stack([sub.row[order], sub.col[order]]).astype(np.int64)


def make_subgraphs(edge_index, num_nodes: int, size: int, seed: int = 0, sampler: str = "partition",
                   val_frac: float = 0.15, test_frac: float = 0.15, halo_hops: int = 0, max_subgraphs: int = 0,
                   use_metis: Optional[bool] = None) -> dict[str, list[dict]]:
    """{split: [{"nodes", "core", "edge_index", "part"}]}: nodes = core then halo (sorted within each),
    core = bool mask over nodes, edge_index local. Parts are ordered by part id within each split."""
    parts = partition(edge_index, num_nodes, size, seed, sampler, use_metis)
    adj = csr_adjacency(edge_index, num_nodes)
    assign = split_parts(int(parts.max()) + 1, val_frac, test_frac, seed, max_subgraphs)
    members = np.argsort(parts, kind="stable")
    bounds = np.searchsorted(parts[members], np.arange(parts.max() + 2))
    out: dict[str, list[dict]] = {}
    for split in ("train", "val", "test"):
        rows = []
        for p in np.sort(assign[split]):
            core = members[bounds[p]:bounds[p + 1]]
            h = halo(adj, core, halo_hops)
            nodes = np.concatenate([core, h])
            mask = np.zeros(nodes.size, dtype=bool)
            mask[:core.size] = True
            rows.append({"nodes": nodes, "core": mask, "edge_index": induced_edges(adj, nodes), "part": int(p)})
        out[split] = rows
    return out


def label_parts(edge_index, num_nodes: int, size: int, seed: int = 0, val_frac: float = 0.15,
                test_frac: float = 0.15, max_subgraphs: int = 0, use_metis: Optional[bool] = None) -> dict:
    """Labelled-node masks for full-graph (transductive) training.

    Same parts (`partition` with sampler "partition") and split assignment (`split_parts`) as
    `make_subgraphs`, so a split seed labels the same nodes in both modes. Returns
    {"part": part id per node [N], "assign": {split: part ids}, "mask": {split: bool [N]}}; the
    masks are disjoint (and cover every node unless `max_subgraphs` caps the parts per split).
    """
    parts = partition(edge_index, num_nodes, size, seed, "partition", use_metis)
    assign = split_parts(int(parts.max()) + 1, val_frac, test_frac, seed, max_subgraphs)
    masks = {split: np.isin(parts, assign[split]) for split in ("train", "val", "test")}
    return {"part": parts, "assign": assign, "mask": masks}
