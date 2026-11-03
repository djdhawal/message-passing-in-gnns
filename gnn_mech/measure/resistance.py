"""Effective resistance, largest connected component and hop distances.

All functions take a PyG-style `edge_index` (LongTensor or array, shape [2, E])
and treat the graph as undirected and unweighted (duplicate edges and self loops
are ignored). Matrices are returned in the order of `nodes`, which defaults to
the sorted node ids of the largest connected component (LCC).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.sparse.linalg import splu


def _edges_np(edge_index) -> np.ndarray:
    if hasattr(edge_index, "detach"):
        edge_index = edge_index.detach().cpu().numpy()
    return np.asarray(edge_index, dtype=np.int64).reshape(2, -1)


def _adjacency(edge_index, num_nodes: int) -> sp.csr_matrix:
    """Symmetric 0/1 adjacency without self loops."""
    ei = _edges_np(edge_index)
    keep = ei[0] != ei[1]
    row, col = ei[0][keep], ei[1][keep]
    a = sp.coo_matrix((np.ones(row.size), (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    a = a + a.T
    a.data[:] = 1.0
    return a.tocsr()


def lcc_nodes(edge_index, num_nodes: int) -> np.ndarray:
    """Sorted node ids of the largest connected component (ties: lowest label)."""
    if num_nodes == 0:
        return np.zeros(0, dtype=np.int64)
    _, labels = connected_components(_adjacency(edge_index, num_nodes), directed=False)
    counts = np.bincount(labels)
    return np.flatnonzero(labels == int(np.argmax(counts))).astype(np.int64)


def _sub_adjacency(edge_index, num_nodes: int, nodes: Optional[np.ndarray]) -> tuple[sp.csr_matrix, np.ndarray]:
    if nodes is None:
        nodes = lcc_nodes(edge_index, num_nodes)
    nodes = np.asarray(nodes, dtype=np.int64)
    a = _adjacency(edge_index, num_nodes)
    return a[nodes][:, nodes].tocsr(), nodes


def effective_resistance_matrix(edge_index, num_nodes: int, nodes: Optional[np.ndarray] = None) -> np.ndarray:
    """R = diag(L+) + diag(L+)^T - 2 L+ on the subgraph induced by `nodes` (default LCC).

    Uses (L + J/n)^-1, which equals L+ + J/n for a connected graph (the J/n term cancels
    in R); falls back to pinv if the induced subgraph is disconnected (R between
    components is then not meaningful).
    """
    a, nodes = _sub_adjacency(edge_index, num_nodes, nodes)
    n = nodes.size
    if n == 0:
        return np.zeros((0, 0))
    lap = np.diag(np.asarray(a.sum(axis=1)).ravel()) - a.toarray()
    n_comp, _ = connected_components(a, directed=False)
    if n_comp == 1:
        g = np.linalg.inv(lap + np.full((n, n), 1.0 / n))
    else:
        g = np.linalg.pinv(lap, hermitian=True)
    d = np.diag(g)
    r = d[:, None] + d[None, :] - 2.0 * g
    np.fill_diagonal(r, 0.0)
    return np.maximum(r, 0.0)


def hop_distance_matrix(edge_index, num_nodes: int, nodes: Optional[np.ndarray] = None,
                        sources: Optional[np.ndarray] = None) -> np.ndarray:
    """Unweighted shortest-path hop counts on the subgraph induced by `nodes` (default LCC).

    Returns [len(nodes), len(nodes)], or only the rows of `sources` (positions into
    `nodes`) as [len(sources), len(nodes)]: one BFS per source instead of per node.
    Unreachable pairs are `inf`.
    """
    a, nodes = _sub_adjacency(edge_index, num_nodes, nodes)
    if sources is not None:
        sources = np.asarray(sources, dtype=np.int64).reshape(-1)
        if nodes.size == 0 or sources.size == 0:
            return np.zeros((sources.size, nodes.size))
        return shortest_path(a, method="D", directed=False, unweighted=True, indices=sources)
    if nodes.size == 0:
        return np.zeros((0, 0))
    return shortest_path(a, method="D", directed=False, unweighted=True)


def node_degrees(edge_index, num_nodes: int) -> np.ndarray:
    """Number of distinct neighbours of each node (undirected, self loops ignored)."""
    return np.asarray(_adjacency(edge_index, num_nodes).sum(axis=1)).ravel()


def excess_resistance(R, deg_u, deg_v):
    """R_uv - (1/d_u + 1/d_v): resistance beyond its large-graph degree limit (von Luxburg et al. 2014)."""
    return np.asarray(R, dtype=float) - (1.0 / np.asarray(deg_u, dtype=float) + 1.0 / np.asarray(deg_v, dtype=float))


# approx_resistance_rows: Laplacian solves by sparse LU up to this many nodes, else by block
# Jacobi-preconditioned conjugate gradients (LU fill-in on a 35k-node social graph takes ~5 min,
# CG ~10 s for k = 256 right-hand sides; both agree to the CG tolerance). Peak memory of the CG
# path is a few n x k float64 arrays (~70 MB each at n = 35k, k = 256) plus one sign chunk.
LU_MAX_NODES = 10_000
CG_TOL = 1e-6
CG_MAX_ITER = 2000
_RHS_CHUNK = 50_000          # edges per chunk of random signs (CG path): 50k x 256 float64 = 100 MB


def _block_pcg(lap: sp.csr_matrix, y: np.ndarray, tol: float = CG_TOL, max_iter: int = CG_MAX_ITER) -> np.ndarray:
    """Solve L z = y column-wise (y [n, k], columns sum to 0) by Jacobi-preconditioned CG; returns the
    mean-zero solution (L+ y for a connected graph). Columns are iterated together (one sparse
    matmul per step) and each stops updating once its relative residual is below `tol`."""
    deg = lap.diagonal()
    dinv = np.where(deg > 0, 1.0 / np.maximum(deg, 1e-300), 0.0)[:, None]
    x = np.zeros_like(y)
    r = y.copy()
    b0 = np.maximum(np.linalg.norm(y, axis=0), 1e-300)
    z = dinv * r
    p = z.copy()
    rz = (r * z).sum(0)
    for _ in range(max_iter):
        active = np.linalg.norm(r, axis=0) / b0 > tol
        if not active.any():
            break
        ap = lap @ p
        pap = (p * ap).sum(0)
        alpha = np.where(active & (pap > 0), rz / np.where(pap > 0, pap, 1.0), 0.0)
        x += alpha * p
        r -= alpha * ap
        z = dinv * r
        rz_new = (r * z).sum(0)
        beta = np.where(active, rz_new / np.where(rz > 0, rz, 1.0), 0.0)
        p = z + beta * p
        rz = rz_new
    return x - x.mean(0, keepdims=True)


def approx_resistance_rows(edge_index, num_nodes: int, nodes: np.ndarray, sources: np.ndarray,
                           k: int = 256, rng: Optional[np.random.Generator] = None,
                           solver: str = "auto") -> np.ndarray:
    """Spielman-Srivastava random-projection estimate of R(sources, nodes), [len(sources), len(nodes)].

    Z = L+ B^T Q^T / sqrt(k) (Q: k x m random signs, B: edge incidence) is obtained from k
    Laplacian solves; R_uv ~ ||Z_u - Z_v||^2. The relative error per pair is O(sqrt(1/k))
    (about 10% at k = 256), so this is for graphs too large for `effective_resistance_matrix`,
    not for excess resistance on dense graphs. `nodes` must induce a connected subgraph;
    `sources` are positions into `nodes`. solver: "lu" (sparse LU of the grounded Laplacian),
    "cg" (block PCG, `_block_pcg`) or "auto" (LU up to LU_MAX_NODES nodes, else CG).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    a, nodes = _sub_adjacency(edge_index, num_nodes, nodes)
    n = nodes.size
    sources = np.asarray(sources, dtype=np.int64).reshape(-1)
    if n < 2:
        return np.zeros((sources.size, n))
    up = sp.triu(a, k=1).tocoo()
    m = up.nnz
    if solver == "auto":
        solver = "lu" if n <= LU_MAX_NODES else "cg"
    if solver == "lu":       # one draw of all m x k signs (the original stream; small graphs)
        signs = rng.choice([-1.0, 1.0], size=(m, k)) / np.sqrt(k)    # rows of Q^T, one per edge
        y = np.zeros((n, k))
        np.add.at(y, up.row, signs)
        np.add.at(y, up.col, -signs)                                 # Y = B^T Q^T / sqrt(k)
    else:                    # large graphs: signs in edge chunks, never the whole m x k array (1 GB at m = 528k)
        y = np.zeros((n, k))
        for s0 in range(0, m, _RHS_CHUNK):
            r, c = up.row[s0:s0 + _RHS_CHUNK], up.col[s0:s0 + _RHS_CHUNK]
            signs = rng.choice([-1.0, 1.0], size=(r.size, k)) / np.sqrt(k)
            ar = np.arange(r.size)
            inc = sp.csr_matrix((np.concatenate([np.ones(r.size), -np.ones(r.size)]),
                                 (np.concatenate([ar, ar]), np.concatenate([r, c]))), shape=(r.size, n))
            y += np.asarray(inc.T @ signs)
    lap = (sp.diags(np.asarray(a.sum(axis=1)).ravel()) - a).tocsr()
    if solver == "lu":
        z = np.zeros((n, k))
        z[:-1] = splu(lap[:-1, :-1].tocsc()).solve(y[:-1])           # ground the last node at 0
    elif solver == "cg":
        z = _block_pcg(lap, y)
    else:
        raise ValueError(f"solver must be 'auto', 'lu' or 'cg', got {solver!r}")
    sq = np.einsum("nk,nk->n", z, z)
    r = np.maximum(sq[sources][:, None] + sq[None, :] - 2.0 * z[sources] @ z.T, 0.0)
    r[np.arange(sources.size), sources] = 0.0
    return r


def avg_effective_resistance(edge_index, num_nodes: int) -> float:
    """Mean effective resistance over unordered node pairs of the LCC; nan if LCC < 2."""
    nodes = lcc_nodes(edge_index, num_nodes)
    if nodes.size < 2:
        return float("nan")
    r = effective_resistance_matrix(edge_index, num_nodes, nodes)
    iu = np.triu_indices(nodes.size, k=1)
    return float(r[iu].mean())
