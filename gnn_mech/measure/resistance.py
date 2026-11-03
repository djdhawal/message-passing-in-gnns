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


def approx_resistance_rows(edge_index, num_nodes: int, nodes: np.ndarray, sources: np.ndarray,
                           k: int = 256, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Spielman-Srivastava random-projection estimate of R(sources, nodes), [len(sources), len(nodes)].

    Z = L+ B^T Q^T / sqrt(k) (Q: k x m random signs, B: edge incidence) is obtained from k
    solves with one sparse LU of the grounded Laplacian; R_uv ~ ||Z_u - Z_v||^2. The
    relative error per pair is O(sqrt(1/k)) (about 10% at k = 256), so this is for graphs
    too large for `effective_resistance_matrix`, not for excess resistance on dense graphs.
    `nodes` must induce a connected subgraph; `sources` are positions into `nodes`.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    a, nodes = _sub_adjacency(edge_index, num_nodes, nodes)
    n = nodes.size
    sources = np.asarray(sources, dtype=np.int64).reshape(-1)
    if n < 2:
        return np.zeros((sources.size, n))
    up = sp.triu(a, k=1).tocoo()
    m = up.nnz
    signs = rng.choice([-1.0, 1.0], size=(m, k)) / np.sqrt(k)        # rows of Q^T, one per edge
    y = np.zeros((n, k))
    np.add.at(y, up.row, signs)
    np.add.at(y, up.col, -signs)                                     # Y = B^T Q^T / sqrt(k)
    lap = (sp.diags(np.asarray(a.sum(axis=1)).ravel()) - a).tocsc()
    z = np.zeros((n, k))
    z[:-1] = splu(lap[:-1, :-1].tocsc()).solve(y[:-1])               # ground the last node at 0
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
