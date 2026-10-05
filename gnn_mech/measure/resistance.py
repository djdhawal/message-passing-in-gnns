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


def hop_distance_matrix(edge_index, num_nodes: int, nodes: Optional[np.ndarray] = None) -> np.ndarray:
    """Unweighted shortest-path hop counts on the subgraph induced by `nodes` (default LCC).

    Unreachable pairs are `inf`.
    """
    a, nodes = _sub_adjacency(edge_index, num_nodes, nodes)
    if nodes.size == 0:
        return np.zeros((0, 0))
    return shortest_path(a, method="D", directed=False, unweighted=True)


def avg_effective_resistance(edge_index, num_nodes: int) -> float:
    """Mean effective resistance over unordered node pairs of the LCC; nan if LCC < 2."""
    nodes = lcc_nodes(edge_index, num_nodes)
    if nodes.size < 2:
        return float("nan")
    r = effective_resistance_matrix(edge_index, num_nodes, nodes)
    iu = np.triu_indices(nodes.size, k=1)
    return float(r[iu].mean())
