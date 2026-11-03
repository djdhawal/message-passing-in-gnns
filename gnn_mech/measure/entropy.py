"""Per-head attention entropy (attention dilution diagnostic).

Entropy is computed separately for every head. Averaging heads before taking
the entropy (as the pilot did) inflates it, since a mixture of sharp heads
looks diffuse.

Softmax attention (`GlobalAttn`): exact, over every query node of the graph.
Linear attention (`PerformerAttn`, any layer with `implicit_rows`): no attention
matrix exists, so each graph's entropy is the mean over at most
`cfg.entropy_queries` sampled query nodes of the exact implicit rows
phi(q_i).phi(K)^T / normaliser used by the forward pass.
Sparse k-hop attention (`RadiusAttn` on its pair-list path, full-graph runs): exact, over every
query node, from the cached per-pair weights (`sparse_entropy`); no dense matrix is formed.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
from torch_geometric.data import Batch

from ..config import MeasureConfig

ArrayLike = Union[torch.Tensor, np.ndarray]


def attention_entropy(weights: ArrayLike, mask: ArrayLike) -> np.ndarray:
    """Mean over real query nodes of -sum_j a_ij log a_ij over real keys.

    weights: [B, H, N, N] attention probabilities; mask: [B, N] bool (True = real node).
    Returns [B, H]. Uses 0 log 0 = 0; graphs with no real nodes get 0.
    """
    w = torch.as_tensor(weights).detach().to(torch.float64).cpu()
    m = torch.as_tensor(mask).detach().to(torch.bool).cpu()
    keys = m[:, None, None, :]                      # [B, 1, 1, N]
    w = torch.where(keys, w, torch.zeros_like(w))
    plogp = torch.where(w > 0, w * torch.log(w.clamp_min(1e-300)), torch.zeros_like(w))
    h_query = -plogp.sum(dim=-1)                    # [B, H, N]
    q = m[:, None, :].to(torch.float64)             # [B, 1, N]
    n_real = q.sum(dim=-1).clamp_min(1.0)           # [B, 1]
    return ((h_query * q).sum(dim=-1) / n_real).numpy()


def row_entropy(rows: ArrayLike) -> np.ndarray:
    """Mean over queries of -sum_j a_ij log a_ij for attention rows [Q, H, N]; returns [H]."""
    w = torch.as_tensor(rows).detach().to(torch.float64).cpu()
    plogp = torch.where(w > 0, w * torch.log(w.clamp_min(1e-300)), torch.zeros_like(w))
    return (-plogp.sum(dim=-1)).mean(dim=0).numpy()


def sparse_entropy(pairs: torch.Tensor, weights: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> np.ndarray:
    """Per-graph mean over query nodes of -sum_j a_ij log a_ij from pair weights.

    pairs [2, P] (query, key), weights [P, H] (each query's weights sum to 1), batch [N].
    Returns [n_graphs, H]."""
    w = weights.detach().to(torch.float64)
    plogp = torch.where(w > 0, w * torch.log(w.clamp_min(1e-300)), torch.zeros_like(w))
    n = batch.numel()
    h_query = torch.zeros(n, w.shape[1], dtype=torch.float64, device=w.device).index_add_(0, pairs[0], -plogp)
    per_graph = torch.zeros(n_graphs, w.shape[1], dtype=torch.float64, device=w.device).index_add_(0, batch, h_query)
    counts = torch.bincount(batch, minlength=n_graphs).clamp_min(1).to(torch.float64)
    return (per_graph / counts[:, None]).cpu().numpy()


# Dense softmax caches [B, H, Nmax, Nmax]: cap B * Nmax^2 per forward so large
# subgraphs (social, ~2k nodes) do not exhaust memory. Peptides batches are unaffected.
_MAX_DENSE_ELEMENTS = 16_000_000


def _chunks(graphs: list, batch_size: int, dense: bool) -> list[list[int]]:
    """Consecutive index groups of <= batch_size graphs (and <= the dense budget)."""
    groups: list[list[int]] = []
    cur: list[int] = []
    n_max = 0
    for i, g in enumerate(graphs):
        n = int(g.num_nodes)
        fits = len(cur) < batch_size and (not dense or (len(cur) + 1) * max(n_max, n) ** 2 <= _MAX_DENSE_ELEMENTS)
        if cur and not fits:
            groups.append(cur)
            cur, n_max = [], 0
        cur.append(i)
        n_max = max(n_max, n)
    if cur:
        groups.append(cur)
    return groups


def measure_entropy(model, dataset, cfg: MeasureConfig, device, batch_size: int = 64) -> list[dict]:
    """Per-graph, per-layer, per-head attention entropy on sampled graphs.

    Rows: {"graph_id", "n_nodes", "layer", "head", "entropy", "entropy_norm"},
    entropy_norm = entropy / log(n_nodes). Rows of linear-attention layers also carry
    "n_queries" (sampled queries, <= cfg.entropy_queries; query seed cfg.seed + 2).
    Graphs with < 2 nodes are skipped. Returns [] for models without attention.
    """
    layers = model.attention_layers()
    if not layers:
        return []
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(dataset), min(cfg.n_graphs_entropy, len(dataset)), replace=False)
    graphs = [dataset[int(i)] for i in indices]
    gids = [int(g.graph_id.reshape(-1)[0]) if getattr(g, "graph_id", None) is not None else int(i)
            for g, i in zip(graphs, indices)]
    implicit = [hasattr(a, "implicit_rows") for a in layers]
    q_rng = np.random.default_rng(cfg.seed + 2)
    n_queries = int(getattr(cfg, "entropy_queries", 64))

    rows: list[dict] = []
    model.eval()
    model.set_attention_cache(True)
    try:
        for group in _chunks(graphs, batch_size, dense=not all(implicit)):
            batch = Batch.from_data_list([graphs[i] for i in group]).to(device)
            with torch.no_grad():
                model(batch)
            n_nodes = torch.bincount(batch.batch, minlength=batch.num_graphs).cpu().numpy()
            starts = np.concatenate([[0], np.cumsum(n_nodes)[:-1]])
            queries = [np.sort(q_rng.choice(int(n), min(n_queries, int(n)), replace=False)) if n >= 2 else None
                       for n in n_nodes]
            for li, attn in enumerate(layers):
                if getattr(attn, "last_pair_weights", None) is not None:      # sparse radius attention
                    ent = list(sparse_entropy(attn.last_pairs, attn.last_pair_weights, batch.batch,
                                              batch.num_graphs))
                elif implicit[li]:
                    ent = [None if q is None else row_entropy(
                        attn.implicit_rows(torch.as_tensor(s + q, device=device))[:, :, s:s + int(n)])
                        for q, s, n in zip(queries, starts, n_nodes)]
                else:
                    dense = attention_entropy(attn.last_weights, attn.last_mask)   # [B, H]
                    ent = list(dense)
                for b, e_b in enumerate(ent):
                    n = int(n_nodes[b])
                    if n < 2:
                        continue
                    for hd in range(len(e_b)):
                        e = float(e_b[hd])
                        row = {"graph_id": gids[group[b]], "n_nodes": n, "layer": li,
                               "head": hd, "entropy": e, "entropy_norm": e / math.log(n)}
                        if implicit[li]:
                            row["n_queries"] = int(len(queries[b]))
                        rows.append(row)
    finally:
        model.set_attention_cache(False)
    return rows
