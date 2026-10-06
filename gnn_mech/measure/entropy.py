"""Per-head attention entropy (attention dilution diagnostic).

Entropy is computed separately for every head. Averaging heads before taking
the entropy (as the pilot did) inflates it, since a mixture of sharp heads
looks diffuse.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
from torch_geometric.loader import DataLoader

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


def measure_entropy(model, dataset, cfg: MeasureConfig, device, batch_size: int = 64) -> list[dict]:
    """Per-graph, per-layer, per-head attention entropy on sampled graphs.

    Rows: {"graph_id", "n_nodes", "layer", "head", "entropy", "entropy_norm"},
    entropy_norm = entropy / log(n_nodes). Graphs with < 2 nodes are skipped.
    Returns [] for models without attention.
    """
    layers = model.attention_layers()
    if not layers:
        return []
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(dataset), min(cfg.n_graphs_entropy, len(dataset)), replace=False)
    graphs = [dataset[int(i)] for i in indices]
    gids = [int(g.graph_id.reshape(-1)[0]) if getattr(g, "graph_id", None) is not None else int(i)
            for g, i in zip(graphs, indices)]

    rows: list[dict] = []
    model.eval()
    model.set_attention_cache(True)
    try:
        offset = 0
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            with torch.no_grad():
                model(batch)
            n_nodes = torch.bincount(batch.batch, minlength=batch.num_graphs).cpu().numpy()
            for li, attn in enumerate(layers):
                ent = attention_entropy(attn.last_weights, attn.last_mask)  # [B, H]
                for b in range(ent.shape[0]):
                    n = int(n_nodes[b])
                    if n < 2:
                        continue
                    for hd in range(ent.shape[1]):
                        e = float(ent[b, hd])
                        rows.append({"graph_id": gids[offset + b], "n_nodes": n, "layer": li,
                                     "head": hd, "entropy": e, "entropy_norm": e / math.log(n)})
            offset += batch.num_graphs
    finally:
        model.set_attention_cache(False)
    return rows
