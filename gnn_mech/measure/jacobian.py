"""Vectorized Jacobian sensitivity ||d h_v / d h0_u||_F.

The Jacobian is taken w.r.t. the embedded input `h0` (not the raw categorical
codes), and all D output dimensions of a target node are differentiated in one
batched backward pass instead of one pass per hidden dimension.
"""
from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Batch

from ..config import MeasureConfig
from .resistance import effective_resistance_matrix, hop_distance_matrix, lcc_nodes

# Upper bound on elements in one batched cotangent tensor ([rows, N, D]). On CPU
# the backward is compute-bound and small chunks are fastest (about one target
# per chunk for a 150-node graph at hidden 96); on GPU large chunks amortize
# kernel launches.
_MAX_BATCH_ELEMENTS = {"cpu": 2_000_000, "cuda": 50_000_000}


def _single_graph_batch(data, device) -> Batch:
    return Batch.from_data_list([data]).to(device)


def _forward_from_h0(model, data, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (h0 leaf with grad, h) for a one-graph batch in eval mode."""
    model.eval()
    batch = _single_graph_batch(data, device)
    with torch.no_grad():
        h0 = model.embed_inputs(batch)
    h0 = h0.detach().requires_grad_(True)
    h = model.node_embeddings_from_h0(h0, batch)
    return h0, h


def _target_jacobians_batched(h: torch.Tensor, h0: torch.Tensor, targets: list[int]) -> dict[int, torch.Tensor]:
    """Full Jacobians {v: [D, N, D0]} of h[v] w.r.t. h0, chunked batched backward passes."""
    n, d = h.shape
    per_target = d * h0.numel()
    budget = _MAX_BATCH_ELEMENTS.get(h.device.type, _MAX_BATCH_ELEMENTS["cpu"])
    chunk = max(1, budget // per_target)
    out: dict[int, torch.Tensor] = {}
    eye = torch.eye(d, dtype=h.dtype, device=h.device)
    for start in range(0, len(targets), chunk):
        vs = targets[start:start + chunk]
        cot = torch.zeros(len(vs), d, n, d, dtype=h.dtype, device=h.device)
        for i, v in enumerate(vs):
            cot[i, :, v, :] = eye
        (grads,) = torch.autograd.grad(h, h0, grad_outputs=cot.reshape(-1, n, d),
                                       retain_graph=True, is_grads_batched=True)
        grads = grads.reshape(len(vs), d, *h0.shape)
        for i, v in enumerate(vs):
            out[v] = grads[i]
    return out


def _target_jacobians_loop(h: torch.Tensor, h0: torch.Tensor, targets: list[int]) -> dict[int, torch.Tensor]:
    """Slow reference: one backward pass per (target, output dim)."""
    out: dict[int, torch.Tensor] = {}
    for v in targets:
        rows = [torch.autograd.grad(h[v, k], h0, retain_graph=True, allow_unused=True)[0]
                for k in range(h.shape[1])]
        out[v] = torch.stack([torch.zeros_like(h0) if r is None else r for r in rows])
    return out


def jacobian_norms(model, data, pairs: list[tuple[int, int]], device, method: str = "auto") -> np.ndarray:
    """||d h_v / d h0_u||_F for each (u, v) in `pairs` (node ids of the single graph `data`).

    `method`: "batched" (vectorized), "loop" (one backward per hidden dim), or
    "auto" (batched, falling back to the loop if batched gradients fail).
    """
    if len(pairs) == 0:
        return np.zeros(0)
    h0, h = _forward_from_h0(model, data, device)
    by_target: dict[int, list[int]] = defaultdict(list)
    for i, (_, v) in enumerate(pairs):
        by_target[int(v)].append(i)
    targets = sorted(by_target)

    if method == "loop":
        jac = _target_jacobians_loop(h, h0, targets)
    elif method == "batched":
        jac = _target_jacobians_batched(h, h0, targets)
    elif method == "auto":
        try:
            jac = _target_jacobians_batched(h, h0, targets)
        except RuntimeError as e:  # some ops lack batched-backward rules
            warnings.warn(f"batched Jacobian failed ({e}); falling back to per-dim loop")
            jac = _target_jacobians_loop(h, h0, targets)
    else:
        raise ValueError(f"unknown method {method!r}")

    out = np.zeros(len(pairs))
    for v, idx in by_target.items():
        j = jac[v]  # [D, N, D0]
        us = torch.tensor([int(pairs[i][0]) for i in idx], device=j.device)
        norms = j[:, us, :].pow(2).sum(dim=(0, 2)).sqrt()
        out[idx] = norms.detach().double().cpu().numpy()
    return out


def sample_stratified_pairs(R: np.ndarray, n_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Distinct unordered pairs (i != j) spread across the quantiles of R.

    Pairs are ranked by resistance and split into `n_pairs` equal-count strata,
    one pair is drawn per stratum (ties cannot empty a stratum). Each pair's
    orientation (which node is the source u) is randomized.
    """
    n = R.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    if len(iu) <= n_pairs:
        chosen = np.arange(len(iu))
    else:
        order = np.argsort(R[iu, ju], kind="stable")
        chosen = np.array([rng.choice(s) for s in np.array_split(order, n_pairs)])
    flip = rng.random(len(chosen)) < 0.5
    return [(int(ju[k]), int(iu[k])) if f else (int(iu[k]), int(ju[k])) for k, f in zip(chosen, flip)]


def _graph_id(data, fallback: int) -> int:
    gid = getattr(data, "graph_id", None)
    if gid is None:
        return int(fallback)
    return int(gid.reshape(-1)[0]) if torch.is_tensor(gid) else int(gid)


def measure_jacobians(model, dataset, cfg: MeasureConfig, device, method: str = "auto",
                      min_lcc: int = 4) -> list[dict]:
    """Jacobian norms for stratified node pairs on the LCC of sampled graphs.

    Rows: {"graph_id", "u", "v", "resistance", "hops", "n_nodes", "lcc_size", "jacobian"}.
    `u`, `v` are node ids in the full graph; `n_nodes` is the full graph size.
    """
    rng = np.random.default_rng(cfg.seed)
    n_graphs = min(cfg.n_graphs_jacobian, len(dataset))
    indices = rng.choice(len(dataset), n_graphs, replace=False)
    rows: list[dict] = []
    for gi in indices:
        data = dataset[int(gi)]
        n = int(data.num_nodes)
        ei = data.edge_index
        lcc = np.asarray(lcc_nodes(ei, n))
        if len(lcc) < min_lcc:
            continue
        R = effective_resistance_matrix(ei, n, nodes=lcc)
        hops = hop_distance_matrix(ei, n, nodes=lcc)
        local = sample_stratified_pairs(R, cfg.pairs_per_graph, rng)
        pairs = [(int(lcc[a]), int(lcc[b])) for a, b in local]
        J = jacobian_norms(model, data, pairs, device, method=method)
        gid = _graph_id(data, gi)
        for (a, b), (u, v), j in zip(local, pairs, J):
            rows.append({"graph_id": gid, "u": u, "v": v, "resistance": float(R[a, b]),
                         "hops": int(hops[a, b]), "n_nodes": n, "lcc_size": int(len(lcc)),
                         "jacobian": float(j)})
    return rows
