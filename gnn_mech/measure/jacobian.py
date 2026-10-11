"""Vectorized Jacobian sensitivity ||d h_v / d h0_u||_F.

The Jacobian is taken w.r.t. the embedded input `h0` (not the raw categorical
codes), and all D output dimensions of a target node are differentiated in one
batched backward pass instead of one pass per hidden dimension.

One sweep of a target v yields d h_v / d h0_u for every source u; `jacobian_sweeps`
keeps those full per-target vectors (not just the sampled pairs) so the node-level
range measure (`measure/range.py`) costs no extra backward passes.
"""
from __future__ import annotations

import warnings

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


def _compute_target_jacobians(h: torch.Tensor, h0: torch.Tensor, targets: list[int],
                              method: str) -> dict[int, torch.Tensor]:
    if method == "loop":
        return _target_jacobians_loop(h, h0, targets)
    if method == "batched":
        return _target_jacobians_batched(h, h0, targets)
    if method == "auto":
        try:
            return _target_jacobians_batched(h, h0, targets)
        except RuntimeError as e:  # some ops lack batched-backward rules
            warnings.warn(f"batched Jacobian failed ({e}); falling back to per-dim loop")
            return _target_jacobians_loop(h, h0, targets)
    raise ValueError(f"unknown method {method!r}")


def target_influence(model, data, targets: list[int], device,
                     method: str = "auto") -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-source sensitivity of each target over *all* nodes of the single graph `data`.

    Returns {v: (fro, l1)}, two float64 arrays of length N: fro[u] = ||d h_v / d h0_u||_F
    and l1[u] = sum_ij |d h_v,i / d h0_u,j| (the entrywise L1 norm that Bamberger et al.
    use for the range measure). One Jacobian sweep per target gives every source at once.
    """
    if len(targets) == 0:
        return {}
    h0, h = _forward_from_h0(model, data, device)
    targets = sorted({int(v) for v in targets})
    jac = _compute_target_jacobians(h, h0, targets, method)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for v in targets:
        j = jac[v].detach()  # [D, N, D0]
        fro = j.pow(2).sum(dim=(0, 2)).sqrt()
        l1 = j.abs().sum(dim=(0, 2))
        out[v] = (fro.double().cpu().numpy(), l1.double().cpu().numpy())
    return out


def jacobian_norms(model, data, pairs: list[tuple[int, int]], device, method: str = "auto") -> np.ndarray:
    """||d h_v / d h0_u||_F for each (u, v) in `pairs` (node ids of the single graph `data`).

    `method`: "batched" (vectorized), "loop" (one backward per hidden dim), or
    "auto" (batched, falling back to the loop if batched gradients fail).
    """
    if len(pairs) == 0:
        return np.zeros(0)
    infl = target_influence(model, data, [int(v) for _, v in pairs], device, method=method)
    return np.array([infl[int(v)][0][int(u)] for u, v in pairs], dtype=float)


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


def sample_target_pairs(R: np.ndarray, n_targets: int, n_sources: int,
                        rng: np.random.Generator) -> list[tuple[int, int]]:
    """Pairs (u, v) sharing a few targets v, sources spread across R(., v) quantiles.

    One Jacobian sweep of a target gives its sensitivity to every source, so
    reusing targets yields n_targets * n_sources pairs for n_targets sweeps.
    """
    n = R.shape[0]
    targets = rng.choice(n, min(n_targets, n), replace=False)
    pairs: list[tuple[int, int]] = []
    for v in targets:
        others = np.array([u for u in range(n) if u != v])
        if len(others) <= n_sources:
            chosen = others
        else:
            order = others[np.argsort(R[others, v], kind="stable")]
            chosen = np.array([rng.choice(s) for s in np.array_split(order, n_sources)])
        pairs.extend((int(u), int(v)) for u in chosen)
    return pairs


def _graph_id(data, fallback: int) -> int:
    gid = getattr(data, "graph_id", None)
    if gid is None:
        return int(fallback)
    return int(gid.reshape(-1)[0]) if torch.is_tensor(gid) else int(gid)


def jacobian_sweeps(model, dataset, cfg: MeasureConfig, device, method: str = "auto",
                    min_lcc: int = 4) -> tuple[list[dict], list[dict]]:
    """Pair rows of `measure_jacobians` plus the full per-target sensitivity vectors.

    Returns (rows, targets). `targets` has one record per swept target v:
    {"graph_id", "v", "n_nodes", "lcc_size", "fro", "l1", "hops", "resistance"}, where
    the last four are arrays over the LCC nodes (in sorted node-id order; v itself is
    included with distance 0). They feed the node-level range measure (`measure/range.py`)
    at no extra backward passes. With targets_per_graph = 0 the targets are the `v`s of
    the independent pairs.
    """
    rng = np.random.default_rng(cfg.seed)
    n_graphs = min(cfg.n_graphs_jacobian, len(dataset))
    indices = rng.choice(len(dataset), n_graphs, replace=False)
    rows: list[dict] = []
    targets: list[dict] = []
    for gi in indices:
        data = dataset[int(gi)]
        n = int(data.num_nodes)
        ei = data.edge_index
        lcc = np.asarray(lcc_nodes(ei, n))
        if len(lcc) < min_lcc:
            continue
        R = effective_resistance_matrix(ei, n, nodes=lcc)
        hops = hop_distance_matrix(ei, n, nodes=lcc)
        if cfg.targets_per_graph > 0:
            local = sample_target_pairs(R, cfg.targets_per_graph, cfg.pairs_per_graph, rng)
        else:
            local = sample_stratified_pairs(R, cfg.pairs_per_graph, rng)
        pairs = [(int(lcc[a]), int(lcc[b])) for a, b in local]
        infl = target_influence(model, data, [v for _, v in pairs], device, method=method)
        gid = _graph_id(data, gi)
        for (a, b), (u, v) in zip(local, pairs):
            rows.append({"graph_id": gid, "u": u, "v": v, "resistance": float(R[a, b]),
                         "hops": int(hops[a, b]), "n_nodes": n, "lcc_size": int(len(lcc)),
                         "jacobian": float(infl[v][0][u])})
        local_of = {int(lcc[b]): b for _, b in local}
        for v in sorted(infl):
            b = local_of[v]
            fro, l1 = infl[v]
            targets.append({"graph_id": gid, "v": v, "n_nodes": n, "lcc_size": int(len(lcc)),
                            "fro": fro[lcc], "l1": l1[lcc], "hops": hops[b], "resistance": R[b]})
    return rows, targets


def measure_jacobians(model, dataset, cfg: MeasureConfig, device, method: str = "auto",
                      min_lcc: int = 4) -> list[dict]:
    """Jacobian norms for node pairs on the LCC of sampled graphs, stratified by resistance.

    See MeasureConfig for how targets_per_graph / pairs_per_graph choose pairs.

    Rows: {"graph_id", "u", "v", "resistance", "hops", "n_nodes", "lcc_size", "jacobian"}.
    `u`, `v` are node ids in the full graph; `n_nodes` is the full graph size.
    """
    return jacobian_sweeps(model, dataset, cfg, device, method=method, min_lcc=min_lcc)[0]
