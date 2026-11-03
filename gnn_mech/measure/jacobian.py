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
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Batch

from ..config import MeasureConfig
from .resistance import (approx_resistance_rows, effective_resistance_matrix, excess_resistance,
                         hop_distance_matrix, lcc_nodes, node_degrees)

# LCCs up to this size get the exact dense resistance matrix (O(n^3) time, a few n^2
# float64 buffers: about 1 s and 0.2 GB at n = 2,000, about 15 s and 1.5 GB at 5,000).
# Larger LCCs use the random-projection estimate for the swept targets' rows only
# (rows flagged res_exact = False; excess_res is then unreliable on dense graphs).
EXACT_RESISTANCE_MAX_NODES = 5000
APPROX_RESISTANCE_K = 256

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


def _pair_scale(model) -> int:
    """Largest (query, key) pair count of the model's last forward (sparse radius attention), or 0."""
    layers = model.attention_layers() if hasattr(model, "attention_layers") else []
    return max([int(getattr(m, "last_n_pairs", 0) or 0) for m in layers] or [0])


def _target_norms_batched(h: torch.Tensor, h0: torch.Tensor, targets: list[int],
                          scale: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """(fro^2, l1) [T, N]: sum over (output dim, input dim) of squared / absolute Jacobian entries.

    Rows (target, output dim) are differentiated in chunks of batched backward passes, and each
    chunk is reduced to per-source norms at once, so no [D, N, D0] Jacobian is ever held (it is
    0.6 GB per target on a 35k-node graph at hidden 64). The chunk size keeps rows * max(N, scale)
    * D0 under the device budget; `scale` is the attention pair count, since sparse attention's
    backward intermediates grow with the pairs, not the nodes.
    """
    n, d = h.shape
    per_row = max(n, int(scale)) * h0.shape[1]
    budget = _MAX_BATCH_ELEMENTS.get(h.device.type, _MAX_BATCH_ELEMENTS["cpu"])
    rows_per = max(1, budget // max(per_row, 1))
    rows = [(t, v, k) for t, v in enumerate(targets) for k in range(d)]
    fro2 = torch.zeros(len(targets), n, dtype=h.dtype, device=h.device)
    l1 = torch.zeros_like(fro2)
    for start in range(0, len(rows), rows_per):
        chunk = rows[start:start + rows_per]
        cot = torch.zeros(len(chunk), n, d, dtype=h.dtype, device=h.device)
        for i, (_, v, k) in enumerate(chunk):
            cot[i, v, k] = 1.0
        if len(chunk) == 1:
            (g,) = torch.autograd.grad(h, h0, grad_outputs=cot[0], retain_graph=True)
            g = g[None]
        else:
            (g,) = torch.autograd.grad(h, h0, grad_outputs=cot, retain_graph=True, is_grads_batched=True)
        tidx = torch.tensor([t for t, _, _ in chunk], device=h.device)
        g = g.reshape(len(chunk), n, -1)
        fro2.index_add_(0, tidx, g.pow(2).sum(-1))
        l1.index_add_(0, tidx, g.abs().sum(-1))
    return fro2, l1


def _target_norms_loop(h: torch.Tensor, h0: torch.Tensor, targets: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference for `_target_norms_batched`: one backward pass per (target, output dim)."""
    n = h.shape[0]
    fro2 = torch.zeros(len(targets), n, dtype=h.dtype, device=h.device)
    l1 = torch.zeros_like(fro2)
    for t, v in enumerate(targets):
        for k in range(h.shape[1]):
            (g,) = torch.autograd.grad(h[v, k], h0, retain_graph=True, allow_unused=True)
            if g is not None:
                fro2[t] += g.reshape(n, -1).pow(2).sum(-1)
                l1[t] += g.reshape(n, -1).abs().sum(-1)
    return fro2, l1


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
    if method == "loop":
        fro2, l1 = _target_norms_loop(h, h0, targets)
    elif method in ("batched", "auto"):
        try:
            fro2, l1 = _target_norms_batched(h, h0, targets, _pair_scale(model))
        except RuntimeError as e:  # some ops lack batched-backward rules
            if method == "batched":
                raise
            warnings.warn(f"batched Jacobian failed ({e}); falling back to per-dim loop")
            fro2, l1 = _target_norms_loop(h, h0, targets)
    else:
        raise ValueError(f"unknown method {method!r}")
    fro = fro2.sqrt().detach().double().cpu().numpy()
    l1 = l1.detach().double().cpu().numpy()
    return {v: (fro[t], l1[t]) for t, v in enumerate(targets)}


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


def _stratified_sources(r_v: np.ndarray, v: int, n_sources: int, rng: np.random.Generator) -> np.ndarray:
    """Sources u != v spread across the quantiles of r_v[u] = R(u, v)."""
    others = np.delete(np.arange(r_v.shape[0]), v)
    if len(others) <= n_sources:
        return others
    order = others[np.argsort(r_v[others], kind="stable")]
    return np.array([rng.choice(s) for s in np.array_split(order, n_sources)])


def sample_target_pairs(R: np.ndarray, n_targets: int, n_sources: int,
                        rng: np.random.Generator, candidates: Optional[np.ndarray] = None) -> list[tuple[int, int]]:
    """Pairs (u, v) sharing a few targets v, sources spread across R(., v) quantiles.

    One Jacobian sweep of a target gives its sensitivity to every source, so
    reusing targets yields n_targets * n_sources pairs for n_targets sweeps.
    `candidates` (indices into R) restricts the targets, e.g. to supervised nodes.
    """
    n = R.shape[0]
    if candidates is None:
        targets = rng.choice(n, min(n_targets, n), replace=False)
    else:
        targets = rng.choice(candidates, min(n_targets, len(candidates)), replace=False)
    pairs: list[tuple[int, int]] = []
    for v in targets:
        chosen = _stratified_sources(R[:, v], int(v), n_sources, rng)
        pairs.extend((int(u), int(v)) for u in chosen)
    return pairs


def _graph_id(data, fallback: int) -> int:
    gid = getattr(data, "graph_id", None)
    if gid is None:
        return int(fallback)
    return int(gid.reshape(-1)[0]) if torch.is_tensor(gid) else int(gid)


def _target_candidates(data, lcc: np.ndarray) -> Optional[np.ndarray]:
    """Positions into `lcc` of supervised nodes (data.y_mask), or None when every node qualifies."""
    mask = getattr(data, "y_mask", None)
    if mask is None:
        return None
    m = mask.detach().cpu().numpy().reshape(-1).astype(bool)[lcc]
    if m.all():
        return None
    return np.flatnonzero(m)


def _sample_local_pairs(ei, n: int, lcc: np.ndarray, cfg: MeasureConfig, rng: np.random.Generator,
                        candidates: Optional[np.ndarray]):
    """(local pairs into lcc, resistance-row lookup {b: R(., b)}, exact?) for one graph."""
    if len(lcc) <= EXACT_RESISTANCE_MAX_NODES:
        R = effective_resistance_matrix(ei, n, nodes=lcc)
        if cfg.targets_per_graph > 0:
            local = sample_target_pairs(R, cfg.targets_per_graph, cfg.pairs_per_graph, rng, candidates)
        else:
            local = sample_stratified_pairs(R, cfg.pairs_per_graph, rng)
        return local, {b: R[b] for b in sorted({b for _, b in local})}, True
    # Large LCC: no dense n x n matrices. Targets first, then their approximate R rows.
    # With targets_per_graph = 0, pairs_per_graph targets get one stratified source each.
    pool = np.arange(len(lcc)) if candidates is None else candidates
    n_t, n_s = (cfg.targets_per_graph, cfg.pairs_per_graph) if cfg.targets_per_graph > 0 \
        else (cfg.pairs_per_graph, 1)
    tgts = np.sort(rng.choice(pool, min(n_t, len(pool)), replace=False))
    warnings.warn(f"LCC of {len(lcc)} nodes > {EXACT_RESISTANCE_MAX_NODES}: approximate resistance "
                  f"(random projection, k={APPROX_RESISTANCE_K}) for {len(tgts)} targets")
    rows = approx_resistance_rows(ei, n, lcc, tgts, k=APPROX_RESISTANCE_K, rng=rng)
    r_of = {int(b): rows[i] for i, b in enumerate(tgts)}
    local = [(int(u), int(b)) for b in tgts for u in _stratified_sources(r_of[int(b)], int(b), n_s, rng)]
    return local, r_of, False


def jacobian_sweeps(model, dataset, cfg: MeasureConfig, device, method: str = "auto",
                    min_lcc: int = 4) -> tuple[list[dict], list[dict]]:
    """Pair rows of `measure_jacobians` plus the full per-target sensitivity vectors.

    Returns (rows, targets). `targets` has one record per swept target v:
    {"graph_id", "v", "n_nodes", "lcc_size", "fro", "l1", "hops", "resistance",
    "excess_res"}, where the last five are arrays over the LCC nodes (in sorted node-id
    order; v itself is included with distance 0). They feed the node-level range measure
    (`measure/range.py`) at no extra backward passes. With targets_per_graph = 0 the
    targets are the `v`s of the independent pairs.

    Node-level graphs with a `y_mask` sweep only supervised targets (targets_per_graph > 0);
    sources range over the whole LCC. Hop distances are computed for the target rows only.
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
        candidates = _target_candidates(data, lcc)
        if candidates is not None and len(candidates) == 0:
            continue
        local, r_of, exact = _sample_local_pairs(ei, n, lcc, cfg, rng, candidates)
        tloc = sorted(r_of)
        hop_rows = hop_distance_matrix(ei, n, nodes=lcc, sources=np.array(tloc, dtype=np.int64))
        hop_of = {b: hop_rows[i] for i, b in enumerate(tloc)}
        deg = node_degrees(ei, n)[lcc]
        pairs = [(int(lcc[a]), int(lcc[b])) for a, b in local]
        infl = target_influence(model, data, [v for _, v in pairs], device, method=method)
        gid = _graph_id(data, gi)
        for (a, b), (u, v) in zip(local, pairs):
            r = float(r_of[b][a])
            rows.append({"graph_id": gid, "u": u, "v": v, "resistance": r,
                         "excess_res": float(excess_resistance(r, deg[a], deg[b])),
                         "res_exact": exact, "hops": int(hop_of[b][a]), "n_nodes": n,
                         "lcc_size": int(len(lcc)), "jacobian": float(infl[v][0][u])})
        local_of = {int(lcc[b]): b for _, b in local}
        for v in sorted(infl):
            b = local_of[v]
            fro, l1 = infl[v]
            targets.append({"graph_id": gid, "v": v, "n_nodes": n, "lcc_size": int(len(lcc)),
                            "fro": fro[lcc], "l1": l1[lcc], "hops": hop_of[b], "resistance": r_of[b],
                            "excess_res": excess_resistance(r_of[b], deg, deg[b])})
    return rows, targets


def measure_jacobians(model, dataset, cfg: MeasureConfig, device, method: str = "auto",
                      min_lcc: int = 4) -> list[dict]:
    """Jacobian norms for node pairs on the LCC of sampled graphs, stratified by resistance.

    See MeasureConfig for how targets_per_graph / pairs_per_graph choose pairs.

    Rows: {"graph_id", "u", "v", "resistance", "excess_res", "res_exact", "hops", "n_nodes",
    "lcc_size", "jacobian"}. `u`, `v` are node ids in the full graph; `n_nodes` is the full
    graph size. excess_res = resistance - (1/deg_u + 1/deg_v), degrees in the graph `data`
    (the subgraph, for social datasets); res_exact is False for LCCs above
    EXACT_RESISTANCE_MAX_NODES (approximate resistance).
    """
    return jacobian_sweeps(model, dataset, cfg, device, method=method, min_lcc=min_lcc)[0]
