"""Range measure (Bamberger et al., "On Measuring Long-Range Interactions in Graph
Neural Networks", ICML 2025, arXiv 2506.05971).

Definitions, transcribed from the official implementation
(github.com/BenGutteridge/range-measure, `longrange/range.py::compute_range` and
`lrgb_exps/graphgps/train/eval_range.py`; the arXiv page itself is not reachable
from the build sandbox):

    Influence matrix. For node-level outputs y_v in R^K and inputs x_u in R^D,
        I_vu = sum_{k, d} | d y_v,k / d x_u,d |            (entrywise L1 of the Jacobian block)
    For graph-level outputs y in R^K the paper uses the Hessian,
        I_uw = sum_{k, d, d'} | d^2 y_k / (d x_u,d  d x_w,d') |
    Each row is a node's influence distribution over the other nodes (itself included).

    Range of node v under a distance d(., .):
        rho_v       = sum_u I_vu d(v, u)                          (unnormalized)
        rho_v^norm  = sum_u I_vu d(v, u) / sum_u I_vu             (normalized)
    The graph's range is the mean of rho_v over its nodes; a dataset's range is the
    mean over graphs. d is shortest-path distance ("spd") or effective resistance
    ("res").

On LRGB graph-level tasks (Peptides) the paper's main results use the node-level
Jacobian of the last layer *before pooling* (`use_hessian=False` in eval_range.py);
the Hessian of the pooled output is used for its synthetic graph-level tasks. Both
are implemented here:

* Node-level (`node_range_rows`): rho^norm from the full per-target Jacobian vectors
  that `jacobian.jacobian_sweeps` already computes (no extra backward passes). Inputs
  are the embedded features h0 (as for every measurement in this package), outputs the
  final node representations. Fields `range_hops` / `range_res` are the paper-faithful
  L1 version; `range_hops_fro` / `range_res_fro` weight by the Frobenius norm used in
  the Jacobian pair rows. `total_influence` = sum_u I_vu recovers the unnormalized rho.
* Graph-level (`measure_graph_range`): rho^norm of sampled source nodes u from the
  Hessian of the pooled logits w.r.t. h0, via batched Hessian-vector products (one
  batch per chunk of sources gives that source's mixed partials against every w).
  Caveat: for piecewise-linear networks (ReLU everywhere, mean pooling, no attention,
  i.e. alpha = 0 and the GCN baseline) the Hessian is zero almost everywhere, so the
  graph-level range is undefined (reported as None and counted in `n_zero`). Only
  softmax attention makes it nonzero here, which is also why the node-level version is
  the one comparable across models.

Distances are restricted to the graph's largest connected component (LCC), as in the
Jacobian measurement; the paper's Peptides graphs are almost all connected.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch
from scipy.stats import pearsonr
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch_geometric.data import Batch

from ..config import MeasureConfig
from .resistance import effective_resistance_matrix, hop_distance_matrix, lcc_nodes

# Upper bound on elements of one batched second-derivative result ([rows, N, D0]).
_MAX_BATCH_ELEMENTS = {"cpu": 2_000_000, "cuda": 50_000_000}


def normalized_range(weights: np.ndarray, dist: np.ndarray) -> Optional[float]:
    """sum_u w_u d_u / sum_u w_u; None if all weights are zero (range undefined)."""
    w = np.asarray(weights, dtype=float)
    d = np.asarray(dist, dtype=float)
    keep = np.isfinite(d)
    total = w[keep].sum()
    if not total > 0:
        return None
    return float((w[keep] * d[keep]).sum() / total)


def node_range_rows(targets: list[dict]) -> list[dict]:
    """One row per swept target from `jacobian.jacobian_sweeps` target records.

    Rows: {"graph_id", "v", "n_nodes", "lcc_size", "range_hops", "range_res",
    "range_hops_fro", "range_res_fro", "total_influence"}.
    """
    rows = []
    for t in targets:
        rows.append({
            "graph_id": t["graph_id"], "v": t["v"], "n_nodes": t["n_nodes"], "lcc_size": t["lcc_size"],
            "range_hops": normalized_range(t["l1"], t["hops"]),
            "range_res": normalized_range(t["l1"], t["resistance"]),
            "range_hops_fro": normalized_range(t["fro"], t["hops"]),
            "range_res_fro": normalized_range(t["fro"], t["resistance"]),
            "total_influence": float(np.sum(t["l1"])),
        })
    return rows


# ------------------------------------------------------------------ graph level (Hessian)


def _graph_output_from_h0(model, data, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(h0 leaf with grad, graph logits [K]) for a single graph in eval mode."""
    model.eval()
    batch = Batch.from_data_list([data]).to(device)
    with torch.no_grad():
        h0 = model.embed_inputs(batch)
    h0 = h0.detach().requires_grad_(True)
    # Fused attention kernels (flash / memory-efficient, used by nn.MultiheadAttention)
    # have no double backward; the math backend is plain differentiable ops.
    with sdpa_kernel(SDPBackend.MATH):
        y = model.forward_from_h0(h0, batch)
    return h0, y.reshape(-1)


def _first_grads(y: torch.Tensor, h0: torch.Tensor) -> Optional[torch.Tensor]:
    """[K, N, D0] gradients of each output with create_graph=True; None if y ignores h0."""
    if not y.requires_grad:
        return None
    rows = []
    for k in range(y.numel()):
        (g,) = torch.autograd.grad(y[k], h0, retain_graph=True, create_graph=True, allow_unused=True)
        rows.append(torch.zeros_like(h0) if g is None else g)
    return torch.stack(rows)


def _second_grads(sel: torch.Tensor, h0: torch.Tensor, method: str) -> torch.Tensor:
    """d sel[r] / d h0 for every entry r of `sel` (flattened), as [R, N, D0]."""
    flat = sel.reshape(-1)
    r = flat.numel()
    if not sel.requires_grad:   # first derivative constant in h0: Hessian is zero
        return torch.zeros(r, *h0.shape, dtype=h0.dtype, device=h0.device)
    if method == "loop":
        out = []
        for i in range(r):
            (g,) = torch.autograd.grad(flat[i], h0, retain_graph=True, allow_unused=True)
            out.append(torch.zeros_like(h0) if g is None else g)
        return torch.stack(out)
    budget = _MAX_BATCH_ELEMENTS.get(h0.device.type, _MAX_BATCH_ELEMENTS["cpu"])
    chunk = max(1, budget // h0.numel())
    out = []
    for start in range(0, r, chunk):
        idx = torch.arange(start, min(start + chunk, r), device=h0.device)
        cot = torch.zeros(len(idx), r, dtype=flat.dtype, device=flat.device)
        cot[torch.arange(len(idx)), idx] = 1.0
        (g,) = torch.autograd.grad(flat, h0, grad_outputs=cot, retain_graph=True,
                                   is_grads_batched=True, allow_unused=True)
        out.append(torch.zeros(len(idx), *h0.shape, dtype=h0.dtype, device=h0.device) if g is None else g)
    return torch.cat(out)


def hessian_influence(model, data, sources: list[int], device, method: str = "auto") -> dict[int, np.ndarray]:
    """{u: I_u [N]} with I_uw = sum_{k, d, d'} |d^2 y_k / d h0_u,d d h0_w,d'| for the graph logits y.

    `method`: "batched" (Hessian-vector products batched over rows), "loop" (one
    double-backward per row), or "auto" (batched, falling back to the loop).
    """
    if len(sources) == 0:
        return {}
    sources = sorted({int(u) for u in sources})
    h0, y = _graph_output_from_h0(model, data, device)
    n = h0.shape[0]
    G = _first_grads(y, h0)
    if G is None:
        return {u: np.zeros(n) for u in sources}
    k, _, d0 = G.shape
    sel = G[:, sources, :]                       # [K, S, D0]
    if method == "auto":
        try:
            H = _second_grads(sel, h0, "batched")
        except RuntimeError as e:  # some ops lack batched double-backward rules
            warnings.warn(f"batched Hessian failed ({e}); falling back to per-row loop")
            H = _second_grads(sel, h0, "loop")
    elif method in ("batched", "loop"):
        H = _second_grads(sel, h0, method)
    else:
        raise ValueError(f"unknown method {method!r}")
    H = H.detach().reshape(k, len(sources), d0, n, d0).abs().sum(dim=(0, 2, 4))  # [S, N]
    return {u: H[i].double().cpu().numpy() for i, u in enumerate(sources)}


def measure_graph_range(model, dataset, cfg: MeasureConfig, device, method: str = "auto",
                        min_lcc: int = 4) -> list[dict]:
    """Graph-level (Hessian) range for `cfg.sources_per_graph` random LCC sources per graph.

    Rows: {"graph_id", "u", "n_nodes", "lcc_size", "range_hops", "range_res",
    "total_hessian"}; ranges are None when the source's Hessian row is all zero.
    Graphs are drawn with their own seed stream (cfg.seed + 1), independent of the
    Jacobian sample. Returns [] when disabled or when the model has no
    `forward_from_h0` (node-level stand-ins).
    """
    if cfg.n_graphs_range <= 0 or cfg.sources_per_graph <= 0 or not hasattr(model, "forward_from_h0"):
        return []
    rng = np.random.default_rng(cfg.seed + 1)
    indices = rng.choice(len(dataset), min(cfg.n_graphs_range, len(dataset)), replace=False)
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
        local = rng.choice(len(lcc), min(cfg.sources_per_graph, len(lcc)), replace=False)
        infl = hessian_influence(model, data, [int(lcc[a]) for a in local], device, method=method)
        gid = _graph_id(data, gi)
        for a in sorted(int(a) for a in local):
            w = infl[int(lcc[a])][lcc]
            rows.append({"graph_id": gid, "u": int(lcc[a]), "n_nodes": n, "lcc_size": int(len(lcc)),
                         "range_hops": normalized_range(w, hops[a]), "range_res": normalized_range(w, R[a]),
                         "total_hessian": float(w.sum())})
    return rows


def _graph_id(data, fallback: int) -> int:
    gid = getattr(data, "graph_id", None)
    if gid is None:
        return int(fallback)
    return int(gid.reshape(-1)[0]) if torch.is_tensor(gid) else int(gid)


# ------------------------------------------------------------------ summary

_QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _distribution(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"n": 0, "mean": None, "median": None, "std": None, "quantiles": None}
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": float(x.std()), "min": float(x.min()), "max": float(x.max()),
            "quantiles": {str(q): float(np.quantile(x, q)) for q in _QUANTILES}}


def _size_corr(rows: list[dict], key: str) -> dict:
    """Pearson r between graph size and the per-graph mean range."""
    per_graph: dict = {}
    for r in rows:
        if r[key] is not None:
            per_graph.setdefault(r["graph_id"], (r["n_nodes"], []))[1].append(r[key])
    n = np.array([v[0] for v in per_graph.values()], dtype=float)
    m = np.array([np.mean(v[1]) for v in per_graph.values()], dtype=float)
    if len(n) < 3 or np.ptp(n) == 0 or np.ptp(m) == 0:
        return {"r": None, "p": None, "n": int(len(n))}
    r, p = pearsonr(n, m)
    return {"r": float(r), "p": float(p), "n": int(len(n))}


def summarize_range(rows: list[dict], keys: tuple[str, ...]) -> dict:
    """Per-key distribution (over rows and over per-graph means) and size correlation."""
    if not rows:
        return {"n": 0}
    out: dict = {"n": len(rows), "n_graphs": len({r["graph_id"] for r in rows})}
    for key in keys:
        vals = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
        per_graph: dict = {}
        for r in rows:
            if r[key] is not None:
                per_graph.setdefault(r["graph_id"], []).append(r[key])
        out[key] = {
            "n_undefined": int(sum(r[key] is None for r in rows)),
            "rows": _distribution(vals),
            "graph_means": _distribution(np.array([np.mean(v) for v in per_graph.values()], dtype=float)),
            "size_corr": _size_corr(rows, key),
        }
    return out


NODE_RANGE_KEYS = ("range_hops", "range_res", "range_hops_fro", "range_res_fro")
GRAPH_RANGE_KEYS = ("range_hops", "range_res")
