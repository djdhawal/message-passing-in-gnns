"""Mechanistic measurements: Jacobian sensitivity, attention entropy and range."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import linregress, pearsonr

from ..config import MeasureConfig
from .entropy import attention_entropy, measure_entropy
from .jacobian import jacobian_norms, jacobian_sweeps, measure_jacobians, sample_stratified_pairs, target_influence
from .range import (GRAPH_RANGE_KEYS, NODE_RANGE_KEYS, hessian_influence, measure_graph_range, node_range_rows,
                    summarize_range)

__all__ = [
    "attention_entropy", "measure_entropy", "jacobian_norms", "jacobian_sweeps", "measure_jacobians",
    "target_influence", "sample_stratified_pairs", "hessian_influence", "measure_graph_range",
    "node_range_rows", "summarize_jacobian", "summarize_entropy", "summarize_range", "run_all",
]


def _loglog_fit(x: np.ndarray, y: np.ndarray) -> dict:
    """log10-log10 linear regression of y on x, over pairs with x > 0 and y > 0."""
    keep = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    lx, ly = np.log10(x[keep]), np.log10(y[keep])
    out: dict = {"n": int(keep.sum()), "slope": None, "intercept": None, "r": None, "p": None}
    if out["n"] < 3 or np.ptp(lx) == 0:
        return out
    fit = linregress(lx, ly)
    out.update(slope=float(fit.slope), intercept=float(fit.intercept),
               r=float(fit.rvalue), p=float(fit.pvalue))
    return out


def _ols(y: np.ndarray, X: np.ndarray, names: list[str]) -> dict:
    """OLS with intercept; returns coefficients and classical standard errors."""
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    names = ["intercept"] + names
    out: dict = {"n": int(n), "coef": None, "se": None, "r2": None}
    if n <= k + 1 or np.linalg.matrix_rank(A) < k + 1:
        return out
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    sigma2 = resid @ resid / (n - k - 1)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A)))
    ss_tot = ((y - y.mean()) ** 2).sum()
    out.update(coef=dict(zip(names, map(float, beta))), se=dict(zip(names, map(float, se))),
               r2=float(1 - resid @ resid / ss_tot) if ss_tot > 0 else None)
    return out


def summarize_jacobian(rows: list[dict]) -> dict:
    """Log-log fits of J vs resistance and hops, plus a size-controlled OLS.

    Pairs with J == 0 (outside the receptive field) cannot enter a log fit;
    their count is reported as `n_zero`.
    """
    if not rows:
        return {"n_pairs": 0, "n_zero": 0}
    J = np.array([r["jacobian"] for r in rows], dtype=float)
    R = np.array([r["resistance"] for r in rows], dtype=float)
    hops = np.array([r["hops"] for r in rows], dtype=float)
    size = np.array([r["n_nodes"] for r in rows], dtype=float)
    keep = (J > 0) & (R > 0) & (size > 0)
    controlled = _ols(np.log10(J[keep]), np.column_stack([np.log10(R[keep]), np.log10(size[keep])]),
                      ["log10_resistance", "log10_n_nodes"])
    return {
        "n_pairs": len(rows),
        "n_zero": int((J == 0).sum()),
        "vs_resistance": _loglog_fit(R, J),
        "vs_hops": _loglog_fit(hops, J),
        "size_controlled": controlled,
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"r": None, "p": None, "n": int(len(x))}
    r, p = pearsonr(x, y)
    return {"r": float(r), "p": float(p), "n": int(len(x))}


def summarize_entropy(rows: list[dict]) -> dict:
    """Size-entropy correlations (per-graph means over layers and heads) and per-layer means."""
    if not rows:
        return {"n_graphs": 0}
    per_graph: dict = defaultdict(lambda: [0, [], []])
    per_layer: dict = defaultdict(lambda: ([], []))
    for r in rows:
        g = per_graph[r["graph_id"]]
        g[0] = r["n_nodes"]
        g[1].append(r["entropy"])
        g[2].append(r["entropy_norm"])
        per_layer[r["layer"]][0].append(r["entropy"])
        per_layer[r["layer"]][1].append(r["entropy_norm"])
    n = np.array([g[0] for g in per_graph.values()], dtype=float)
    ent = np.array([np.mean(g[1]) for g in per_graph.values()])
    ent_norm = np.array([np.mean(g[2]) for g in per_graph.values()])
    return {
        "n_graphs": len(per_graph),
        "size_vs_entropy": _pearson(n, ent),
        "size_vs_entropy_norm": _pearson(n, ent_norm),
        "mean_entropy": float(ent.mean()),
        "mean_entropy_norm": float(ent_norm.mean()),
        "per_layer": [{"layer": int(l), "entropy": float(np.mean(e)), "entropy_norm": float(np.mean(en))}
                      for l, (e, en) in sorted(per_layer.items())],
    }


def run_all(model, dataset, cfg: MeasureConfig, device) -> dict:
    """Run every measurement.

    Returns {"jacobian": rows, "entropy": rows, "range_node": rows, "range_graph": rows,
    "summary": {"jacobian", "entropy", "range_node", "range_graph"}}. Node-level range
    rows (one per Jacobian target) reuse the Jacobian sweeps; graph-level (Hessian)
    range rows come from `cfg.n_graphs_range` x `cfg.sources_per_graph` sources.
    """
    jac, targets = jacobian_sweeps(model, dataset, cfg, device)
    rng_node = node_range_rows(targets)
    ent = measure_entropy(model, dataset, cfg, device)
    rng_graph = measure_graph_range(model, dataset, cfg, device)
    return {"jacobian": jac, "entropy": ent, "range_node": rng_node, "range_graph": rng_graph,
            "summary": {"jacobian": summarize_jacobian(jac), "entropy": summarize_entropy(ent),
                        "range_node": summarize_range(rng_node, NODE_RANGE_KEYS),
                        "range_graph": summarize_range(rng_graph, GRAPH_RANGE_KEYS)}}
