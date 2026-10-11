"""Phase 2 analysis: tables and figures from run folders written by `RunDir`.

    python -m gnn_mech.analysis <runs_dir> [--exp NAME ...] [--out DIR] [--n-boot 1000]

Every function reads `<out_dir>/<exp>/<hash>/seed<k>/` folders (config.json,
metrics.json, preds_<split>.npz, measurements.json). A "model" is one config hash
(all seeds of one config); it is labelled "GCN" or "alpha=<a>".

Confidence intervals:
    * over seeds: mean +- t_{0.975, n-1} * std / sqrt(n);
    * graph-clustered bootstrap: graphs are resampled with replacement (all rows of a
      graph, from every seed, move together), because pairs, targets and heads within
      one graph are correlated. Slope differences between models are paired: both
      models are measured on the same sampled graphs (same `measure.seed`), so one
      resample of graph ids is applied to both. Every bootstrap takes an explicit seed
      and is reproducible.
Log-log Jacobian fits use pairs with J > 0 only (as in `measure.summarize_jacobian`);
the fraction of zero pairs (outside an MPNN's receptive field) is reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats as sps

from .measure import _loglog_fit
from .train import macro_ap

# Published Peptides-func test AP (Tönshoff et al., TMLR 2024).
PUBLISHED = {"GCN (Tönshoff et al.)": 0.686, "GPS (Tönshoff et al.)": 0.6535}

# Pilot-notebook style (notebooks/pilot/Data_Analysis_Thesis.ipynb).
RC_PARAMS = {
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}
COLORS = {"GCN": "#888780", "alpha=0": "#1D9E75", "alpha=0.5": "#534AB7", "alpha=1": "#993C1D"}


# --------------------------------------------------------------------------- runs


def model_label(cfg: dict) -> str:
    m = cfg["model"]
    if m["arch"] == "gcn":
        return "GCN"
    return f"alpha={float(m['alpha']):g}"


def _model_order(labels: Iterable[str]) -> list[str]:
    def key(lab: str):
        return (0, -1.0) if lab == "GCN" else (1, float(lab.split("=")[1])) if lab.startswith("alpha=") else (2, lab)
    return sorted(set(labels), key=key)


def _color(label: str) -> str:
    if label in COLORS:
        return COLORS[label]
    import matplotlib.pyplot as plt
    if label.startswith("alpha="):
        return plt.get_cmap("viridis")(float(label.split("=")[1]))
    return "#444441"


def collect_runs(out_dir: str, exps: Optional[list[str]] = None) -> pd.DataFrame:
    """One row per run folder: config, metrics and paths.

    Columns: exp, hash, seed, model, arch, alpha, layers, metric, test, val, best_epoch,
    n_params, epochs_run, seconds, complete, has_preds, has_measurements, path, config.
    """
    rows = []
    for cfg_path in sorted(Path(out_dir).glob("*/*/seed*/config.json")):
        run = cfg_path.parent
        exp, hsh = run.parent.parent.name, run.parent.name
        if exps and exp not in exps:
            continue
        cfg = json.loads(cfg_path.read_text())
        metrics = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").exists() else None
        metric = None
        if metrics:
            metric = "ap" if "ap" in (metrics.get("test") or {}) else "mae"
        rows.append({
            "exp": exp, "hash": hsh, "seed": int(cfg["seed"]), "model": model_label(cfg),
            "arch": cfg["model"]["arch"], "alpha": float(cfg["model"]["alpha"]),
            "layers": int(cfg["model"]["layers"]), "metric": metric,
            "test": metrics["test"].get(metric) if metrics else np.nan,
            "val": metrics["val"].get(metric) if metrics else np.nan,
            "best_epoch": metrics.get("best_epoch") if metrics else None,
            "n_params": metrics.get("n_params") if metrics else None,
            "epochs_run": metrics.get("epochs_run") if metrics else None,
            "seconds": metrics.get("seconds") if metrics else None,
            "complete": metrics is not None,
            "has_preds": (run / "preds_test.npz").exists(),
            "has_measurements": (run / "measurements.json").exists(),
            "path": str(run), "config": cfg,
        })
    cols = ["exp", "hash", "seed", "model", "arch", "alpha", "layers", "metric", "test", "val", "best_epoch",
            "n_params", "epochs_run", "seconds", "complete", "has_preds", "has_measurements", "path", "config"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values(["exp", "model", "seed"]).reset_index(drop=True)


def _seed_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float, float, float]:
    """(mean, std, ci_low, ci_high) over seeds with a t interval; CI is nan for n < 2."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan,) * 4
    mean = float(x.mean())
    if x.size < 2:
        return mean, np.nan, np.nan, np.nan
    std = float(x.std(ddof=1))
    half = float(sps.t.ppf(0.5 + level / 2, x.size - 1) * std / np.sqrt(x.size))
    return mean, std, mean - half, mean + half


def performance_table(runs: pd.DataFrame, published: Optional[dict] = None) -> pd.DataFrame:
    """Test metric mean +- std and 95% CI over seeds per model, next to published numbers."""
    published = PUBLISHED if published is None else published
    done = runs[runs["complete"]]
    rows = []
    for (exp, hsh), g in done.groupby(["exp", "hash"], sort=False):
        mean, std, lo, hi = _seed_ci(g["test"].to_numpy())
        vmean = float(np.nanmean(g["val"].to_numpy()))
        rows.append({"model": g["model"].iloc[0], "exp": exp, "hash": hsh, "metric": g["metric"].iloc[0],
                     "n_seeds": len(g), "test_mean": mean, "test_std": std, "ci_low": lo, "ci_high": hi,
                     "val_mean": vmean, "n_params": g["n_params"].iloc[0], "source": "ours"})
    for name, val in published.items():
        rows.append({"model": name, "exp": None, "hash": None, "metric": "ap", "n_seeds": None,
                     "test_mean": val, "test_std": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                     "val_mean": np.nan, "n_params": None, "source": "published"})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {m: i for i, m in enumerate(_model_order(df["model"]))}
    return df.sort_values(["source", "model"], key=lambda s: s.map(order) if s.name == "model" else s
                          ).reset_index(drop=True)


# --------------------------------------------------------------------------- predictions


def load_preds(run_path: str, split: str = "test") -> dict[str, np.ndarray]:
    with np.load(Path(run_path) / f"preds_{split}.npz") as z:
        return {k: z[k] for k in ("graph_id", "y", "logits")}


def load_structure_stats(cfg: dict, split: str = "test") -> dict[str, np.ndarray]:
    """Cached per-graph structure stats (`data.structure_stats`) for a run's dataset split."""
    from .config import from_dict
    from .data import load_dataset, structure_stats

    data_cfg = from_dict(cfg).data
    tr, va, te, _ = load_dataset(data_cfg)
    return structure_stats({"train": tr, "val": va, "test": te}[split], split, data_cfg)


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision of one column, same definition as sklearn (ties share a threshold)."""
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(s)), y.size - 1]
    tps = np.cumsum(y)[last]
    if tps[-1] == 0:
        return np.nan
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def fast_macro_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    """`train.macro_ap` without sklearn overhead (for bootstraps): mean AP over columns with a positive."""
    aps = [_ap(y_true[:, c], scores[:, c]) for c in range(y_true.shape[1]) if y_true[:, c].sum() > 0]
    return float(np.mean(aps)) if aps else float("nan")


def _tertile_edges(values: np.ndarray, n_bins: int = 3) -> np.ndarray:
    return np.quantile(values[np.isfinite(values)], np.linspace(0, 1, n_bins + 1))


def _bin_index(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges[1:-1], values, side="right")
    return np.where(np.isfinite(values), idx, -1)


def stratified_ap(preds: dict, structure_stats: dict, keys: tuple[str, ...] = ("avg_resistance", "n_nodes"),
                  n_bins: int = 3, edges: Optional[dict] = None) -> pd.DataFrame:
    """Macro AP within quantile bins (tertiles by default) of each structure stat for one run.

    `edges` ({key: bin edges}) fixes the bins across runs; by default they are the
    quantiles of the stat over the graphs in `preds`. Rows: stat, bin, lo, hi, n, ap.
    """
    gid = preds["graph_id"]
    scores = 1.0 / (1.0 + np.exp(-preds["logits"]))
    rows = []
    for key in keys:
        vals = np.asarray(structure_stats[key], dtype=float)[gid]
        e = edges[key] if edges and key in edges else _tertile_edges(vals, n_bins)
        b = _bin_index(vals, e)
        for i in range(len(e) - 1):
            m = b == i
            ap = macro_ap(preds["y"][m], scores[m]) if m.sum() > 0 else np.nan
            rows.append({"stat": key, "bin": i, "lo": float(e[i]), "hi": float(e[i + 1]), "n": int(m.sum()), "ap": ap})
    return pd.DataFrame(rows)


def _cluster_bootstrap(keys: np.ndarray, stat: Callable[[np.ndarray], float], n_boot: int,
                       seed: int) -> np.ndarray:
    """Bootstrap distribution of stat(row_indices) resampling unique `keys` (clusters)."""
    uniq, inv = np.unique(keys, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[b] = stat(np.concatenate([members[i] for i in pick]))
    return out


def _pct_ci(boot: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return np.nan, np.nan
    a = (1 - level) / 2
    return float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def stratified_ap_table(runs: pd.DataFrame, structure: dict, split: str = "test",
                        keys: tuple[str, ...] = ("avg_resistance", "n_nodes"), n_bins: int = 3,
                        n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Stratified AP per model and bin: mean / CI over seeds and a graph-bootstrap CI.

    Bins are fixed from the structure stats of the split's graphs, so every model
    is compared on the same graphs. The bootstrap resamples graphs within a bin and
    averages AP over seeds in each resample.
    """
    runs = runs[runs["complete"] & runs["has_preds"]]
    if runs.empty:
        return pd.DataFrame()
    preds = {p: load_preds(p, split) for p in runs["path"]}
    gid_all = next(iter(preds.values()))["graph_id"]
    edges = {k: _tertile_edges(np.asarray(structure[k], dtype=float)[gid_all], n_bins) for k in keys}
    rows = []
    for (_, _), g in runs.groupby(["exp", "hash"], sort=False):
        per_seed = [stratified_ap(preds[p], structure, keys, n_bins, edges) for p in g["path"]]
        for key in keys:
            for i in range(n_bins):
                aps = np.array([float(t[(t.stat == key) & (t.bin == i)]["ap"].iloc[0]) for t in per_seed])
                mean, std, lo, hi = _seed_ci(aps)
                p0 = preds[g["path"].iloc[0]]
                vals = np.asarray(structure[key], dtype=float)[p0["graph_id"]]
                in_bin = np.flatnonzero(_bin_index(vals, edges[key]) == i)
                seeds = [preds[p] for p in g["path"]]
                sig = [1.0 / (1.0 + np.exp(-s["logits"])) for s in seeds]

                def stat(idx, seeds=seeds, sig=sig, in_bin=in_bin):
                    rows_ = in_bin[idx]
                    return float(np.nanmean([fast_macro_ap(s["y"][rows_], q[rows_]) for s, q in zip(seeds, sig)]))

                boot = _cluster_bootstrap(np.arange(len(in_bin)), stat, n_boot, seed) if len(in_bin) else np.array([])
                blo, bhi = _pct_ci(boot)
                rows.append({"model": g["model"].iloc[0], "stat": key, "bin": i, "lo": float(edges[key][i]),
                             "hi": float(edges[key][i + 1]), "n_graphs": int(len(in_bin)), "n_seeds": len(g),
                             "ap_mean": mean, "ap_std": std, "seed_ci_low": lo, "seed_ci_high": hi,
                             "boot_ci_low": blo, "boot_ci_high": bhi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- measurements


def _measurement_frames(runs: pd.DataFrame, key: str) -> pd.DataFrame:
    """Concatenate `measurements.json[key]` rows of every run, tagged with model and seed."""
    frames = []
    for _, r in runs[runs["has_measurements"]].iterrows():
        meas = json.loads((Path(r["path"]) / "measurements.json").read_text())
        rows = meas.get(key) or []
        if rows:
            df = pd.DataFrame(rows)
            df["model"], df["seed"], df["exp"], df["hash"] = r["model"], r["seed"], r["exp"], r["hash"]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _slope(lx: np.ndarray, ly: np.ndarray) -> Optional[float]:
    """OLS slope of ly on lx (same estimate as `_loglog_fit`, without scipy overhead)."""
    if lx.size < 3:
        return None
    dx = lx - lx.mean()
    sxx = dx @ dx
    if sxx <= 1e-12 * lx.size:
        return None
    return float(dx @ (ly - ly.mean()) / sxx)


def _jac_arrays(df: pd.DataFrame) -> dict:
    """log10 columns of the J > 0 pairs, for fast repeated fits on bootstrap resamples."""
    J, R, H, N = (df[c].to_numpy(dtype=float) for c in ("jacobian", "resistance", "hops", "n_nodes"))
    keep = (J > 0) & (R > 0) & (H > 0) & (N > 0)
    return {"lj": np.log10(J[keep]), "lr": np.log10(R[keep]), "lh": np.log10(H[keep]),
            "ln": np.log10(N[keep]), "gid": df["graph_id"].to_numpy()[keep]}


def _jac_stats(a: dict, idx: Optional[np.ndarray] = None) -> dict:
    """Log-log slopes vs resistance / hops and the size-controlled OLS coefficients."""
    lj, lr, lh, ln = (a[k] if idx is None else a[k][idx] for k in ("lj", "lr", "lh", "ln"))
    ctrl_res = ctrl_size = None
    if lj.size > 3:
        X = np.column_stack([np.ones(lj.size), lr, ln])
        if np.linalg.matrix_rank(X) == 3:
            beta = np.linalg.lstsq(X, lj, rcond=None)[0]
            ctrl_res, ctrl_size = float(beta[1]), float(beta[2])
    return {"slope_res": _slope(lr, lj), "slope_hops": _slope(lh, lj), "ctrl_res": ctrl_res, "ctrl_size": ctrl_size}


def _members(gid: np.ndarray, keys: np.ndarray) -> dict:
    return {g: np.flatnonzero(gid == g) for g in keys}


def _nan(x) -> float:
    return np.nan if x is None else float(x)


def jacobian_analysis(runs: pd.DataFrame, n_boot: int = 1000, seed: int = 0,
                      baseline: str = "alpha=1") -> dict:
    """Pooled-seed Jacobian slopes per model with graph-clustered bootstrap CIs.

    Returns {"slopes": DataFrame, "differences": DataFrame, "collinearity": dict}.
    slopes: log-log slope vs resistance and vs hops, size-controlled OLS coefficients,
    and per-seed slope spread. differences: slope minus the `baseline` model's slope
    (the transformer, alpha=1, is the size-confound baseline) with a paired bootstrap
    CI over shared graph ids.
    """
    jac = _measurement_frames(runs, "jacobian")
    out: dict = {"slopes": pd.DataFrame(), "differences": pd.DataFrame(), "collinearity": {}}
    if jac.empty:
        return out
    rows, arrays = [], {}
    for model in _model_order(jac["model"]):
        df = jac[jac["model"] == model].reset_index(drop=True)
        a = _jac_arrays(df)
        arrays[model] = a
        point = _jac_stats(a)
        uniq = np.unique(df["graph_id"].to_numpy())
        members = _members(a["gid"], uniq)
        rng = np.random.default_rng(seed)
        bs = {k: np.empty(n_boot) for k in point}
        for b in range(n_boot):
            pick = uniq[rng.integers(0, uniq.size, uniq.size)]
            s_ = _jac_stats(a, np.concatenate([members[g] for g in pick]))
            for k in point:
                bs[k][b] = _nan(s_[k])
        per_seed = [_nan(_jac_stats(_jac_arrays(g))["slope_res"]) for _, g in df.groupby("seed")]
        row = {"model": model, "n_pairs": len(df), "n_graphs": int(uniq.size), "n_seeds": df["seed"].nunique(),
               "frac_zero": float((df["jacobian"] == 0).mean()),
               "slope_res_seed_mean": float(np.nanmean(per_seed)),
               "slope_res_seed_std": float(np.nanstd(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan}
        for k in point:
            lo, hi = _pct_ci(bs[k])
            row.update({k: _nan(point[k]), f"{k}_ci_low": lo, f"{k}_ci_high": hi})
        rows.append(row)
    out["slopes"] = pd.DataFrame(rows)

    if baseline in arrays:
        b = arrays[baseline]
        diff_rows = []
        diff_keys = ("slope_res", "slope_hops", "ctrl_res")
        for model, a in arrays.items():
            if model == baseline:
                continue
            shared = np.intersect1d(np.unique(a["gid"]), np.unique(b["gid"]))
            if shared.size == 0:
                continue
            ma, mb = _members(a["gid"], shared), _members(b["gid"], shared)
            ia = np.concatenate([ma[g] for g in shared])
            ib = np.concatenate([mb[g] for g in shared])
            pa, pb = _jac_stats(a, ia), _jac_stats(b, ib)
            rng = np.random.default_rng(seed)
            bd = {k: np.empty(n_boot) for k in diff_keys}
            for i in range(n_boot):         # one paired resample of graph ids for both models
                pick = shared[rng.integers(0, shared.size, shared.size)]
                sa = _jac_stats(a, np.concatenate([ma[g] for g in pick]))
                sb = _jac_stats(b, np.concatenate([mb[g] for g in pick]))
                for k in diff_keys:
                    bd[k][i] = _nan(sa[k]) - _nan(sb[k])
            for k in diff_keys:
                lo, hi = _pct_ci(bd[k])
                diff_rows.append({"model": model, "baseline": baseline, "stat": k,
                                  "diff": _nan(pa[k]) - _nan(pb[k]), "ci_low": lo, "ci_high": hi,
                                  "n_graphs": int(shared.size),
                                  "excludes_zero": bool(np.isfinite(lo) and (lo > 0 or hi < 0))})
        out["differences"] = pd.DataFrame(diff_rows)

    pairs = jac.drop_duplicates(["graph_id", "u", "v"])
    R, H = pairs["resistance"].to_numpy(float), pairs["hops"].to_numpy(float)
    ok = (R > 0) & (H > 0)
    out["collinearity"] = {
        "n_pairs": int(ok.sum()),
        "pearson_r": float(np.corrcoef(R[ok], H[ok])[0, 1]) if ok.sum() > 2 else None,
        "pearson_r_log": float(np.corrcoef(np.log10(R[ok]), np.log10(H[ok]))[0, 1]) if ok.sum() > 2 else None,
        "spearman_r": float(sps.spearmanr(R[ok], H[ok])[0]) if ok.sum() > 2 else None,
        "frac_r_equals_hops": float(np.mean(np.isclose(R[ok], H[ok]))) if ok.sum() else None,
    }
    r = out["collinearity"]["pearson_r_log"]
    out["collinearity"]["vif_log"] = (1.0 / (1.0 - r ** 2)) if r is not None and abs(r) < 1 else None
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def entropy_analysis(runs: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Correlation of normalized entropy with graph size per model, layer and head.

    Pooled over seeds with a graph-clustered bootstrap CI, plus the spread of the
    per-seed correlations.
    """
    ent = _measurement_frames(runs, "entropy")
    if ent.empty:
        return pd.DataFrame()
    rows = []
    for (model, layer, head), g in ent.groupby(["model", "layer", "head"], sort=False):
        g = g.reset_index(drop=True)
        x, y = g["n_nodes"].to_numpy(float), g["entropy_norm"].to_numpy(float)
        boot = _cluster_bootstrap(g["graph_id"].to_numpy(), lambda i: _pearson(x[i], y[i]), n_boot, seed)
        lo, hi = _pct_ci(boot)
        per_seed = [_pearson(s["n_nodes"].to_numpy(float), s["entropy_norm"].to_numpy(float))
                    for _, s in g.groupby("seed")]
        rows.append({"model": model, "layer": int(layer), "head": int(head), "n_graphs": int(g["graph_id"].nunique()),
                     "n_seeds": int(g["seed"].nunique()), "mean_entropy_norm": float(y.mean()),
                     "r_size": _pearson(x, y), "ci_low": lo, "ci_high": hi,
                     "r_seed_mean": float(np.nanmean(per_seed)),
                     "r_seed_std": float(np.nanstd(per_seed, ddof=1)) if len(per_seed) > 1 else np.nan})
    df = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(_model_order(df["model"]))}
    return df.sort_values(["model", "layer", "head"], key=lambda s: s.map(order) if s.name == "model" else s
                          ).reset_index(drop=True)


def range_analysis(runs: pd.DataFrame) -> dict:
    """Range per model (node-level from Jacobian sweeps, graph-level from Hessians).

    Returns {"node": DataFrame, "graph": DataFrame, "reading": str}. Each row holds the
    mean / median over targets (pooled seeds), the seed CI of per-seed means, the
    correlation of per-graph mean range with graph size and, for the Hessian, the number
    of sources with an all-zero Hessian (range undefined). `reading` states the
    task-range interpretation, using a stated rule of thumb: if every model's mean hop
    range is at most half its depth, that supports Bamberger et al.'s finding that
    Peptides tasks are short-range.
    """
    out: dict = {"node": pd.DataFrame(), "graph": pd.DataFrame(), "reading": ""}
    layers = runs.groupby("model")["layers"].first().to_dict() if not runs.empty else {}
    for level, key in (("node", "range_node"), ("graph", "range_graph")):
        df = _measurement_frames(runs, key)
        if df.empty:
            continue
        rows = []
        for model in _model_order(df["model"]):
            g = df[df["model"] == model]
            row = {"model": model, "layers": layers.get(model), "n": len(g), "n_graphs": int(g["graph_id"].nunique())}
            for col in ("range_hops", "range_res"):
                v = pd.to_numeric(g[col], errors="coerce")
                seed_means = v.groupby(g["seed"]).mean().to_numpy()
                mean, std, lo, hi = _seed_ci(seed_means)
                per_graph = v.groupby(g["graph_id"]).mean()
                size = g.groupby("graph_id")["n_nodes"].first().loc[per_graph.index]
                row.update({f"{col}_mean": float(v.mean()), f"{col}_median": float(v.median()),
                            f"{col}_q90": float(v.quantile(0.9)), f"{col}_seed_ci_low": lo,
                            f"{col}_seed_ci_high": hi, f"{col}_size_r": _pearson(size.to_numpy(float),
                                                                                 per_graph.to_numpy(float)),
                            f"{col}_n_undefined": int(v.isna().sum())})
            rows.append(row)
        out[level] = pd.DataFrame(rows)
    node = out["node"]
    if not node.empty:
        parts = [f"{r.model}: {r.range_hops_mean:.2f} hops" + (f" (depth {r.layers})" if r.layers else "")
                 for r in node.itertuples()]
        short = [r.layers is not None and r.range_hops_mean <= r.layers / 2 for r in node.itertuples()]
        verdict = ("Every model's mean range is at most half its depth, consistent with Bamberger et al.'s "
                   "finding that Peptides-func is short-range." if all(short) else
                   "Not every model stays within half its depth ("
                   + ", ".join(r.model for r, ok in zip(node.itertuples(), short) if not ok)
                   + "), so the short-range reading is not supported by all models.")
        out["reading"] = "Mean node-level range (L1-normalized, hops): " + "; ".join(parts) + ". " + verdict
    return out


# --------------------------------------------------------------------------- figures


def _save(fig, fig_dir: Path, name: str) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = fig_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(str(p))
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def plot_ap_bars(perf: pd.DataFrame, fig_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    ours = perf[perf["source"] == "ours"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(ours))
    err = np.clip(np.vstack([ours["test_mean"] - ours["ci_low"], ours["ci_high"] - ours["test_mean"]]), 0, None)
    ax.bar(x, ours["test_mean"], 0.6, color=[_color(m) for m in ours["model"]],
           yerr=np.nan_to_num(err), error_kw={"elinewidth": 1.0, "ecolor": "black"})
    for name, val in perf[perf["source"] == "published"][["model", "test_mean"]].itertuples(index=False):
        ax.axhline(val, color="#444441", linewidth=1.0, linestyle="--")
        ax.text(len(ours) - 0.5, val, f" {name}", va="center", fontsize=8)
    ax.set_xticks(x, [f"{m}\n(n={int(n)})" for m, n in zip(ours["model"], ours["n_seeds"])])
    ax.set_ylabel("test AP (mean, 95% CI over seeds)")
    if len(ours):
        lo = np.nanmin(ours["ci_low"].fillna(ours["test_mean"]))
        ax.set_ylim(max(0.0, lo - 0.05), None)
    return _save(fig, fig_dir, "ap_bars")


def plot_stratified_ap(strat: pd.DataFrame, fig_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    keys = list(dict.fromkeys(strat["stat"]))
    fig, axes = plt.subplots(1, len(keys), figsize=(7.0 * len(keys) / 1.4, 4.2), squeeze=False)
    models = _model_order(strat["model"])
    w = 0.8 / max(1, len(models))
    for ax, key in zip(axes[0], keys):
        s = strat[strat["stat"] == key]
        for j, m in enumerate(models):
            sm = s[s["model"] == m].sort_values("bin")
            err = np.clip(np.vstack([sm["ap_mean"] - sm["boot_ci_low"], sm["boot_ci_high"] - sm["ap_mean"]]), 0, None)
            ax.bar(sm["bin"] + (j - (len(models) - 1) / 2) * w, sm["ap_mean"], w, color=_color(m), label=m,
                   yerr=np.nan_to_num(err), error_kw={"elinewidth": 0.8, "ecolor": "black"})
        sm = s[s["model"] == models[0]].sort_values("bin")
        ax.set_xticks(sm["bin"], [f"{lo:.3g}-{hi:.3g}" for lo, hi in zip(sm["lo"], sm["hi"])])
        ax.set_xlabel(f"{key} tertile")
        ax.set_ylabel("test AP (graph-bootstrap 95% CI)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, fig_dir, "stratified_ap")


def plot_jacobian_loglog(runs: pd.DataFrame, fig_dir: Path, max_points: int = 4000, seed: int = 0) -> list[str]:
    import matplotlib.pyplot as plt
    jac = _measurement_frames(runs, "jacobian")
    if jac.empty:
        return []
    models = _model_order(jac["model"])
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.6), squeeze=False, sharey=True)
    rng = np.random.default_rng(seed)
    for ax, m in zip(axes[0], models):
        df = jac[(jac["model"] == m) & (jac["jacobian"] > 0) & (jac["resistance"] > 0)]
        if len(df) > max_points:
            df = df.iloc[rng.choice(len(df), max_points, replace=False)]
        ax.scatter(df["resistance"], df["jacobian"], s=4, alpha=0.3, color=_color(m), linewidths=0)
        fit = _loglog_fit(df["resistance"].to_numpy(float), df["jacobian"].to_numpy(float))
        if fit["slope"] is not None:
            xs = np.logspace(np.log10(df["resistance"].min()), np.log10(df["resistance"].max()), 50)
            ax.plot(xs, 10 ** fit["intercept"] * xs ** fit["slope"], color="black", linewidth=1.0,
                    label=f"slope {fit['slope']:.2f}")
            ax.legend(loc="lower left")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(m)
        ax.set_xlabel("effective resistance")
    axes[0][0].set_ylabel(r"$\|\partial h_v / \partial h^0_u\|_F$")
    return _save(fig, fig_dir, "jacobian_vs_resistance")


def plot_entropy_vs_size(runs: pd.DataFrame, fig_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    ent = _measurement_frames(runs, "entropy")
    if ent.empty:
        return []
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for m in _model_order(ent["model"]):
        g = ent[ent["model"] == m].groupby(["seed", "graph_id"]).agg(n=("n_nodes", "first"),
                                                                     e=("entropy_norm", "mean"))
        ax.scatter(g["n"], g["e"], s=6, alpha=0.35, color=_color(m), label=m, linewidths=0)
    ax.set_xlabel("graph size (nodes)")
    ax.set_ylabel("normalized attention entropy\n(mean over layers and heads)")
    ax.legend()
    return _save(fig, fig_dir, "entropy_vs_size")


def plot_range_distributions(runs: pd.DataFrame, fig_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt
    rng_df = _measurement_frames(runs, "range_node")
    if rng_df.empty:
        return []
    models = _model_order(rng_df["model"])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for ax, col, label in zip(axes, ("range_hops", "range_res"), ("hops", "effective resistance")):
        data = [pd.to_numeric(rng_df[rng_df["model"] == m][col], errors="coerce").dropna().to_numpy() for m in models]
        parts = ax.violinplot([d if d.size else np.array([np.nan]) for d in data], showmedians=True)
        for body, m in zip(parts["bodies"], models):
            body.set_facecolor(_color(m))
            body.set_alpha(0.6)
        ax.set_xticks(np.arange(1, len(models) + 1), models)
        ax.set_ylabel(f"node-level range ({label})")
    return _save(fig, fig_dir, "range_distributions")


# --------------------------------------------------------------------------- driver


def run_analysis(out_dir: str, report_dir: str, exps: Optional[list[str]] = None, n_boot: int = 1000,
                 seed: int = 0, split: str = "test") -> dict:
    """Every table (CSV / JSON) and figure (PDF + PNG) into `report_dir`; returns them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC_PARAMS)
    rep = Path(report_dir)
    fig_dir = rep / "figures"
    rep.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(out_dir, exps)
    if runs.empty:
        raise FileNotFoundError(f"no runs under {out_dir}")
    perf = performance_table(runs)
    done = runs[runs["complete"]]
    structure = load_structure_stats(done["config"].iloc[0], split) if done["has_preds"].any() else None
    strat = stratified_ap_table(done, structure, split, n_boot=n_boot, seed=seed) if structure else pd.DataFrame()
    jac = jacobian_analysis(done, n_boot=n_boot, seed=seed)
    ent = entropy_analysis(done, n_boot=n_boot, seed=seed)
    rng = range_analysis(done)

    tables = {"runs": runs.drop(columns=["config"]), "performance": perf, "stratified_ap": strat,
              "jacobian_slopes": jac["slopes"], "jacobian_differences": jac["differences"],
              "entropy": ent, "range_node": rng["node"], "range_graph": rng["graph"]}
    for name, df in tables.items():
        df.to_csv(rep / f"{name}.csv", index=False)
    with open(rep / "summary.json", "w") as f:
        json.dump({"collinearity": jac["collinearity"], "range_reading": rng["reading"],
                   "n_boot": n_boot, "seed": seed, "split": split}, f, indent=2)

    figures = plot_ap_bars(perf, fig_dir)
    if not strat.empty:
        figures += plot_stratified_ap(strat, fig_dir)
    figures += plot_jacobian_loglog(done, fig_dir, seed=seed)
    figures += plot_entropy_vs_size(done, fig_dir)
    figures += plot_range_distributions(done, fig_dir)
    return {**tables, "collinearity": jac["collinearity"], "range_reading": rng["reading"], "figures": figures}


def main(argv: Optional[list[str]] = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", help="out_dir the runs were written to")
    parser.add_argument("--exp", nargs="*", default=None, help="only these exp_names")
    parser.add_argument("--out", default="reports/phase2", help="directory for tables and figures")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)
    res = run_analysis(args.runs_dir, args.out, args.exp, args.n_boot, args.seed, args.split)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(res["performance"][["model", "n_seeds", "test_mean", "test_std", "ci_low", "ci_high", "source"]])
    print(f"tables and {len(res['figures'])} figure files in {args.out}")
    return res


if __name__ == "__main__":
    main()
