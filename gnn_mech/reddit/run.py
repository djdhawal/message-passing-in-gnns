"""Reddit experiments (Phase 2, B4): distance split (main) and random split (control).

    python -m gnn_mech.reddit.run --target 18 [--models sage vmn] [--splits distance random]
                                  [--seeds 0:30] [--root data/reddit] [--cached-pt PT]
                                  [--out-dir runs] [--synthetic] [key=value ...]

Main: train on 0-1 hops from the anchor, validate on 2, test on 3+ (as the pilot).
Control: a random split of the same nodes with the same split sizes; test error is
then broken down by BFS distance band. If the VMN's advantage still grows with
distance under the random split, it tracks graph distance rather than extrapolation
away from the training region; if it vanishes, the pilot gain was distribution shift.

Per band (0, 1, 2, 3, 4, 5+) we report n, MSE, the MSE of the constant train-mean
predictor on the same nodes, and their ratio `rel_mse` (unit-free, so bands with
different target scales compare). Errors are in the modelled target units
(standardized, log1p first when the target is skewed, as in target selection).
Inputs are every feature except the target and the columns with |r| > 0.9 to it.

Each (model, split, seed) is one `RunDir` folder `<out_dir>/<exp_name>/<hash>/seed<k>/`
with config.json, metrics.json and history.json; finished runs are skipped.
`aggregate` collects them into a band table with bootstrap CIs over seeds and the
paired VMN gain (1 - MSE_vmn / MSE_sage, matched by seed).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import skew
from torch_geometric.utils import to_undirected

from ..results import RunDir
from ..train import resolve_device, set_seed
from .data import bfs_hops, make_split, pick_anchor
from .models import build_reddit_model
from .target_select import drop_correlated

BANDS = ((0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, None))   # (lo, hi) hops, hi None = open


@dataclass
class RedditConfig:
    """Duck-types the parts of `Config` that `RunDir` uses (to_dict, hash, seed, out_dir, exp_name)."""
    exp_name: str = "reddit"
    out_dir: str = "./runs"
    seed: int = 0
    root: str = "./data/reddit"
    cached_pt: Optional[str] = None
    synthetic: bool = False
    target: int = 0
    corr_thresh: float = 0.9
    split: str = "distance"          # "distance" | "random"
    model: str = "sage"              # "sage" | "vmn"
    hidden: int = 64
    layers: int = 3
    dropout: float = 0.3
    lr: float = 5e-3
    weight_decay: float = 5e-4
    epochs: int = 500
    patience: int = 80
    eval_every: int = 10
    device: str = "auto"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        d = self.to_dict()
        for k in ("seed", "out_dir", "exp_name", "root", "cached_pt", "device"):
            d.pop(k)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


def apply_overrides(cfg: RedditConfig, overrides: list[str]) -> RedditConfig:
    import yaml
    d = cfg.to_dict()
    for item in overrides:
        key, eq, raw = item.partition("=")
        if not eq or key not in d:
            raise KeyError(f"bad override {item!r}; keys: {sorted(d)}")
        d[key] = yaml.safe_load(raw)
    return RedditConfig(**d)


# ------------------------------------------------------------------ data prep


@dataclass
class Prepared:
    x: torch.Tensor            # [N, F] inputs (target and near-duplicates removed), not standardized
    y: torch.Tensor            # [N] modelled target (log1p if skewed)
    edge_index: torch.Tensor   # undirected
    dist: np.ndarray           # BFS hops from the anchor, -1 unreachable
    is_source: np.ndarray
    anchor: int
    inputs: list[int]
    log_target: bool


def prepare(data, target: int, corr_thresh: float = 0.9) -> Prepared:
    x_all = data.x.numpy().astype(np.float64)
    is_source = data.is_source.numpy().astype(bool)
    y = x_all[:, target].copy()
    ys = y[is_source]
    log = bool(ys.min() >= 0 and skew(ys) > 1)
    if log:
        y = np.log1p(np.maximum(y, 0))
    others = np.array([j for j in range(x_all.shape[1]) if j != target])
    keep = others[drop_correlated(x_all[is_source][:, others], x_all[is_source, target], corr_thresh)]
    n = data.num_nodes
    anchor = pick_anchor(data.edge_index, n)
    return Prepared(
        x=torch.tensor(x_all[:, keep], dtype=torch.float), y=torch.tensor(y, dtype=torch.float),
        edge_index=to_undirected(data.edge_index, num_nodes=n), dist=bfs_hops(data.edge_index, n, anchor),
        is_source=is_source, anchor=anchor, inputs=[int(j) for j in keep], log_target=log)


# ------------------------------------------------------------------ metrics


def band_mask(dist: np.ndarray, lo: int, hi: Optional[int]) -> np.ndarray:
    return (dist >= lo) & ((dist <= hi) if hi is not None else True)


def band_label(lo: int, hi: Optional[int]) -> str:
    return f"{lo}" if hi == lo else f"{lo}+" if hi is None else f"{lo}-{hi}"


def band_metrics(pred: np.ndarray, y: np.ndarray, dist: np.ndarray, eval_mask: np.ndarray,
                 train_mask: np.ndarray, bands=BANDS) -> list[dict]:
    """Per band within `eval_mask`: n, mse, const_mse (train-mean predictor), rel_mse = mse / const_mse."""
    const = float(y[train_mask].mean())
    rows = []
    for lo, hi in bands:
        m = eval_mask & band_mask(dist, lo, hi)
        n = int(m.sum())
        if n == 0:
            continue
        mse = float(np.mean((pred[m] - y[m]) ** 2))
        cmse = float(np.mean((const - y[m]) ** 2))
        rows.append({"band": band_label(lo, hi), "lo": lo, "n": n, "mse": mse, "const_mse": cmse,
                     "rel_mse": mse / cmse if cmse > 0 else None})
    return rows


# ------------------------------------------------------------------ training


def train_reddit(cfg: RedditConfig, prep: Prepared, masks: dict[str, np.ndarray]) -> dict:
    """Full-batch Adam with early stopping on val MSE (checked every `eval_every` epochs)."""
    device = resolve_device(cfg.device)
    tr = torch.from_numpy(masks["train"]).to(device)
    va = torch.from_numpy(masks["val"]).to(device)
    x, y = prep.x.to(device), prep.y.to(device)
    x_mu, x_sd = x[tr].mean(0, keepdim=True), x[tr].std(0, keepdim=True).clamp_min(1e-6)
    y_mu, y_sd = y[tr].mean(), y[tr].std().clamp_min(1e-6)
    xn, yn = (x - x_mu) / x_sd, (y - y_mu) / y_sd
    ei = prep.edge_index.to(device)

    model = build_reddit_model(cfg.model, x.size(1), cfg.hidden, cfg.layers, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val, best_state, best_epoch, bad, history = math.inf, None, 0, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        loss = F.mse_loss(model(xn, ei)[tr], yn[tr])
        loss.backward()
        opt.step()
        if epoch == 1 or epoch % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(xn, ei)[va], yn[va]).item()
            history.append({"epoch": epoch, "train_loss": loss.item(), "val_mse": val})
            if val < best_val - 1e-6:
                best_val, best_epoch, bad = val, epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                bad += cfg.eval_every
                if bad >= cfg.patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = (model(xn, ei) * y_sd + y_mu).cpu().numpy()
    return {"pred": pred, "best_epoch": best_epoch, "epochs_run": history[-1]["epoch"], "history": history,
            "n_params": sum(p.numel() for p in model.parameters())}


def run_reddit(cfg: RedditConfig, data=None, prep: Optional[Prepared] = None) -> dict:
    """Train/evaluate one (model, split, seed); skipped when metrics.json exists."""
    run = RunDir(cfg)
    if run.is_complete():
        return {**run.load_json("metrics.json"), "run_dir": str(run.path)}
    if prep is None:
        prep = prepare(data, cfg.target, cfg.corr_thresh)
    set_seed(cfg.seed)
    masks = make_split(cfg.split, prep.dist, prep.is_source, cfg.seed)
    out = train_reddit(cfg, prep, masks)
    y = prep.y.numpy()
    eligible = masks["train"] | masks["val"] | masks["test"]
    mse = {k: float(np.mean((out["pred"][m] - y[m]) ** 2)) for k, m in masks.items()}
    metrics = {
        "target": cfg.target, "model": cfg.model, "split": cfg.split, "log_target": prep.log_target,
        "anchor": prep.anchor, "inputs": prep.inputs, "split_sizes": {k: int(m.sum()) for k, m in masks.items()},
        "mse": mse, "best_epoch": out["best_epoch"], "epochs_run": out["epochs_run"], "n_params": out["n_params"],
        "test_bands": band_metrics(out["pred"], y, prep.dist, masks["test"], masks["train"]),
        "all_bands": band_metrics(out["pred"], y, prep.dist, eligible, masks["train"]),
    }
    run.save_json("history.json", out["history"])
    run.save_json("metrics.json", metrics)
    return {**metrics, "run_dir": str(run.path)}


# ------------------------------------------------------------------ aggregation


def _boot_mean_ci(x: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, (n_boot, x.size))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def collect(out_dir: str, exp_name: str = "reddit", bands_key: str = "test_bands") -> pd.DataFrame:
    rows = []
    for mpath in Path(out_dir, exp_name).glob("*/seed*/metrics.json"):
        m = json.loads(mpath.read_text())
        cfg = json.loads((mpath.parent / "config.json").read_text())
        for b in m[bands_key]:
            rows.append({"target": m["target"], "model": m["model"], "split": m["split"], "seed": cfg["seed"], **b})
    return pd.DataFrame(rows)


def aggregate(out_dir: str, exp_name: str = "reddit", bands_key: str = "test_bands", n_boot: int = 2000,
              seed: int = 0) -> pd.DataFrame:
    """Per (target, split, band, model): n (mean over seeds), mean rel_mse with bootstrap CI over seeds, and the
    seed-paired VMN gain 1 - mse_vmn / mse_sage with its CI (on the model == "vmn" rows)."""
    df = collect(out_dir, exp_name, bands_key)
    if df.empty:
        return df
    rows = []
    for (target, split, band), g in df.groupby(["target", "split", "band"], sort=False):
        for model, gm in g.groupby("model"):
            rel = gm["rel_mse"].to_numpy(float)
            lo, hi = _boot_mean_ci(rel, n_boot, seed)
            row = {"target": target, "split": split, "band": band, "lo": int(gm["lo"].iloc[0]), "model": model,
                   "n": int(round(gm["n"].mean())), "n_seeds": len(gm), "rel_mse": float(np.nanmean(rel)),
                   "rel_ci_low": lo, "rel_ci_high": hi, "mse": float(gm["mse"].mean())}
            if model == "vmn" and "sage" in set(g["model"]):
                paired = gm.merge(g[g["model"] == "sage"][["seed", "mse"]], on="seed", suffixes=("", "_sage"))
                gain = 1.0 - paired["mse"].to_numpy(float) / paired["mse_sage"].to_numpy(float)
                glo, ghi = _boot_mean_ci(gain, n_boot, seed)
                row.update(vmn_gain=float(gain.mean()) if gain.size else np.nan, gain_ci_low=glo, gain_ci_high=ghi)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "split", "lo", "model"]).reset_index(drop=True)


def plot_bands(table: pd.DataFrame, fig_dir: str, name: str = "reddit_bands") -> list[str]:
    """rel_mse per band (one panel per split), SAGE vs VMN, pilot style; PDF + PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..analysis import RC_PARAMS, _save
    plt.rcParams.update(RC_PARAMS)
    colors = {"sage": "#888780", "vmn": "#534AB7"}
    splits = list(dict.fromkeys(table["split"]))
    fig, axes = plt.subplots(1, len(splits), figsize=(7.0 * len(splits) / 1.3, 4.2), squeeze=False)
    for ax, split in zip(axes[0], splits):
        t = table[table["split"] == split]
        bands = list(dict.fromkeys(t.sort_values("lo")["band"]))
        x = np.arange(len(bands))
        for j, model in enumerate(["sage", "vmn"]):
            tm = t[t["model"] == model].set_index("band").reindex(bands)
            err = np.clip(np.vstack([tm["rel_mse"] - tm["rel_ci_low"], tm["rel_ci_high"] - tm["rel_mse"]]), 0, None)
            ax.bar(x + (j - 0.5) * 0.38, tm["rel_mse"], 0.38, color=colors[model],
                   label=f"{'GraphSAGE' if model == 'sage' else 'VMN'} (n={int(tm['n_seeds'].max())} seeds)",
                   yerr=np.nan_to_num(err), error_kw={"elinewidth": 0.8, "ecolor": "black"})
        ax.axhline(1.0, color="#993C1D", linewidth=1.0, linestyle="--")
        ax.set_xticks(x, [f"dist {b}" for b in bands])
        ax.set_ylabel("MSE / constant-predictor MSE")
        ax.set_title(f"{split} split")
    axes[0][0].legend()
    return _save(fig, Path(fig_dir), name)


# ------------------------------------------------------------------ CLI


def _seeds(spec: str) -> list[int]:
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b)))
    return [int(s) for s in spec.split(",")]


def main(argv: Optional[list[str]] = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, required=True, help="feature column to predict")
    parser.add_argument("--models", nargs="+", default=["sage", "vmn"])
    parser.add_argument("--splits", nargs="+", default=["distance", "random"])
    parser.add_argument("--seeds", default="0:30", help="'a:b' range or comma list")
    parser.add_argument("--root", default="./data/reddit")
    parser.add_argument("--cached-pt", default=None)
    parser.add_argument("--out-dir", default="./runs")
    parser.add_argument("--exp-name", default="reddit")
    parser.add_argument("--synthetic", action="store_true", help="tiny synthetic graph (smoke test)")
    parser.add_argument("--report", default=None, help="dir for the band table and figure (default <out-dir>/<exp>/report)")
    parser.add_argument("overrides", nargs="*", help="RedditConfig overrides, e.g. epochs=200 hidden=64")
    args = parser.parse_args(argv)

    if args.synthetic:
        from .data import synthetic_graph
        data = synthetic_graph()
    else:
        from .data import load_reddit_graph
        data = load_reddit_graph(args.root, args.cached_pt)
    base = apply_overrides(RedditConfig(exp_name=args.exp_name, out_dir=args.out_dir, root=args.root,
                                        cached_pt=args.cached_pt, synthetic=args.synthetic, target=args.target),
                           args.overrides)
    prep = prepare(data, base.target, base.corr_thresh)
    print(f"target {base.target} (log1p={prep.log_target}), {len(prep.inputs)} inputs, anchor {prep.anchor}")
    for split in args.splits:
        for model in args.models:
            for seed in _seeds(args.seeds):
                cfg = dataclasses.replace(base, split=split, model=model, seed=seed)
                m = run_reddit(cfg, prep=prep)
                print(f"{split:8s} {model:4s} seed {seed:3d}  test MSE {m['mse']['test']:.4f}")
    report = args.report or str(Path(args.out_dir) / args.exp_name / "report")
    table = aggregate(args.out_dir, args.exp_name)
    table = table[table["target"] == base.target]
    Path(report).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(report) / f"bands_target{base.target}.csv", index=False)
    figs = plot_bands(table, report, f"bands_target{base.target}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(table)
    return {"table": table, "figures": figs}


if __name__ == "__main__":
    main()
