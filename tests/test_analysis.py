"""Tests for gnn_mech.analysis on hand-built run folders (CPU, fast, no training)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from gnn_mech import analysis as A
from gnn_mech.config import Config
from gnn_mech.train import macro_ap

N_GRAPHS = 60


def _cfg(arch: str, alpha: float, seed: int, out: Path) -> Config:
    cfg = Config(exp_name="exp", out_dir=str(out), seed=seed)
    cfg.data.name = "synthetic-tiny"
    cfg.model.arch, cfg.model.alpha, cfg.model.layers = arch, alpha, 3
    return cfg


def _fake_run(out: Path, arch: str, alpha: float, seed: int, slope: float, ap_shift: float,
              entropy_r: float = 0.0) -> Path:
    """A run folder with metrics, predictions and measurements of known structure."""
    cfg = _cfg(arch, alpha, seed, out)
    path = out / cfg.exp_name / cfg.hash() / f"seed{seed}"
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(cfg.to_dict()))
    rng = np.random.default_rng(seed)
    y = (rng.random((N_GRAPHS, 3)) < 0.4).astype(float)
    logits = 3 * (y - 0.5) * ap_shift + rng.normal(size=y.shape)
    np.savez(path / "preds_test.npz", graph_id=np.arange(N_GRAPHS), y=y, logits=logits)
    np.savez(path / "preds_val.npz", graph_id=np.arange(N_GRAPHS), y=y, logits=logits)
    ap = macro_ap(y, 1 / (1 + np.exp(-logits)))
    (path / "metrics.json").write_text(json.dumps({"test": {"ap": ap}, "val": {"ap": ap}, "best_epoch": 1,
                                                   "n_params": 10, "epochs_run": 1, "seconds": 1.0}))
    jac, ent, rng_rows = [], [], []
    for g in range(20):
        n = 10 + g
        for k in range(8):
            R = 0.5 + k + 0.1 * g
            hops = k + 1
            J = 10 ** (slope * np.log10(R) + 0.05 * rng.normal())
            jac.append({"graph_id": g, "u": k, "v": 0, "resistance": R, "hops": hops, "n_nodes": n,
                        "lcc_size": n, "jacobian": J})
        rng_rows.append({"graph_id": g, "v": 0, "n_nodes": n, "lcc_size": n, "range_hops": 1.0 + 0.01 * n,
                         "range_res": 1.0, "range_hops_fro": 1.0, "range_res_fro": 1.0, "total_influence": 1.0})
        if arch == "hybrid" and alpha > 0:
            ent.append({"graph_id": g, "n_nodes": n, "layer": 0, "head": 0, "entropy": 1.0,
                        "entropy_norm": entropy_r * n + 0.01 * rng.normal()})
    (path / "measurements.json").write_text(json.dumps(
        {"jacobian": jac, "entropy": ent, "range_node": rng_rows, "range_graph": [], "summary": {}}))
    return path


@pytest.fixture
def runs_dir(tmp_path):
    for seed in range(3):
        _fake_run(tmp_path, "hybrid", 0.0, seed, slope=-2.0, ap_shift=1.0)
        _fake_run(tmp_path, "hybrid", 1.0, seed, slope=-0.5, ap_shift=0.2, entropy_r=0.02)
        _fake_run(tmp_path, "gcn", 0.5, seed, slope=-1.5, ap_shift=0.6)
    return tmp_path


def test_fast_ap_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = (rng.random(50) < 0.3).astype(float)
        s = np.round(rng.random(50), 1)          # many ties
        if y.sum() == 0:
            continue
        assert A._ap(y, s) == pytest.approx(average_precision_score(y, s))
    Y, S = (rng.random((40, 4)) < 0.5).astype(float), rng.random((40, 4))
    assert A.fast_macro_ap(Y, S) == pytest.approx(macro_ap(Y, S))


def test_collect_and_performance(runs_dir):
    runs = A.collect_runs(str(runs_dir))
    assert len(runs) == 9 and runs["complete"].all() and runs["has_preds"].all()
    assert set(runs["model"]) == {"alpha=0", "alpha=1", "GCN"}
    perf = A.performance_table(runs)
    ours = perf[perf["source"] == "ours"].set_index("model")
    assert list(ours.index) == ["GCN", "alpha=0", "alpha=1"]
    assert (ours["n_seeds"] == 3).all()
    assert ours.loc["alpha=0", "test_mean"] > ours.loc["alpha=1", "test_mean"]
    assert (ours["ci_low"] <= ours["test_mean"]).all() and (ours["test_mean"] <= ours["ci_high"]).all()
    assert set(perf[perf["source"] == "published"]["model"]) == set(A.PUBLISHED)


def test_stratified_ap_bins():
    preds = {"graph_id": np.arange(9), "y": np.eye(3)[np.arange(9) % 3], "logits": np.eye(3)[np.arange(9) % 3]}
    stats = {"n_nodes": np.arange(9, dtype=float), "avg_resistance": np.arange(9, dtype=float)[::-1]}
    t = A.stratified_ap(preds, stats)
    assert list(t["n"]) == [3, 3, 3, 3, 3, 3]
    assert np.allclose(t["ap"], 1.0)


def test_stratified_table_and_bootstrap_reproducible(runs_dir):
    runs = A.collect_runs(str(runs_dir))
    stats = {"n_nodes": np.arange(N_GRAPHS, dtype=float), "avg_resistance": np.linspace(1, 5, N_GRAPHS)}
    a = A.stratified_ap_table(runs, stats, n_boot=50, seed=1)
    b = A.stratified_ap_table(runs, stats, n_boot=50, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 3 * 2 * 3 and (a["n_graphs"] == 20).all()
    assert (a["boot_ci_low"] <= a["boot_ci_high"]).all()


def test_jacobian_analysis_recovers_slopes(runs_dir):
    runs = A.collect_runs(str(runs_dir))
    res = A.jacobian_analysis(runs, n_boot=100, seed=0)
    s = res["slopes"].set_index("model")
    assert s.loc["alpha=0", "slope_res"] == pytest.approx(-2.0, abs=0.05)
    assert s.loc["alpha=1", "slope_res"] == pytest.approx(-0.5, abs=0.05)
    assert s.loc["alpha=0", "slope_res_ci_low"] < -2.0 + 0.05 and s.loc["alpha=0", "slope_res_ci_high"] > -2.05
    d = res["differences"]
    row = d[(d["model"] == "alpha=0") & (d["stat"] == "slope_res")].iloc[0]
    assert row["diff"] == pytest.approx(-1.5, abs=0.07) and row["excludes_zero"]
    assert set(d["model"]) == {"alpha=0", "GCN"}
    col = res["collinearity"]
    assert col["n_pairs"] == 160 and col["pearson_r"] > 0.9
    again = A.jacobian_analysis(runs, n_boot=100, seed=0)
    pd.testing.assert_frame_equal(res["slopes"], again["slopes"])


def test_entropy_and_range(runs_dir):
    runs = A.collect_runs(str(runs_dir))
    e = A.entropy_analysis(runs, n_boot=100)
    assert list(e["model"]) == ["alpha=1"] and e["r_size"].iloc[0] > 0.9
    assert e["ci_low"].iloc[0] > 0.5 and e["n_seeds"].iloc[0] == 3
    r = A.range_analysis(runs)
    node = r["node"].set_index("model")
    assert node.loc["alpha=0", "range_res_mean"] == pytest.approx(1.0)
    assert node.loc["alpha=0", "range_hops_size_r"] == pytest.approx(1.0)
    assert r["graph"].empty and "hops" in r["reading"]


def test_run_analysis_writes_everything(runs_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(A, "load_structure_stats", lambda cfg, split="test": {
        "n_nodes": np.arange(N_GRAPHS, dtype=float), "avg_resistance": np.linspace(1, 5, N_GRAPHS)})
    out = tmp_path / "report"
    res = A.run_analysis(str(runs_dir), str(out), n_boot=30)
    for name in ("performance", "stratified_ap", "jacobian_slopes", "jacobian_differences", "entropy",
                 "range_node", "summary"):
        assert (out / f"{name}.{'json' if name == 'summary' else 'csv'}").exists(), name
    figs = {Path(f).name for f in res["figures"]}
    for stem in ("ap_bars", "stratified_ap", "jacobian_vs_resistance", "entropy_vs_size", "range_distributions"):
        assert {f"{stem}.pdf", f"{stem}.png"} <= figs
