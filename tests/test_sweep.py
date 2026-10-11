"""Tests for gnn_mech.sweep (CPU, fast): expansion, status, dry run, skip and resume."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gnn_mech import sweep as sweep_mod
from gnn_mech.config import load_config

REPO = Path(__file__).resolve().parents[1]


def write_sweep(tmp_path, runs, **top) -> str:
    p = tmp_path / "sweep.yaml"
    p.write_text(yaml.safe_dump({**top, "runs": runs}))
    return str(p)


def test_load_sweep_expands_seeds_and_overrides(tmp_path):
    path = write_sweep(tmp_path, [
        {"name": "a", "config": "configs/smoke.yaml", "overrides": ["model.alpha=0.0"], "seeds": [0, 1],
         "est_minutes": 10},
        {"config": "configs/smoke.yaml", "seeds": [2]},
    ], measure_minutes=5)
    runs, mm = sweep_mod.load_sweep(path, [f"out_dir={tmp_path}"])
    assert mm == 5 and len(runs) == 3
    assert [(r.name, r.seed) for r in runs] == [("a", 0), ("a", 1), ("smoke", 2)]
    assert runs[0].cfg.model.alpha == 0.0 and runs[2].cfg.model.alpha == 0.5
    assert all(r.cfg.out_dir == str(tmp_path) for r in runs)
    assert runs[0].cfg.hash() == runs[1].cfg.hash() != runs[2].cfg.hash()
    with pytest.raises(KeyError):
        sweep_mod.load_sweep(write_sweep(tmp_path, [{"config": "configs/smoke.yaml", "seed": 0}]))
    with pytest.raises(FileNotFoundError):
        sweep_mod.load_sweep(write_sweep(tmp_path, [{"config": "configs/nope.yaml"}]))


def test_phase2_sweep_file():
    runs, mm = sweep_mod.load_sweep(str(REPO / "configs/sweeps/phase2.yaml"))
    assert len(runs) == 12 and mm > 0
    alphas = sorted({(r.cfg.model.arch, r.cfg.model.alpha) for r in runs})
    assert alphas == [("gcn", 0.5), ("hybrid", 0.0), ("hybrid", 0.5), ("hybrid", 1.0)]
    assert all(r.cfg.data.pe_dim == 20 and r.cfg.data.name == "Peptides-func" for r in runs)
    assert sorted({r.seed for r in runs}) == [0, 1, 2]
    text = sweep_mod.describe(runs, mm)
    assert "12 runs, 0 done" in text


def test_dry_run_creates_nothing(tmp_path, capsys):
    path = str(REPO / "configs/sweeps/smoke.yaml")
    out = sweep_mod.main([path, "--dry-run", f"out_dir={tmp_path / 'runs'}"])
    assert out["done"] == [] and len(out["runs"]) == 2
    assert not (tmp_path / "runs").exists()
    assert "estimated remaining" in capsys.readouterr().out


def test_status_and_failure_handling(tmp_path, monkeypatch):
    cfg = load_config(str(REPO / "configs/smoke.yaml"), [f"out_dir={tmp_path}"])
    assert sweep_mod.run_status(cfg) == "todo"
    path = sweep_mod._run_path(cfg)
    path.mkdir(parents=True)
    (path / "checkpoint.pt").write_bytes(b"")
    assert sweep_mod.run_status(cfg) == "partial"
    for name in ("metrics.json", "preds_val.npz", "preds_test.npz"):
        (path / name).write_text("{}")
    assert sweep_mod.run_status(cfg, measure=False) == "done"
    assert sweep_mod.run_status(cfg) == "partial"          # measurements missing
    (path / "measurements.json").write_text("{}")
    assert sweep_mod.run_status(cfg) == "done"

    import gnn_mech.run as run_mod
    calls = []

    def fake_run(c, measure=True):
        calls.append(c.seed)
        if c.seed == 1:
            raise RuntimeError("boom")
        return {"run_dir": f"seed{c.seed}"}

    monkeypatch.setattr(run_mod, "run", fake_run)
    sweep = write_sweep(tmp_path, [{"config": str(REPO / "configs/smoke.yaml"), "seeds": [0, 1, 2]}])
    res = sweep_mod.main([sweep, f"out_dir={tmp_path}"])
    assert calls == [0, 1, 2]                              # a failure does not stop the sweep
    assert res["done"] == ["seed0", "seed2"] and [f["seed"] for f in res["failed"]] == [1]


def test_smoke_sweep_end_to_end(tmp_path, monkeypatch):
    """Two-entry smoke sweep trains, predicts and measures; a re-run skips both runs."""
    import json
    monkeypatch.chdir(REPO)
    argv = [str(REPO / "configs/sweeps/smoke.yaml"), f"out_dir={tmp_path}", f"data.root={tmp_path / 'data'}",
            "train.device=cpu"]
    res = sweep_mod.main(argv)
    assert len(res["done"]) == 2 and not res["failed"]
    for d in res["done"]:
        d = Path(d)
        assert (d / "preds_test.npz").exists() and (d / "preds_val.npz").exists()
        meas = json.loads((d / "measurements.json").read_text())
        assert meas["range_node"] and "range_node" in meas["summary"]
    runs, _ = sweep_mod.load_sweep(argv[0], argv[1:])
    assert [sweep_mod.run_status(r.cfg) for r in runs] == ["done", "done"]
    mtimes = [(Path(d) / "metrics.json").stat().st_mtime for d in res["done"]]
    assert sweep_mod.main(argv)["done"] == res["done"]
    assert [(Path(d) / "metrics.json").stat().st_mtime for d in res["done"]] == mtimes


def test_sweep_resumes_interrupted_run(tmp_path, monkeypatch):
    """A run killed mid-training resumes from checkpoint.pt when the sweep is re-run."""
    import json
    import gnn_mech.train as train_mod
    monkeypatch.chdir(REPO)
    argv = [str(REPO / "configs/sweeps/smoke.yaml"), f"out_dir={tmp_path}", f"data.root={tmp_path / 'data'}",
            "train.device=cpu", "--no-measure"]
    real_epoch = train_mod._train_epoch
    count = {"n": 0}

    def dying_epoch(*a, **kw):
        count["n"] += 1
        if count["n"] == 3:                                # die during epoch 3 of the first run
            raise KeyboardInterrupt
        return real_epoch(*a, **kw)

    monkeypatch.setattr(train_mod, "_train_epoch", dying_epoch)
    with pytest.raises(KeyboardInterrupt):
        sweep_mod.main(argv)
    monkeypatch.setattr(train_mod, "_train_epoch", real_epoch)
    runs, _ = sweep_mod.load_sweep(argv[0], argv[1:4])
    assert sweep_mod.run_status(runs[0].cfg, measure=False) == "partial"
    res = sweep_mod.main(argv)
    assert len(res["done"]) == 2
    hist = json.loads((Path(res["done"][0]) / "history.json").read_text())
    assert [h["epoch"] for h in hist] == [1, 2, 3]
