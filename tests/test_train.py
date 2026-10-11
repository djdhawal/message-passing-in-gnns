"""Tests for the training loop, resume logic and CLI (CPU, fast).

Uses a tiny stand-in model and hand-built datasets so they do not depend on
data.py / models.py; only the end-to-end smoke test needs those.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

import gnn_mech.train as train_mod
from gnn_mech import run as run_mod
from gnn_mech.config import Config, load_config
from gnn_mech.results import RunDir
from gnn_mech.train import evaluate, macro_ap, train, warmup_cosine

REPO = Path(__file__).resolve().parents[1]
OUT_DIM = 3


@dataclass
class Info:
    node_vocab: list
    edge_vocab: list
    out_dim: int
    pe_dim: int
    task: str


class TinyModel(nn.Module):
    """Embedding + dropout + mean pool + linear: enough to exercise train()."""

    def __init__(self, out_dim: int = OUT_DIM):
        super().__init__()
        self.emb = nn.Embedding(5, 8)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(8, out_dim)

    def forward(self, batch):
        h = self.drop(self.emb(batch.x[:, 0]))
        return self.head(global_mean_pool(h, batch.batch))


def make_ds(n: int, seed: int, task: str = "multilabel") -> list[Data]:
    g = torch.Generator().manual_seed(seed)
    graphs = []
    for i in range(n):
        k = int(torch.randint(3, 7, (1,), generator=g))
        x = torch.randint(0, 5, (k, 1), generator=g)
        ei = torch.stack([torch.arange(k - 1), torch.arange(1, k)])
        ei = torch.cat([ei, ei.flip(0)], 1)
        if task == "multilabel":
            y = torch.stack([(x == c).any().float() for c in range(OUT_DIM)]).view(1, -1)
        else:
            y = torch.stack([(x == c).float().mean() for c in range(OUT_DIM)]).view(1, -1)
        graphs.append(Data(x=x, edge_index=ei, y=y, graph_id=torch.tensor([i])))
    return graphs


def make_cfg(tmp_path, **train_kw) -> Config:
    cfg = Config(exp_name="t", out_dir=str(tmp_path))
    cfg.train.device = "cpu"
    cfg.train.epochs = 4
    cfg.train.batch_size = 8
    cfg.train.warmup_epochs = 1
    cfg.train.checkpoint_every = 1
    for k, v in train_kw.items():
        setattr(cfg.train, k, v)
    return cfg


def fit(cfg: Config, task: str = "multilabel") -> tuple[dict, RunDir]:
    torch.manual_seed(0)
    info = Info([5], [1], OUT_DIM, 0, task)
    run = RunDir(cfg)
    metrics = train(cfg, TinyModel(), make_ds(24, 0, task), make_ds(12, 1, task),
                    make_ds(12, 2, task), info, run)
    return metrics, run


# ---------------------------------------------------------------- schedule

def test_warmup_cosine_values():
    lrs = [warmup_cosine(e, 2, 10) for e in range(10)]
    assert lrs[0] == pytest.approx(0.5)
    assert lrs[1] == pytest.approx(1.0)
    assert lrs[2] == pytest.approx(1.0)                      # cosine starts at its peak
    assert lrs[6] == pytest.approx(0.5)                      # halfway through decay
    assert warmup_cosine(10, 2, 10) == pytest.approx(0.0)    # reaches 0 at `epochs`
    assert all(a >= b for a, b in zip(lrs[1:], lrs[2:]))     # monotone after warmup
    assert warmup_cosine(0, 0, 4) == pytest.approx(1.0)      # no warmup


def test_history_lr_follows_schedule(tmp_path):
    cfg = make_cfg(tmp_path, epochs=5, warmup_epochs=2, lr=1e-2)
    _, run = fit(cfg)
    lrs = [h["lr"] for h in run.load_json("history.json")]
    expected = [1e-2 * warmup_cosine(e, 2, 5) for e in range(5)]
    assert lrs == pytest.approx(expected)


# ---------------------------------------------------------------- evaluate

class ScoreModel(nn.Module):
    """Returns per-graph logits stored on the data, for exact metric checks."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        return batch.logit.view(batch.num_graphs, -1)


def test_evaluate_ap_toy():
    # col 0 perfectly ranked (AP 1); col 1 has no positives (skipped);
    # col 2: y=[1,0,1,0] ranked by scores .9,.8,.7,.1 -> AP = (1 + 2/3) / 2.
    y = torch.tensor([[1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float)
    p = torch.tensor([[.9, .5, .9], [.8, .5, .8], [.2, .5, .7], [.1, .5, .1]])
    logit = torch.logit(p)
    ds = [Data(x=torch.zeros(1, 1, dtype=torch.long), edge_index=torch.empty(2, 0, dtype=torch.long),
               y=y[i:i + 1], logit=logit[i:i + 1]) for i in range(4)]
    out = evaluate(ScoreModel(), DataLoader(ds, batch_size=3), Info([1], [1], 3, 0, "multilabel"), "cpu")
    assert out["ap"] == pytest.approx((1.0 + (1 + 2 / 3) / 2) / 2)
    bce = nn.functional.binary_cross_entropy_with_logits(logit, y).item()
    assert out["loss"] == pytest.approx(bce, rel=1e-5)
    assert math.isnan(macro_ap(np.zeros((3, 2)), np.ones((3, 2))))


def test_evaluate_regression_mae():
    y = torch.tensor([[1.0, 2.0], [0.0, 0.0]])
    pred = torch.tensor([[1.5, 2.0], [0.0, -1.0]])
    ds = [Data(x=torch.zeros(1, 1, dtype=torch.long), edge_index=torch.empty(2, 0, dtype=torch.long),
               y=y[i:i + 1], logit=pred[i:i + 1]) for i in range(2)]
    out = evaluate(ScoreModel(), DataLoader(ds, batch_size=2), Info([1], [1], 2, 0, "regression"), "cpu")
    assert out["mae"] == pytest.approx(1.5 / 4)
    assert out["loss"] == pytest.approx(1.5 / 4)


# ---------------------------------------------------------------- train outputs

def test_train_writes_outputs(tmp_path):
    metrics, run = fit(make_cfg(tmp_path))
    for name in ("history.json", "metrics.json", "best.pt", "checkpoint.pt", "config.json"):
        assert run.file(name).exists(), name
    hist = run.load_json("history.json")
    assert [h["epoch"] for h in hist] == [1, 2, 3, 4]
    assert {"epoch", "lr", "train_loss", "val_loss", "val_ap", "seconds"} <= set(hist[0])
    on_disk = run.load_json("metrics.json")
    assert on_disk["epochs_run"] == 4 and on_disk["config_hash"] == run.cfg.hash()
    assert set(on_disk) == {"best_epoch", "val", "test", "n_params", "epochs_run", "seconds", "config_hash"}
    assert on_disk["n_params"] == sum(p.numel() for p in TinyModel().parameters())
    assert "ap" in on_disk["test"] and "loss" in on_disk["test"]
    best_val = max(h["val_ap"] for h in hist)
    assert on_disk["val"]["ap"] == pytest.approx(best_val)
    assert hist[on_disk["best_epoch"] - 1]["val_ap"] == pytest.approx(best_val)
    # best.pt reproduces the best-epoch validation score
    model = TinyModel()
    model.load_state_dict(run.load_torch("best.pt"))
    val = evaluate(model, DataLoader(make_ds(12, 1), batch_size=8), Info([5], [1], OUT_DIM, 0, "multilabel"), "cpu")
    assert val["ap"] == pytest.approx(best_val)


def test_train_regression(tmp_path):
    metrics, run = fit(make_cfg(tmp_path), task="regression")
    hist = run.load_json("history.json")
    assert "val_mae" in hist[0]
    assert metrics["val"]["mae"] == pytest.approx(min(h["val_mae"] for h in hist))


# ---------------------------------------------------------------- best epoch / early stop

def _scripted_evaluate(monkeypatch, val_scores, metric="ap"):
    """Replace evaluate(): val returns scripted scores, test returns 100 + epoch of the call."""
    state = {"epoch": 0, "test_calls": []}

    def fake(model, loader, info, device):
        if loader.dataset.tag == "val":
            state["epoch"] += 1
            return {"loss": 0.0, metric: val_scores[state["epoch"] - 1]}
        state["test_calls"].append(state["epoch"])
        return {"loss": 0.0, metric: 100.0 + state["epoch"]}

    monkeypatch.setattr(train_mod, "evaluate", fake)
    return state


class Tagged(list):
    tag = ""


def _tagged(ds, tag):
    out = Tagged(ds)
    out.tag = tag
    return out


def _fit_tagged(cfg, task="multilabel"):
    info = Info([5], [1], OUT_DIM, 0, task)
    return train(cfg, TinyModel(), _tagged(make_ds(16, 0, task), "train"),
                 _tagged(make_ds(8, 1, task), "val"), _tagged(make_ds(8, 2, task), "test"),
                 info, RunDir(cfg))


def test_best_epoch_selection_ap(tmp_path, monkeypatch):
    state = _scripted_evaluate(monkeypatch, [0.3, 0.7, 0.5, 0.6])
    m = _fit_tagged(make_cfg(tmp_path))
    assert m["best_epoch"] == 2
    assert m["val"]["ap"] == 0.7
    assert m["test"]["ap"] == 102.0              # test@best, not test@last
    assert state["test_calls"] == [1, 2]         # test evaluated only at improvements


def test_best_epoch_selection_mae(tmp_path, monkeypatch):
    _scripted_evaluate(monkeypatch, [0.9, 0.4, 0.2, 0.5], metric="mae")
    m = _fit_tagged(make_cfg(tmp_path), task="regression")
    assert m["best_epoch"] == 3 and m["test"]["mae"] == 103.0


def test_early_stopping(tmp_path, monkeypatch):
    _scripted_evaluate(monkeypatch, [0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    m = _fit_tagged(make_cfg(tmp_path, epochs=6, patience=2))
    assert m["best_epoch"] == 1 and m["epochs_run"] == 3
    # resuming a stopped run does nothing more
    m2 = _fit_tagged(make_cfg(tmp_path, epochs=6, patience=2))
    assert m2["epochs_run"] == 3


# ---------------------------------------------------------------- resume

class Interrupt(Exception):
    pass


def _interrupt_at(monkeypatch, epoch: int):
    real = train_mod._train_epoch
    calls = {"n": 0}

    def wrapped(*a, **kw):
        calls["n"] += 1
        if calls["n"] == epoch:
            raise Interrupt
        return real(*a, **kw)

    monkeypatch.setattr(train_mod, "_train_epoch", wrapped)


@pytest.mark.parametrize("checkpoint_every,interrupt_at", [(1, 3), (2, 4)])
def test_resume_matches_uninterrupted(tmp_path, monkeypatch, checkpoint_every, interrupt_at):
    ref_metrics, ref_run = fit(make_cfg(tmp_path / "ref", checkpoint_every=checkpoint_every))
    ref_hist = ref_run.load_json("history.json")

    cfg = make_cfg(tmp_path / "int", checkpoint_every=checkpoint_every)
    with monkeypatch.context() as mp:
        _interrupt_at(mp, interrupt_at)
        with pytest.raises(Interrupt):
            fit(cfg)
    run = RunDir(cfg)
    assert not run.is_complete()
    assert torch.load(run.file("checkpoint.pt"), weights_only=False)["epoch"] == 2

    metrics, run = fit(cfg)   # fresh model, resumes from checkpoint.pt
    hist = run.load_json("history.json")
    assert [h["epoch"] for h in hist] == [1, 2, 3, 4]
    assert metrics["epochs_run"] == 4
    for a, b in zip(hist, ref_hist):
        assert a["lr"] == pytest.approx(b["lr"])
        assert a["train_loss"] == pytest.approx(b["train_loss"], rel=1e-6)
        assert a["val_ap"] == pytest.approx(b["val_ap"], rel=1e-6)
    assert metrics["best_epoch"] == ref_metrics["best_epoch"]
    assert metrics["test"]["ap"] == pytest.approx(ref_metrics["test"]["ap"], rel=1e-6)


def test_resume_extends_epochs(tmp_path):
    cfg = make_cfg(tmp_path, epochs=2)
    fit(cfg)
    cfg4 = make_cfg(tmp_path, epochs=4)
    cfg4.train.epochs = 4
    # epochs is part of the config hash, so point the new config at the same folder
    run = RunDir(cfg)
    run.cfg = cfg4
    metrics = train(cfg4, TinyModel(), make_ds(24, 0), make_ds(12, 1), make_ds(12, 2),
                    Info([5], [1], OUT_DIM, 0, "multilabel"), run)
    hist = run.load_json("history.json")
    assert [h["epoch"] for h in hist] == [1, 2, 3, 4]
    assert metrics["epochs_run"] == 4


# ---------------------------------------------------------------- CLI

def test_cli_overrides():
    args, cfg = run_mod.parse_args([
        "--config", str(REPO / "configs/smoke.yaml"), "--seed", "3", "--no-measure",
        "model.alpha=0.25", "train.epochs=7", "out_dir=/tmp/x"])
    assert cfg.seed == 3 and cfg.model.alpha == 0.25 and cfg.train.epochs == 7
    assert cfg.out_dir == "/tmp/x" and cfg.data.name == "synthetic-tiny"
    assert args.no_measure and not args.force
    with pytest.raises(KeyError):
        run_mod.parse_args(["--config", str(REPO / "configs/smoke.yaml"), "model.nope=1"])


@pytest.mark.parametrize("name", ["smoke", "peptides_func_hybrid", "peptides_func_gcn", "pilot_equivalence"])
def test_configs_load(name):
    cfg = load_config(str(REPO / f"configs/{name}.yaml"))
    assert cfg.exp_name


def test_run_skip_force_measure(tmp_path, monkeypatch):
    """run(): skip when complete, measure-only when measurements are missing, --force retrains."""
    import sys
    import types

    calls = {"train": 0, "measure": 0}
    info = Info([5], [1], OUT_DIM, 0, "multilabel")
    data_mod = types.SimpleNamespace(load_dataset=lambda c: (make_ds(16, 0), make_ds(8, 1), make_ds(8, 2), info))
    models_mod = types.SimpleNamespace(build_model=lambda c, i: TinyModel())

    def fake_run_all(model, dataset, cfg, device):
        calls["measure"] += 1
        return {"jacobian": [], "entropy": [], "summary": {"n": len(dataset)}}

    def counting_train(*a, **kw):
        calls["train"] += 1
        return train(*a, **kw)

    monkeypatch.setitem(sys.modules, "gnn_mech.data", types.ModuleType("gnn_mech.data"))
    monkeypatch.setitem(sys.modules, "gnn_mech.models", types.ModuleType("gnn_mech.models"))
    monkeypatch.setitem(sys.modules, "gnn_mech.measure", types.ModuleType("gnn_mech.measure"))
    sys.modules["gnn_mech.data"].load_dataset = data_mod.load_dataset
    sys.modules["gnn_mech.models"].build_model = models_mod.build_model
    sys.modules["gnn_mech.measure"].run_all = fake_run_all
    monkeypatch.setattr(run_mod, "train", counting_train)

    cfg = make_cfg(tmp_path, epochs=2)
    cfg.measure.split = "val"
    m = run_mod.run(cfg, measure=False)
    assert calls == {"train": 1, "measure": 0} and m["epochs_run"] == 2
    m = run_mod.run(cfg)                       # trained already: measure only
    assert calls == {"train": 1, "measure": 1}
    assert json.loads((Path(m["run_dir"]) / "measurements.json").read_text())["summary"]["n"] == 8
    run_mod.run(cfg)                           # fully complete: skip
    assert calls == {"train": 1, "measure": 1}
    run_mod.run(cfg, force=True)               # force: retrain from scratch and re-measure
    assert calls == {"train": 2, "measure": 2}
    assert [h["epoch"] for h in json.loads((Path(m["run_dir"]) / "history.json").read_text())] == [1, 2]

    # predictions: written for val and test, regenerated when missing, deleted by --force
    path = Path(m["run_dir"])
    for split, n in (("val", 8), ("test", 8)):
        with np.load(path / f"preds_{split}.npz") as z:
            assert z["graph_id"].tolist() == list(range(n))
            assert z["y"].shape == z["logits"].shape == (n, OUT_DIM)
    (path / "preds_test.npz").unlink()
    run_mod.run(cfg)                           # preds missing: regenerate without training/measuring
    assert calls == {"train": 2, "measure": 2} and (path / "preds_test.npz").exists()
    assert "preds_val.npz" in run_mod._RUN_FILES and "preds_test.npz" in run_mod._RUN_FILES
    (path / "preds_val.npz").write_bytes(b"stale")
    run_mod.run(cfg, force=True)               # --force deletes and rewrites the prediction files
    with np.load(path / "preds_val.npz") as z:
        assert z["logits"].shape == (8, OUT_DIM)


def test_predict_matches_evaluate():
    ds = make_ds(20, 3)
    model = TinyModel().eval()
    out = train_mod.predict(model, ds, batch_size=6, device=torch.device("cpu"))
    assert out["graph_id"].tolist() == list(range(20))
    ev = evaluate(model, DataLoader(ds, batch_size=6), Info([5], [1], OUT_DIM, 0, "multilabel"), torch.device("cpu"))
    assert macro_ap(out["y"], 1 / (1 + np.exp(-out["logits"]))) == pytest.approx(ev["ap"])
    for d in ds:
        del d.graph_id
    assert train_mod.predict(model, ds, 7, torch.device("cpu"))["graph_id"].tolist() == list(range(20))


def _deps_available() -> bool:
    try:
        import gnn_mech.data  # noqa: F401
        import gnn_mech.measure  # noqa: F401
        import gnn_mech.models  # noqa: F401
        from gnn_mech.data import load_dataset  # noqa: F401
        from gnn_mech.measure import run_all  # noqa: F401
        from gnn_mech.models import build_model  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _deps_available(), reason="data.py / models.py / measure not available")
def test_end_to_end_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    argv = ["--config", "configs/smoke.yaml", "--seed", "0", f"out_dir={tmp_path}",
            f"data.root={tmp_path / 'data'}", "train.device=cpu"]
    metrics = run_mod.main(argv)
    path = Path(metrics["run_dir"])
    for name in ("config.json", "history.json", "metrics.json", "best.pt", "checkpoint.pt", "measurements.json",
                 "preds_val.npz", "preds_test.npz"):
        assert (path / name).exists(), name
    assert metrics["epochs_run"] == 3
    meas = json.loads((path / "measurements.json").read_text())
    assert {"jacobian", "entropy", "range_node", "range_graph", "summary"} <= set(meas)
    assert meas["range_node"] and {"range_hops", "range_res"} <= set(meas["range_node"][0])
    with np.load(path / "preds_test.npz") as z:
        assert z["logits"].shape == (32, 10)
        assert macro_ap(z["y"], 1 / (1 + np.exp(-z["logits"]))) == pytest.approx(metrics["test"]["ap"], abs=1e-6)
    assert run_mod.main(argv)["seconds"] == metrics["seconds"]   # second call is skipped
