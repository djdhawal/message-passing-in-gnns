"""End-to-end node-level pipeline on a small synthetic dataset built here (no gnn_mech.social)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import gnn_mech.data as data_mod
from gnn_mech import run as run_mod
from gnn_mech.config import Config, ModelConfig
from gnn_mech.data import DataInfo
from gnn_mech.models import MLPBaseline, build_model
from gnn_mech.train import evaluate, node_metrics, predict

FEAT, PE = 5, 4


def _graph(rng: np.random.Generator, gid: int, task: str) -> Data:
    n = int(rng.integers(30, 50))
    edges = [(int(rng.integers(0, i)), i) for i in range(1, n)]
    edges += [(int(a), int(b)) for a, b in rng.integers(0, n, (n // 2, 2)) if a != b]
    e = np.unique(np.sort(np.array(edges), axis=1), axis=0).T
    ei = torch.from_numpy(np.concatenate([e, e[::-1]], axis=1).copy())
    x = torch.from_numpy(rng.normal(size=(n, FEAT)).astype(np.float32))
    nb = torch.zeros(n).index_add(0, ei[1], x[ei[0], 0])          # neighbour signal
    score = x[:, 1] + 0.5 * nb
    if task == "binary":
        y = (score > 0).float()
    elif task == "multiclass":
        y = torch.bucketize(score, torch.tensor([-0.7, 0.7]))
    else:
        y = score
    mask = torch.from_numpy(rng.random(n) < 0.8)
    return Data(x=x, edge_index=ei, edge_attr=torch.zeros(ei.shape[1], 1, dtype=torch.long),
                pe=torch.rand(n, PE), y=y, y_mask=mask, node_id=torch.arange(n) + 1000 * gid,
                graph_id=torch.tensor([gid]), num_nodes=n)


def make_node_dataset(task: str = "binary", sizes=(12, 4, 4)):
    rng = np.random.default_rng(0)
    out, gid = [], 0
    for s in sizes:
        out.append([_graph(rng, gid + i, task) for i in range(s)])
        gid += s
    info = DataInfo(node_vocab=[], edge_vocab=[1], out_dim=3 if task == "multiclass" else 1, pe_dim=PE,
                    task=task, task_level="node", feat_dim=FEAT)
    return (*out, info)


def _cfg(tmp_path, **model_kw) -> Config:
    cfg = Config(exp_name="node", out_dir=str(tmp_path))
    cfg.data.name = "node-synthetic"
    cfg.train.device = "cpu"
    cfg.train.epochs = 3
    cfg.train.batch_size = 4
    cfg.train.warmup_epochs = 1
    cfg.model = ModelConfig(hidden=16, layers=2, heads=2, performer_features=16, **model_kw)
    cfg.measure.n_graphs_jacobian = 2
    cfg.measure.targets_per_graph = 2
    cfg.measure.pairs_per_graph = 4
    cfg.measure.n_graphs_entropy = 3
    cfg.measure.entropy_queries = 5
    cfg.measure.n_graphs_range = 0
    return cfg


VARIANTS = [dict(alpha=a, attn="performer") for a in (0.0, 0.5, 1.0)] + [dict(arch="mlp"), dict(arch="gcn")]


@pytest.mark.parametrize("kw", VARIANTS, ids=lambda kw: "-".join(f"{k}={v}" for k, v in kw.items()))
def test_run_end_to_end(tmp_path, monkeypatch, kw):
    tr, va, te, info = make_node_dataset()
    monkeypatch.setattr(data_mod, "load_dataset", lambda c: (tr, va, te, info))
    cfg = _cfg(tmp_path, **kw)
    m = run_mod.run(cfg)
    path = Path(m["run_dir"])
    metrics = json.loads((path / "metrics.json").read_text())
    assert {"loss", "auroc", "ap"} <= set(metrics["val"]) and {"auroc", "ap"} <= set(metrics["test"])
    assert 0.0 <= metrics["test"]["auroc"] <= 1.0
    n_sup = sum(int(d.y_mask.sum()) for d in te)
    with np.load(path / "preds_test.npz") as z:
        assert set(z.files) == {"graph_id", "node_id", "y", "logits"}
        assert z["logits"].shape == (n_sup, 1) and z["y"].shape == (n_sup,)
        assert set(z["graph_id"].tolist()) == {int(d.graph_id) for d in te}
        exp_nid = np.concatenate([d.node_id[d.y_mask].numpy() for d in te])
        np.testing.assert_array_equal(z["node_id"], exp_nid)
    meas = json.loads((path / "measurements.json").read_text())
    assert meas["range_graph"] == []
    assert meas["jacobian"] and all("excess_res" in r for r in meas["jacobian"])
    assert meas["range_node"]
    sup = {(int(d.graph_id), int(v)) for d in te for v in torch.nonzero(d.y_mask).view(-1)}
    assert all((r["graph_id"], r["v"]) in sup for r in meas["range_node"])
    if kw.get("arch", "hybrid") == "hybrid" and kw["alpha"] > 0:
        ent = meas["entropy"]
        assert ent and all(r["n_queries"] == 5 for r in ent)
        assert all(0 <= r["entropy_norm"] <= 1 + 1e-9 for r in ent)
    else:
        assert meas["entropy"] == []
    if kw.get("arch") == "mlp":   # no message passing: only self-influence
        assert all(r["jacobian"] == 0 for r in meas["jacobian"] if r["u"] != r["v"])
        assert all(r["range_hops"] in (0.0, None) for r in meas["range_node"])


def test_predict_matches_evaluate_and_shapes():
    tr, va, te, info = make_node_dataset()
    torch.manual_seed(0)
    model = build_model(ModelConfig(hidden=16, layers=2, heads=2, alpha=0.5, attn="performer"), info).eval()
    out = model(next(iter(DataLoader(te, batch_size=2))))
    assert out.shape[1] == 1 and out.shape[0] == te[0].num_nodes + te[1].num_nodes
    p = predict(model, te, 3, torch.device("cpu"), level="node")
    ev = evaluate(model, DataLoader(te, batch_size=3), info, torch.device("cpu"))
    again = node_metrics(torch.from_numpy(p["logits"]), torch.from_numpy(p["y"]), "binary")
    assert again["auroc"] == pytest.approx(ev["auroc"]) and again["ap"] == pytest.approx(ev["ap"])


@pytest.mark.parametrize("task,keys", [("multiclass", {"loss", "acc"}), ("regression", {"loss", "r2", "mae"})])
def test_other_tasks_train(tmp_path, monkeypatch, task, keys):
    tr, va, te, info = make_node_dataset(task)
    monkeypatch.setattr(data_mod, "load_dataset", lambda c: (tr, va, te, info))
    cfg = _cfg(tmp_path, alpha=0.5, attn="performer")
    cfg.measure.enabled = False
    m = run_mod.run(cfg)
    assert set(m["val"]) == keys
    with np.load(Path(m["run_dir"]) / "preds_val.npz") as z:
        assert z["logits"].shape[1] == info.out_dim


def test_node_metrics_values():
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])
    m = node_metrics(torch.tensor([[-2.0], [3.0], [1.0], [0.5]]), y, "binary")
    assert m["auroc"] == 1.0 and m["ap"] == 1.0
    assert np.isnan(node_metrics(torch.zeros(3, 1), torch.ones(3), "binary")["auroc"])
    mc = node_metrics(torch.tensor([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0]]), torch.tensor([0, 1, 1]), "multiclass")
    assert mc["acc"] == pytest.approx(2 / 3)
    t = torch.tensor([1.0, 2.0, 3.0])
    r = node_metrics(t[:, None], t, "regression")
    assert r["r2"] == pytest.approx(1.0) and r["mae"] == pytest.approx(0.0)
    assert np.isnan(node_metrics(torch.zeros(0, 1), torch.zeros(0), "regression")["r2"])


def test_train_selects_higher_r2(tmp_path, monkeypatch):
    import gnn_mech.train as train_mod
    assert train_mod.HIGHER_IS_BETTER["r2"] and train_mod.HIGHER_IS_BETTER["auroc"]
    assert not train_mod.HIGHER_IS_BETTER["mae"]
    tr, va, te, info = make_node_dataset("regression")
    scores = iter([0.1, 0.5, 0.3, 0.2])
    real = train_mod.evaluate

    def fake(model, loader, info_, device):
        out = real(model, loader, info_, device)
        if loader.dataset is va:
            out["r2"] = next(scores)
        return out
    monkeypatch.setattr(train_mod, "evaluate", fake)
    from gnn_mech.results import RunDir
    cfg = _cfg(tmp_path, arch="mlp")
    cfg.train.epochs = 4
    torch.manual_seed(0)
    metrics = train_mod.train(cfg, build_model(cfg.model, info), tr, va, te, info, RunDir(cfg))
    assert metrics["best_epoch"] == 2


def test_mlp_graph_level_and_float_encoder():
    info = DataInfo(node_vocab=[], edge_vocab=[1], out_dim=2, pe_dim=0, task="regression", feat_dim=FEAT)
    tr, *_ = make_node_dataset()
    m = build_model(ModelConfig(arch="mlp", hidden=8, layers=2), info)
    assert isinstance(m, MLPBaseline) and m.attention_layers() == []
    b = next(iter(DataLoader(tr, batch_size=3)))
    assert m(b).shape == (3, 2)
    with pytest.raises(ValueError):
        build_model(ModelConfig(arch="mlp"), DataInfo([], [1], 1, 0, "binary", "node", 0))
