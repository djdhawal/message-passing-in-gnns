"""Tests for gnn_mech.measure (CPU, fast).

Unit tests use small stand-in models implementing the HybridGNN measurement API
(embed_inputs, node_embeddings_from_h0, attention_layers, set_attention_cache).
Integration tests against the real HybridGNN are skipped if it cannot be imported.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from gnn_mech.config import MeasureConfig
from gnn_mech.measure import (attention_entropy, jacobian_norms, measure_entropy, measure_jacobians,
                              run_all, summarize_jacobian)

CPU = torch.device("cpu")


# ---------------------------------------------------------------- helpers

def path_edges(n: int, offset: int = 0) -> torch.Tensor:
    src = torch.arange(offset, offset + n - 1)
    return torch.cat([torch.stack([src, src + 1]), torch.stack([src + 1, src])], dim=1)


def path_graph(n: int, gid: int = 0, n_feat: int = 1) -> Data:
    return Data(x=torch.randint(0, 4, (n, n_feat)), edge_index=path_edges(n),
                graph_id=torch.tensor([gid]), num_nodes=n)


class PropModel(nn.Module):
    """K rounds of h <- act((A + I) h W_k): receptive field is exactly K hops."""

    def __init__(self, k: int = 3, dim: int = 6, nonlinear: bool = True):
        super().__init__()
        self.emb = nn.Embedding(4, dim)
        self.ws = nn.ParameterList([nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim)) for _ in range(k)])
        self.nonlinear = nonlinear

    def embed_inputs(self, batch) -> torch.Tensor:
        return self.emb(batch.x[:, 0])

    def node_embeddings_from_h0(self, h0, batch) -> torch.Tensor:
        src, dst = batch.edge_index
        h = h0
        for w in self.ws:
            h = (h + torch.zeros_like(h).index_add(0, dst, h[src])) @ w
            if self.nonlinear:
                h = torch.tanh(h)
        return h

    def attention_layers(self) -> list:
        return []

    def set_attention_cache(self, on: bool) -> None:
        pass

    def forward(self, batch):
        return self.node_embeddings_from_h0(self.embed_inputs(batch), batch)


class FixedAttn(nn.Module):
    """Stand-in GlobalAttn: head 0 uniform over real nodes, head 1 attends to self."""

    def __init__(self):
        super().__init__()
        self.cache = False
        self.last_weights = None
        self.last_mask = None

    def forward(self, x, batch):
        xd, mask = to_dense_batch(x, batch)
        b, n = mask.shape
        uni = mask[:, None, :].float().expand(b, n, n) / mask.sum(-1)[:, None, None]
        eye = torch.eye(n).expand(b, n, n) * mask[:, None, :]
        w = torch.stack([uni, eye], dim=1)
        if self.cache:
            self.last_weights, self.last_mask = w, mask
        return (w.mean(1) @ xd)[mask]


class AttnModel(PropModel):
    def __init__(self, n_layers: int = 2):
        super().__init__(k=1)
        self.attns = nn.ModuleList([FixedAttn() for _ in range(n_layers)])

    def node_embeddings_from_h0(self, h0, batch):
        h = super().node_embeddings_from_h0(h0, batch)
        for a in self.attns:
            h = h + a(h, batch.batch)
        return h

    def attention_layers(self):
        return list(self.attns)

    def set_attention_cache(self, on: bool) -> None:
        for a in self.attns:
            a.cache = on


def hop(u: int, v: int) -> int:
    return abs(u - v)


# ---------------------------------------------------------------- attention_entropy

def test_entropy_uniform_and_onehot():
    n = 7
    uni = torch.full((1, 1, n, n), 1.0 / n)
    onehot = torch.eye(n).reshape(1, 1, n, n)
    mask = torch.ones(1, n, dtype=torch.bool)
    assert attention_entropy(uni, mask)[0, 0] == pytest.approx(math.log(n))
    assert attention_entropy(onehot, mask)[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_entropy_ignores_padding():
    model = AttnModel(n_layers=1)
    model.set_attention_cache(True)
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([path_graph(3), path_graph(8)])
    model(batch)
    a = model.attns[0]
    ent = attention_entropy(a.last_weights, a.last_mask)
    assert ent.shape == (2, 2)
    np.testing.assert_allclose(ent[:, 0], [math.log(3), math.log(8)], atol=1e-10)
    np.testing.assert_allclose(ent[:, 1], [0.0, 0.0], atol=1e-12)
    # garbage in padded keys/queries must not matter
    w = a.last_weights.clone()
    w[0, :, 3:, :] = 0.3
    w[0, :, :, 3:] = 0.3
    np.testing.assert_allclose(attention_entropy(w, a.last_mask), ent, atol=1e-10)


def test_entropy_is_per_head_not_averaged():
    n = 5
    w = torch.stack([torch.full((n, n), 1.0 / n), torch.eye(n)])[None]
    ent = attention_entropy(w, torch.ones(1, n, dtype=torch.bool))
    assert ent[0, 0] == pytest.approx(math.log(n))
    assert ent[0, 1] == pytest.approx(0.0, abs=1e-12)
    averaged = attention_entropy(w.mean(1, keepdim=True), torch.ones(1, n, dtype=torch.bool))[0, 0]
    assert averaged > ent[0].mean() + 0.1


def test_measure_entropy_rows():
    ds = [path_graph(n, gid=100 + i) for i, n in enumerate([1, 3, 6, 9, 12])]
    model = AttnModel(n_layers=2)
    rows = measure_entropy(model, ds, MeasureConfig(n_graphs_entropy=5), CPU, batch_size=2)
    assert {"graph_id", "n_nodes", "layer", "head", "entropy", "entropy_norm"} <= set(rows[0])
    assert len(rows) == 4 * 2 * 2  # single-node graph skipped
    for r in rows:
        assert r["graph_id"] == 100 + [1, 3, 6, 9, 12].index(r["n_nodes"])
        target = 1.0 if r["head"] == 0 else 0.0
        assert r["entropy_norm"] == pytest.approx(target, abs=1e-6)
    assert all(not a.cache for a in model.attns)
    assert measure_entropy(PropModel(), ds, MeasureConfig(), CPU) == []


# ---------------------------------------------------------------- jacobian_norms

def test_jacobian_receptive_field():
    torch.manual_seed(0)
    k, n = 3, 10
    model = PropModel(k=k)
    data = path_graph(n)
    pairs = [(u, 0) for u in range(1, n)] + [(u, 5) for u in range(n) if u != 5]
    J = jacobian_norms(model, data, pairs, CPU)
    for (u, v), j in zip(pairs, J):
        if hop(u, v) > k:
            assert j == 0.0
        else:
            assert j > 0.0


def test_jacobian_batched_matches_loop():
    torch.manual_seed(1)
    model = PropModel(k=4, dim=8)
    data = path_graph(12)
    rng = np.random.default_rng(0)
    pairs = [tuple(int(a) for a in rng.choice(12, 2, replace=False)) for _ in range(20)]
    fast = jacobian_norms(model, data, pairs, CPU, method="batched")
    slow = jacobian_norms(model, data, pairs, CPU, method="loop")
    np.testing.assert_allclose(fast, slow, atol=1e-6, rtol=1e-5)


def test_jacobian_analytic_linear():
    torch.manual_seed(2)
    k, n = 3, 8
    model = PropModel(k=k, dim=5, nonlinear=False)
    data = path_graph(n)
    A = torch.zeros(n, n)
    A[data.edge_index[1], data.edge_index[0]] = 1.0
    P = torch.linalg.matrix_power(A + torch.eye(n), k)
    W = (model.ws[0] @ model.ws[1] @ model.ws[2]).detach()
    pairs = [(u, v) for u in range(n) for v in range(n) if u != v]
    J = jacobian_norms(model, data, pairs, CPU)
    expected = [float(P[v, u] * torch.linalg.norm(W)) for u, v in pairs]
    np.testing.assert_allclose(J, expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- measure_jacobians

KEYS = {"graph_id", "u", "v", "resistance", "hops", "n_nodes", "jacobian"}


def test_measure_jacobians_rows_on_paths():
    ds = [path_graph(n, gid=10 + i) for i, n in enumerate([3, 6, 9, 15, 20])]
    cfg = MeasureConfig(n_graphs_jacobian=5, pairs_per_graph=8, seed=3)
    rows = measure_jacobians(PropModel(k=2), ds, cfg, CPU)
    assert rows and all(KEYS <= set(r) for r in rows)
    assert {r["graph_id"] for r in rows} == {11, 12, 13, 14}  # n=3 graph has LCC < 4
    seen = set()
    for r in rows:
        assert r["u"] != r["v"]
        assert r["resistance"] == pytest.approx(r["hops"], abs=1e-8)  # tree: R = hop distance
        assert r["hops"] == hop(r["u"], r["v"])
        assert (r["jacobian"] == 0.0) == (r["hops"] > 2)
        key = (r["graph_id"], min(r["u"], r["v"]), max(r["u"], r["v"]))
        assert key not in seen
        seen.add(key)
    per_graph = {g: sum(r["graph_id"] == g for r in rows) for g in (11, 12, 13, 14)}
    assert all(c == 8 for c in per_graph.values())
    big = [r["resistance"] for r in rows if r["graph_id"] == 14]
    assert max(big) - min(big) >= 10  # stratification spans the R range


def test_measure_jacobians_lcc_only():
    ei = torch.cat([path_edges(6), path_edges(3, offset=6)], dim=1)
    data = Data(x=torch.zeros(9, 1, dtype=torch.long), edge_index=ei, num_nodes=9)
    cfg = MeasureConfig(n_graphs_jacobian=1, pairs_per_graph=50)
    rows = measure_jacobians(PropModel(k=2), [data], cfg, CPU)
    assert len(rows) == 15  # all C(6, 2) pairs of the LCC
    assert all(r["u"] < 6 and r["v"] < 6 for r in rows)
    assert all(r["graph_id"] == 0 and r["n_nodes"] == 9 for r in rows)


# ---------------------------------------------------------------- run_all / summary

def test_summary_recovers_power_law():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        R, n = rng.uniform(0.5, 30), int(rng.integers(20, 200))
        J = 3.0 * R ** -1.5 * n ** 0.4 * 10 ** rng.normal(0, 0.01)
        rows.append({"graph_id": i, "u": 0, "v": 1, "resistance": R, "hops": max(1, round(R)),
                     "n_nodes": n, "jacobian": J})
    rows.append({**rows[0], "jacobian": 0.0})
    s = summarize_jacobian(rows)
    assert s["n_zero"] == 1
    sc = s["size_controlled"]
    assert sc["coef"]["log10_resistance"] == pytest.approx(-1.5, abs=0.01)
    assert sc["coef"]["log10_n_nodes"] == pytest.approx(0.4, abs=0.01)
    assert sc["se"]["log10_resistance"] < 0.01
    assert s["vs_resistance"]["n"] == 300
    assert s["vs_resistance"]["slope"] < -1.0
    assert summarize_jacobian([]) == {"n_pairs": 0, "n_zero": 0}


def test_run_all_without_attention():
    ds = [path_graph(n, gid=i) for i, n in enumerate([5, 8, 12, 16])]
    out = run_all(PropModel(k=2), ds, MeasureConfig(n_graphs_jacobian=4, pairs_per_graph=6), CPU)
    assert set(out) == {"jacobian", "entropy", "summary"}
    assert out["entropy"] == [] and out["summary"]["entropy"] == {"n_graphs": 0}
    sj = out["summary"]["jacobian"]
    assert {"n_pairs", "n_zero", "vs_resistance", "vs_hops", "size_controlled"} <= set(sj)
    assert {"slope", "intercept", "r", "p", "n"} <= set(sj["vs_resistance"])


def test_run_all_with_attention():
    ds = [path_graph(n, gid=i) for i, n in enumerate([5, 8, 12, 16, 20])]
    cfg = MeasureConfig(n_graphs_jacobian=3, pairs_per_graph=4, n_graphs_entropy=5)
    se = run_all(AttnModel(), ds, cfg, CPU)["summary"]["entropy"]
    assert se["n_graphs"] == 5
    assert se["size_vs_entropy"]["r"] > 0.9
    assert [d["layer"] for d in se["per_layer"]] == [0, 1]
    assert se["mean_entropy_norm"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------- integration with HybridGNN

def _real_model(alpha: float, layers: int):
    models = pytest.importorskip("gnn_mech.models")
    data_mod = pytest.importorskip("gnn_mech.data")
    from gnn_mech.config import ModelConfig
    info = data_mod.DataInfo(node_vocab=[5] * 9, edge_vocab=[4] * 3, out_dim=3, pe_dim=4, task="multilabel")
    cfg = ModelConfig(alpha=alpha, hidden=16, layers=layers, heads=2, dropout=0.1, attn_dropout=0.1)
    torch.manual_seed(0)
    return models.HybridGNN(cfg, info).eval()


def real_path_graph(n: int, gid: int = 0) -> Data:
    ei = path_edges(n)
    return Data(x=torch.randint(0, 5, (n, 9)), edge_index=ei, edge_attr=torch.randint(0, 4, (ei.size(1), 3)),
                pe=torch.rand(n, 4), y=torch.zeros(1, 3), graph_id=torch.tensor([gid]), num_nodes=n)


def test_integration_mpnn_receptive_field():
    L = 3
    model = _real_model(0.0, L)
    data = real_path_graph(14)
    pairs = [(u, 0) for u in range(1, 14)]
    J = jacobian_norms(model, data, pairs, CPU)
    for (u, _), j in zip(pairs, J):
        assert (j == 0.0) == (u > L), (u, j)
    slow = jacobian_norms(model, data, pairs, CPU, method="loop")
    np.testing.assert_allclose(J, slow, atol=1e-6, rtol=1e-5)


def test_integration_attention_far_pair():
    model = _real_model(1.0, 2)
    J = jacobian_norms(model, real_path_graph(14), [(13, 0)], CPU, method="batched")
    assert J[0] > 0


def test_integration_entropy_hybrid():
    L = 3
    model = _real_model(0.5, L)
    ds = [real_path_graph(n, gid=i) for i, n in enumerate([4, 7, 10, 13])]
    rows = measure_entropy(model, ds, MeasureConfig(n_graphs_entropy=4), CPU)
    assert {(r["layer"], r["head"]) for r in rows} == {(l, h) for l in range(L) for h in range(2)}
    assert len(rows) == 4 * L * 2
    assert all(0.0 <= r["entropy_norm"] <= 1.0 + 1e-6 for r in rows)
    out = run_all(model, ds, MeasureConfig(n_graphs_jacobian=4, pairs_per_graph=4, n_graphs_entropy=4), CPU)
    assert out["summary"]["entropy"]["n_graphs"] == 4
