"""Full-graph (transductive) mode and sparse k-hop radius attention, on social-tiny."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn_mech import run as run_mod
from gnn_mech.config import Config, DataConfig, MeasureConfig, ModelConfig, TrainConfig
from gnn_mech.layers import RadiusAttn, hop_pairs, hop_reach
from gnn_mech.social import sources, subgraph
from gnn_mech.social.dataset import data_key, load_meta, load_social, rwse, rwse_sampled, tiny_graph


@pytest.fixture(autouse=True)
def _seeded():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        yield


def _cfg(tmp_path, **kw):
    base = dict(name="social-tiny", root=str(tmp_path), pe_dim=0, subgraph_size=50, sampler="full")
    base.update(kw)
    return DataConfig(**base)


# ----------------------------------------------------------------------------- data


def test_each_split_is_one_whole_graph(tmp_path):
    tr, va, te, info = load_social(_cfg(tmp_path))
    g = sources.load_graph("social-tiny", str(tmp_path))
    assert info.task_level == "node" and info.pe_dim == 0
    for ds in (tr, va, te):
        assert len(ds) == 1
        d = ds[0]
        assert d.num_nodes == g.num_nodes == 600
        assert torch.equal(d.edge_index, g.edge_index)
        assert torch.equal(d.node_id, torch.arange(600)) and d.graph_id.tolist() == [0]
        assert d.x.shape == (600, info.feat_dim) and d.y.shape == (600,) and d.y_mask.dtype == torch.bool
        assert d.edge_attr.shape == (d.edge_index.size(1), 1)
    # features and targets are the same tensor content in every split (transductive)
    assert torch.equal(tr[0].x, te[0].x) and torch.equal(tr[0].y, va[0].y)


def test_masks_disjoint_and_match_parts(tmp_path):
    cfg = _cfg(tmp_path)
    tr, va, te, _ = load_social(cfg)
    masks = [ds[0].y_mask.numpy() for ds in (tr, va, te)]
    assert not (masks[0] & masks[1]).any() and not (masks[0] & masks[2]).any() and not (masks[1] & masks[2]).any()
    assert (masks[0] | masks[1] | masks[2]).all()          # social-tiny `label` is defined everywhere
    g = sources.load_graph("social-tiny", str(tmp_path))
    lab = subgraph.label_parts(g.edge_index.numpy(), g.num_nodes, 50, seed=0)
    for m, split in zip(masks, ("train", "val", "test")):
        np.testing.assert_array_equal(m, np.isin(lab["part"], lab["assign"][split]))
    # the same parts label the same nodes as the subgraph sampler's cores
    sub = load_social(_cfg(tmp_path, sampler="partition"))[2]
    core = np.zeros(600, dtype=bool)
    for i in range(len(sub)):
        core[sub[i].node_id[sub[i].y_mask].numpy()] = True
    np.testing.assert_array_equal(core, masks[2])
    meta = load_meta(cfg)
    assert meta["sampler"] == "full" and meta["test"]["n_labelled"] == int(masks[2].sum())


def test_cache_key_differs_from_subgraph_mode(tmp_path):
    assert data_key(_cfg(tmp_path)) != data_key(_cfg(tmp_path, sampler="partition"))


def test_standardization_uses_train_labelled_only(tmp_path):
    tr, va, te, _ = load_social(_cfg(tmp_path))
    x, m = tr[0].x.numpy(), tr[0].y_mask.numpy()
    np.testing.assert_allclose(x[m].mean(0), 0, atol=1e-5)
    np.testing.assert_allclose(x[m].std(0), 1, atol=1e-4)
    assert np.abs(x[~m].mean(0)).max() > 1e-3           # other nodes are not standardized to themselves


def test_standardization_skips_unlabelled_train_nodes(tmp_path, monkeypatch):
    real = sources.node_target

    def masked(name, graph, target):
        y, task = real(name, graph, target)
        y = np.asarray(y, dtype=float).copy()
        y[::3] = np.nan                                   # every third node has no label
        return y, task

    monkeypatch.setattr(sources, "node_target", masked)
    tr, _, _, _ = load_social(_cfg(tmp_path))
    x, m = tr[0].x.numpy(), tr[0].y_mask.numpy()
    assert not m[::3].any()
    np.testing.assert_allclose(x[m].mean(0), 0, atol=1e-5)


def test_rwse_sampled_close_to_exact():
    g = tiny_graph()
    ei = g.edge_index.numpy()
    exact = rwse(ei, g.num_nodes, 4)
    est = rwse_sampled(ei, g.num_nodes, 4, walks=2000, seed=0)
    assert est.shape == exact.shape and np.abs(est - exact).mean() < 0.01


# ------------------------------------------------------------------ sparse radius attention


def _path(n):
    s = torch.arange(n - 1)
    return torch.stack([torch.cat([s, s + 1]), torch.cat([s + 1, s])])


def _batch():
    return Batch.from_data_list([Data(x=torch.randn(7, 8), edge_index=_path(7)),
                                 Data(x=torch.randn(5, 8), edge_index=_path(5))])


def test_hop_pairs_match_dense_reach():
    b = _batch()
    for r in (0, 1, 2, 3):
        pairs = hop_pairs(b.edge_index, b.num_nodes, r)
        dense = hop_reach(b.edge_index, b.batch, r, 7)
        want = set()
        for gi, (s, n) in enumerate(((0, 7), (7, 5))):
            for i in range(n):
                for j in range(n):
                    if dense[gi, i, j]:
                        want.add((s + i, s + j))
        assert set(map(tuple, pairs.t().tolist())) == want


@pytest.mark.parametrize("radius", [0, 1, 2, -1])
def test_sparse_radius_equals_dense(radius):
    b = _batch()
    attn = RadiusAttn(8, 2, radius=radius).eval()
    attn.cache = True
    attn.sparse = False
    dense = attn(b.x, b.batch, b.edge_index)
    w_dense = attn.last_weights
    attn.sparse = True
    sparse = attn(b.x, b.batch, b.edge_index)
    assert torch.allclose(dense, sparse, atol=1e-5)
    # cached pair weights equal the dense weights at those pairs
    qi, kj = attn.last_pairs
    ptr = b.ptr
    gi = b.batch[qi]
    wd = w_dense[gi, :, qi - ptr[gi], kj - ptr[gi]]
    assert torch.allclose(wd, attn.last_pair_weights, atol=1e-5)
    assert attn.last_weights is None


def test_sparse_radius_gradients_equal_dense():
    b = _batch()
    attn = RadiusAttn(8, 2, radius=1).eval()
    grads = []
    for sp in (False, True):
        attn.sparse = sp
        x = b.x.clone().requires_grad_(True)
        attn(x, b.batch, b.edge_index).pow(2).sum().backward()
        grads.append(x.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


def test_sparse_radius_isolated_beyond_k_hops():
    b = _batch()
    attn = RadiusAttn(8, 2, radius=2, sparse=True).eval()
    x = b.x.clone().requires_grad_(True)
    out = attn(x, b.batch, b.edge_index)
    out[3].sum().backward()                                   # node 3 of the 7-path sees nodes 1..5
    nz = set((x.grad.abs().sum(1) > 0).nonzero().flatten().tolist())
    assert nz <= {1, 2, 3, 4, 5} and 3 in nz


def test_auto_path_picks_sparse_for_large_graphs(monkeypatch):
    import gnn_mech.layers as layers

    b = _batch()
    attn = RadiusAttn(8, 2, radius=1).eval()
    attn.cache = True
    attn(b.x, b.batch, b.edge_index)
    assert attn.last_weights is not None and attn.last_pair_weights is None     # small: dense
    monkeypatch.setattr(layers, "RADIUS_DENSE_MAX_ELEMENTS", 10)
    attn(b.x, b.batch, b.edge_index)
    assert attn.last_pair_weights is not None and attn.last_weights is None     # "large": sparse


def test_pair_budget_error():
    b = _batch()
    attn = RadiusAttn(8, 2, radius=2, max_pairs=20, sparse=True)
    with pytest.raises(ValueError, match="attn_max_pairs"):
        attn(b.x, b.batch, b.edge_index)
    with pytest.raises(ValueError, match="attn_max_pairs"):
        hop_pairs(b.edge_index, b.num_nodes, -1, max_pairs=10, batch=b.batch)
    assert hop_pairs(b.edge_index, b.num_nodes, 2, max_pairs=10_000).size(1) == 7 + 2 * 6 + 2 * 5 + 5 + 2 * 4 + 2 * 3


def test_attn_max_pairs_keeps_default_hash():
    assert Config(model=ModelConfig(attn_max_pairs=ModelConfig().attn_max_pairs)).hash() == Config().hash()
    assert Config(model=ModelConfig(attn_max_pairs=5)).hash() != Config().hash()
    assert Config(data=DataConfig(sampler="full")).hash() != Config().hash()


# -------------------------------------------------------------------------- end to end


def _run_cfg(tmp_path, **model_kw):
    return Config(
        exp_name="full_test", out_dir=str(tmp_path / "runs"), seed=0,
        data=_cfg(tmp_path),
        model=ModelConfig(hidden=16, layers=2, heads=2, performer_features=8, local="gcn", branch_affine=False,
                          **model_kw),
        train=TrainConfig(epochs=3, batch_size=1, warmup_epochs=1, checkpoint_every=1, device="cpu"),
        measure=MeasureConfig(n_graphs_jacobian=1, targets_per_graph=3, pairs_per_graph=6, n_graphs_entropy=1,
                              entropy_queries=7, n_graphs_range=0))


@pytest.mark.parametrize("kw", [dict(alpha=0.5, attn="performer"), dict(alpha=0.5, attn="radius", attn_radius=1)],
                         ids=["performer", "radius1"])
def test_run_end_to_end_full(tmp_path, monkeypatch, kw):
    import gnn_mech.layers as layers

    sparse = kw["attn"] == "radius"
    if sparse:   # force the sparse path on the 600-node graph
        monkeypatch.setattr(layers, "RADIUS_DENSE_MAX_ELEMENTS", 1000)
    m = run_mod.run(_run_cfg(tmp_path, **kw))
    path = Path(m["run_dir"])
    metrics = json.loads((path / "metrics.json").read_text())
    assert 0.0 <= metrics["test"]["auroc"] <= 1.0
    te = load_social(_cfg(tmp_path))[2][0]
    with np.load(path / "preds_test.npz") as z:
        np.testing.assert_array_equal(z["node_id"], torch.nonzero(te.y_mask).view(-1).numpy())
        assert set(z["graph_id"].tolist()) == {0}
    meas = json.loads((path / "measurements.json").read_text())
    test_nodes = set(torch.nonzero(te.y_mask).view(-1).tolist())
    assert meas["jacobian"] and len({r["v"] for r in meas["jacobian"]}) == 3
    assert all(r["v"] in test_nodes and r["n_nodes"] == 600 for r in meas["jacobian"])
    assert meas["range_node"] and meas["range_graph"] == []
    ent = meas["entropy"]
    assert len(ent) == 2 * 2 and all(0 <= r["entropy_norm"] <= 1 + 1e-9 for r in ent)
    if sparse:
        assert all("n_queries" not in r for r in ent)
        # 2 layers, each reaching 1 hop (GCN and radius-1 attention): nothing beyond 2 hops
        assert all(r["jacobian"] == 0 for r in meas["jacobian"] if r["hops"] > 2)
    else:
        assert all(r["n_queries"] == 7 for r in ent)


def test_approx_resistance_cg_matches_lu_and_exact():
    from gnn_mech.measure.resistance import approx_resistance_rows, effective_resistance_matrix, lcc_nodes

    g = tiny_graph()
    lcc = lcc_nodes(g.edge_index, g.num_nodes)
    src = np.array([0, 5, 17])
    lu = approx_resistance_rows(g.edge_index, g.num_nodes, lcc, src, k=64, rng=np.random.default_rng(1), solver="lu")
    cg = approx_resistance_rows(g.edge_index, g.num_nodes, lcc, src, k=64, rng=np.random.default_rng(1), solver="cg")
    exact = effective_resistance_matrix(g.edge_index, g.num_nodes, lcc)[src]
    # same estimator (different sign draws in chunks), both within JL error of the exact rows
    for est in (lu, cg):
        rel = np.abs(est - exact)[exact > 0] / exact[exact > 0]
        assert np.median(rel) < 0.2
