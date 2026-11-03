"""Social data layer on social-tiny (no download): schema, splits, determinism, standardization,
halos, partition fallback, structure stats with known answers."""
import os.path as osp

import numpy as np
import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import AddRandomWalkPE

from gnn_mech.config import DataConfig
from gnn_mech.social import SOCIAL_DATASETS
from gnn_mech.social import sources, structure, subgraph
from gnn_mech.social.dataset import data_key, load_meta, load_social, processed_path, rwse, tiny_graph


def _cfg(tmp_path, **kw):
    base = dict(name="social-tiny", root=str(tmp_path), pe_dim=8, subgraph_size=50)
    base.update(kw)
    return DataConfig(**base)


def _all(ds):
    return [ds[i] for i in range(len(ds))]


def test_dataset_names():
    assert {"twitch-gamers", "reddit-body", "social-tiny", "higgs-follow", "twitch-de"} <= SOCIAL_DATASETS
    assert set(sources.DATASETS) | {"social-tiny"} == set(SOCIAL_DATASETS)
    assert set(sources.DEFAULT_TARGETS) == set(SOCIAL_DATASETS)
    assert sources.source_of("higgs-follow") == "higgs" and sources.source_of("twitch-de") == "twitch-de"


def test_schema(tmp_path):
    tr, va, te, info = load_social(_cfg(tmp_path))
    assert info.node_vocab == [] and info.edge_vocab == [1] and info.task == "binary"
    assert info.task_level == "node" and info.feat_dim == 8 and info.out_dim == 1 and info.pe_dim == 8
    assert len(tr) >= 1 and len(va) >= 1 and len(te) >= 1
    gids = []
    for d in _all(tr) + _all(va) + _all(te):
        n, e = d.num_nodes, d.edge_index.size(1)
        assert d.x.shape == (n, info.feat_dim) and d.x.dtype == torch.float32
        assert d.pe.shape == (n, 8) and d.pe.dtype == torch.float32
        assert d.edge_index.dtype == torch.long and d.edge_index.shape == (2, e)
        assert d.edge_index.max() < n and (d.edge_index[0] != d.edge_index[1]).all()
        # undirected: every edge appears in both directions
        fw = set(map(tuple, d.edge_index.t().tolist()))
        assert fw == {(b, a) for a, b in fw}
        assert d.edge_attr.shape == (e, 1) and d.edge_attr.dtype == torch.long and (d.edge_attr == 0).all()
        assert d.y.shape == (n,) and d.y.dtype == torch.float32
        assert d.y_mask.shape == (n,) and d.y_mask.dtype == torch.bool
        assert d.node_id.shape == (n,) and d.node_id.dtype == torch.long
        assert d.graph_id.shape == (1,) and d.graph_id.dtype == torch.long
        gids.append(int(d.graph_id))
    assert len(set(gids)) == len(gids)


def test_multiclass_and_regression(tmp_path):
    _, _, te, info = load_social(_cfg(tmp_path, target="block"))
    assert info.task == "multiclass" and info.out_dim == 6 and te[0].y.dtype == torch.long
    tr, _, _, info = load_social(_cfg(tmp_path, target="reg"))
    assert info.task == "regression"
    y = torch.cat([d.y[d.y_mask] for d in _all(tr)])
    assert abs(float(y.mean())) < 1e-4 and abs(float(y.std(unbiased=False)) - 1) < 1e-3
    assert load_meta(_cfg(tmp_path, target="reg"))["y_log1p"]


def test_disjoint_splits(tmp_path):
    tr, va, te, _ = load_social(_cfg(tmp_path))
    cores = [set(torch.cat([d.node_id[d.y_mask] for d in _all(ds)]).tolist()) for ds in (tr, va, te)]
    assert not (cores[0] & cores[1]) and not (cores[0] & cores[2]) and not (cores[1] & cores[2])
    assert sum(map(len, cores)) == 600          # every node is in exactly one core
    allnodes = torch.cat([d.node_id for ds in (tr, va, te) for d in _all(ds)])
    assert allnodes.unique().numel() == allnodes.numel()   # no halo: node sets fully disjoint


def test_determinism(tmp_path):
    a = load_social(_cfg(tmp_path / "a", split_seed=3))
    b = load_social(_cfg(tmp_path / "b", split_seed=3))
    c = load_social(_cfg(tmp_path / "c", split_seed=4))
    for da, db in zip(_all(a[2]), _all(b[2])):
        assert torch.equal(da.node_id, db.node_id) and torch.equal(da.x, db.x) and torch.equal(da.pe, db.pe)
    ida = sorted(torch.cat([d.node_id for d in _all(a[2])]).tolist())
    idc = sorted(torch.cat([d.node_id for d in _all(c[2])]).tolist())
    assert ida != idc


def test_cache_hit_and_key(tmp_path):
    cfg = _cfg(tmp_path)
    load_social(cfg)
    assert osp.exists(processed_path(cfg))
    assert data_key(cfg) == data_key(_cfg(tmp_path, root="/elsewhere", num_workers=4))
    assert data_key(cfg) != data_key(_cfg(tmp_path, split_seed=1))
    assert data_key(cfg) == data_key(_cfg(tmp_path, target="label"))   # "" resolves to the default


def test_standardization_uses_train_only(tmp_path):
    cfg = _cfg(tmp_path)
    tr, va, te, _ = load_social(cfg)
    g = tiny_graph()
    raw = g.attrs["features"]
    train_ids = torch.cat([d.node_id[d.y_mask] for d in _all(tr)]).numpy()
    mu, sd = raw[train_ids].mean(0), raw[train_ids].std(0)
    for d in _all(tr) + _all(te):
        np.testing.assert_allclose(d.x.numpy(), (raw[d.node_id.numpy()] - mu) / sd, rtol=1e-4, atol=1e-5)
    xtr = torch.cat([d.x[d.y_mask] for d in _all(tr)])
    assert xtr.mean(0).abs().max() < 1e-4 and (xtr.std(0, unbiased=False) - 1).abs().max() < 1e-3


def test_halo(tmp_path):
    tr, va, te, _ = load_social(_cfg(tmp_path, halo_hops=1))
    g = tiny_graph()
    adj = subgraph.csr_adjacency(g.edge_index.numpy(), g.num_nodes)
    cores = []
    for d in _all(tr) + _all(va) + _all(te):
        core = d.node_id[d.y_mask].numpy()
        halo = d.node_id[~d.y_mask].numpy()
        assert halo.size > 0
        expect = subgraph.halo(adj, core, 1)
        assert np.array_equal(np.sort(halo), expect)
        cores.append(set(core.tolist()))
    assert sum(map(len, cores)) == 600 and len(set().union(*cores)) == 600


def test_partition_fallback_without_metis(tmp_path):
    g = tiny_graph()
    for sampler, use in (("partition", False), ("bfs", None)):
        parts = subgraph.partition(g.edge_index.numpy(), g.num_nodes, 50, seed=0, sampler=sampler, use_metis=use)
        assert parts.shape == (600,) and parts.min() == 0
        sizes = np.bincount(parts)
        assert sizes.size == 12 and (sizes > 0).all() and sizes.max() <= 75
        again = subgraph.partition(g.edge_index.numpy(), g.num_nodes, 50, seed=0, sampler=sampler, use_metis=use)
        assert np.array_equal(parts, again)
    # end to end through load_social with the BFS sampler
    tr, va, te, _ = load_social(_cfg(tmp_path, sampler="bfs"))
    ids = torch.cat([d.node_id for ds in (tr, va, te) for d in _all(ds)])
    assert ids.unique().numel() == 600 == ids.numel()


def test_split_parts_and_cap():
    s = subgraph.split_parts(20, 0.15, 0.15, seed=0)
    assert len(s["val"]) == 3 and len(s["test"]) == 3 and len(s["train"]) == 14
    allp = np.concatenate(list(s.values()))
    assert np.array_equal(np.sort(allp), np.arange(20))
    capped = subgraph.split_parts(20, 0.15, 0.15, seed=0, max_subgraphs=2)
    assert all(len(v) == 2 for v in capped.values())
    assert set(capped["test"]) <= set(s["test"])


def test_rwse_matches_pyg():
    g = tiny_graph()
    d = Data(edge_index=g.edge_index, num_nodes=g.num_nodes)
    ref = AddRandomWalkPE(walk_length=6, attr_name="pe")(d).pe.numpy()
    np.testing.assert_allclose(rwse(g.edge_index.numpy(), g.num_nodes, 6), ref, atol=1e-5)


def _path(n):
    s = np.arange(n - 1)
    return sources.undirected(s, s + 1, n).numpy()


def _complete(n):
    u, v = np.triu_indices(n, 1)
    return sources.undirected(u, v, n).numpy()


def test_resistance_known_answers():
    # path: R(u, v) = |u - v|
    a = structure.adjacency(_path(6), 6)
    lp = structure.laplacian_pinv(a)
    u, v = np.triu_indices(6, 1)
    np.testing.assert_allclose(structure.resistance(lp, u, v), np.abs(u - v), atol=1e-9)
    # complete graph K_n: R = 2 / n = 1/d_u + 1/d_v - 2/(n(n-1))... exactly 2/n, excess = 2/n - 2/(n-1)
    n = 8
    s = structure.subgraph_summary(_complete(n), n, n_pairs=1000)
    assert s["mean_res"] == pytest.approx(2 / n)
    assert s["mean_excess_res"] == pytest.approx(2 / n - 2 / (n - 1))
    assert s["kirchhoff"] == pytest.approx(n - 1)              # Kf(K_n) = n - 1
    assert s["spectral_gap"] == pytest.approx(n / (n - 1))     # normalized Laplacian of K_n
    assert s["mean_degree"] == n - 1 and s["n_edges"] == n * (n - 1) // 2
    # path of 5: Kirchhoff = sum |u-v| = 20, effective diameter over hop distances
    p = structure.subgraph_summary(_path(5), 5)
    assert p["kirchhoff"] == pytest.approx(20.0) and p["mean_res"] == pytest.approx(2.0)


def test_resistance_lcc_only():
    # path of 4 plus 2 isolated nodes: stats on the LCC (4 nodes)
    s = structure.subgraph_summary(_path(4), 6)
    assert s["lcc_size"] == 4 and s["n_nodes"] == 6
    assert s["kirchhoff"] == pytest.approx(10.0)


def test_structure_table(tmp_path):
    t = structure.structure_table(["social-tiny"], str(tmp_path), max_graphs=2, splits=("train", "test"),
                                  data_overrides={"subgraph_size": 100},
                                  out_csv=str(tmp_path / "s0" / "structure.csv"))
    assert list(t["dataset"]) == ["social-tiny"] and int(t["n_subgraphs"][0]) == 3
    for k in structure.SUMMARY_KEYS:
        assert k in t.columns and np.isfinite(t[k][0])
    assert osp.exists(tmp_path / "s0" / "structure.csv") and osp.exists(tmp_path / "s0" / "structure_subgraphs.csv")


def test_undirected_dedup():
    ei = sources.undirected(np.array([0, 1, 1, 2, 3]), np.array([1, 0, 1, 3, 2]), 4).numpy()
    assert sorted(map(tuple, ei.T.tolist())) == [(0, 1), (1, 0), (2, 3), (3, 2)]
