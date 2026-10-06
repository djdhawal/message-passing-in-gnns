"""Data, encoders and resistance tests (CPU, no network)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

import gnn_mech.data as data_mod
from gnn_mech.config import DataConfig
from gnn_mech.data import OGB_ATOM_VOCAB, OGB_BOND_VOCAB, load_dataset, structure_stats
from gnn_mech.encoders import CategoricalEncoder, EdgeEncoder, InputEncoder
from gnn_mech.measure.resistance import (
    avg_effective_resistance,
    effective_resistance_matrix,
    hop_distance_matrix,
    lcc_nodes,
)


def undirected(edges: list[tuple[int, int]]) -> torch.Tensor:
    e = torch.tensor(edges, dtype=torch.long).t()
    return torch.cat([e, e.flip(0)], dim=1)


def path_edges(n: int) -> torch.Tensor:
    return undirected([(i, i + 1) for i in range(n - 1)])


# ------------------------------------------------------------------ resistance


@pytest.mark.parametrize("n", [2, 5, 12])
def test_path_resistance_equals_hops(n):
    ei = path_edges(n)
    idx = np.arange(n)
    expected = np.abs(idx[:, None] - idx[None, :]).astype(float)
    np.testing.assert_allclose(effective_resistance_matrix(ei, n), expected, atol=1e-9)
    np.testing.assert_allclose(hop_distance_matrix(ei, n), expected)
    pairs = expected[np.triu_indices(n, 1)]
    assert avg_effective_resistance(ei, n) == pytest.approx(pairs.mean())


@pytest.mark.parametrize("n", [3, 6, 10])
def test_complete_graph_resistance(n):
    ei = undirected(list(itertools.combinations(range(n), 2)))
    r = effective_resistance_matrix(ei, n)
    expected = np.full((n, n), 2.0 / n)
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(r, expected, atol=1e-9)


@pytest.mark.parametrize("n", [3, 7, 10])
def test_cycle_resistance(n):
    ei = undirected([(i, (i + 1) % n) for i in range(n)])
    r = effective_resistance_matrix(ei, n)
    for i in range(n):
        for j in range(n):
            k = abs(i - j)
            assert r[i, j] == pytest.approx(k * (n - k) / n, abs=1e-9)


def test_disconnected_uses_lcc():
    # component A: path 0-1-2-3 (LCC); component B: edge 4-5; isolated node 6
    ei = undirected([(0, 1), (1, 2), (2, 3), (4, 5)])
    n = 7
    np.testing.assert_array_equal(lcc_nodes(ei, n), [0, 1, 2, 3])
    r = effective_resistance_matrix(ei, n)
    assert r.shape == (4, 4)
    assert r[0, 3] == pytest.approx(3.0)
    assert avg_effective_resistance(ei, n) == pytest.approx(np.mean([1, 2, 3, 1, 2, 1]))
    # explicit node subset, in the given order
    r_b = effective_resistance_matrix(ei, n, nodes=np.array([5, 4]))
    np.testing.assert_allclose(r_b, [[0, 1], [1, 0]], atol=1e-9)
    assert np.isnan(avg_effective_resistance(torch.zeros(2, 0, dtype=torch.long), 3))


def test_hop_distance_subset_order():
    ei = path_edges(4)
    h = hop_distance_matrix(ei, 4, nodes=np.array([3, 0, 2, 1]))
    np.testing.assert_array_equal(h, [[0, 3, 1, 2], [3, 0, 2, 1], [1, 2, 0, 1], [2, 1, 1, 0]])
    # the subgraph is induced: dropping node 1 disconnects 0 from the rest
    assert np.isinf(hop_distance_matrix(ei, 4, nodes=np.array([0, 2, 3]))[0, 1])


def test_tree_resistance_equals_hops():
    # on a tree, effective resistance equals the hop distance
    ei = undirected([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6)])
    np.testing.assert_allclose(effective_resistance_matrix(ei, 7), hop_distance_matrix(ei, 7), atol=1e-9)


# --------------------------------------------------------------- synthetic data


@pytest.fixture(scope="module")
def tiny():
    return load_dataset(DataConfig(name="synthetic-tiny", pe_dim=8))


def test_synthetic_schema(tiny):
    train, val, test, info = tiny
    assert (len(train), len(val), len(test)) == (128, 32, 32)
    assert info.node_vocab == OGB_ATOM_VOCAB and info.edge_vocab == OGB_BOND_VOCAB
    assert info.out_dim == 10 and info.pe_dim == 8 and info.task == "multilabel"
    for ds in (train, val, test):
        for i in range(len(ds)):
            d = ds[i]
            n, e = d.num_nodes, d.edge_index.size(1)
            assert 8 <= n <= 30
            assert d.x.dtype == torch.long and d.x.shape == (n, 9)
            assert d.edge_attr.dtype == torch.long and d.edge_attr.shape == (e, 3)
            x, ea = d.x.numpy(), d.edge_attr.numpy()
            assert x.min() >= 0 and (x.max(0) < np.array(OGB_ATOM_VOCAB)).all()
            assert ea.min() >= 0 and (ea.max(0) < np.array(OGB_BOND_VOCAB)).all()
            assert d.pe.dtype == torch.float32 and d.pe.shape == (n, 8)
            assert d.y.shape == (1, 10) and set(d.y.unique().tolist()) <= {0.0, 1.0}
            assert d.graph_id.dtype == torch.long and d.graph_id.tolist() == [i]
            # undirected: both directions present; connected
            pairs = set(map(tuple, d.edge_index.t().tolist()))
            assert all((b, a) in pairs for a, b in pairs)
            assert len(lcc_nodes(d.edge_index, n)) == n


def test_synthetic_deterministic_and_labels_vary(tiny):
    train2, *_ = load_dataset(DataConfig(name="synthetic-tiny", pe_dim=8))
    a, b = tiny[0][3], train2[3]
    assert torch.equal(a.x, b.x) and torch.equal(a.edge_index, b.edge_index) and torch.equal(a.y, b.y)
    y = torch.cat([tiny[0][i].y for i in range(len(tiny[0]))])
    rate = y.mean(0)
    assert ((rate > 0.05) & (rate < 0.95)).sum() >= 6


def test_synthetic_no_pe():
    train, _, _, info = load_dataset(DataConfig(name="synthetic-tiny", pe_dim=0))
    assert info.pe_dim == 0 and "pe" not in train[0]


def test_dataloader_graph_id(tiny):
    train = tiny[0]
    batch = next(iter(DataLoader(train, batch_size=16, shuffle=False)))
    assert batch.graph_id.shape == (16,)
    assert batch.graph_id.tolist() == list(range(16))
    assert batch.y.shape == (16, 10)
    assert batch.x.dtype == torch.long


def test_unknown_dataset():
    with pytest.raises(ValueError):
        load_dataset(DataConfig(name="nope"))


# -------------------------------------------------------------------- encoders


def test_categorical_encoder_is_sum_of_embeddings():
    enc = CategoricalEncoder([4, 3], 5)
    codes = torch.tensor([[1, 2], [3, 0]])
    out = enc(codes)
    assert out.shape == (2, 5)
    expected = enc.embs[0].weight[codes[:, 0]] + enc.embs[1].weight[codes[:, 1]]
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("hidden", [16, 33])
def test_input_encoder_shapes(tiny, hidden):
    train, _, _, info = tiny
    batch = next(iter(DataLoader(train, batch_size=8)))
    h0 = InputEncoder(info, hidden)(batch)
    assert h0.shape == (batch.num_nodes, hidden) and h0.dtype == torch.float32
    e = EdgeEncoder(info, hidden)(batch.edge_attr)
    assert e.shape == (batch.edge_index.size(1), hidden)


def test_input_encoder_without_pe():
    train, _, _, info = load_dataset(DataConfig(name="synthetic-tiny", pe_dim=0))
    batch = next(iter(DataLoader(train, batch_size=4)))
    assert InputEncoder(info, 12)(batch).shape == (batch.num_nodes, 12)


# ------------------------------------------------------------- structure stats


def test_structure_stats_and_cache(tiny, tmp_path, monkeypatch):
    test = tiny[2]
    cfg = DataConfig(name="synthetic-tiny", root=str(tmp_path / "root"), structure_cache=str(tmp_path / "cache"))
    s = structure_stats(test, "test", cfg)
    assert (tmp_path / "cache" / "synthetic-tiny_test.npz").exists()
    assert set(s) == {"n_nodes", "n_edges", "avg_resistance", "diameter", "lcc_size"}
    for k, v in s.items():
        assert v.shape == (len(test),) and not np.isnan(v).any(), k
    d = test[4]
    assert s["n_nodes"][4] == d.num_nodes
    assert s["n_edges"][4] == d.edge_index.size(1) // 2
    assert s["lcc_size"][4] == d.num_nodes
    assert s["avg_resistance"][4] == pytest.approx(avg_effective_resistance(d.edge_index, d.num_nodes))
    assert s["diameter"][4] == hop_distance_matrix(d.edge_index, d.num_nodes).max()

    def boom(*_):
        raise AssertionError("recomputed instead of loading cache")

    monkeypatch.setattr(data_mod, "_graph_stats", boom)
    s2 = structure_stats(test, "test", cfg)
    for k in s:
        np.testing.assert_array_equal(s[k], s2[k])


def test_structure_stats_default_dir(tiny, tmp_path):
    cfg = DataConfig(name="synthetic-tiny", root=str(tmp_path))
    structure_stats(tiny[1], "val", cfg)
    assert (tmp_path / "structure" / "synthetic-tiny_val.npz").exists()
