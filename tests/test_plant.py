"""Planted targets (gnn_mech.social.plant): exact shells, determinism, NaN where undefined, retrieval with
no matching key inside radius k-1, analytic range == k, and the sources.py hooks end to end on social-tiny."""
import numpy as np
import pytest
import torch
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data

from gnn_mech.config import DataConfig
from gnn_mech.social import plant, sources
from gnn_mech.social.dataset import load_meta, load_social, tiny_graph


def _graph(src, dst, n):
    return Data(edge_index=sources.undirected(np.asarray(src), np.asarray(dst), n), num_nodes=n)


def path_graph(n=9):
    return _graph(np.arange(n - 1), np.arange(1, n), n)


def grid_graph(r=6, c=7):
    idx = np.arange(r * c).reshape(r, c)
    src = np.r_[idx[:, :-1].ravel(), idx[:-1, :].ravel()]
    dst = np.r_[idx[:, 1:].ravel(), idx[1:, :].ravel()]
    return _graph(src, dst, r * c)


def path_plus_isolated():
    g = path_graph(5)
    return Data(edge_index=g.edge_index, num_nodes=6)      # node 5 has no edges


def _dist(g):
    return shortest_path(plant.adjacency(g), unweighted=True, directed=False)


GRAPHS = {"path": path_graph, "grid": grid_graph, "tiny": tiny_graph}


@pytest.fixture(autouse=True)
def _fresh_cache():
    plant.clear_cache()
    yield
    plant.clear_cache()


def test_parse_and_names():
    assert plant.parse("plant:retrieve:3") == ("retrieve", 3, 0)
    assert plant.parse("plant:shell_mean:2:s7") == ("shell_mean", 2, 7)
    assert plant.name("retrieve", 3, 0) == "plant:retrieve:3" and plant.name("retrieve", 3, 2) == "plant:retrieve:3:s2"
    for bad in ("plant:retrieve", "plant:nope:2", "plant:retrieve:0", "plant:retrieve:2:7", "retrieve:2"):
        with pytest.raises(ValueError):
            plant.parse(bad)
    assert plant.is_plant("plant:x") and not plant.is_plant("active_after")


@pytest.mark.parametrize("gname", sorted(GRAPHS))
def test_shells_exact(gname):
    g = GRAPHS[gname]()
    D = _dist(g)
    a = plant.adjacency(g)
    n = g.num_nodes
    for src in plant._blocks(n, budget=n * 5):          # several blocks
        lev = plant._levels(a, src, 4)
        for d in range(5):
            np.testing.assert_array_equal(lev[d], (D[src] == d).T)
    sizes = plant._level_sizes(g, 4)
    for d in range(5):
        np.testing.assert_array_equal(sizes[d], (D == d).sum(1))


def test_path_shell_mean_by_hand():
    g = path_graph(9)
    Z, y, meta = plant.make(g, "shell_mean", 2)
    z = Z[:, 0]
    assert Z.shape == (9, 1) and meta["names"] == ["plant_z0"]
    assert np.isclose(y[0], z[2]) and np.isclose(y[4], (z[2] + z[6]) / 2) and np.isclose(y[8], z[6])


@pytest.mark.parametrize("family", plant.FAMILIES)
def test_determinism(family):
    g = grid_graph()
    Z1, y1, _ = plant.make(g, family, 2, seed=3)
    plant.clear_cache()
    Z2, y2, _ = plant.make(g, family, 2, seed=3)
    np.testing.assert_array_equal(Z1, Z2)
    np.testing.assert_array_equal(y1, y2)
    Z3, y3, _ = plant.make(g, family, 2, seed=4)
    assert not np.allclose(Z1, Z3)
    # block layout does not change the target
    plant.clear_cache()
    old = plant.BLOCK_ELEMENTS
    try:
        plant.BLOCK_ELEMENTS = g.num_nodes * 16
        Z4, y4, _ = plant.make(g, family, 2, seed=3)
    finally:
        plant.BLOCK_ELEMENTS = old
    np.testing.assert_array_equal(Z1, Z4)
    np.testing.assert_allclose(y1, y4, rtol=1e-12, atol=1e-12)    # BLAS rounding only


@pytest.mark.parametrize("family", plant.FAMILIES)
def test_nan_where_undefined(family):
    g = path_plus_isolated()                  # path 0-1-2-3-4 and isolated node 5
    _, y, meta = plant.make(g, family, 3)
    D = _dist(g)
    has_shell = (D == 3).any(1)
    assert not has_shell[2] and not has_shell[5]
    assert np.isnan(y[2]) and np.isnan(y[5])
    if family == "shell_mean":
        assert np.isfinite(y[[0, 1, 3, 4]]).all()   # shells {3}, {4}, {0}, {1}
    assert not (np.isfinite(y) & ~has_shell).any()
    assert meta["eligible"] == int(np.isfinite(y).sum())


@pytest.mark.parametrize("gname", sorted(GRAPHS))
@pytest.mark.parametrize("k", [1, 2, 3])
def test_retrieve_is_k_range(gname, k):
    g = GRAPHS[gname]()
    D = _dist(g)
    Z, y, meta = plant.make(g, "retrieve", k, seed=1, key_dim=8)
    keys, query, values = meta["keys"], meta["query"], meta["values"]
    assert Z.shape == (g.num_nodes, 17) and meta["names"][-1] == "plant_value"
    codes_q = Z[:, 8:16]
    ok = np.isfinite(y)
    assert ok.any()
    for v in range(g.num_nodes):
        if not ok[v]:
            assert query[v] == -1 and np.allclose(codes_q[v], 0)
            # undefined only if every shell key also occurs in the inner ball (or the shell is empty)
            inner_keys = set(keys[D[v] < k])
            assert all(keys[u] in inner_keys for u in np.flatnonzero(D[v] == k))
            continue
        same = keys == query[v]
        assert not (same & (D[v] < k)).any(), "a node closer than k carries the query key"
        match = same & (D[v] == k)
        assert match.any()
        assert np.isclose(y[v], values[match].mean())
        # the query channel is the code of the query key, the key channel the code of each node's key
        u = np.flatnonzero(match)[0]
        np.testing.assert_allclose(codes_q[v], Z[u, :8])
    assert 2 <= meta["n_keys"] <= g.num_nodes


def test_retrieve_targets_ignore_values_off_shell():
    """Changing values of nodes not in v's matching set never changes y_v (Jacobian support check)."""
    g = tiny_graph()
    _, y, meta = plant.make(g, "retrieve", 2)
    D = _dist(g)
    v = int(np.flatnonzero(np.isfinite(y))[0])
    J = plant.jacobian(g, "retrieve", 2, 0, v)
    assert (J[:, :-1] == 0).all()
    assert set(np.flatnonzero(J[:, -1])) <= set(np.flatnonzero(D[v] == 2))
    assert np.isclose(J[:, -1] @ meta["values"], y[v])       # linear in the values, exactly


@pytest.mark.parametrize("family", plant.FAMILIES)
@pytest.mark.parametrize("gname", sorted(GRAPHS))
def test_analytic_range_equals_k(family, gname):
    g = GRAPHS[gname]()
    for k in (1, 2, 3):
        r = plant.analytic_range(g, family, k, n_sample=64)
        assert r["n"] > 0
        assert np.isclose(r["range_hops"], k) and np.isclose(r["range_hops_min"], k)
        assert np.isclose(r["range_hops_max"], k)
        assert r["range_res"] is not None and r["range_res"] > 0     # small graphs: exact resistance


def test_jacobian_reproduces_shell_mean():
    g = grid_graph()
    Z, y, _ = plant.make(g, "shell_mean", 2)
    for v in range(g.num_nodes):
        J = plant.jacobian(g, "shell_mean", 2, 0, v)
        assert np.isclose(J[:, 0] @ Z[:, 0], y[v])


def test_shell_stats():
    g = path_plus_isolated()
    rows = plant.shell_stats(g, [1, 3])
    assert [r["k"] for r in rows] == [1, 3]
    assert rows[0]["eligible"] == 5 and rows[1]["eligible"] == 4
    assert rows[1]["shell"]["max"] == 1
    assert np.isclose(rows[0]["eligible_frac"], 5 / 6)


def test_target_columns_accepts_plant():
    tasks = sources.target_columns("social-tiny")
    assert "plant:shell_mean:2" in tasks and tasks["plant:retrieve:4:s3"] == "regression"
    assert "label" in tasks and "nope" not in tasks
    with pytest.raises(ValueError):
        "plant:bogus:2" in tasks
    assert "plant:retrieve:1" in sources.target_columns("higgs-follow")


def test_hooks_direct():
    g = tiny_graph()
    x, names = sources.node_inputs("social-tiny", g, "plant:retrieve:2")
    assert x.shape == (g.num_nodes, 8 + 2 * plant.KEY_DIM + 1)
    assert names[:8] == [f"f{i}" for i in range(8)] and all(nm.startswith("plant_") for nm in names[8:])
    y, task = sources.node_target("social-tiny", g, "plant:retrieve:2")
    assert task == "regression" and y.shape == (g.num_nodes,)
    _, y2, _ = plant.make(g, "retrieve", 2)
    np.testing.assert_array_equal(y, y2)


@pytest.mark.parametrize("target", ["plant:shell_mean:2", "plant:retrieve:2:s1"])
def test_load_social_end_to_end(tmp_path, target):
    cfg = DataConfig(name="social-tiny", root=str(tmp_path), pe_dim=0, subgraph_size=50, target=target)
    tr, va, te, info = load_social(cfg)
    family, k, seed = plant.parse(target)
    n_plant = len(plant.channel_names(family))
    assert info.task == "regression" and info.out_dim == 1 and info.feat_dim == 8 + n_plant
    meta = load_meta(cfg)
    assert meta["feat_names"][-n_plant:] == plant.channel_names(family)
    g = tiny_graph()
    _, y, _ = plant.make(g, family, k, seed)
    # labelled nodes are exactly the core nodes with a defined planted target, and y is the standardized y
    mu, sd = meta["y_mean"], meta["y_std"]
    n_lab = 0
    for ds in (tr, va, te):
        for i in range(len(ds)):
            d = ds[i]
            nid = d.node_id.numpy()
            m = d.y_mask.numpy()
            assert np.isfinite(y[nid[m]]).all()
            np.testing.assert_allclose(d.y.numpy()[m], ((y[nid[m]] - mu) / sd).astype(np.float32), rtol=1e-4,
                                       atol=1e-5)
            n_lab += int(m.sum())
    assert 0 < n_lab <= int(np.isfinite(y).sum())
    assert torch.isfinite(tr[0].x).all()
