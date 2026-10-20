"""Tests for gnn_mech.reddit on tiny synthetic node-regression graphs (CPU, fast, no download)."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gnn_mech.reddit import data as rdata
from gnn_mech.reddit import run as rrun
from gnn_mech.reddit import target_select as ts
from gnn_mech.reddit.models import SAGEBaseline, VMNModel, add_virtual_master, build_reddit_model


@pytest.fixture(scope="module")
def graph():
    return rdata.synthetic_graph(n=600, seed=0)


def test_feature_names():
    assert len(rdata.FEATURE_NAMES) == 86 and len(set(rdata.FEATURE_NAMES)) == 86
    assert len(rdata.LIWC_COLS) == 65 and rdata.FEATURE_NAMES[rdata.SENTIMENT_COLS[2]] == "vader_compound"


def test_build_graph_from_tsv(tmp_path):
    props = lambda v: ",".join(str(v + i) for i in range(86))   # noqa: E731
    lines = ["SOURCE_SUBREDDIT\tTARGET_SUBREDDIT\tPOST_ID\tTIMESTAMP\tLINK_SENTIMENT\tPROPERTIES",
             f"a\tb\tp1\tt\t1\t{props(0)}", f"a\tc\tp2\tt\t1\t{props(2)}", f"b\tc\tp3\tt\t-1\t{props(10)}"]
    tsv = tmp_path / "x.tsv"
    tsv.write_text("\n".join(lines) + "\n")
    d = rdata.build_graph(str(tsv))
    assert d.num_nodes == 3 and d.edge_index.tolist() == [[0, 0, 1], [1, 2, 2]]
    assert d.is_source.tolist() == [True, True, False]
    assert d.x[0, 0].item() == pytest.approx(1.0) and d.x[1, 5].item() == pytest.approx(15.0)
    assert (d.x[2] == 0).all()                                # receive-only node: zero features
    root = tmp_path / "root"
    root.mkdir()
    (root / rdata.TSV_NAME).write_text(tsv.read_text())
    loaded = rdata.load_reddit_graph(str(root), url="http://invalid.invalid/")
    assert (root / rdata.CACHE_NAME).exists() and loaded.num_nodes == 3


def test_bfs_and_anchor():
    ei = torch.tensor([[0, 0, 0, 1, 3], [1, 2, 3, 4, 5]])      # 6 isolated
    assert rdata.pick_anchor(ei, 7) == 0
    assert rdata.bfs_hops(ei, 7, 0).tolist() == [0, 1, 1, 1, 2, 2, -1]
    assert rdata.bfs_hops(ei, 7, 4).tolist() == [2, 1, 3, 3, 0, 4, -1]   # undirected view


def test_split_masks_disjoint_and_complete(graph):
    dist = rdata.bfs_hops(graph.edge_index, graph.num_nodes, rdata.pick_anchor(graph.edge_index, graph.num_nodes))
    src = graph.is_source.numpy()
    ok = rdata.eligible_nodes(dist, src)
    d = rdata.make_split("distance", dist, src, seed=0)
    for seed in (0, 1):
        r = rdata.make_split("random", dist, src, seed=seed)
        for masks in (d, r):
            tot = masks["train"].astype(int) + masks["val"] + masks["test"]
            assert tot.max() == 1                              # disjoint
            assert ((tot == 1) == ok).all()                    # complete over reachable sources
        assert {k: int(v.sum()) for k, v in r.items()} == {k: int(v.sum()) for k, v in d.items()}
    assert (dist[d["train"]] <= 1).all() and (dist[d["val"]] == 2).all() and (dist[d["test"]] >= 3).all()
    assert not np.array_equal(rdata.make_split("random", dist, src, 0)["test"],
                              rdata.make_split("random", dist, src, 1)["test"])
    assert not (d["train"] & ~src).any()


def test_target_selection_recovers_planted_target(graph):
    table = ts.rank_targets(graph, candidates=[0, 1], k_max=3, mlp=False)
    by = {r["target"]: r for r in table}
    assert table[0]["target"] == 0
    assert by[0]["neighbour_gain"] > 0.3 and by[0]["multi_hop_gain"] > 0.05
    assert abs(by[1]["neighbour_gain"]) < 0.05 and by[1]["r2"]["own"] > 0.9
    assert 5 in by[1]["dropped_correlated"]                    # near-duplicate input removed
    assert ts.select_target(table)["target"]["target"] == 0
    assert ts.select_target([by[1]])["target"] is None


def test_propagate_means():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])                  # path 0-1-2-3
    X = np.arange(4, dtype=float)[:, None]
    valid = np.array([True, False, True, True])                # node 1 has no features
    h1, h2 = ts.propagate(X, ei, valid, 2)
    np.testing.assert_allclose(h1[:, 0], [0, 1, 3, 2])         # node 1: mean(0, 2); node 0: no valid neighbour
    np.testing.assert_allclose(h2[:, 0], [1, 3, 1.5, 3])       # node 0 invalid after hop 1


def test_vmn_adds_one_node_and_2n_edges():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    aug, n = add_virtual_master(ei, 4)
    assert n == 5 and aug.size(1) == 3 + 2 * 4
    new = aug[:, 3:]
    assert set(map(tuple, new.t().tolist())) == {(i, 4) for i in range(4)} | {(4, i) for i in range(4)}


def test_models_matched():
    sage, vmn = SAGEBaseline(7, 16, 3, 0.1), VMNModel(7, 16, 3, 0.1)
    assert hasattr(sage, "input_proj") and hasattr(vmn, "input_proj")
    assert sum(p.numel() for p in vmn.parameters()) - sum(p.numel() for p in sage.parameters()) == 16
    x, ei = torch.randn(6, 7), torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    assert sage.eval()(x, ei).shape == vmn.eval()(x, ei).shape == (6,)
    # the master node connects node 0 to node 5 within 2 hops; SAGE with 2 layers cannot
    for m in (build_reddit_model("sage", 7, 8, 2, 0.0), build_reddit_model("vmn", 7, 8, 2, 0.0)):
        x0 = x.clone().requires_grad_(True)
        m.eval()(x0, ei)[0].backward()
        reached = x0.grad[5].abs().sum() > 0
        assert reached == isinstance(m, VMNModel)
    with pytest.raises(ValueError):
        build_reddit_model("gat", 7, 8, 2, 0.0)


def test_band_metrics():
    y = np.array([0.0, 1, 2, 3, 4, 5, 6, 7])
    dist = np.array([0, 1, 2, 3, 3, 4, 5, 9])
    pred = y + np.array([0, 0, 0, 1, -1, 2, 0, 3])
    train = np.array([1, 1, 0, 0, 0, 0, 0, 0], bool)         # constant predictor = 0.5
    ev = ~train
    rows = {r["band"]: r for r in rrun.band_metrics(pred, y, dist, ev, train)}
    assert set(rows) == {"2", "3", "4", "5+"}
    assert rows["3"]["n"] == 2 and rows["3"]["mse"] == pytest.approx(1.0)
    assert rows["3"]["const_mse"] == pytest.approx((2.5 ** 2 + 3.5 ** 2) / 2)
    assert rows["5+"]["n"] == 2 and rows["5+"]["mse"] == pytest.approx(4.5)
    assert rows["4"]["rel_mse"] == pytest.approx(4.0 / 4.5 ** 2)


def test_reddit_config_duck_types_rundir(tmp_path):
    from gnn_mech.results import RunDir
    cfg = rrun.RedditConfig(out_dir=str(tmp_path), seed=3)
    run = RunDir(cfg)
    assert run.path == tmp_path / "reddit" / cfg.hash() / "seed3"
    assert dataclasses.replace(cfg, seed=4).hash() == cfg.hash()
    assert dataclasses.replace(cfg, model="vmn").hash() != cfg.hash()
    assert rrun.apply_overrides(cfg, ["epochs=7"]).epochs == 7
    with pytest.raises(KeyError):
        rrun.apply_overrides(cfg, ["nope=1"])


def test_reddit_cli_end_to_end(tmp_path):
    out = tmp_path / "runs"
    res = rrun.main(["--synthetic", "--target", "0", "--seeds", "0:2", "--out-dir", str(out), "epochs=30",
                     "eval_every=5"])
    t = res["table"]
    assert set(t["split"]) == {"distance", "random"} and set(t["model"]) == {"sage", "vmn"}
    assert (t[t["model"] == "vmn"]["vmn_gain"].notna()).all()
    assert (t["n_seeds"] == 2).all()
    assert {Path(f).suffix for f in res["figures"]} == {".pdf", ".png"}
    runs = sorted(out.glob("reddit/*/seed*/metrics.json"))
    assert len(runs) == 8
    m = json.loads(runs[0].read_text())
    assert {"test_bands", "all_bands", "split_sizes", "mse", "inputs"} <= set(m)
    assert 0 not in m["inputs"]                                # target removed from inputs
    mtime = runs[0].stat().st_mtime
    rrun.main(["--synthetic", "--target", "0", "--seeds", "0:2", "--out-dir", str(out), "epochs=30", "eval_every=5"])
    assert runs[0].stat().st_mtime == mtime                    # finished runs are skipped
