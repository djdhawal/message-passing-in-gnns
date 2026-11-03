"""Tests for the Higgs-check knobs: affine-free branch norms, GCN local branch, k-hop radius attention."""
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn_mech.config import Config, ModelConfig
from gnn_mech.layers import GCNLocal, GlobalAttn, HybridLayer, RadiusAttn, hop_reach


@pytest.fixture(autouse=True)
def _seeded():
    with torch.random.fork_rng():
        torch.manual_seed(0)
        yield


def path_graph(n: int) -> torch.Tensor:
    src = torch.arange(n - 1)
    return torch.stack([torch.cat([src, src + 1]), torch.cat([src + 1, src])])


def two_paths_batch(n1: int = 6, n2: int = 4, hidden: int = 8):
    g1 = Data(x=torch.randn(n1, hidden), edge_index=path_graph(n1))
    g2 = Data(x=torch.randn(n2, hidden), edge_index=path_graph(n2))
    return Batch.from_data_list([g1, g2])


def test_hop_reach_on_paths():
    b = two_paths_batch()
    for r in (0, 1, 2, 3):
        reach = hop_reach(b.edge_index, b.batch, r, 6)
        for i in range(6):
            for j in range(6):
                assert reach[0, i, j].item() == (abs(i - j) <= r)
        # second graph has 4 real nodes; padded positions are never reachable from real queries
        assert not reach[1, :4, 4:].any()


def test_radius_weights_zero_outside_radius_and_rows_sum_to_one():
    b = two_paths_batch()
    attn = RadiusAttn(8, 2, radius=2).eval()
    attn.cache = True
    out = attn(b.x, b.batch, b.edge_index)
    assert out.shape == (10, 8)
    w = attn.last_weights                                    # [B, heads, N, N]
    for i in range(6):
        for j in range(6):
            if abs(i - j) > 2:
                assert torch.all(w[0, :, i, j] == 0)
    real = attn.last_mask
    sums = w.sum(-1)[real[:, None, :].expand(-1, 2, -1)]
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert torch.isfinite(out).all()


def test_large_or_negative_radius_equals_full_attention():
    b = two_paths_batch()
    full = GlobalAttn(8, 2).eval()
    for r in (-1, 50):
        rad = RadiusAttn(8, 2, radius=r).eval()
        rad.load_state_dict(full.state_dict())
        assert torch.allclose(rad(b.x, b.batch, b.edge_index), full(b.x, b.batch), atol=1e-5)


def test_radius_attention_isolated_per_graph_and_local():
    b = two_paths_batch()
    attn = RadiusAttn(8, 2, radius=1).eval()
    x = b.x.clone().requires_grad_(True)
    out = attn(x, b.batch, b.edge_index)
    out[0].sum().backward()                                  # node 0 of graph 1: sees nodes 0, 1 only
    nz = (x.grad.abs().sum(1) > 0).nonzero().flatten().tolist()
    assert set(nz) <= {0, 1}


def test_branch_affine_false_has_no_norm_parameters():
    layer = HybridLayer(8, 2, 0.5, branch_affine=False)
    assert layer.norm_loc.weight is None and layer.norm_glb.weight is None
    assert layer.norm_ffn.weight is not None                 # the FFN norm keeps its affine
    layer_on = HybridLayer(8, 2, 0.5)
    n_off = sum(p.numel() for p in layer.parameters())
    n_on = sum(p.numel() for p in layer_on.parameters())
    assert n_on - n_off == 4 * 8


def test_gcn_local_branch_and_radius_in_hybrid_layer():
    b = two_paths_batch()
    e = torch.zeros(b.edge_index.size(1), 8)
    layer = HybridLayer(8, 2, 0.5, local="gcn", attn="radius", attn_radius=1, branch_affine=False).eval()
    assert isinstance(layer.local, GCNLocal) and isinstance(layer.glob, RadiusAttn)
    out = layer(b.x, b.edge_index, e, b.batch)
    assert out.shape == b.x.shape and torch.isfinite(out).all()
    with pytest.raises(ValueError):
        HybridLayer(8, 2, 0.5, local="sage")


def test_new_fields_keep_default_hash():
    base = Config()
    assert Config(model=ModelConfig(attn_radius=-1, branch_affine=True)).hash() == base.hash()
    assert Config(model=ModelConfig(branch_affine=False)).hash() != base.hash()
    assert Config(model=ModelConfig(attn="radius", attn_radius=2)).hash() != base.hash()


def test_activate_passive_target_masks_active_users():
    from gnn_mech.social.sources import node_target

    attrs = {"active_after": np.array([1.0, 0.0, 1.0, 0.0]),
             "rt_out_pre": np.array([0.0, 2.0, 0.0, 0.0]),
             "mt_out_pre": np.array([0.0, 0.0, 1.0, 0.0]),
             "re_out_pre": np.array([0.0, 0.0, 0.0, 0.0])}
    g = Data(num_nodes=4)
    g.attrs = attrs
    y, task = node_target("higgs-follow", g, "activate_passive")
    assert task == "binary"
    assert y[0] == 1.0 and y[3] == 0.0 and np.isnan(y[1]) and np.isnan(y[2])
    y2, _ = node_target("higgs-follow", g, "active_after")
    assert not np.isnan(y2).any()
