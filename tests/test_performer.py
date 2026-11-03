"""Tests for PerformerAttn (FAVOR+ linear attention) and its use in HybridGNN (CPU, fast)."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gnn_mech.config import ModelConfig
from gnn_mech.data import DataInfo
from gnn_mech.layers import AttentionBase, GlobalAttn, HybridLayer, PerformerAttn, orthogonal_gaussian
from gnn_mech.models import build_model

HIDDEN, HEADS = 32, 4


@pytest.fixture(autouse=True)
def _restore_torch_rng():
    """Do not leak global RNG state into later (unseeded) test modules."""
    with torch.random.fork_rng():
        yield


def _inputs(sizes=(7, 12, 3), seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    n = sum(sizes)
    x = torch.randn(n, HIDDEN, generator=g) * scale
    batch = torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(sizes)])
    return x, batch


def _attn(m=64, seed=0) -> PerformerAttn:
    torch.manual_seed(seed)
    return PerformerAttn(HIDDEN, HEADS, n_features=m).eval()


def test_shape_and_attention_base():
    a = _attn()
    x, b = _inputs()
    out = a(x, b)
    assert out.shape == x.shape and torch.isfinite(out).all()
    assert isinstance(a, AttentionBase) and issubclass(GlobalAttn, AttentionBase)
    assert a.omega.shape == (HEADS, 64, HIDDEN // HEADS)
    assert "omega" in a.state_dict()


def test_per_graph_isolation():
    a = _attn()
    x, b = _inputs()
    out = a(x, b)
    x2 = x.clone()
    x2[b == 1] += 5 * torch.randn_like(x2[b == 1])        # perturb graph 1 only
    out2 = a(x2, b)
    assert torch.equal(out[b != 1], out2[b != 1])
    assert not torch.allclose(out[b == 1], out2[b == 1])
    # and in gradient terms: graph-0 outputs do not depend on graph-1/2 inputs
    xg = x.clone().requires_grad_(True)
    (g,) = torch.autograd.grad(a(xg, b)[b == 0].sum(), xg)
    assert torch.all(g[b != 0] == 0) and g[b == 0].abs().sum() > 0


def test_single_graph_equals_batched():
    a = _attn()
    x, b = _inputs()
    out = a(x, b)
    for gi in range(3):
        sel = b == gi
        alone = a(x[sel], torch.zeros(int(sel.sum()), dtype=torch.long))
        torch.testing.assert_close(alone, out[sel], rtol=1e-5, atol=1e-6)


def test_orthogonal_features_rows_orthogonal_within_block():
    w = orthogonal_gaussian(2, 20, 8, generator=torch.Generator().manual_seed(0))
    assert w.shape == (2, 20, 8)
    blk = w[0, :8]
    unit = blk / blk.norm(dim=1, keepdim=True)
    torch.testing.assert_close(unit @ unit.T, torch.eye(8), atol=1e-5, rtol=0)


@pytest.mark.parametrize("m", [256])
def test_agrees_with_exact_softmax(m):
    a = _attn(m=m)
    x, b = _inputs(sizes=(30, 45, 10), seed=1)
    bias = a.out_proj.bias
    with torch.no_grad():
        approx = a(x, b) - bias
        exact = a.exact_attention(x, b) - bias
    cos = F.cosine_similarity(approx.flatten(), exact.flatten(), dim=0)
    assert cos > 0.95, float(cos)
    assert F.cosine_similarity(approx, exact, dim=1).mean() > 0.9


def test_approximation_improves_with_features():
    x, b = _inputs(sizes=(40,), seed=2)
    errs = []
    for m in (4, 512):
        a = _attn(m=m)
        with torch.no_grad():
            errs.append((a(x, b) - a.exact_attention(x, b)).norm().item())
    assert errs[1] < errs[0]


def test_implicit_rows_are_distributions_and_reproduce_forward():
    a = _attn()
    x, b = _inputs()
    a.cache = True
    with torch.no_grad():
        out = a(x, b)
    q = torch.arange(x.shape[0])
    rows = a.implicit_rows(q)                                        # [N, H, N]
    assert rows.shape == (x.shape[0], HEADS, x.shape[0])
    assert (rows >= 0).all()
    torch.testing.assert_close(rows.sum(-1), torch.ones(x.shape[0], HEADS), atol=1e-5, rtol=0)
    cross = b[:, None] != b[None, :]
    assert torch.all(rows.permute(0, 2, 1)[cross] == 0)
    # rows @ V through the output projection is exactly the forward output
    with torch.no_grad():
        v = a.v_proj(x).view(-1, HEADS, HIDDEN // HEADS)
        recon = a.out_proj(torch.einsum("qhn,nhd->qhd", rows, v).reshape(x.shape[0], -1))
    torch.testing.assert_close(recon, out, rtol=1e-4, atol=1e-5)
    sub = a.implicit_rows(torch.tensor([3, 10]))
    torch.testing.assert_close(sub, rows[[3, 10]])
    a.clear_cache()
    assert a.last_q_feat is None and a.last_weights is None
    with pytest.raises(RuntimeError):
        a.implicit_rows(q)


def test_double_backward():
    a = _attn(m=16).double()
    x, b = _inputs(sizes=(4, 3))
    x = x.double().requires_grad_(True)
    y = a(x, b).pow(2).sum()
    (g,) = torch.autograd.grad(y, x, create_graph=True)
    (h,) = torch.autograd.grad(g.pow(2).sum(), x)
    assert torch.isfinite(h).all() and h.abs().sum() > 0
    small = x[:5].detach().clone().requires_grad_(True)
    assert torch.autograd.gradgradcheck(lambda z: a(z, b[:5]), (small,), atol=1e-5)


def test_eval_determinism_and_train_redraw():
    a = _attn()
    x, b = _inputs()
    with torch.no_grad():
        e1 = a(x, b)
        a.train()
        t1, t2 = a(x, b), a(x, b)
        a.eval()
        e2 = a(x, b)
    assert torch.equal(e1, e2)                 # eval: fixed buffer, untouched by training forwards
    assert not torch.allclose(t1, t2)          # training: features redrawn each step
    a.redraw = False
    a.train()
    with torch.no_grad():
        assert torch.equal(a(x, b), e1)


def test_buffer_restored_from_state_dict():
    a, c = _attn(seed=0), _attn(seed=5)
    x, b = _inputs()
    assert not torch.allclose(a(x, b), c(x, b))
    c.load_state_dict(a.state_dict())
    assert torch.equal(a(x, b), c(x, b))


def test_large_inputs_stay_finite():
    a = _attn()
    x, b = _inputs(scale=20.0)
    xg = x.requires_grad_(True)
    out = a(xg, b)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(xg.grad).all()


def test_hybrid_layer_and_model_use_performer():
    layer = HybridLayer(HIDDEN, HEADS, 0.5, attn="performer", performer_features=8)
    assert isinstance(layer.glob, PerformerAttn) and layer.glob.n_features == 8
    with pytest.raises(ValueError):
        HybridLayer(HIDDEN, HEADS, 0.5, attn="nope")
    info = DataInfo(node_vocab=[5, 3], edge_vocab=[2], out_dim=2, pe_dim=0, task="multilabel")
    torch.manual_seed(0)
    m = build_model(ModelConfig(alpha=1.0, hidden=HIDDEN, layers=3, heads=HEADS, attn="performer"), info)
    att = m.attention_layers()
    assert len(att) == 3 and all(isinstance(t, PerformerAttn) for t in att)
    soft = build_model(ModelConfig(alpha=1.0, hidden=HIDDEN, layers=2, heads=HEADS), info)
    assert all(isinstance(t, GlobalAttn) for t in soft.attention_layers())
    m.set_attention_cache(True)
    assert all(t.cache for t in att)
    m.set_attention_cache(False)
    assert all(not t.cache and t.last_q_feat is None for t in att)
