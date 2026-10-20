"""CPU tests for gnn_mech.layers / gnn_mech.models."""
from __future__ import annotations

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv

from gnn_mech.config import ModelConfig
from gnn_mech.data import DataInfo
from gnn_mech.layers import GlobalAttn, HybridLayer
from gnn_mech.models import GCNBaseline, HybridGNN, build_model, count_parameters

NODE_VOCAB = [119, 5, 12, 12, 10, 6, 6, 2, 2]
EDGE_VOCAB = [5, 6, 2]
PE_DIM = 20
OUT_DIM = 10
INFO = DataInfo(node_vocab=NODE_VOCAB, edge_vocab=EDGE_VOCAB, out_dim=OUT_DIM, pe_dim=PE_DIM, task="multilabel")


def _codes(n: int, vocab: list[int], g: torch.Generator) -> torch.Tensor:
    return torch.stack([torch.randint(0, v, (n,), generator=g) for v in vocab], dim=1)


def make_graph(edges: list[tuple[int, int]], n: int, seed: int = 0) -> Data:
    """Undirected graph from an edge list, with random categorical features."""
    g = torch.Generator().manual_seed(seed)
    ei = torch.tensor(edges, dtype=torch.long).t()
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    e_half = _codes(len(edges), EDGE_VOCAB, g)
    return Data(
        x=_codes(n, NODE_VOCAB, g),
        edge_index=ei,
        edge_attr=torch.cat([e_half, e_half], dim=0),
        pe=torch.rand(n, PE_DIM, generator=g),
        y=torch.randint(0, 2, (1, OUT_DIM), generator=g).float(),
        num_nodes=n,
    )


def random_connected(n: int, extra: int, seed: int) -> Data:
    g = torch.Generator().manual_seed(seed)
    edges = [(int(torch.randint(0, i, (1,), generator=g)), i) for i in range(1, n)]   # random tree
    for _ in range(extra):
        a, b = torch.randint(0, n, (2,), generator=g).tolist()
        if a != b and (a, b) not in edges and (b, a) not in edges:
            edges.append((a, b))
    return make_graph(edges, n, seed)


def path_graph(n: int, seed: int = 0) -> Data:
    return make_graph([(i, i + 1) for i in range(n - 1)], n, seed)


def mixed_batch() -> Batch:
    """Graphs of very different sizes, so padding is heavy."""
    return Batch.from_data_list([random_connected(n, 3, s) for s, n in enumerate([3, 40, 7, 2, 25])])


def cfg(alpha=0.5, arch="hybrid", hidden=32, layers=2, heads=4, **kw) -> ModelConfig:
    return ModelConfig(alpha=alpha, arch=arch, hidden=hidden, layers=layers, heads=heads, **kw)


def model(alpha=0.5, arch="hybrid", seed=0, **kw) -> nn.Module:
    torch.manual_seed(seed)
    return build_model(cfg(alpha, arch, **kw), INFO)


ALL = [(a, "hybrid") for a in (0.0, 0.25, 0.5, 1.0)] + [(0.0, "gcn")]


@pytest.mark.parametrize("alpha,arch", ALL)
def test_forward_shapes_no_nan(alpha, arch):
    b = mixed_batch()
    m = model(alpha, arch)
    for train in (True, False):
        m.train(train)
        out = m(b)
        assert out.shape == (b.num_graphs, OUT_DIM)
        assert torch.isfinite(out).all()
        h = m.node_embeddings(b)
        assert h.shape == (b.num_nodes, 32) and torch.isfinite(h).all()
    out.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_build_model_types_and_errors():
    assert isinstance(model(0.5), HybridGNN)
    assert isinstance(model(arch="gcn"), GCNBaseline)
    with pytest.raises(ValueError):
        model(arch="nope")
    with pytest.raises(NotImplementedError):
        model("learned")
    with pytest.raises(ValueError):
        model(1.5)
    with pytest.raises(ValueError):
        model(0.5, pool="max")
    assert model(0.5, pool="add")(mixed_batch()).shape == (5, OUT_DIM)


def test_alpha_extremes_skip_modules():
    m0 = model(0.0)
    assert m0.attention_layers() == []
    assert not any(isinstance(x, (nn.MultiheadAttention, GlobalAttn)) for x in m0.modules())
    assert any(isinstance(x, GINEConv) for x in m0.modules())
    m1 = model(1.0)
    assert not any(isinstance(x, GINEConv) for x in m1.modules())
    assert len(m1.attention_layers()) == 2
    assert model(arch="gcn").attention_layers() == []


@pytest.mark.parametrize("alpha,arch", ALL)
def test_eval_and_seed_determinism(alpha, arch):
    b = mixed_batch()
    m1, m2 = model(alpha, arch, seed=3).eval(), model(alpha, arch, seed=3).eval()
    with torch.no_grad():
        o1, o1b, o2 = m1(b), m1(b), m2(b)
    assert torch.equal(o1, o1b)
    assert torch.equal(o1, o2)


def test_attention_cache():
    b = mixed_batch()
    m = model(0.5, heads=4).eval()
    for a in m.attention_layers():
        assert a.last_weights is None
    m.set_attention_cache(True)
    with torch.no_grad():
        m(b)
    nmax = 40
    for a in m.attention_layers():
        w, mask = a.last_weights, a.last_mask
        assert w.shape == (b.num_graphs, 4, nmax, nmax)
        assert mask.shape == (b.num_graphs, nmax) and mask.dtype == torch.bool
        assert torch.isfinite(w).all()
        key_pad = ~mask[:, None, None, :].expand_as(w)
        assert (w[key_pad] == 0).all()                       # padded keys get zero weight
        row_sums = w.sum(-1)                                  # [B, H, Nmax]
        real_q = mask[:, None, :].expand_as(row_sums)
        assert torch.allclose(row_sums[real_q], torch.ones(()), atol=1e-5)
        assert (row_sums[~real_q] == 0).all()                 # padded query rows zeroed
    m.set_attention_cache(False)
    assert all(a.last_weights is None for a in m.attention_layers())


def _jacobian_block(m: nn.Module, data, u: int, v: int) -> torch.Tensor:
    """d node_embeddings_from_h0[v] / d h0[u], shape [hidden, hidden]."""
    h0 = m.embed_inputs(data).detach()

    def f(h):
        return m.node_embeddings_from_h0(h, data)[v]

    return torch.autograd.functional.jacobian(f, h0)[:, u, :]


def test_receptive_field_exact():
    """alpha == 0 with L layers: zero sensitivity beyond L hops, nonzero within."""
    L = 3
    data = Batch.from_data_list([path_graph(12)])
    m = model(0.0, layers=L).eval()
    J = torch.autograd.functional.jacobian(
        lambda h: m.node_embeddings_from_h0(h, data), m.embed_inputs(data).detach()
    )  # [N, hid, N, hid]
    norms = J.abs().sum(dim=(1, 3))                          # [v, u]
    for v in range(12):
        for u in range(12):
            if abs(u - v) > L:
                assert norms[v, u] == 0, (u, v)
            else:
                assert norms[v, u] > 0, (u, v)
    # GCN baseline obeys the same bound
    g = model(arch="gcn", layers=L).eval()
    assert (_jacobian_block(g, data, 0, L + 1) == 0).all()
    assert _jacobian_block(g, data, 0, L).abs().sum() > 0


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0])
def test_attention_reaches_far_nodes(alpha):
    data = Batch.from_data_list([path_graph(12)])
    m = model(alpha, layers=3).eval()
    assert _jacobian_block(m, data, 0, 11).abs().sum() > 0


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_permutation_equivariance(alpha):
    d = random_connected(15, 5, seed=7)
    perm = torch.randperm(15, generator=torch.Generator().manual_seed(1))
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(15)
    dp = Data(x=d.x[perm], edge_index=inv[d.edge_index], edge_attr=d.edge_attr, pe=d.pe[perm], y=d.y, num_nodes=15)
    m = model(alpha).eval()
    with torch.no_grad():
        h = m.node_embeddings(Batch.from_data_list([d]))
        hp = m.node_embeddings(Batch.from_data_list([dp]))
        o, op = m(Batch.from_data_list([d])), m(Batch.from_data_list([dp]))
    assert torch.allclose(hp, h[perm], atol=1e-5)
    assert torch.allclose(o, op, atol=1e-5)


@pytest.mark.parametrize("alpha", [0.5, 1.0])
def test_graph_isolation(alpha):
    """Graph A's node embeddings don't depend on graph B's inputs (no attention leakage)."""
    b = Batch.from_data_list([random_connected(6, 2, seed=1), random_connected(9, 2, seed=2)])
    m = model(alpha).eval()
    h0 = m.embed_inputs(b).detach().requires_grad_(True)
    h = m.node_embeddings_from_h0(h0, b)
    a_nodes, b_nodes = b.batch == 0, b.batch == 1
    (grad,) = torch.autograd.grad(h[a_nodes].sum(), h0)
    assert (grad[b_nodes] == 0).all()
    assert grad[a_nodes].abs().sum() > 0


def test_single_data_without_batch_vector():
    d = random_connected(8, 2, seed=4)
    m = model(0.5).eval()
    assert m(d).shape == (1, OUT_DIM)


def test_hybrid_layer_standalone():
    layer = HybridLayer(16, 2, 0.5, edge_dim=16).eval()
    b = mixed_batch()
    x = torch.randn(b.num_nodes, 16)
    e = torch.randn(b.num_edges, 16)
    assert layer(x, b.edge_index, e, b.batch).shape == (b.num_nodes, 16)


def test_parameter_budget(capsys):
    rows = [
        ("hybrid alpha=0", cfg(0.0, hidden=96, layers=5, heads=4)),
        ("hybrid alpha=0.5", cfg(0.5, hidden=96, layers=5, heads=4)),
        ("hybrid alpha=1", cfg(1.0, hidden=96, layers=5, heads=4)),
        ("gcn h=240 L=6", cfg(arch="gcn", hidden=240, layers=6)),
        ("gcn Tönshoff", cfg(arch="gcn", hidden=235, layers=6, act="gelu", norm="none", head_layers=3)),
    ]
    with capsys.disabled():
        print("\nmodel                params")
        for name, c in rows:
            n = count_parameters(build_model(c, INFO))
            print(f"{name:<18} {n:>8,}")
            assert n < 500_000, (name, n)


def test_gcn_tonshoff_options():
    """act / norm / head_layers: GELU, no norm, 3-layer head; defaults keep the Phase 1 structure."""
    m = model(arch="gcn", act="gelu", norm="none", head_layers=3)
    assert all(isinstance(n, nn.Identity) for n in m.norms)
    assert isinstance(m.act, nn.GELU)
    assert sum(isinstance(x, nn.Linear) for x in m.head) == 3
    assert sum(isinstance(x, nn.GELU) for x in m.head) == 2
    assert m(mixed_batch()).shape[1] == OUT_DIM
    old = model(arch="gcn")
    assert all(isinstance(n, nn.BatchNorm1d) for n in old.norms)
    assert [type(x) for x in old.head] == [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]
    with pytest.raises(ValueError):
        model(arch="gcn", norm="layer")
    with pytest.raises(ValueError):
        model(arch="gcn", act="tanh")


def test_forward_from_h0_matches_forward():
    b = mixed_batch()
    for m in (model(0.5).eval(), model(arch="gcn").eval()):
        with torch.no_grad():
            assert torch.allclose(m(b), m.forward_from_h0(m.embed_inputs(b), b))


def test_config_hash_stable_for_new_fields():
    """Fields added in Phase 2 do not change the hash of Phase 1 configs at their defaults."""
    from pathlib import Path
    from gnn_mech.config import load_config
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(str(repo / "configs/pilot_equivalence.yaml"))
    assert cfg.hash() == "d5c95c8973"          # value before the Phase 2 fields existed
    assert load_config(str(repo / "configs/pilot_equivalence.yaml"), ["model.alpha=0.0"]).hash() == "69c07e224f"
    assert load_config(str(repo / "configs/pilot_equivalence.yaml"), ["model.head_layers=3"]).hash() != cfg.hash()
