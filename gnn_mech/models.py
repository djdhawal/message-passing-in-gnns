"""Graph-level models: alpha-blended hybrid GNN and a Tönshoff-style GCN baseline.

Both expose the same API so training and measurement code is model-agnostic:
`embed_inputs` (batch -> h0), `node_embeddings_from_h0` (a pure function of h0
given the batch structure, so Jacobians w.r.t. h0 are meaningful),
`node_embeddings`, `forward` (graph logits), `attention_layers` and
`set_attention_cache`.
"""
from __future__ import annotations

from torch import Tensor, nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool

from .config import ModelConfig
from .data import DataInfo
from .encoders import CategoricalEncoder, InputEncoder
from .layers import GlobalAttn, HybridLayer

_POOLS = {"mean": global_mean_pool, "add": global_add_pool}
_ACTS = {"relu": nn.ReLU, "gelu": nn.GELU}


def _batch_vector(batch) -> Tensor:
    """Node-to-graph assignment; a single un-batched `Data` maps to graph 0."""
    b = getattr(batch, "batch", None)
    if b is None:
        return torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.edge_index.device)
    return b


def _act(name: str) -> nn.Module:
    if name not in _ACTS:
        raise ValueError(f"act must be one of {sorted(_ACTS)}, got {name!r}")
    return _ACTS[name]()


def _mlp_head(hidden: int, out_dim: int, dropout: float, layers: int = 2, act: str = "relu") -> nn.Sequential:
    """(layers - 1) x [Linear, act, Dropout] then Linear. layers=2 is the Phase 1 head."""
    if layers < 1:
        raise ValueError(f"head_layers must be >= 1, got {layers}")
    mods: list[nn.Module] = []
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), _act(act), nn.Dropout(dropout)]
    return nn.Sequential(*mods, nn.Linear(hidden, out_dim))


def _pool_fn(name: str):
    if name not in _POOLS:
        raise ValueError(f"pool must be one of {sorted(_POOLS)}, got {name!r}")
    return _POOLS[name]


class _GraphModel(nn.Module):
    """Shared encode -> node reps -> pool -> head plumbing."""

    def __init__(self, cfg: ModelConfig, info: DataInfo, head_act: str = "relu"):
        super().__init__()
        self.input_encoder = InputEncoder(info, cfg.hidden)
        self.pool = _pool_fn(cfg.pool)
        self.head = _mlp_head(cfg.hidden, info.out_dim, cfg.dropout, cfg.head_layers, head_act)

    def embed_inputs(self, batch) -> Tensor:
        return self.input_encoder(batch)

    def node_embeddings_from_h0(self, h0: Tensor, batch) -> Tensor:
        raise NotImplementedError

    def node_embeddings(self, batch) -> Tensor:
        return self.node_embeddings_from_h0(self.embed_inputs(batch), batch)

    def forward(self, batch) -> Tensor:
        h = self.node_embeddings(batch)
        return self.head(self.pool(h, _batch_vector(batch)))

    def attention_layers(self) -> list[GlobalAttn]:
        return [m for m in self.modules() if isinstance(m, GlobalAttn)]

    def set_attention_cache(self, on: bool) -> None:
        for m in self.attention_layers():
            m.cache = on
            if not on:
                m.last_weights = None
                m.last_mask = None


class HybridGNN(_GraphModel):
    """InputEncoder -> L x HybridLayer(alpha) -> pool -> MLP head."""

    def __init__(self, cfg: ModelConfig, info: DataInfo):
        super().__init__(cfg, info)
        if cfg.local != "gine":
            raise ValueError(f"HybridGNN supports local='gine' only, got {cfg.local!r}")
        self.edge_encoder = CategoricalEncoder(info.edge_vocab, cfg.hidden)
        self.layers = nn.ModuleList(
            HybridLayer(cfg.hidden, cfg.heads, cfg.alpha, cfg.dropout, cfg.attn_dropout, edge_dim=cfg.hidden)
            for _ in range(cfg.layers)
        )

    def node_embeddings_from_h0(self, h0: Tensor, batch) -> Tensor:
        e = self.edge_encoder(batch.edge_attr)       # independent of h0
        b = _batch_vector(batch)
        x = h0
        for layer in self.layers:
            x = layer(x, batch.edge_index, e, b)
        return x


class GCNBaseline(_GraphModel):
    """Tönshoff et al. GCN: L x (GCNConv -> Norm -> act -> Dropout) with residual, pool, MLP head.

    `cfg.norm` is "batch" (BatchNorm1d) or "none" (Tönshoff's tuned config has no
    norm); `cfg.act` ("relu" | "gelu") is used in the layers and the head.
    Edge features are ignored (GCNConv has no edge-feature input).
    """

    def __init__(self, cfg: ModelConfig, info: DataInfo):
        super().__init__(cfg, info, head_act=cfg.act)
        if cfg.norm not in ("batch", "none"):
            raise ValueError(f"norm must be 'batch' or 'none', got {cfg.norm!r}")
        self.dropout = cfg.dropout
        self.act = _act(cfg.act)
        self.convs = nn.ModuleList(GCNConv(cfg.hidden, cfg.hidden) for _ in range(cfg.layers))
        self.norms = nn.ModuleList(
            nn.BatchNorm1d(cfg.hidden) if cfg.norm == "batch" else nn.Identity() for _ in range(cfg.layers))

    def node_embeddings_from_h0(self, h0: Tensor, batch) -> Tensor:
        x = h0
        for conv, norm in zip(self.convs, self.norms):
            h = self.act(norm(conv(x, batch.edge_index)))
            x = x + F.dropout(h, p=self.dropout, training=self.training)
        return x


def build_model(cfg: ModelConfig, info: DataInfo) -> nn.Module:
    if cfg.arch == "hybrid":
        return HybridGNN(cfg, info)
    if cfg.arch == "gcn":
        return GCNBaseline(cfg, info)
    raise ValueError(f"unknown model arch {cfg.arch!r} (expected 'hybrid' or 'gcn')")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
