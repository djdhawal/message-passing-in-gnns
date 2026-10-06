"""Message-passing and attention layers.

`HybridLayer` blends a local GINE branch and a global attention branch with a
fixed weight `alpha`: 0 is a pure MPNN (with residuals and FFN), 1 is a pure
graph transformer, values in between are GPS-style hybrids.

Normalization: all three norms are `BatchNorm1d`, as in GraphGPS. In `eval()`
mode BatchNorm is a fixed per-feature affine map applied to each node
independently, so it neither mixes nodes nor graphs; Jacobian and isolation
measurements (done in eval mode) are therefore unaffected by it. In training
mode BN statistics couple the nodes of a mini-batch, which is the standard GPS
behaviour.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch


class GlobalAttn(nn.Module):
    """Dense multi-head self-attention restricted to nodes of the same graph.

    Returns the attention output only (no residual, no norm). When `cache` is
    True, the per-head weights `[B, heads, Nmax, Nmax]` and the node mask
    `[B, Nmax]` of the last forward are stored (detached). Padded query rows are
    zeroed in the stored weights; padded keys always get weight 0.
    """

    def __init__(self, hidden: int, heads: int, attn_dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=attn_dropout, batch_first=True)
        self.cache: bool = False
        self.last_weights: Optional[Tensor] = None
        self.last_mask: Optional[Tensor] = None

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        dense, mask = to_dense_batch(x, batch)          # [B, Nmax, h], [B, Nmax]
        out, w = self.attn(
            dense, dense, dense,
            key_padding_mask=~mask,
            need_weights=self.cache,
            average_attn_weights=False,
        )
        if self.cache:
            # w: [B, heads, Nmax, Nmax]; zero rows of padded queries
            self.last_weights = (w * mask[:, None, :, None]).detach()
            self.last_mask = mask.detach()
        # Every graph has >= 1 real node, so no query row is fully masked and
        # no NaNs arise; padded query rows are dropped here.
        return out[mask]


class HybridLayer(nn.Module):
    """One alpha-blended local/global layer.

        h_loc = Norm_loc(GINE(x, edge_index, e))       (not built when alpha == 1)
        h_glb = Norm_glb(Attn(x, batch))               (not built when alpha == 0)
        x = x + dropout((1 - alpha) * h_loc + alpha * h_glb)
        x = Norm_ffn(x + FFN(x))
    """

    def __init__(
        self,
        hidden: int,
        heads: int,
        alpha: Union[float, str],
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()
        if isinstance(alpha, str):
            if alpha == "learned":
                raise NotImplementedError("alpha='learned' is reserved for later phases")
            raise ValueError(f"alpha must be a float in [0, 1] or 'learned', got {alpha!r}")
        alpha = float(alpha)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        # Edge features already of width `hidden` are fed to GINE directly (as in
        # GraphGPS); only a different width gets a per-layer projection.
        gine_edge_dim = None if edge_dim in (None, hidden) else edge_dim

        self.local: Optional[GINEConv] = None
        self.norm_loc: Optional[nn.Module] = None
        if alpha < 1.0:
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.local = GINEConv(mlp, train_eps=True, edge_dim=gine_edge_dim)
            self.norm_loc = nn.BatchNorm1d(hidden)

        self.glob: Optional[GlobalAttn] = None
        self.norm_glb: Optional[nn.Module] = None
        if alpha > 0.0:
            self.glob = GlobalAttn(hidden, heads, attn_dropout)
            self.norm_glb = nn.BatchNorm1d(hidden)

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(2 * hidden, hidden)
        )
        self.norm_ffn = nn.BatchNorm1d(hidden)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, batch: Tensor) -> Tensor:
        mix = 0.0
        if self.local is not None:
            mix = mix + (1.0 - self.alpha) * self.norm_loc(self.local(x, edge_index, edge_attr))
        if self.glob is not None:
            mix = mix + self.alpha * self.norm_glb(self.glob(x, batch))
        x = x + self.dropout(mix)
        return self.norm_ffn(x + self.ffn(x))
