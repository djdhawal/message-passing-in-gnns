"""Message-passing and attention layers.

`HybridLayer` blends a local GINE branch and a global attention branch with a
fixed weight `alpha`: 0 is a pure MPNN (with residuals and FFN), 1 is a pure
graph transformer, values in between are GPS-style hybrids. The global branch is
dense softmax attention (`GlobalAttn`) or FAVOR+ linear attention (`PerformerAttn`);
both derive from `AttentionBase`, which is what `attention_layers()` collects.

Normalization: all three norms are `BatchNorm1d`, as in GraphGPS. In `eval()`
mode BatchNorm is a fixed per-feature affine map applied to each node
independently, so it neither mixes nodes nor graphs; Jacobian and isolation
measurements (done in eval mode) are therefore unaffected by it. In training
mode BN statistics couple the nodes of a mini-batch, which is the standard GPS
behaviour.
"""
from __future__ import annotations

import math
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

ATTN_TYPES = ("softmax", "performer")


class AttentionBase(nn.Module):
    """Common base of the global attention modules (`cache` / `last_*` attributes)."""

    def __init__(self):
        super().__init__()
        self.cache: bool = False
        self.last_weights: Optional[Tensor] = None
        self.last_mask: Optional[Tensor] = None

    def clear_cache(self) -> None:
        self.last_weights = None
        self.last_mask = None


class GlobalAttn(AttentionBase):
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


def orthogonal_gaussian(heads: int, m: int, d: int, generator: Optional[torch.Generator] = None,
                        device=None, dtype=torch.float32) -> Tensor:
    """[heads, m, d] random features: blocks of orthogonal rows (QR of Gaussians), each row
    rescaled to the norm of an independent Gaussian vector, so every row is marginally
    N(0, I_d) (Choromanski et al. 2021, "orthogonal random features")."""
    out = []
    for _ in range(heads):
        blocks, left = [], m
        while left > 0:
            g = torch.randn(d, d, generator=generator, dtype=torch.float32)
            q, _ = torch.linalg.qr(g)
            blocks.append(q.T[:min(left, d)])
            left -= d
        w = torch.cat(blocks, dim=0)                                         # [m, d], unit rows
        norms = torch.randn(m, d, generator=generator, dtype=torch.float32).norm(dim=1, keepdim=True)
        out.append(w * norms)
    return torch.stack(out).to(device=device, dtype=dtype)


class PerformerAttn(AttentionBase):
    """FAVOR+ linear multi-head self-attention restricted to nodes of the same graph.

    With q' = W_q x / d^(1/4), k' = W_k x / d^(1/4) (d = head width) and positive random
    features phi(z) = exp(w_r . z - |z|^2 / 2) / sqrt(m), r = 1..m, the output of node i in
    graph g is
        out_i = phi(q'_i)^T S_g / (phi(q'_i)^T z_g),
        S_g = sum_{j in g} phi(k'_j) v_j^T,   z_g = sum_{j in g} phi(k'_j),
    an unbiased estimate of softmax(q k^T / sqrt(d)) V in O(N m d). The per-graph sums are
    segment sums over `batch` (computed as an einsum on the zero-padded dense batch, which
    is the same sum). phi is stabilised by subtracting the (detached) max of each query's
    exponent and, for keys, the max over the graph's keys, which cancel in the ratio, plus
    `eps` inside phi (as in the reference implementation) so no row can underflow to 0/0.

    Random features: `self.omega` [heads, m, d] is a persistent buffer drawn at
    construction (from the global torch RNG, so it is fixed by the model seed and saved
    with the state dict). In training mode with `redraw=True` fresh features are drawn
    every forward; in eval mode the buffer is always used, so evaluation and Jacobians are
    deterministic. Pure PyTorch: double backward works.

    `attn_dropout` is accepted for signature parity with `GlobalAttn` and ignored (there
    is no explicit attention matrix to drop entries from).

    Cache: when `cache` is True, forward stores the detached features `last_q_feat`,
    `last_k_feat` [N, heads, m] and `last_batch` [N]; `implicit_rows(query_index)` then
    returns the exact attention rows the forward used. `last_weights` stays None.
    """

    def __init__(self, hidden: int, heads: int, n_features: int = 64, redraw: bool = True,
                 attn_dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible by heads ({heads})")
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        self.heads = heads
        self.head_dim = hidden // heads
        self.n_features = n_features
        self.redraw = redraw
        self.eps = eps
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.register_buffer("omega", orthogonal_gaussian(heads, n_features, self.head_dim))
        self.last_q_feat: Optional[Tensor] = None
        self.last_k_feat: Optional[Tensor] = None
        self.last_batch: Optional[Tensor] = None

    def clear_cache(self) -> None:
        super().clear_cache()
        self.last_q_feat = self.last_k_feat = self.last_batch = None

    def features(self) -> Tensor:
        """Random features for this forward: fresh in training (if redraw), else the buffer."""
        if self.training and self.redraw:
            return orthogonal_gaussian(self.heads, self.n_features, self.head_dim,
                                       device=self.omega.device, dtype=self.omega.dtype)
        return self.omega

    def _qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n = x.shape[0]
        scale = self.head_dim ** -0.25
        q = self.q_proj(x).view(n, self.heads, self.head_dim) * scale
        k = self.k_proj(x).view(n, self.heads, self.head_dim) * scale
        v = self.v_proj(x).view(n, self.heads, self.head_dim)
        return q, k, v

    def _phi(self, z: Tensor, omega: Tensor, batch: Optional[Tensor], n_graphs: int) -> Tensor:
        """Positive features [N, H, m]; batch=None stabilises per row (queries), else per graph (keys)."""
        logits = torch.einsum("nhd,hmd->nhm", z, omega) - 0.5 * z.pow(2).sum(-1, keepdim=True)
        if batch is None:
            stab = logits.detach().amax(dim=-1, keepdim=True)
        else:
            row_max = logits.detach().amax(dim=-1)                               # [N, H]
            g_max = torch.full((n_graphs, self.heads), -math.inf, dtype=row_max.dtype, device=row_max.device)
            g_max = g_max.scatter_reduce(0, batch[:, None].expand(-1, self.heads), row_max, "amax")
            stab = g_max[batch][..., None]
        return (torch.exp(logits - stab) + self.eps) / math.sqrt(self.n_features)

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        n = x.shape[0]
        n_graphs = int(batch.max()) + 1 if n > 0 else 0
        q, k, v = self._qkv(x)
        omega = self.features()
        qf = self._phi(q, omega, None, n_graphs)                              # [N, H, m]
        kf = self._phi(k, omega, batch, n_graphs)                             # [N, H, m]
        if self.cache:
            self.last_q_feat, self.last_k_feat = qf.detach(), kf.detach()
            self.last_batch = batch.detach()
        # Segment sums per graph on the zero-padded dense batch (padding contributes 0).
        kf_d, mask = to_dense_batch(kf.reshape(n, -1), batch, batch_size=n_graphs)  # [B, Nmax, H*m]
        v_d, _ = to_dense_batch(v.reshape(n, -1), batch, batch_size=n_graphs)
        kf_d = kf_d.view(n_graphs, -1, self.heads, self.n_features)
        v_d = v_d.view(n_graphs, -1, self.heads, self.head_dim)
        s = torch.einsum("bnhm,bnhd->bhmd", kf_d, v_d)                        # [B, H, m, d]
        zsum = kf_d.sum(dim=1)                                                # [B, H, m]
        qf_d, _ = to_dense_batch(qf.reshape(n, -1), batch, batch_size=n_graphs)
        qf_d = qf_d.view(n_graphs, -1, self.heads, self.n_features)
        num = torch.einsum("bnhm,bhmd->bnhd", qf_d, s)
        den = torch.einsum("bnhm,bhm->bnh", qf_d, zsum)
        den = den + (~mask)[..., None].to(den.dtype)          # padded queries: avoid 0/0 (dropped below)
        out = (num / den[..., None])[mask]                                    # [N, H, d]
        return self.out_proj(out.reshape(n, -1))

    @torch.no_grad()
    def implicit_rows(self, query_index: Tensor) -> Tensor:
        """Exact attention rows A[i, h, j] = phi(q_i).phi(k_j) / sum_{j' in g(i)} phi(q_i).phi(k_j')
        of the last cached forward, for the given query node indices (into that forward's
        nodes). Returns [Q, heads, N]; entries for keys outside the query's graph are 0 and
        each row sums to 1 over its graph."""
        if self.last_q_feat is None:
            raise RuntimeError("implicit_rows needs a forward pass with cache=True first")
        qi = torch.as_tensor(query_index, dtype=torch.long, device=self.last_q_feat.device).reshape(-1)
        qf = self.last_q_feat[qi]                                             # [Q, H, m]
        scores = torch.einsum("qhm,nhm->qhn", qf, self.last_k_feat)           # [Q, H, N]
        same = (self.last_batch[qi][:, None] == self.last_batch[None, :])     # [Q, N]
        scores = scores * same[:, None, :].to(scores.dtype)
        return scores / scores.sum(dim=-1, keepdim=True)

    def exact_attention(self, x: Tensor, batch: Tensor) -> Tensor:
        """Reference output with exact softmax(q k^T / sqrt(d)) on the same projections (tests, gate S1a)."""
        n = x.shape[0]
        q, k, v = self._qkv(x)
        logits = torch.einsum("ihd,jhd->hij", q, k)                           # q, k carry d^(-1/4) each
        same = batch[:, None] == batch[None, :]
        logits = logits.masked_fill(~same[None], -math.inf)
        a = torch.softmax(logits, dim=-1)
        out = torch.einsum("hij,jhd->ihd", a, v)
        return self.out_proj(out.reshape(n, -1))


def build_attention(kind: str, hidden: int, heads: int, attn_dropout: float = 0.0,
                    performer_features: int = 64) -> AttentionBase:
    """GlobalAttn for kind="softmax", PerformerAttn for kind="performer"."""
    if kind == "softmax":
        return GlobalAttn(hidden, heads, attn_dropout)
    if kind == "performer":
        return PerformerAttn(hidden, heads, n_features=performer_features, attn_dropout=attn_dropout)
    raise ValueError(f"attn must be one of {ATTN_TYPES}, got {kind!r}")


class HybridLayer(nn.Module):
    """One alpha-blended local/global layer.

        h_loc = Norm_loc(GINE(x, edge_index, e))       (not built when alpha == 1)
        h_glb = Norm_glb(Attn(x, batch))               (not built when alpha == 0;
                                                        Attn = GlobalAttn or PerformerAttn per `attn`)
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
        attn: str = "softmax",
        performer_features: int = 64,
    ):
        super().__init__()
        if attn not in ATTN_TYPES:
            raise ValueError(f"attn must be one of {ATTN_TYPES}, got {attn!r}")
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

        self.glob: Optional[AttentionBase] = None
        self.norm_glb: Optional[nn.Module] = None
        if alpha > 0.0:
            self.glob = build_attention(attn, hidden, heads, attn_dropout, performer_features)
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
