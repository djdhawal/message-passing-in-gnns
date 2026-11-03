"""Message-passing and attention layers.

`HybridLayer` blends a local GINE branch and a global attention branch with a
fixed weight `alpha`: 0 is a pure MPNN (with residuals and FFN), 1 is a pure
graph transformer, values in between are GPS-style hybrids. The global branch is
dense softmax attention (`GlobalAttn`) or FAVOR+ linear attention (`PerformerAttn`);
both derive from `AttentionBase`, which is what `attention_layers()` collects.

Normalization: all three norms are `BatchNorm1d`, as in GraphGPS. With
`branch_affine=False` the two branch norms have no learnable scale/shift, so the
(1 - alpha) / alpha weights cannot be rescaled away by training. In `eval()`
mode BatchNorm is a fixed per-feature affine map applied to each node
independently, so it neither mixes nodes nor graphs; Jacobian and isolation
measurements (done in eval mode) are therefore unaffected by it. In training
mode BN statistics couple the nodes of a mini-batch, which is the standard GPS
behaviour.

`RadiusAttn` has two exact implementations of the same function: a dense one (masked
`nn.MultiheadAttention` on the padded batch, for subgraphs) and a sparse one over the k-hop
pair list (`hop_pairs`, softmax per query with `torch_geometric.utils.softmax`), picked
automatically when B * Nmax^2 exceeds `RADIUS_DENSE_MAX_ELEMENTS` (full-graph runs).
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, GINEConv
from torch_geometric.utils import softmax as segment_softmax
from torch_geometric.utils import to_dense_adj, to_dense_batch

ATTN_TYPES = ("softmax", "performer", "radius")

# RadiusAttn: above this many dense mask entries (B * Nmax^2) the sparse pair-list path is used.
# Higgs 500-node subgraphs at batch 8 (2M) stay dense; a 35k-node full graph (1.3e9) goes sparse.
RADIUS_DENSE_MAX_ELEMENTS = 16_000_000
DEFAULT_MAX_PAIRS = 20_000_000        # ModelConfig.attn_max_pairs default
_PAIR_CACHE: "OrderedDict[tuple, Tensor]" = OrderedDict()
_PAIR_CACHE_SIZE = 4
LOCAL_TYPES = ("gine", "gcn")


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


def hop_reach(edge_index: Tensor, batch: Tensor, radius: int, max_nodes: int) -> Tensor:
    """[B, Nmax, Nmax] bool: True where key j is within `radius` hops of query i in the same graph.

    Hops follow `edge_index` (treated as undirected), the diagonal (0 hops) is always True, and
    padded positions are False. Built by `radius` boolean adjacency products on the dense batch,
    so it depends on the graph only (never on node features) and Jacobians w.r.t. h0 ignore it.
    """
    adj = to_dense_adj(edge_index, batch, max_num_nodes=max_nodes) > 0          # [B, N, N]
    adj = adj | adj.transpose(1, 2)
    b, n = adj.shape[0], adj.shape[1]
    reach = torch.eye(n, dtype=torch.bool, device=adj.device).expand(b, n, n).clone()
    adj_f = adj.to(torch.float32)
    for _ in range(radius):
        nxt = reach | (torch.bmm(reach.to(torch.float32), adj_f) > 0)
        if torch.equal(nxt, reach):                                             # converged: whole component
            break
        reach = nxt
    return reach


def _edge_key(edge_index: Tensor, num_nodes: int) -> tuple:
    """Structure fingerprint of an edge_index (shape, node count and two checksums).

    Datasets clone their graphs on every access, so tensor identity / data_ptr cannot key the cache."""
    if edge_index.numel() == 0:
        return (int(num_nodes), 0, 0, 0, str(edge_index.device))
    e = edge_index.to(torch.int64)
    s1 = int(e[0].sum() * 1_000_003 + e[1].sum())
    s2 = int(((e[0] % 9_973 + 1) * (e[1] % 10_007 + 3)).sum())
    return (int(num_nodes), int(e.size(1)), s1, s2, str(edge_index.device))


def hop_pairs(edge_index: Tensor, num_nodes: int, radius: int, max_pairs: int = DEFAULT_MAX_PAIRS,
              batch: Optional[Tensor] = None) -> Tensor:
    """[2, P] long (query i, key j) pairs with j within `radius` hops of i (diagonal included),
    sorted by query then key, on edge_index's device. Structure only, cached per graph.

    Hops follow `edge_index` treated as undirected; batched graphs have no edges between them, so
    no cross-graph pairs arise. radius < 0 gives every same-graph pair (needs `batch`). Raises
    ValueError as soon as the count exceeds `max_pairs` (checked in row chunks, before the full
    list is materialised).
    """
    n = int(num_nodes)
    key = (_edge_key(edge_index, n), int(radius), int(max_pairs),
           None if radius >= 0 or batch is None else int(batch.max()) + 1)
    hit = _PAIR_CACHE.get(key)
    if hit is not None:
        _PAIR_CACHE.move_to_end(key)
        return hit

    def too_many(count: int) -> ValueError:
        return ValueError(
            f"RadiusAttn: the {'all-pairs' if radius < 0 else f'{radius}-hop'} pair list on a {n}-node graph "
            f"has more than model.attn_max_pairs={max_pairs:,} pairs (>= {count:,}); lower model.attn_radius, "
            f"raise model.attn_max_pairs (GPU memory grows ~linearly in the pairs) or use model.attn=performer")

    if radius < 0:
        if batch is None:
            raise ValueError("hop_pairs with radius < 0 needs the batch vector")
        b = batch.detach().cpu().numpy()
        counts = np.bincount(b)
        total = int((counts.astype(np.int64) ** 2).sum())
        if total > max_pairs:
            raise too_many(total)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        order = np.argsort(b, kind="stable")
        rows, cols = [], []
        for s0, c in zip(starts, counts):
            idx = order[s0:s0 + c]
            rows.append(np.repeat(idx, c))
            cols.append(np.tile(idx, c))
        qi, kj = np.concatenate(rows), np.concatenate(cols)
    else:
        ei = edge_index.detach().cpu().numpy()
        a = sp.csr_matrix((np.ones(ei.shape[1], dtype=np.float32), (ei[0], ei[1])), shape=(n, n))
        m = (a + a.T + sp.eye(n, dtype=np.float32, format="csr")).tocsr()
        m.data[:] = 1.0
        reach = sp.eye(n, dtype=np.float32, format="csr")
        chunk = max(1, min(n, 2_000_000 // max(1, int(m.nnz / max(n, 1)) ** 2 + 1)))
        for _ in range(int(radius)):
            parts, count = [], 0
            for s0 in range(0, n, chunk):
                blk = reach[s0:s0 + chunk] @ m
                count += blk.nnz
                if count > max_pairs:
                    raise too_many(count)
                parts.append(blk)
            nxt = sp.vstack(parts).tocsr()
            if nxt.nnz == reach.nnz:                     # converged: whole components
                break
            reach = nxt
        if reach.nnz > max_pairs:
            raise too_many(reach.nnz)
        reach.sort_indices()
        coo = reach.tocoo()
        qi, kj = coo.row, coo.col
        order = np.lexsort((kj, qi))
        qi, kj = qi[order], kj[order]
    pairs = torch.from_numpy(np.stack([qi, kj]).astype(np.int64)).to(edge_index.device)
    _PAIR_CACHE[key] = pairs
    while len(_PAIR_CACHE) > _PAIR_CACHE_SIZE:
        _PAIR_CACHE.popitem(last=False)
    return pairs


class RadiusAttn(GlobalAttn):
    """Softmax attention restricted to nodes within `radius` hops (same graph).

    The attention-radius knob: radius 0 is self-attention only (a per-node map), radius >= the
    graph diameter (or -1) is full attention within the graph, as `GlobalAttn`. Unlike the alpha
    weight, the mask cannot be undone by learned scales: it limits which nodes attention sees.
    Cached weights are exactly zero outside the radius.
    """

    def __init__(self, hidden: int, heads: int, attn_dropout: float = 0.0, radius: int = -1,
                 max_pairs: int = DEFAULT_MAX_PAIRS, sparse: Optional[bool] = None):
        super().__init__(hidden, heads, attn_dropout)
        self.radius = int(radius)
        self.max_pairs = int(max_pairs)
        self.sparse = sparse                      # None = auto by size; True / False force a path
        self.last_pairs: Optional[Tensor] = None
        self.last_pair_weights: Optional[Tensor] = None
        self.last_n_pairs: int = 0

    def clear_cache(self) -> None:
        super().clear_cache()
        self.last_pairs = self.last_pair_weights = None

    def use_sparse(self, batch: Tensor) -> bool:
        if self.sparse is not None:
            return bool(self.sparse)
        counts = torch.bincount(batch)
        return int(counts.numel()) * int(counts.max()) ** 2 > RADIUS_DENSE_MAX_ELEMENTS

    def forward(self, x: Tensor, batch: Tensor, edge_index: Optional[Tensor] = None) -> Tensor:
        if x.size(0) > 0 and self.use_sparse(batch):
            if edge_index is None and self.radius >= 0:
                raise ValueError("RadiusAttn needs edge_index")
            return self._sparse_forward(x, batch, edge_index)
        if self.radius < 0:
            return super().forward(x, batch)
        if edge_index is None:
            raise ValueError("RadiusAttn needs edge_index")
        dense, mask = to_dense_batch(x, batch)                                  # [B, Nmax, h], [B, Nmax]
        n = dense.size(1)
        allowed = hop_reach(edge_index, batch, self.radius, n) & mask[:, None, :]
        # padded query rows would be fully masked (NaN); let them see themselves, they are dropped below
        eye = torch.eye(n, dtype=torch.bool, device=x.device)
        allowed = allowed | (eye & ~mask[:, :, None])
        attn_mask = (~allowed).repeat_interleave(self.heads, dim=0)            # [B*heads, N, N], True = blocked
        out, w = self.attn(dense, dense, dense, attn_mask=attn_mask, need_weights=self.cache,
                           average_attn_weights=False)
        self.last_n_pairs = int(allowed.sum())
        if self.cache:
            self.last_weights = (w * mask[:, None, :, None]).detach()
            self.last_mask = mask.detach()
        return out[mask]

    def _sparse_forward(self, x: Tensor, batch: Tensor, edge_index: Optional[Tensor]) -> Tensor:
        """Same function as the dense path (same nn.MultiheadAttention weights), over the k-hop pair
        list: per-pair scores q_i . k_j / sqrt(d), softmax over each query's pairs, weighted sum of v_j.
        With cache on, stores `last_pairs` [2, P] and `last_pair_weights` [P, heads] (last_weights None)."""
        n, hidden = x.shape
        d = hidden // self.heads
        ei = edge_index if edge_index is not None else torch.zeros(2, 0, dtype=torch.long, device=x.device)
        pairs = hop_pairs(ei, n, self.radius, self.max_pairs, batch=batch)
        qi, kj = pairs[0], pairs[1]
        mha = self.attn
        qkv = nn.functional.linear(x, mha.in_proj_weight, mha.in_proj_bias)
        q, k, v = qkv.view(n, 3, self.heads, d).unbind(1)                      # [N, H, d] each
        scores = (q[qi] * k[kj]).sum(-1) * (d ** -0.5)                         # [P, H]
        w = segment_softmax(scores, qi, num_nodes=n)
        self.last_n_pairs = int(qi.numel())
        if self.cache:
            self.last_pairs, self.last_pair_weights = pairs, w.detach()
            self.last_weights = self.last_mask = None
        w = nn.functional.dropout(w, p=mha.dropout, training=self.training)
        out = torch.zeros(n, self.heads, d, dtype=v.dtype, device=x.device)
        out = out.index_add(0, qi, w[..., None] * v[kj])
        return mha.out_proj(out.reshape(n, hidden))


class GCNLocal(nn.Module):
    """Degree-normalised local branch: GCNConv (symmetric normalisation, self loops) -> ReLU -> Linear.

    For social graphs, where GINE's sum aggregation over hubs with hundreds of neighbours gives
    large, high-variance messages. Edge features are ignored.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.conv = GCNConv(hidden, hidden)
        self.out = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None) -> Tensor:
        return self.out(self.conv(x, edge_index))


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
                    performer_features: int = 64, radius: int = -1,
                    max_pairs: int = DEFAULT_MAX_PAIRS) -> AttentionBase:
    """GlobalAttn for kind="softmax", PerformerAttn for "performer", RadiusAttn for "radius"."""
    if kind == "radius":
        return RadiusAttn(hidden, heads, attn_dropout, radius, max_pairs=max_pairs)
    if kind == "softmax":
        return GlobalAttn(hidden, heads, attn_dropout)
    if kind == "performer":
        return PerformerAttn(hidden, heads, n_features=performer_features, attn_dropout=attn_dropout)
    raise ValueError(f"attn must be one of {ATTN_TYPES}, got {kind!r}")


class HybridLayer(nn.Module):
    """One alpha-blended local/global layer.

        h_loc = Norm_loc(Local(x, edge_index, e))      (not built when alpha == 1; Local = GINE or GCNLocal)
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
        local: str = "gine",
        branch_affine: bool = True,
        attn_radius: int = -1,
        attn_max_pairs: int = DEFAULT_MAX_PAIRS,
    ):
        super().__init__()
        if local not in LOCAL_TYPES:
            raise ValueError(f"local must be one of {LOCAL_TYPES}, got {local!r}")
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

        self.local: Optional[nn.Module] = None
        self.norm_loc: Optional[nn.Module] = None
        if alpha < 1.0:
            if local == "gcn":
                self.local = GCNLocal(hidden)
            else:
                mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                self.local = GINEConv(mlp, train_eps=True, edge_dim=gine_edge_dim)
            self.norm_loc = nn.BatchNorm1d(hidden, affine=branch_affine)

        self.glob: Optional[AttentionBase] = None
        self.norm_glb: Optional[nn.Module] = None
        if alpha > 0.0:
            self.glob = build_attention(attn, hidden, heads, attn_dropout, performer_features, attn_radius,
                                        attn_max_pairs)
            self.norm_glb = nn.BatchNorm1d(hidden, affine=branch_affine)

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
            g = self.glob(x, batch, edge_index) if isinstance(self.glob, RadiusAttn) else self.glob(x, batch)
            mix = mix + self.alpha * self.norm_glb(g)
        x = x + self.dropout(mix)
        return self.norm_ffn(x + self.ffn(x))
