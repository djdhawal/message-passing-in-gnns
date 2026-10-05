"""Input encoders: categorical atom/bond embeddings and RWSE projection.

    CategoricalEncoder(vocab, dim)  sum of one nn.Embedding per categorical column
    InputEncoder(info, hidden)      h0 = CategoricalEncoder(x) + Linear(pe)   -> [N, hidden]
    EdgeEncoder(info, hidden)       CategoricalEncoder over edge_attr         -> [E, hidden]

Models embed edges with `EdgeEncoder(info, hidden)(batch.edge_attr)`, or build a
`CategoricalEncoder(info.edge_vocab, dim)` directly when they need another width.
The PE projection is summed (not concatenated) so `hidden` can be any size.
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .data import DataInfo


class CategoricalEncoder(nn.Module):
    """Sum of per-column embeddings for LongTensor codes of shape [*, len(vocab)]."""

    def __init__(self, vocab: list[int], dim: int):
        super().__init__()
        self.vocab = list(vocab)
        self.embs = nn.ModuleList(nn.Embedding(v, dim) for v in self.vocab)
        for e in self.embs:
            nn.init.xavier_uniform_(e.weight)

    def forward(self, codes: Tensor) -> Tensor:
        if codes.dim() == 1:
            codes = codes.unsqueeze(-1)
        if codes.size(-1) != len(self.embs):
            raise ValueError(f"expected {len(self.embs)} categorical columns, got {codes.size(-1)}")
        codes = codes.long()
        out = self.embs[0](codes[..., 0])
        for i in range(1, len(self.embs)):
            out = out + self.embs[i](codes[..., i])
        return out


class InputEncoder(nn.Module):
    """Node codes (+ RWSE when info.pe_dim > 0) -> h0 [N, hidden]."""

    def __init__(self, info: DataInfo, hidden: int):
        super().__init__()
        self.node = CategoricalEncoder(info.node_vocab, hidden)
        self.pe_dim = info.pe_dim
        self.pe = nn.Linear(info.pe_dim, hidden) if info.pe_dim > 0 else None

    def forward(self, batch) -> Tensor:
        h = self.node(batch.x)
        if self.pe is not None:
            h = h + self.pe(batch.pe.to(h.dtype))
        return h


class EdgeEncoder(nn.Module):
    """Edge codes -> [E, hidden]."""

    def __init__(self, info: DataInfo, hidden: int):
        super().__init__()
        self.enc = CategoricalEncoder(info.edge_vocab, hidden)

    def forward(self, edge_attr: Tensor) -> Tensor:
        return self.enc(edge_attr)
