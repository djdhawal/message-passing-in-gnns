"""Reddit node-regression models with matched architecture (Phase 2, B3).

Both models are input_proj (Linear in -> hidden) -> L x SAGEConv -> Linear head,
with ReLU + dropout between convolutions and the same depth, width and dropout.
The only difference is the virtual master node (VMN): one extra node with a
learned initial embedding, linked both ways to every real node, which gives every
pair of nodes a 2-hop path. In the pilot only the VMN model had `input_proj`, so
the architecture changed along with the shortcut.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv


def add_virtual_master(edge_index: Tensor, num_real_nodes: int) -> tuple[Tensor, int]:
    """Append master node `num_real_nodes` with edges real -> master and master -> real (2N edges)."""
    real = torch.arange(num_real_nodes, device=edge_index.device)
    master = torch.full((num_real_nodes,), num_real_nodes, dtype=torch.long, device=edge_index.device)
    aug = torch.cat([edge_index, torch.stack([real, master]), torch.stack([master, real])], dim=1)
    return aug, num_real_nodes + 1


class SAGEBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(SAGEConv(hidden, hidden) for _ in range(layers))
        self.head = nn.Linear(hidden, 1)

    def _convs(self, h: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.dropout(F.relu(h), p=self.dropout, training=self.training)
        return h

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Predictions [N] for the real nodes."""
        return self.head(self._convs(self.input_proj(x), edge_index)).squeeze(-1)


class VMNModel(SAGEBaseline):
    """SAGEBaseline on the graph augmented with a virtual master node (ported from the pilot HGNN_VMN)."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3, dropout: float = 0.3):
        super().__init__(in_dim, hidden, layers, dropout)
        self.master_init = nn.Parameter(torch.randn(1, hidden) * 0.1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        n = x.size(0)
        aug, _ = add_virtual_master(edge_index, n)
        h = torch.cat([self.input_proj(x), self.master_init], dim=0)
        return self.head(self._convs(h, aug)[:n]).squeeze(-1)


MODELS = {"sage": SAGEBaseline, "vmn": VMNModel}


def build_reddit_model(name: str, in_dim: int, hidden: int, layers: int, dropout: float) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, got {name!r}")
    return MODELS[name](in_dim, hidden, layers, dropout)
