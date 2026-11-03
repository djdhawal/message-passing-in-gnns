"""Experiment configuration.

A run is fully described by a `Config`. Configs are loaded from YAML and can be
overridden from the command line with dotted keys, e.g. `model.alpha=0.25`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import yaml


@dataclass
class DataConfig:
    name: str = "Peptides-func"      # LRGB dataset name, or "synthetic-tiny" for smoke tests
    root: str = "./data/LRGB"
    pe_dim: int = 20                  # RWSE walk length; 0 disables positional encoding
    structure_cache: Optional[str] = None  # dir for cached per-graph stats; defaults to <root>/structure
    num_workers: int = 0
    # Social-graph (node-level) datasets; see docs/social_plan.md "Build contract".
    target: str = ""                  # dataset-specific target name; "" = dataset default
    subgraph_size: int = 2000         # target nodes per subgraph (core, before halo)
    sampler: str = "partition"        # "partition" (METIS, fallback BFS-grown parts) | "bfs" | "full" (transductive:
                                      # whole graph per split, parts only choose the labelled nodes)
    halo_hops: int = 0                # context-only hops around each core (y_mask False); 0 = strict
    val_frac: float = 0.15            # fraction of subgraphs (whole parts) per split; train gets the rest
    test_frac: float = 0.15
    split_seed: int = 0               # partition/split seed, fixed across model seeds
    max_subgraphs: int = 0            # 0 = all; else cap per split (quick runs), applied after splitting


@dataclass
class ModelConfig:
    alpha: Union[float, str] = 0.5    # 0 = pure MPNN, 1 = pure attention; "learned" reserved for later phases
    local: str = "gine"               # HybridGNN local branch: "gine" (sum aggregation, edge-aware) | "gcn"
                                      # (symmetric degree-normalised GCNConv + MLP; for high-degree social graphs)
    arch: str = "hybrid"              # "hybrid" (HybridGNN), "gcn" (Tönshoff-style GCN) or "mlp" (no graph)
    hidden: int = 96
    layers: int = 5
    heads: int = 4
    dropout: float = 0.1
    attn_dropout: float = 0.1
    pool: str = "mean"
    # GCN-baseline-only knobs (HybridGNN layers always use ReLU + BatchNorm), added in
    # Phase 2 to match Tönshoff et al.'s tuned GCN. Defaults reproduce the Phase 1 GCN.
    act: str = "relu"                 # GCN nonlinearity: "relu" | "gelu"
    norm: str = "batch"               # GCN per-layer norm: "batch" | "none"
    head_layers: int = 2              # Linear layers in the graph-level MLP head (both models)
    attn: str = "softmax"             # global branch: "softmax" (GlobalAttn) | "performer" (PerformerAttn)
                                      # | "radius" (RadiusAttn: softmax masked to attn_radius hops)
    performer_features: int = 64      # FAVOR+ random features m per head
    attn_radius: int = -1             # attn="radius": attend only to nodes within this many hops (-1 = whole graph)
    attn_max_pairs: int = 20_000_000  # attn="radius", sparse path (large graphs): max k-hop (query, key) pairs
                                      # per forward; more raises a ValueError instead of running out of memory
    branch_affine: bool = True        # learnable affine in the two branch norms. False makes alpha a real,
                                      # fixed mixing weight (with affine on, alpha is absorbed by the norm scales)


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    patience: int = 0                 # 0 disables early stopping
    checkpoint_every: int = 5         # epochs between resumable checkpoints
    device: str = "auto"


@dataclass
class MeasureConfig:
    enabled: bool = True
    split: str = "test"
    n_graphs_jacobian: int = 200
    # Each target node costs one full Jacobian sweep, which yields its sensitivity to
    # every source at once. targets_per_graph > 0 samples that many targets per graph
    # and pairs_per_graph sources per target (stratified by resistance to the target);
    # targets_per_graph = 0 samples pairs_per_graph independent pairs per graph.
    targets_per_graph: int = 4
    pairs_per_graph: int = 32
    n_graphs_entropy: int = 500
    # Graph-level (Hessian) range, Bamberger et al. 2025: n_graphs_range graphs, each
    # with sources_per_graph sampled source nodes (one batch of Hessian-vector products
    # per source). 0 graphs disables it. The node-level range needs no setting: it is
    # computed from the Jacobian sweeps of the targets above.
    n_graphs_range: int = 30
    sources_per_graph: int = 4
    hessian_channels: int = 16        # source-side h0 channels sampled per graph (0 = all D0)
    entropy_queries: int = 64         # Performer: sampled query nodes per graph for implicit-row entropy
    seed: int = 0


@dataclass
class Config:
    exp_name: str = "default"
    out_dir: str = "./runs"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    measure: MeasureConfig = field(default_factory=MeasureConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """Short hash of everything except seed, out_dir and exp_name.

        Runs that differ only by seed share a hash, so seeds group naturally.
        Fields added after Phase 1 (`_ADDED_FIELDS`) enter the hash only when they
        differ from their default, so run folders written before they existed keep
        their hash.
        """
        d = self.to_dict()
        for k in ("seed", "out_dir", "exp_name"):
            d.pop(k)
        d["train"].pop("device")
        d["data"].pop("root")
        d["data"].pop("structure_cache")
        d["data"].pop("num_workers")
        for section, names in _ADDED_FIELDS.items():
            defaults = _SECTIONS[section]()
            for name in names:
                if d[section][name] == getattr(defaults, name):
                    d[section].pop(name)
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:10]


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig, "measure": MeasureConfig}

# Fields added after Phase 1 runs were written; omitted from Config.hash() at their default.
_ADDED_FIELDS = {
    "data": ("target", "subgraph_size", "sampler", "halo_hops", "val_frac", "test_frac", "split_seed",
             "max_subgraphs"),
    "model": ("act", "norm", "head_layers", "attn", "performer_features", "attn_radius",
              "branch_affine", "attn_max_pairs"),
    "measure": ("n_graphs_range", "sources_per_graph", "hessian_channels", "entropy_queries"),
}


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def from_dict(d: dict) -> Config:
    d = dict(d)
    kwargs = {}
    for name, cls in _SECTIONS.items():
        section = d.pop(name, {}) or {}
        unknown = set(section) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise KeyError(f"unknown keys in '{name}': {sorted(unknown)}")
        kwargs[name] = cls(**section)
    unknown = set(d) - {f.name for f in dataclasses.fields(Config)}
    if unknown:
        raise KeyError(f"unknown top-level keys: {sorted(unknown)}")
    return Config(**d, **kwargs)


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply `section.key=value` (or `key=value` for top-level) overrides."""
    d = cfg.to_dict()
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        parts = key.split(".")
        target = d
        for p in parts[:-1]:
            target = target[p]
        if parts[-1] not in target:
            raise KeyError(f"unknown config key: {key}")
        target[parts[-1]] = _coerce(raw)
    return from_dict(d)


def load_config(path: Optional[str] = None, overrides: Optional[list[str]] = None) -> Config:
    d = {}
    if path:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
    cfg = from_dict(d)
    return apply_overrides(cfg, overrides or [])
