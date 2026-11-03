"""Social-graph track (docs/social_plan.md): SNAP sources, subgraph partitions, structure stats, datasets.

`SOCIAL_DATASETS` is the set of valid `data.name` values; `gnn_mech.data.load_dataset` dispatches them
to `gnn_mech.social.dataset.load_social`. Kept import-light (no torch_geometric import here).
"""
from __future__ import annotations

MUSAE_LANGS = ("de", "en", "es", "fr", "pt", "ru")
HIGGS_LAYERS = ("follow", "retweet", "mention", "reply")

SOCIAL_DATASETS: frozenset[str] = frozenset(
    ["twitch-gamers", "reddit-body", "social-tiny"]
    + [f"twitch-{l}" for l in MUSAE_LANGS]
    + [f"higgs-{l}" for l in HIGGS_LAYERS]
)

__all__ = ["SOCIAL_DATASETS", "MUSAE_LANGS", "HIGGS_LAYERS"]
