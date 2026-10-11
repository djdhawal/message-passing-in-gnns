"""Reddit hyperlink node-regression track (Phase 2, Track B).

Separate from the graph-level Peptides pipeline: one 35,776-node graph, node
regression, full-batch training. Reuses `RunDir` (via the duck-typed
`RedditConfig`) and `set_seed`.

    data.py           SNAP graph construction, FEATURE_NAMES, BFS bands, splits
    target_select.py  pick a target whose value depends on multi-hop neighbourhoods
    models.py         SAGE baseline and virtual-master-node (VMN) model, same input projection
    run.py            distance-split main experiment, random-split control, per-band metrics
"""
