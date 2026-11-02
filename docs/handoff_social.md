# Handoff: starting the social-graph phases (S0–S6)

For the agent picking up the project after the 2026-11-02 change of direction. Read this, then
`docs/roadmap.md` (big picture), `docs/social_plan.md` (what to build, gates, interfaces) and
`docs/phase1_plan.md` (the package interface you will extend). `docs/handoff_phase2.md` is
superseded; keep it only for the "learned the hard way" notes, which still apply.

## The decision

- All main experiments now run on **large SNAP social graphs**: Reddit hyperlinks, Twitch
  (twitch-gamers and the 6 musae language graphs), Twitter (higgs-twitter's 4 layers, ego-Twitter).
  **Subgraphs are allowed** and are the unit of training and evaluation.
- **Peptides-func and the river / tree experiments are appendix background only.** Do not spend
  GPU time on Peptides. G0 ran once (0.6259 test AP vs the 0.652 threshold, overfitting) and stays as it is.
- Order of work (the user's): α sweeps on real targets → plant semi-synthetic targets on the same
  graphs → compute range → phase diagram → GFM benchmarking → release the planted tasks as a benchmark.
- The attention branch for the α sweeps is **Performer-style linear attention** (FAVOR+), not dense softmax.

## Why this direction

- LRGB (PascalVOC-SP, COCO-SP, PCQM-Contact, Peptides) and ECHO (arXiv 2512.17762: synthetic
  long-diameter graphs plus molecules) contain **no social graphs**, so a long-range social benchmark is new.
- Gutteridge, Bronstein & Dong (arXiv 2603.22984) show that fixed-backbone GFMs under-reach on tasks
  whose range differs from training, but only for message passing. Testing attention-based GFMs
  (GraphPFN, arXiv 2509.21489) on planted-range social tasks extends that. Gutteridge also wrote the
  range-measure code that `gnn_mech/measure/range.py` follows.
- NodeGround (arXiv 2609.39673): tuned GNNs still beat GFMs overall on node classification, and it
  does no long-range analysis.

## Where things stand

- Branch `main` on `djdhawal/message-passing-in-gnns`. The earlier `claude/bold-thompson-1dbz9u`
  branch is merged (PR #2).
- `gnn_mech/` exists: config, data (Peptides), encoders, layers (GINE + dense softmax `GlobalAttn`),
  models (graph-level `HybridGNN`, `GCNBaseline`), train, run, sweep, analysis, results, and
  `measure/` (resistance, Jacobian, entropy, node- and graph-level range). Phase 2 docs report 83+ CPU
  tests passing; not re-run at handoff time.
- `gnn_mech/reddit/` is a node-level track on the full Reddit graph (SAGE vs virtual-node model,
  distance and random splits, target selection). It has **never run on the real graph**: SNAP was
  blocked from the old build sandbox. Check whether it is reachable from your environment; otherwise
  data downloads happen on Colab.
- `report/main.tex` is the capstone draft. Its main text still centres on Peptides / rivers and must be
  restructured (roadmap correction 8) once S2+ results exist. `docs/presentation/` is the Beamer deck.
- Nothing in `docs/social_plan.md` is built yet.

## First task: S0 (data layer), then S1 (Performer + node-level model)

Follow `docs/social_plan.md` §S0 and §S1. In short:

1. `gnn_mech/social/sources.py`: download + parse twitch-gamers, the twitch musae graphs, higgs (all 4 layers),
   ego-Twitter, and Reddit (reuse `reddit/data.py`). Cache one full-graph PyG `Data` per source under `./data/`
   (gitignored). Start with **twitch-gamers**: real targets, one file, mid-size.
2. `social/subgraph.py`: partition into disjoint parts (METIS if `torch_geometric`'s METIS / `pymetis` is
   available, else label propagation), assign whole parts to train/val/test, and allow BFS balls inside a part.
3. `social/structure.py`: degree, hops, exact effective resistance per subgraph (pseudo-inverse, ≤ 10k nodes)
   and **excess resistance** R − (1/d_u + 1/d_v). Gate S0 needs the correlation of R with 1/d_u + 1/d_v per dataset.
4. `social/dataset.py`: `SubgraphDataset` of `Data` with node-level `y`, a supervision mask and `graph_id`,
   so the existing batching, `RunDir` and sweep code work.
5. S1: `PerformerAttn` in `layers.py` with the same `(x, batch)` signature as `GlobalAttn`, plus
   `ModelConfig.attn` and `ModelConfig.task_level`. **New config fields must default to the old behaviour so
   existing config hashes stay unchanged** (pinned in `tests/test_models.py`).

Write the shared interfaces (function signatures, config fields, row schemas) into `docs/social_plan.md`
**before** splitting work among subagents, as was done in Phases 1–2.

## Things to know

1. **Resistance collapses to degree on big social graphs** (von Luxburg et al. 2014). Always report excess
   resistance next to raw R and hops; the phase diagram's x-axis is excess resistance.
2. **Real social labels are mostly local.** Expect a low best α and short measured range in S2. Say so;
   the planted targets in S3 exist to give the range axis some spread.
3. **Leakage.** Splits must be by disjoint node sets (whole partitions). For Higgs, a target derived from the
   activity layers must not appear in the inputs or the graph used for message passing.
4. **Performer and the measures.** No explicit attention matrix exists: entropy comes from exact implicit rows
   for sampled queries. Implement in pure PyTorch so double backward works (fused SDPA has none; see
   `range.py`). Fix the random features in eval mode so Jacobians are deterministic.
5. **Old notes still valid** (`docs/handoff_phase2.md` § Things you must know): Jacobian cost and target-sharing
   sweeps, thread oversubscription (`OMP_NUM_THREADS=1` when several processes run tests), venv setup.
6. **Data licensing.** The S6 release ships generator code, node-index files and seeds, never raw Twitter /
   Twitch data.
7. **Compute** is a Colab T4 (about 15 GB of GPU memory). Add new Colab driver sections after §10 of
   `notebooks/colab_driver.ipynb` and leave the Peptides sections as they are.

## Open questions to settle with the user

1. Higgs real target: which activity definition, and how leakage is excluded.
2. Default subgraph sampler and size (proposal: partition for splits, BFS ball within parts, 2k nodes).
3. Whether to also run α=1 on full graphs (Performer is linear) as a check on subgraphing.

## How to work

- Commit and push only when the user asks. Branch for feature work unless the user says to use `main`.
- The commit message trailer is `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Parallel subagents: give each disjoint file ownership and the contract docs; they must not run
  state-changing git commands. You integrate, run the full test suite, then commit per workstream.
- Keep reports to the user short: what passed, what did not, and what they need to run on Colab.
