# Handoff: starting Phase 2

For the orchestrator agent picking up Phase 2. Read this, then
`docs/roadmap.md` (big picture), `docs/phase2_plan.md` (what to build) and
`docs/phase1_plan.md` (the package's interface contract).

## Where things stand

- Branch: `claude/bold-thompson-1dbz9u` on `djdhawal/message-passing-in-gnns`.
  Develop and push there. Do not open a pull request unless the user asks.
- Phase 1 is built: `gnn_mech/` package, 83 CPU tests passing. Pilot notebooks
  are preserved unchanged in `notebooks/pilot/`.
- **The Phase 1 GPU check (gate G0) has not been reported yet.** The user runs
  it on Colab via `notebooks/colab_driver.ipynb`. Phase 2 *code* can be built
  now; Phase 2 *GPU runs* wait for G0. Ask the user for the results table if
  you need it.
- The user approved the Phase 2 plan's scope choices: range measure in Phase 2,
  Reddit as a parallel track, Reddit target chosen from the data, depth
  ablation deferred.

## Package map (what exists)

| File | What it does |
|---|---|
| `gnn_mech/config.py` | Dataclass configs, YAML + `key=value` overrides, seed-independent hash |
| `gnn_mech/results.py` | `RunDir`: `<out>/<exp>/<hash>/seed<k>/` with config/history/metrics/measurements JSON, `checkpoint.pt`, `best.pt` |
| `gnn_mech/data.py` | Peptides-func/struct with RWSE (`processed_pe<k>` cache dirs), categorical codes kept as Long, `graph_id` on every graph, `structure_stats` (cached per-graph size/resistance/diameter), `synthetic-tiny` for tests |
| `gnn_mech/encoders.py` | Per-column embeddings (OGB vocabs) + RWSE projection → `h0` |
| `gnn_mech/layers.py` | `GlobalAttn` (per-head weight cache), `HybridLayer` (α blend of normalized GINE and attention branches; branch not built when its weight is 0) |
| `gnn_mech/models.py` | `HybridGNN`, `GCNBaseline`, `build_model`; `embed_inputs` / `node_embeddings_from_h0` split for Jacobians |
| `gnn_mech/train.py` | AdamW, warmup+cosine, grad clip, best-val checkpoint, resumable |
| `gnn_mech/run.py` | `python -m gnn_mech.run --config X --seed N [k=v]`; skips finished runs, resumes interrupted ones |
| `gnn_mech/measure/` | `resistance.py`, `jacobian.py` (target-sharing sweeps), `entropy.py` (per head), `__init__.run_all` with log-log fits and size-controlled OLS |
| `configs/` | `smoke`, `pilot_equivalence`, `peptides_func_hybrid` (α via override), `peptides_func_gcn` |

## Things you must know (learned the hard way)

1. **Parameter budget is tight.** α=0.5, hidden 96, 5 layers = 498,255 params
   (LRGB budget 500k). GINE layers have no per-layer edge projection for this
   reason. Any widening goes over.
2. **Jacobian cost.** One target node's full Jacobian costs D backward passes
   but gives sensitivity to every source. `measure_jacobians` therefore samples
   `targets_per_graph` targets × `pairs_per_graph` sources (67 pairs/s on CPU vs
   2 pairs/s for independent pairs). The range measure should reuse these full
   per-target vectors (see plan A1), not add new sweeps.
3. **Thread oversubscription.** When several agents run pytest or data
   processing at once on this 4-core box, torch/BLAS threads oversubscribe and
   things run 10–45× slower. Use `OMP_NUM_THREADS=1` for tests while subagents
   share the machine. `structure_stats` and RWSE processing already limit threads.
4. **Environment.** No system torch. Create a venv if missing:
   `python3 -m venv /home/user/venv && /home/user/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch && /home/user/venv/bin/pip install torch_geometric networkx scipy scikit-learn pyyaml pytest && /home/user/venv/bin/pip install -e ".[dev]"`.
   Peptides downloads work from this sandbox (Dropbox reachable); data goes in
   `./data/` (gitignored). The Reddit SNAP URL has not been checked yet.
5. **Peptides is near-tree.** Effective resistance and hop distance are highly
   collinear there, so "resistance beats hops" can't be shown cleanly on it.
   Report the collinearity; the synthetic suite (Phase 5) is where that gets tested.
6. **Pilot numbers to compare against.** Pilot LRGB notebook (RWSE, 4 layers):
   GCN 0.6263, GPS 0.6518 test AP, 1 seed. Pilot mechanistic notebook (broken
   setup): GINE 0.54, transformer 0.27, GPS 0.24; Jacobian slopes −1.76 / −0.48 /
   −0.81. Published: Tönshoff GCN ≈ 0.686, GPS 0.6535.
7. **Reddit pilot facts.** Graph from SNAP `soc-redditHyperlinks-body.tsv`
   (35,776 nodes, 286,561 directed edges, 86 post properties averaged per source
   subreddit; 7,913 nodes have no outgoing posts, so zero features). Target was
   feature 0 = character count, range 49–40,276, with near-duplicate columns in
   the inputs. Distance split from the highest out-degree anchor: train 0–1 hops
   (1,455), val 2 (17,876), test 3+ (7,939). Over 100 seeds the virtual-node model
   cut MSE 12% / 38% / 51% / 62% / 81% at 1 / 2 / 3 / 4 / 5+ hops. In the pilot
   only the VMN model had an input projection layer (unfair baseline).

## How to work (what worked in Phase 1)

- Spawn Opus subagents in parallel with **disjoint file ownership**; the plan's
  Execution section lists the three workstreams and their files. Tell each
  subagent to read the contract docs, code against them, never run state-changing
  git commands, and report contract deviations instead of silently changing
  signatures.
- Write any shared contract (new function signatures, config fields, row
  schemas) yourself *before* spawning, and add it to `docs/phase2_plan.md`.
- You integrate: run the whole suite, run a smoke sweep end to end, review each
  diff, then commit per workstream with clear messages and push.
- Commit message trailer used on this branch:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
  (plus a `Claude-Session:` line with your own session URL if your environment gives one).
- The user follows along in a Claude app; keep reports short and say plainly
  what passed, what didn't, and what they need to run on Colab.
