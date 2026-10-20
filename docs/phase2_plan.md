# Phase 2 plan: re-baseline Peptides-func, add the range measure, fix Reddit

## Context

The thesis draft's §6 rests on pilot runs we now know are flawed. The models trained for only 30 epochs on one seed, and the hybrid model (GPS) reached 0.24 AP. Categorical atom codes went through `nn.Linear`, no model had positional encoding, and entropy was computed on head-averaged attention. Phase 1 fixed all of that in the `gnn_mech` package.

Phase 2 produces the numbers the thesis will actually report:
1. Properly trained local, hybrid and global models plus a GCN baseline on Peptides-func, 3 seeds each, with every mechanistic measurement.
2. The Bamberger et al. (ICML 2025) range measure, built now as agreed, so each model also reports how far its predictions reach.
3. A corrected Reddit experiment, as a parallel track: the target is chosen from the data, the baseline is fair, and a random-split control is added.

Decisions already made: range measure in Phase 2; Reddit as a parallel track; Reddit target chosen from the data; depth ablation deferred.

**Prerequisite (gate G0):** the Phase 1 step-7 GPU check passes. The 4-layer α=0.5 model must reach ≥0.652 test AP, and α=0 and α=1 must train. Nothing in Track A runs before this.

---

## Track A: Peptides-func re-baseline (GPU on Colab)

### A1. Range measure (`gnn_mech/measure/range.py`)

1. Fetch arXiv 2506.05971 and transcribe the exact definitions into the module docstring. Two variants matter:
   - Node-level: a sensitivity-weighted mean distance from each target to its sources.
   - Graph-level: Hessian or mixed second derivative of the graph output. Peptides-func is a graph-level task, so this is the paper-faithful one.

   Implement whichever the paper uses for graph tasks. Also implement the node-level variant, which is cheap.
2. **Node-level range from existing sweeps.** `jacobian_norms` (`gnn_mech/measure/jacobian.py`) already computes each target's full Jacobian over all sources, then keeps only the 32 sampled ones. Refactor `_target_jacobians_batched` so `measure_jacobians` also gets the full per-target norm vector over the LCC, and compute per target:
   - `range_hops = Σ_u J_uv·hops(u,v) / Σ_u J_uv`
   - `range_res`, the same with effective resistance as the distance.

   These are new row-level fields, one record per target, with no extra backward passes.
3. **Graph-level range** (if the paper defines it via Hessians): sample sources u per graph and use Hessian-vector products of the pooled output with respect to `h0`. One HVP batch per source gives that source's mixed partials against every v. Reuse the batched-autograd pattern in `jacobian.py`. Add config fields to `MeasureConfig` in `gnn_mech/config.py`: `n_graphs_range`, `sources_per_graph`.
4. Add range results to `run_all` and its summary (`gnn_mech/measure/__init__.py`): per-model mean, median and distribution of range, and correlation with graph size.
5. **Tests** (`tests/test_measure.py`, using the existing stand-in `PropModel` pattern):
   - A self-only model has range 0.
   - A model that reads exactly the hop-d neighbour has range d.
   - The K-hop linear propagation model matches its closed form.
   - Range from the full vector is invariant to which sources were sampled.
   - The real HybridGNN at α=0 has range ≤ L.

### A2. Save per-graph predictions (`gnn_mech/run.py`)

After reloading `best.pt`, write `preds_<split>.npz` (graph_id, y, logits) for val and test. This is needed for AP stratified by resistance and size, and for bootstrap confidence intervals, without retraining. Update `--force` cleanup to delete these files too. Add a test in `tests/test_train.py` on the smoke config.

### A3. Sweep runner (`gnn_mech/sweep.py`, `configs/sweeps/phase2.yaml`)

`python -m gnn_mech.sweep configs/sweeps/phase2.yaml [out_dir=...]` reads a list of `{config, overrides, seeds}` entries and runs them in order through `gnn_mech.run.run()`. It skips runs that are already complete (`RunDir.is_complete` in `gnn_mech/results.py`) and resumes interrupted ones. A Colab disconnect then costs a re-run of one cell. Add a dry-run flag that prints the run list and estimated hours.

Phase 2 sweep (all `pe_dim=20`, seeds 0, 1, 2):

| Run | Config | Overrides | Est. GPU time / seed |
|---|---|---|---|
| Local (GINE) | `peptides_func_hybrid.yaml` | `model.alpha=0.0` | ~10 min |
| Hybrid (GPS-like) | `peptides_func_hybrid.yaml` | `model.alpha=0.5` | ~25 min |
| Global (transformer) | `peptides_func_hybrid.yaml` | `model.alpha=1.0` | ~25 min |
| GCN baseline | `peptides_func_gcn.yaml` | none | ~15 min |

Total is about 4 hours of training plus about 1 hour of measurement (12 runs × ~5 min). That fits in two or three Colab sessions.

### A4. GCN reference alignment (before launching the sweep)

Fetch Tönshoff et al.'s Peptides-func GCN config from their public repo (`toenshoff/LRGB`). Diff it against `configs/peptides_func_gcn.yaml`: layers, hidden size, dropout, epochs, lr, head depth, positional encoding. Align ours where it differs, within the parameter budget. Record their reported number, about 0.686 AP, in the config header.

### A5. Analysis module (`gnn_mech/analysis.py` + `notebooks/phase2_analysis.ipynb`)

Add `pandas` and `matplotlib` to `pyproject.toml`. Functions, all reading run folders written by `RunDir`:

- `collect_runs(out_dir)`: one DataFrame of config, metrics and paths, grouped by exp, config hash and α.
- `performance_table`: test AP mean ± std and 95% CI over seeds, next to published numbers (Tönshoff GCN, GPS 0.6535).
- `stratified_ap(preds, structure_stats)`: AP within resistance tertiles and within size tertiles, per model. Uses `structure_stats` from `gnn_mech/data.py`, which is already cached. This replaces pilot §6 with confidence intervals.
- `jacobian_analysis`:
  - Per model, pool the seeds and report the log-log slope against resistance and against hops.
  - Size-controlled OLS (reuse `_ols` and `_loglog_fit` from `gnn_mech/measure/__init__.py`).
  - Confidence intervals from a **graph-clustered bootstrap**: resample graphs, not pairs, because pairs within a graph are correlated.
  - **Slope difference vs. the transformer (α=1)** with a bootstrap CI. The transformer's slope is the confound baseline from the thesis review.
  - Report the collinearity of resistance and hops on Peptides, since these are near-tree molecules.
- `entropy_analysis`: per model, layer and head, the correlation of normalized entropy with graph size, with a bootstrap CI, plus per-seed variation.
- `range_analysis`: range per model, plus the task-range reading. If every well-trained model has short range, that supports Bamberger's finding that Peptides tasks are short-range.
- Figures saved as PDF and PNG:
  - AP bars;
  - stratified AP;
  - Jacobian against resistance on log-log axes, one panel per model;
  - normalized entropy against graph size;
  - range distributions.

  Style follows `notebooks/pilot/Data_Analysis_Thesis.ipynb`.

### A6. Gates

- **G1, training is sound:**
  - α=0.5 mean test AP ≥ 0.63 (published GPS 0.6535);
  - α=1 ≥ 0.60;
  - GCN ≥ 0.66, or the remaining gap is documented after A4 alignment.
- **If G1 fails,** try fixes in this order:
  1. Training curves (underfitting vs. overfitting).
  2. lr 5e-4 / 1e-3 and warmup 5 / 10.
  3. Dropout 0.1 / 0.2.
  4. BatchNorm vs. LayerNorm in `HybridLayer` (`gnn_mech/layers.py`).
  5. Epochs 300.

  Change one thing at a time on seed 0 only, then rerun the sweep.
- **G2, measurements are interpretable:** the Jacobian slope difference vs. the transformer has a bootstrap CI excluding 0, or we report clearly that it doesn't. Either outcome goes in the thesis, but it decides the §6 wording.

---

## Track B: Reddit fixes (CPU-friendly, runs in parallel)

New subpackage `gnn_mech/reddit/`. It is separate from the graph-level pipeline because this is node regression on one 35,776-node graph. It reuses `RunDir` by duck-typing a small `RedditConfig` with `to_dict()`, `hash()`, `seed`, `out_dir` and `exp_name`, and `set_seed` from `gnn_mech/train.py`.

### B1. Data (`reddit/data.py`)

Port graph construction from `notebooks/pilot/Reddit_Network_Data_EDA.ipynb`: SNAP `soc-redditHyperlinks-body.tsv`, 86 post properties averaged per source subreddit. Check that the SNAP URL is reachable from Colab and from here; if not, read the cached `.pt` from Drive. Ship a `FEATURE_NAMES` list (86 names from SNAP's documentation) and cache the graph. Keep `is_source` masking and BFS from the highest out-degree anchor, ported from `notebooks/pilot/GNN.ipynb`.

### B2. Target selection (`reddit/target_select.py`)

For each candidate target column (sentiment, LIWC categories):
1. Drop input columns with |r| > 0.9 to the target. This removes near-duplicates like "characters without spaces".
2. On a **random** node split, fit ridge regression (plus a small-MLP check) on:
   - (a) the node's own features;
   - (b) own + mean of 1-hop neighbours;
   - (c) own + 1..k-hop propagated means, for k up to 4 (SGC-style).
3. Score `neighbour_gain = R²(c) − R²(a)` and `multi_hop_gain = R²(k≥2) − R²(k=1)`, on log-transformed targets where the data is skewed.

Output a ranked table to `runs/reddit/target_selection.json`. Choose a target with clear multi-hop gain and moderate own-feature R². If none shows multi-hop gain, that becomes the finding, and Experiment 1 is reframed as generalization to distant nodes.

### B3. Fair models (`reddit/models.py`)

- **SAGE baseline:** gets the same `input_proj` layer the virtual-master-node (VMN) model has. In the pilot only the VMN model had it, so architecture changed along with the shortcut.
- **VMN model:** ported from `notebooks/pilot/GNN.ipynb` (`add_virtual_master`, `HGNN_VMN`).

Both use the same depth, hidden size, dropout and optimizer.

### B4. Experiments (`reddit/run.py`)

- **Main:** distance-based split, as in the pilot (train 0–1 hops, val 2, test 3+), on the selected target, 30 seeds per model.
- **Control:** a **random** split, with test MSE then broken down by BFS distance band.
  - If VMN still gains more at far bands, the gain tracks distance in the graph, not extrapolation from the training region.
  - If the gain vanishes, it was distribution shift.
- **Metrics per band:** MSE relative to the constant baseline for that band, which fixes the raw-unit comparison problem; n per band; bootstrap CI over seeds.
- Write results with `RunDir` and produce figures in the pilot style.

### B5. Tests (`tests/test_reddit.py`)

On a tiny synthetic node-regression graph:
- target-selection math recovers a planted neighbour-dependent target;
- the VMN adds exactly one node and 2N edges;
- band metrics are correct;
- random-split and distance-split masks are disjoint and complete over source nodes.

---

## Track C: Docs and thesis

- **`docs/roadmap.md`:** all phases (0–6) with gates, each literature-review item mapped to its phase. Done first.
- **`docs/phase2_plan.md`:** this plan, committed to the repo.
- **After results, `docs/results/phase2.md`:** tables and figures, plus draft replacement text for thesis §4 (Reddit data description), §5 (methods: embeddings, RWSE, α-blend, range measure) and §6. The §6 rewrite covers the entropy claim narrowed to what the data show, the AP of every measured model, Jacobian slopes with the confound baseline, and range.

---

## Execution

Three Opus subagents in parallel, each owning disjoint files, as in Phase 1:
1. **Range measure:** A1. Files: `measure/range.py`, `measure/jacobian.py` refactor, `measure/__init__.py`, `config.py` MeasureConfig fields, `tests/test_measure.py`.
2. **Run tooling:** A2, A3, A5. Files: `run.py`, `sweep.py`, `analysis.py`, `configs/sweeps/phase2.yaml`, `notebooks/phase2_analysis.ipynb`, `tests/test_train.py`, `tests/test_analysis.py`, and a Phase 2 section in `notebooks/colab_driver.ipynb`.
3. **Reddit:** B1–B5, all under `gnn_mech/reddit/` and `tests/test_reddit.py`.

I do A4 (GCN config alignment), the roadmap and plan docs, integration, review of each subagent's diff, and commit/push to `claude/bold-thompson-1dbz9u`.

Then you run on Colab:
1. G0, if not already done.
2. `python -m gnn_mech.sweep configs/sweeps/phase2.yaml`.
3. Reddit target selection plus the experiments (CPU is fine; Colab or here).

I then run the analysis, check G1 and G2, and write `docs/results/phase2.md` with the thesis text drafts.

## Contract additions (Phase 2)

Written into the code; `docs/phase1_plan.md` signatures are unchanged.

```python
# config.py
ModelConfig.act: str = "relu"          # GCN baseline nonlinearity: "relu" | "gelu"
ModelConfig.norm: str = "batch"        # GCN per-layer norm: "batch" | "none"
ModelConfig.head_layers: int = 2       # Linear layers in the graph MLP head (both models)
MeasureConfig.n_graphs_range: int = 30 # graph-level (Hessian) range: graphs ...
MeasureConfig.sources_per_graph: int = 4  # ... sampled source nodes per graph ...
MeasureConfig.hessian_channels: int = 16  # ... and sampled source-side h0 channels (0 = all)
# Fields added after Phase 1 enter Config.hash() only when they differ from their
# default, so Phase 1 run folders (G0) keep their hash.

# models.py: both models
def forward_from_h0(self, h0, batch) -> Tensor   # graph logits from h0; forward() uses it

# measure/jacobian.py
def target_influence(model, data, targets, device, method="auto") -> dict[int, tuple[ndarray, ndarray]]
    # {v: (fro [N], l1 [N])} per-source sensitivity over all nodes from one sweep per target
def jacobian_sweeps(model, dataset, cfg, device, method="auto", min_lcc=4) -> tuple[list[dict], list[dict]]
    # (pair rows as measure_jacobians, per-target records with LCC vectors fro, l1, hops, resistance)

# measure/range.py
def node_range_rows(targets) -> list[dict]
    # {"graph_id", "v", "n_nodes", "lcc_size", "range_hops", "range_res",
    #  "range_hops_fro", "range_res_fro", "total_influence"}
def hessian_influence(model, data, sources, device, method="auto") -> dict[int, ndarray]
def measure_graph_range(model, dataset, cfg, device) -> list[dict]
    # {"graph_id", "u", "n_nodes", "lcc_size", "range_hops", "range_res", "total_hessian"};
    # range is None when the Hessian row is all zero
def summarize_range(rows, keys) -> dict

# measure/__init__.py
run_all(...) -> {"jacobian", "entropy", "range_node", "range_graph", "summary"}

# train.py
def predict(model, dataset, batch_size, device) -> {"graph_id", "y", "logits"}   # numpy, dataset order

# run.py: after best.pt reload, writes preds_val.npz / preds_test.npz (graph_id, y, logits)
```

## Status (Phase 2 code, built 2026-10-11)

| Item | State | Where |
|---|---|---|
| A1 range measure | Done; node level from existing sweeps (no extra backward passes), graph level via batched HVPs | `gnn_mech/measure/range.py`, `jacobian.py` |
| A2 predictions | Done | `gnn_mech/run.py`, `train.predict` |
| A3 sweep runner | Done, with `--dry-run` | `gnn_mech/sweep.py`, `configs/sweeps/{phase2,smoke}.yaml` |
| A4 GCN alignment | Done (see config header for the remaining RWSE-encoder difference) | `configs/peptides_func_gcn.yaml` |
| A5 analysis | Done; notebook executed on CPU smoke runs | `gnn_mech/analysis.py`, `notebooks/phase2_analysis.ipynb` |
| A6 gates G1/G2 | **Waiting for the GPU sweep** (and G0) | Colab driver section 9 |
| B1-B5 Reddit | Code and tests done on a synthetic graph; **not yet run on the real graph** | `gnn_mech/reddit/`, Colab driver section 10 |
| C results doc | Waiting for results | `docs/results/phase2.md` |

Decisions taken where the plan left a choice:
- **Range definition.** arXiv was blocked from the build sandbox, so the definitions were
  transcribed from the authors' code (`BenGutteridge/range-measure`), recorded in the
  `range.py` docstring. Influence is the entrywise **L1** norm of the Jacobian/Hessian
  block (not Frobenius); `range_hops` / `range_res` follow that, and `*_fro` variants use
  the Frobenius norm of the Jacobian pair rows, as written in A1.2. Sums run over the LCC
  and include the target itself (distance 0), as in the paper, so a self-only model has range 0.
- **Which variant is "paper-faithful" for Peptides.** The paper's LRGB (Peptides) numbers use
  the node-level Jacobian of the last pre-pooling layer; Hessians are used for its synthetic
  graph-level tasks. Both are implemented. The Hessian is zero almost everywhere for ReLU-only
  models (alpha = 0); its range is then reported as undefined. The Tönshoff-aligned GCN uses
  GELU, so its Hessian is nonzero, but through the mean-pool + nonlinear head it couples all
  node pairs and can exceed the receptive field (about 20 hops for an untrained GCN on a
  150-node path), so use the node-level range to compare models.
- **Hessian cost.** On CPU a 150-node Peptides-size graph takes about 50 s (attention models)
  for the default 4 sources x 16 sampled channels x 10 outputs = 640 double-backward rows
  (batched; the per-row loop is about 2x slower). Hence `hessian_channels = 16` (the paper
  also samples channels) and `n_graphs_range = 30`. GPU time is not measured yet; if the
  measurement step runs long on Colab, lower `measure.n_graphs_range`, or set it to 0 to skip.
  The node-level range costs nothing extra (about 1-2 s per graph on CPU with the Jacobian sweeps).
- **Hessian through attention** runs with the math SDPA backend: the fused attention kernels
  have no double backward.
- **GCN alignment** also needed new model options (GELU, no norm, 3-layer head) rather than
  only config values; added as `ModelConfig` fields whose defaults keep the Phase 1 model.
- **Sweep failure policy:** a failing run is logged and the sweep continues (exit status 1);
  `KeyboardInterrupt` / a Colab disconnect still stops it, and a rerun resumes.
- **Reddit inputs** drop the same |r| > 0.9 near-duplicates of the target that target
  selection drops; skewed non-negative targets are modelled as log1p; band metrics use the
  modelled units (the ratio to the constant predictor is unit-free). Propagated features for
  target selection average only over neighbours that have features (non-source nodes have none).
- **Reddit random split** has the same split sizes as the distance split, drawn from the same
  pool (reachable source nodes), reseeded per seed.

What remains (needs Colab GPU or the real data):
1. G0, then `python -m gnn_mech.sweep configs/sweeps/phase2.yaml` (section 9 of the Colab driver).
2. Range sanity on real Peptides: alpha = 0 mean node-level range <= 5 hops (guaranteed by the
   receptive field; tested on synthetic graphs), alpha = 1 reported.
3. Reddit on the real graph: check the SNAP download from Colab (blocked here), run target
   selection, and confirm one seed of the old target's distance split reproduces the pilot's
   per-band ordering before switching targets (`--target 0 --seeds 0:1 --splits distance`).
4. Analysis, G1/G2, and `docs/results/phase2.md` with the thesis text drafts.

## Verification

- `pytest -q`: all existing 83 tests plus the new range, prediction-saving, sweep, analysis and Reddit tests pass on CPU.
- Smoke: `python -m gnn_mech.sweep` on a 2-entry smoke sweep produces metrics, measurements with range fields, and `preds_test.npz`. Re-running it skips both runs, and killing it mid-run then re-running resumes.
- Analysis on the smoke runs produces every table and figure without errors, with the bootstrap seeded and reproducible.
- Range sanity on real Peptides: the α=0 model's mean range is ≤ 5 hops (5 layers). The α=1 range is reported, not asserted.
- Reddit: target selection runs end to end on the real graph; one seed of the main experiment matches the pilot's per-band ordering before the switch to the new target.
- After the GPU sweep: G1 and G2 checked and reported, with numbers, in `docs/results/phase2.md`.
