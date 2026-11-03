# Social-graph plan (phases S0–S6)

Decided 2026-11-02. Main experiments on large SNAP social graphs and on subgraphs sampled
from them. Peptides and rivers are appendix background. Big picture: `docs/roadmap.md`.
Package interface from Phases 1–2: `docs/phase1_plan.md`, `docs/phase2_plan.md`.

Order of work (as agreed): α sweeps on real targets → plant targets → range → phase
diagram → GFM benchmarking → benchmark release.

## Datasets

| Key | Source (SNAP) | Graph | Node inputs | Real targets |
|---|---|---|---|---|
| `reddit` | soc-redditHyperlinks (body; title optional) | 35,776 subreddits / 286,561 directed links (body) | 86 post properties averaged per source subreddit (existing `gnn_mech/reddit/data.py`) | Chosen by existing `reddit/target_select.py` (not feature 0) |
| `twitch-gamers` | twitch_gamers | 168,114 users / 6.8M mutual follows | Account columns not used as the target (views, lifetime, created/updated dates, language, flags) | `mature` (binary), `views` (log1p regression), `life_time`, `affiliate`, `dead_account`, `language` |
| `twitch-<lang>` | twitch-social-networks (musae) | 6 graphs, 1.9k–9.5k nodes | Shared game/streaming features (same across languages) | `mature` (binary) |
| `higgs-<layer>` | higgs-twitter | 456k users; layers: follow (14.9M), retweet (328k), mention (151k), reply (33k) | None: structural inputs only (degree, RWSE) + planted channels | Derived from activity timestamps, e.g. "active (RT/MT/RE) after the 4 July announcement", computed so the predicted activity never appears in the inputs or the graph |
| `ego-twitter` | ego-Twitter | 81k nodes / 1.77M edges, 973 ego nets | Binary hashtag / mention profile features (anonymised) | Circle membership (optional) |

Notes:
- Higgs gives the **same nodes under different structure** (follow layer: 90% effective diameter 3.7;
  reply layer: 10). This is the cleanest structural intervention available on real social data.
- twitter7 has been withdrawn; ego-Gplus is a fallback with the same format as ego-Twitter.
- Undirected view for message passing and distances, as in the Reddit track; keep direction
  as an optional edge feature.
- Licensing: SNAP data is research-use; release only **generator code + node-index files +
  seeds**, never raw Twitter/Twitch data (S6).

## S0. Data layer and subgraphs (`gnn_mech/social/`)

```
social/
  sources.py     # download + parse each SNAP source -> one cached full-graph PyG Data
  subgraph.py    # samplers: partition (METIS / label-propagation), BFS ball, random-walk induced
  structure.py   # per-node and per-pair structure: degree, hops, effective resistance, excess resistance
  dataset.py     # SubgraphDataset: list of Data, node-level y + masks, graph_id, split
```

- **Subgraphs** are the unit of training and evaluation. Default size N ≈ 2,000 nodes (sweep
  500 / 2k / 10k as a dilution axis). Train / val / test subgraphs come from **disjoint node
  sets** (partition first, then assign parts to splits) so neither labels nor features leak.
  This reuses the existing graph-level batching (`to_dense_batch`, `graph_id`) with a node-level loss.
- **Effective resistance**: exact via the Laplacian pseudo-inverse up to about 10k nodes per subgraph;
  random-projection approximation (Spielman–Srivastava) for full-graph checks.
- **Degree correction (required).** On large, well-connected graphs R_uv → 1/d_u + 1/d_v
  (von Luxburg et al. 2014), so raw resistance mostly measures degree. Report and bin on
  **excess resistance** R̃_uv = R_uv − (1/d_u + 1/d_v), alongside raw R and hops.
- Per-subgraph summaries: N, mean degree, effective diameter, spectral gap, mean R̃, Kirchhoff index.
- **Gate S0:** a table of these summaries for every dataset, and the Spearman correlation of R with
  1/d_u + 1/d_v reported, which tells us how much the degree correction matters on each graph.

## S1. Performer branch and node-level model

- `layers.py`: new `PerformerAttn(hidden, heads, n_features=m, redraw=…)` with FAVOR+ positive
  random features, the same call signature as `GlobalAttn(x, batch)`. Attention runs within each subgraph
  via scatter sums over `batch` (per graph: φ(K)ᵀV and φ(K)ᵀ1), at O(N·m·d) cost. Pure PyTorch,
  so double backward (Hessian range) works.
- `ModelConfig.attn: "softmax" | "performer"` (default `"softmax"` keeps old config hashes),
  `ModelConfig.task_level: "graph" | "node"` (node = no pooling, per-node head).
- Local branch: GINE where edge features exist, otherwise GIN / SAGE-style (`ModelConfig.local`).
- **Entropy with linear attention:** Performer has no explicit attention matrix. For a sampled
  query i the implicit row A_i· = φ(q_i)·φ(K)ᵀ / normaliser is exact and costs O(N·m), so the
  per-head entropy is computed on sampled queries (`measure/entropy.py` gains a performer path).
- Parameter budget: keep α-variants within ±10% of each other (report counts; unused branches
  are still not built, as before).
- **Gate S1:**
  (a) on small graphs, Performer output vs softmax output has cosine similarity ≥ 0.95 at the chosen m;
  (b) on one real target (`twitch-gamers` `mature`), α=0 and α=1 both beat an MLP on node features
  and the constant predictor.

## S2. α sweep on real targets

- α ∈ {0, 0.25, 0.5, 0.75, 1} × 3 seeds × {reddit, twitch-gamers, twitch-<lang>, higgs-follow}, using the
  existing `sweep.py` (new `configs/sweeps/social_s2.yaml`).
- Measurements on test subgraphs: Jacobian sensitivity vs hops / R / R̃, sampled-query entropy,
  node-level range (`range_hops`, `range_res`).
- Baselines: MLP (no graph), tuned GCN / SAGE endpoint (Tönshoff-style), optional gated attention.
- **Expectation stated up front:** real social labels are mostly homophilous and local, so the best α
  should be low and the measured range short. That is a finding, and it is the reason for S3.
- **Gate S2:** performance-vs-α curves with bootstrap CIs over subgraphs, per dataset.

## S3. Planted semi-synthetic targets

Real graph and real features stay as they are. Add `p` planted input channels z_u ~ N(0, 1), and define the
target exactly from them, so the **ground-truth range is known**. Families (`social/plant.py`):

| Family | Target y_v | Stresses |
|---|---|---|
| `shell-mean(k)` | mean of z_u over nodes u at exactly hop k from v | Pure reach; compressible (sanity) |
| `res-shell(r)` | mean of z_u over u in the excess-resistance band r around v | Bottleneck, not just distance |
| `retrieve(k)` | v carries a query key; y_v = value of the unique node at hop k whose key matches | Over-squashing of *associations*; the main test |
| `mix(λ, k)` | (1−λ)·real-local signal + λ·`retrieve(k)` | Continuous task range |
| `distract(k, D)` | `retrieve(k)` with D extra key-bearing distractor nodes per subgraph | Attention dilution |

- k ∈ {1, 2, 3, 4, 6, 8} (bounded by what exists in each subgraph; record the count of eligible
  targets per k and skip a target whose shell is empty). Model depth L ≥ k_max for MPNN, so failure is not
  simply a receptive-field limit.
- Validation of the measure: for every family, the **analytic range of the target function** (from its
  exact Jacobian w.r.t. z) is computed and compared with the measured range of the trained model.
- Targets are generated with a seeded, saved specification (`plant_spec.json`) so S5 and S6 reuse them exactly.
- **Gate S3:** measured range rises with planted k (Spearman > 0.8 for the best model per k); α sweep repeated on
  `retrieve(k)` and `distract(k, D)`.

## S4. Phase diagram

- Unit: a test node v and its planted source set. x-axis: excess resistance between v and its sources
  (binned by quantile within dataset); y-axis: task range (planted k, and measured range of the best model).
  Cell colour: best α chosen on validation; overlay: mean normalised entropy of the α=1 model.
- Separate panels per dataset and for Higgs follow vs reply (same nodes, different structure).
- Uncertainty: bootstrap over test subgraphs (the independent unit), seeds nested.
- **Gate S4:** regions where the best α differs, with CIs that exclude a single flat α; stable across seeds.

## S5. GFM benchmark

- Models: GraphPFN (`yandex-research/graphpfn`, in-context, attention-based neighbourhood aggregation),
  GraphAny if it installs cleanly; the α-sweep models as reference.
- Run on the exact S3 tasks and splits (in-context or light finetune, per each model's protocol).
- Measure each GFM's **range** with the same Jacobian-based node range (input = node features incl.
  planted channels), testing Gutteridge et al.'s under-reach claim on attention-based GFMs and on social graphs.
- Risk: GraphPFN's context attention is quadratic, so subgraph size may have to drop (2k → 500); report it.
- **Gate S5:** table of accuracy and measured range vs required k for each GFM, next to the best-α model.

## S6. Benchmark release and writeup

- `social/benchmark.py`: `build(dataset, family, k, seed) -> SubgraphDataset`, deterministic from the
  public SNAP files + released index/seed files. Datasheet: sources, licences, splits, task families,
  baseline numbers (S2–S5).
- Thesis: main text = S0–S5; appendix = Peptides (pilot, G0 calibration) and the river / tree experiments.

## Reused vs new

| Reused unchanged | Extended | New |
|---|---|---|
| `results.py`, `sweep.py`, `train.py` loop, `measure/jacobian.py`, `measure/range.py` (node level), `analysis.py` bootstrap | `layers.py` (Performer), `models.py` (node-level head), `config.py` (`attn`, `task_level`, dataset keys), `measure/entropy.py` (sampled-query path), `measure/resistance.py` (excess R, approx R) | `gnn_mech/social/` (sources, subgraph, structure, dataset, plant, benchmark) |

Peptides-only pieces (`PeptidesDataset`, Hessian graph range, G0/G1 gates) stay in place for the appendix
and are not developed further.

## Compute (Colab T4, rough, to be re-estimated after S1)

- S2: 5 α × 3 seeds × 4 datasets = 60 runs on 2k-node subgraphs.
- S3–S4: 5 α × 3 seeds × (5 families × ~6 k) on 2 datasets for the main grid, plus Higgs follow/reply.
- Prioritise: twitch-gamers and reddit for the full grid; others for the main contrasts only.

## Open questions

1. Higgs real target: which activity definition, and how to guarantee no leakage from the activity layers.
2. Subgraph sampler default: partition-based (realistic boundaries) vs BFS ball (controls radius). Proposal:
   partition for splits, BFS-ball within parts when a target needs a full k-shell.
3. Whether α=1 should also be run on the full graph (Performer is linear) as a check that subgraphing does
   not cut off the global signal.

## Build contract: S0–S2 (written 2026-11-02, before the parallel build)

Goal of this build: **launch S2 α-sweep runs on Colab** (real targets on social subgraphs).
Planted targets (S3) and the GFM benchmark (S5) are out of scope for this build. Consolidate code
once runs have produced data.

### Ownership (disjoint files)

| Workstream | Owns | Must not edit |
|---|---|---|
| **A. Pipeline modules** | `gnn_mech/config.py`, `data.py` (only `DataInfo` + the dispatch in `load_dataset`), `encoders.py`, `layers.py`, `models.py`, `train.py`, `run.py`, `measure/*`, `tests/test_models.py`, `tests/test_train.py`, `tests/test_measure.py`, new `tests/test_performer.py`, `tests/test_node_level.py` | `gnn_mech/social/`, notebooks |
| **B. Social data + subgraphs** | new `gnn_mech/social/` (`__init__.py`, `sources.py`, `subgraph.py`, `structure.py`, `dataset.py`), `configs/social/*.yaml`, `configs/sweeps/social_*.yaml`, new `tests/test_social.py` | everything in A's column |
| **C. Notebooks** | new `notebooks/social/*.ipynb` | all `.py` and configs |

### Dataset names (`data.name`)

`twitch-gamers`, `twitch-de|en|es|fr|pt|ru` (musae), `reddit-body`, `higgs-follow`,
`higgs-retweet`, `higgs-reply`, `higgs-mention`, and `social-tiny` (synthetic, CPU tests,
no download). `ego-twitter` is reserved for later and not required in this build.
`data.py::load_dataset` dispatches every name in `gnn_mech.social.SOCIAL_DATASETS` to
`gnn_mech.social.dataset.load_social(cfg)`; it imports lazily.

### `DataConfig` fields added (defaults leave old config hashes unchanged; listed in `_ADDED_FIELDS`)

```python
target: str = ""            # dataset-specific target name; "" = dataset default
subgraph_size: int = 2000   # target nodes per subgraph (core, before halo)
sampler: str = "partition"  # "partition" (METIS via pymetis, fallback: BFS-grown parts) | "bfs"
halo_hops: int = 0          # context-only hops added around each core (y_mask False); 0 = strict
val_frac: float = 0.15      # fraction of subgraphs (whole parts) per split; train gets the rest
test_frac: float = 0.15
split_seed: int = 0         # partition/split seed, fixed across model seeds
max_subgraphs: int = 0      # 0 = all; else cap per split (quick runs), applied after splitting
```

### `ModelConfig` fields added (same rule)

```python
attn: str = "softmax"         # "softmax" (GlobalAttn) | "performer" (PerformerAttn)
performer_features: int = 64  # random features m per head
```
`arch` gains `"mlp"` (no message passing, no attention: per-node MLP on h0, the no-graph baseline).

### `DataInfo` (extended, positional order of the old fields unchanged)

```python
@dataclass
class DataInfo:
    node_vocab: list[int]        # [] for social (no categorical node columns)
    edge_vocab: list[int]        # social: [1] (a single constant edge code, so GINE works unchanged)
    out_dim: int                 # 1 for binary/regression, n_classes for multiclass
    pe_dim: int
    task: str                    # "multilabel" | "regression" | "binary" | "multiclass"
    task_level: str = "graph"    # "graph" (pool + head) | "node" (per-node head, no pool)
    feat_dim: int = 0            # width of float node features batch.x (social); 0 = categorical x
```

### Social subgraph `Data` objects (what `load_social` returns per split, as `torch_geometric` Datasets of `Data`)

| Field | Shape / dtype | Meaning |
|---|---|---|
| `x` | `[N, feat_dim]` float32 | node features, standardized with **train-split** statistics |
| `pe` | `[N, pe_dim]` float32 | RWSE on the subgraph (when `pe_dim > 0`) |
| `edge_index` | `[2, E]` long | undirected, no self loops, local indices |
| `edge_attr` | `[E, 1]` long zeros | constant edge code |
| `y` | `[N]` float32 (binary, regression) or long (multiclass) | target; regression targets are transformed (log1p if skewed) and standardized with train stats |
| `y_mask` | `[N]` bool | nodes that are supervised/evaluated (core nodes with a valid label) |
| `node_id` | `[N]` long | index of the node in the full cached graph |
| `graph_id` | `[1]` long | subgraph id, unique across splits |

`load_social(cfg: DataConfig) -> (train_ds, val_ds, test_ds, DataInfo)` with
`task_level="node"`, `node_vocab=[]`, `edge_vocab=[1]`. Splits contain **disjoint core node sets**.
Raw downloads go to `<data.root>/social/<source>/raw/`, the processed full graph to
`<data.root>/social/<source>/graph.pt`, and the split subgraphs to
`<data.root>/social/<source>/processed/<hash of the data fields>.pt`. Each step is skipped when its
file exists (idempotent; Colab-safe: write to a temp file and `os.replace`).
`gnn_mech.social.sources.fetch(name, root)` downloads and parses only (used by the fetch notebook).

### Training / metrics (A)

- Node level: loss and metrics only over `y_mask` nodes, pooled across the batch.
  binary → BCE-with-logits, metric `auroc` (also report `ap`); multiclass → CE, metric `acc`;
  regression → MSE, metric `r2` (also report `mae`). Higher is better for `ap`/`auroc`/`acc`/`r2`; lower for `mae`.
  Graph-level behaviour (Peptides) is unchanged.
- `train.batch_size` counts **subgraphs** per batch.
- `preds_<split>.npz` for node-level runs: `graph_id`, `node_id`, `y`, `logits` over `y_mask` nodes.
- `metrics.json` keys stay `{best_epoch, val: {...}, test: {...}, ...}` with the new metric names.

### Measurements (A)

- Jacobian pair rows gain `excess_res = res - (1/deg_u + 1/deg_v)` (degrees in the subgraph).
- Node-level range (`range_node`) works unchanged on the final node representations.
- Entropy with Performer: `PerformerAttn` exposes the exact implicit attention rows for sampled
  query nodes (`φ(q_i)·φ(K)ᵀ / normaliser`); `measure_entropy` uses them, at most `n_queries` per subgraph (new
  `MeasureConfig.entropy_queries: int = 64`, added-field rule).
- Social configs set `measure.n_graphs_range=0` (no Hessian), and keep Jacobian sampling small
  (`n_graphs_jacobian` 20, `targets_per_graph` 8).

### Performer (A)

FAVOR+ positive random features (Choromanski et al. 2021), multi-head, attention restricted to each
subgraph via scatter sums over `batch` (per graph: Σ φ(k_j) v_jᵀ and Σ φ(k_j)), O(N·m·d). Random
features are redrawn each training step only if `self.training`, and fixed (seeded buffer) in
eval, so Jacobians are deterministic. Pure PyTorch (double backward must work). Same `(x, batch)` signature and
`cache` / `last_*` attributes as `GlobalAttn`, so `attention_layers()` covers both (common base class or a tuple).

### Configs and sweeps (B)

- `configs/social/<dataset>.yaml` for `twitch-gamers`, `reddit-body`, `twitch-de` (representative musae graph),
  `higgs-follow`: node-level hybrid, `attn: performer`, hidden 64, layers 4, heads 4, `pe_dim: 16`,
  batch_size 8, 100 epochs, patience 20, `measure.n_graphs_range: 0`.
- `configs/sweeps/social_s2_<dataset>.yaml`: α ∈ {0, 0.25, 0.5, 0.75, 1} × seeds [0, 1, 2], plus `arch=mlp` and
  `arch=gcn` baselines (3 seeds), with `est_minutes`. One sweep file per dataset, so separate Colab sessions never
  run the same run directory.
- `configs/social/quickstart.yaml`: twitch-gamers, `max_subgraphs` small, 20 epochs, α = 0.5, ≤ 15 min on a T4.

### Default targets (B; provisional, documented in each config header)

- `twitch-gamers`: `mature` (binary). Other columns are inputs (log-scaled where skewed); `numeric_id` is dropped.
- `twitch-<lang>`: `mature` (binary); inputs are the multi-hot game features reduced to the K most frequent columns (K recorded).
- `reddit-body`: a target chosen by `gnn_mech/reddit/target_select.py` if it runs on the real graph; otherwise a
  documented provisional choice that is not character count. Nodes without outgoing posts get `y_mask = False`.
- `higgs-*`: message passing on the named layer; target `active_after` = the user has any RT/MT/RE activity after
  the cutoff (default 2012-07-04 00:00 UTC, config `data.target=active_after`); inputs = structural features
  (degree in each layer) and **activity counts before the cutoff only**. No post-cutoff information in the inputs
  or the graph.

### Colab (C)

- Every notebook: mount Drive, clone `djdhawal/message-passing-in-gnns` at `BRANCH` (default `main`), install,
  set `DRIVE = /content/drive/MyDrive/Thesis/gnn_mech`, `DATA_ROOT = {DRIVE}/data`, `OUT_DIR = {DRIVE}/runs`.
  Data is downloaded **inside the notebook** into Drive once and reused; the processed subgraph file is copied
  to local `/content` disk before training for I/O speed when it exists.
- `00_quickstart.ipynb`: one short run (`configs/social/quickstart.yaml`), print metrics.
- `01_fetch_datasets.ipynb`: fetch and process every dataset into Drive (CPU runtime is fine).
- `02_s2_<dataset>.ipynb` (one per dataset): its sweep file, `--dry-run` first, then the sweep; re-running resumes.
- `03_s2_results.ipynb`: collect `metrics.json` across all S2 runs into a table and α-curve plots.

## Build contract: full-graph mode + planted targets (2026-11-03)

Why: notebook 04 (Higgs, 500-node METIS subgraphs) showed no graph effect, and a local check showed the cause:
only 32% of follow edges fall inside 500-node parts, and the neighbour signal (logistic regression, +0.018 AUROC with
full-graph neighbours) drops to +0.001 with in-part neighbours. Subgraphing removes the signal the knobs are meant to
probe. Planted long-range targets (S3) would be hurt more, since k-hop shells cross part boundaries.

### Workstream F: full-graph mode (owns `gnn_mech/social/dataset.py`, `social/subgraph.py`, `gnn_mech/layers.py`,
`models.py`, `config.py`, `train.py`, `measure/*`, `configs/social/higgs_full*.yaml`, `configs/sweeps/higgs_full_*.yaml`,
`notebooks/social/06_higgs_full_graph.ipynb`, `tests/test_full_graph.py`)

- `data.sampler: "full"` (existing field, new value). METIS parts (`subgraph_size`, `split_seed`, `val_frac`, `test_frac`)
  only decide **which nodes are labelled** in each split. Each split's dataset is a one-element list holding the **whole
  graph** with `y_mask` = labelled nodes of that split's parts (transductive: every node's features are visible, only
  train labels are trained on). x standardized with train-labelled-node statistics. `graph_id` 0, `node_id` = arange.
- RWSE on large graphs: no dense eigh. Use pe_dim 0 or a sparse / sampled estimate; document the choice.
- Full-batch training: one optimizer step per epoch, so full-graph configs need far more epochs (e.g. 500, patience 100).
- Attention on the full graph: k=∞ uses `attn: performer` (linear). `attn: radius` must work sparsely on large graphs:
  softmax over the k-hop pair list (structure-only, computed once and cached), with a pair budget
  (`model.attn_max_pairs`, new ModelConfig field, default-neutral for hashes); dense path unchanged for small graphs.
  Report pair counts for k=1, 2 on Higgs-follow.
- Measurements must run on a 35.6k-node graph (approximate resistance rows already exist above 5,000 nodes; entropy for
  sparse radius attention via its sparse rows, or skipped with a recorded reason).
- Sweeps (both targets `active_after`, `activate_passive` × split seeds 0,1,2 × model seeds 0,1; local gcn, branch_affine
  false): `higgs_full_affine.yaml` (alpha 0, .25, .5, .75, 1 with Performer, + mlp, gcn) and `higgs_full_radius.yaml`
  (k = 0, 1, 2, inf (Performer) + mlp, gcn). exp_names `higgs_full_affine`, `higgs_full_radius`.

### Workstream P: planted targets (owns new `gnn_mech/social/plant.py`, the planted-target hooks in
`gnn_mech/social/sources.py` (`target_columns`, `node_target`, `node_inputs` only), `configs/social/higgs_plant.yaml`,
`configs/sweeps/higgs_full_plant.yaml`, `notebooks/social/07_higgs_planted.ipynb`, `tests/test_plant.py`)

- Target names: `data.target = "plant:<family>:<k>"`, optional `:s<seed>` suffix (default seed 0), on **any** social
  dataset. Families (first build): `shell_mean` (y_v = mean of planted z_u over nodes at exactly k hops; sanity) and
  `retrieve` (keyed retrieval from the k-hop shell; the main test), regression. Planted channels are appended to the real
  node inputs by `node_inputs`; the real features stay as nuisance inputs. Nodes whose target is undefined get NaN
  (→ y_mask False). Deterministic from (graph, family, k, seed); computed on the full graph `graph.edge_index`
  (follow layer for Higgs), so it works with any sampler.
- Ground truth: `plant.analytic_range(...)` gives the Bamberger range of the target function itself (from its exact
  Jacobian w.r.t. the planted channels), which must equal k for both families.
- Report on Higgs-follow: shell sizes by k (effective diameter ≈ 3.7), eligible nodes per k, and how the retrieval key
  count was chosen so the expected number of matches stays small at every k.
- Sweep `higgs_full_plant.yaml` uses F's `data.sampler: full` and names only: k ∈ {1, 2, 3, 4} × both families ×
  variants {mlp, gcn, hybrid alpha 0, hybrid alpha 1 (performer), hybrid alpha 0.5 (performer), radius 1, radius 2}
  × model seeds 0, 1, split seed 0. exp_name `higgs_full_plant`.
