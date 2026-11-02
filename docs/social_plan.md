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
