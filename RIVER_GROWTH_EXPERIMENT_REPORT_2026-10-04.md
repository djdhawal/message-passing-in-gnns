# River hybrid GNN results and controlled tree-growth experiment

**Date:** 2026-10-04  
**Project:** [message-passing-in-gnns](https://github.com/djdhawal/message-passing-in-gnns)

This report records the completed river experiment, the subsequent analysis of graph growth, and the controlled experiment built in response. River training ran on the project owner's laptop; the supplied result archives were reviewed and the analysis/code checks described below were performed separately. The new controlled-tree main sweep has **not** been run as part of this work.

## 1. Research question and current conclusion

The question is whether graph structure and task information can predict when message passing, attention, or a hybrid will work best. In particular: does faster upstream volume growth increase the relative benefit of attention or a hybrid?

The completed river results support three observations:

1. With weather inputs, the attention-heavy hybrid improves on the message-passing endpoint, but its advantage over attention and simple pooling is small.
2. With weather plus historical flow, message passing and simple pooling are already competitive; the tested hybrids do not improve the aggregate score.
3. The measured relationship between volume growth and hybrid benefit changes with the comparison baseline and input regime. These data do not establish a reliable graph-to-alpha rule or demonstrate oversquashing.

This motivated a controlled synthetic experiment that varies tree growth while fixing graph size, depth, and the number of relevant source values.

## 2. What was completed

| Work | Status |
| --- | --- |
| Laptop CUDA setup | User confirmed forward/backward computation on an RTX 3080 Laptop GPU after replacing the incompatible CUDA 13 wheel with a CUDA 12.6 PyTorch build. |
| River training and context ablations | Uploaded runtime records show 73 completed records: 43 full-context records and 30 context ablations; 72 trained fits plus one persistence baseline. |
| River result review | Compared held-out scores, endpoints, simple baselines, context ablations, and the existing alpha selector. |
| Forecast diagnostics | Produced/reviewed 12 forecast plots for two outlets, three horizons, and two configurations. |
| Graph-reader repair | Fixed the growth-analysis loader to recover graph identity from `node_ids[outlet_index]` and the processed manifest instead of requiring a missing NPZ `graph_id` field. |
| Actual graph-growth analysis | Loaded all 32 graphs; verified directed topology and distances; paired model results by outlet, task, and seed. |
| Controlled-tree package | Built and CPU-tested the separately delivered `tree_growth_lab.zip`, including generation, training, resume, analysis, and reporting. |
| Fresh river-group census | Implemented and fixture-tested a structural candidate census. The actual unused raw river data were not available for a new evaluation here. |

The runtime records sum to approximately **28 hours 32 minutes** of reported wall time. This is recorded experiment time, not a guarantee for future runs or elapsed calendar time including pauses.

## 3. River experiment and meaning of alpha

The real-data experiment uses LamaH-CE daily gauge-network data. The reviewed setup uses 90-day histories, forecasts at 1, 3, and 7 days, and training seeds **42, 43, and 44**. Training has a 50-epoch cap and early-stopping patience of eight epochs. Finishing before epoch 50 is therefore expected.

The model uses a shared per-node GRU temporal encoder followed by eight spatial layers, with width 64 and four attention heads. Each hybrid layer combines:

- directed upstream-to-downstream message passing with mean aggregation;
- attention over upstream ancestors and the receiving node itself, including learned graph-distance bias.

The branch combination is `(1 - alpha) * normalized_MP + alpha * normalized_attention`, followed by the surrounding residual/feed-forward computation. Alpha 0 is the message-passing endpoint; alpha 1 is the attention endpoint. **Alpha is a mixing coefficient, not a measured percentage of useful information, accuracy, or computation.**

The study also includes an outlet-only temporal model, topology-free upstream pooling, and persistence. Active parameter counts differ between configurations, so this is not a parameter-matched comparison or an exhaustive tuning study.

## 4. Held-out river results

The following are **held-out test NRMSE**, with full input context, averaged equally over outlets, forecast horizons, and the three training seeds. Lower is better. NRMSE normalizes RMSE by the global development-training specific-discharge standard deviation scaled to each outlet's drainage area, rather than an independently estimated standard deviation at each outlet. It is not the `dev_val_mse` printed in the training logs. Persistence has one deterministic baseline record.

| Configuration | Weather | Weather + flow |
| --- | ---: | ---: |
| Message passing, alpha 0 | 0.266572 | 0.169096 |
| Hybrid, alpha 0.25 | 0.262828 | 0.172186 |
| Hybrid, alpha 0.50 | 0.248851 | 0.170592 |
| Hybrid, alpha 0.75 | **0.240945** | 0.171644 |
| Attention, alpha 1 | 0.242612 | 0.171251 |
| Outlet-only temporal model | 0.260040 | 0.169640 |
| Upstream pooling | 0.243693 | **0.168917** |
| Persistence | — | 0.189063 |

For weather inputs, alpha 0.75 has **9.61% lower mean NRMSE than message passing**, **0.69% lower than attention**, and **1.13% lower than pooling**. Thus the strongest improvement is against message passing; the incremental benefit over simpler alternatives is modest.

With flow history, pooling is the best aggregate configuration in this table and message passing is close behind. Every nonzero-alpha configuration has a slightly higher aggregate NRMSE than message passing. This makes input information an essential part of the research question.

The weather alpha-0.75 model has mean NSE **-0.2962**, whereas weather-plus-flow pooling has mean NSE **0.3728**. Aggregate NRMSE improvements should therefore not be described as uniformly strong hydrological performance. NSE and NRMSE have different denominators and averaging; an aggregate NRMSE cannot be converted directly into aggregate NSE.

### Validation selection versus descriptive test comparisons

Alpha 0.75 is the best aggregate weather configuration observed in the table, not a demonstrated universal optimum. The original development-validation choices were:

| Input regime | 1-day alpha | 3-day alpha | 7-day alpha |
| --- | ---: | ---: | ---: |
| Weather | 0.75 | 0.75 | 0.50 |
| Weather + flow | 0.00 | 1.00 | 1.00 |

Keep these validation-selected choices separate from any coefficient that looks best after inspecting test outcomes.

### Context, selector, and forecast checks

For weather alpha 0.75, mean held-out NRMSE is **0.240945** with full context, **0.274070** with radius-zero dynamic input context, and **0.269518** with two-hop dynamic input context. Full context improves these scores by approximately **12.09%** and **10.60%**, respectively. The ablations mask dynamic histories while retaining graph/static information; they are not graph-rewiring experiments. They support the usefulness of upstream context, not a particular oversquashing mechanism.

The existing graph-conditioned alpha selector did not outperform its validation-selected fixed comparator. Its reported macro NRMSE was **0.212202**, versus **0.207372**, about **2.33% worse**. The reported paired bootstrap interval for selected-minus-fixed NRMSE was **[-0.001550, 0.016134]**. It used only four held-out dependency groups and is unstable at that sample size. This selector summary combines both input regimes and uses its own seed-error aggregation; it is not the same aggregation as the table above. Its fitted rules split on river length and mean branching, rather than measured volume-growth profiles.

The forecast diagnostics cover outlets **499 and 609**, all three horizons, and seed 42, using the last 365 eligible dates. They compare weather alpha 0.75 with weather-plus-flow alpha 0. Both architecture and input information change, so the plots illustrate prediction behavior rather than isolate a causal feature effect. They also cover only a subset of the evaluation period.

## 5. Graph volume growth: what was measured

For an outlet, `V(r)` is the number of unique upstream gauge nodes within `r` directed hops, **including the outlet**. A shell count includes only nodes exactly `r` hops away. A log-growth increment is `log(V(r) / V(r-1))`.

These quantities describe the sampled gauge graph. They are not water volume, physical river length, drainage area, or the full river-reach network. In particular, `V(2)` is a cumulative two-hop volume, not itself a growth rate.

The corrected analysis loaded **32/32 graphs**, with zero missing graphs or distance-only fallbacks. It processed **12,921 metric rows** and found no incomplete seed-pairing tasks. The graphs have **5–35 nodes** and depths **1–8**; the original selection limits depth to eight.

There are **27 development outlets in three dependency groups**, and **five held-out outlets in four groups**. The selected graph node sets do not overlap. Outlets 499 and 519 nevertheless share dependency group 521 because the grouping accounts for transfer-linked hydrological dependence. Disjoint graph nodes do not automatically establish statistical independence.

### Per-outlet hybrid benefit

For alpha 0.75, the following percentages compare the mean NRMSE against message passing, averaging seeds and horizons first. Positive means the hybrid is better; negative means it is worse.

| Outlet | Dependency group | Nodes | Depth | V(2) | Weather improvement | Weather + flow improvement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 444 | 444 | 9 | 6 | 3 | +13.14% | -0.11% |
| 609 | 609 | 7 | 3 | 5 | +18.79% | +1.30% |
| 452 | 456 | 8 | 3 | 6 | +26.17% | -2.86% |
| 519 | 521 | 11 | 3 | 9 | +2.90% | +0.25% |
| 499 | 521 | 29 | 7 | 13 | -4.40% | -8.92% |

The largest two-hop volume in this sample does not correspond to the greatest hybrid improvement. That observation alone also does not establish that large volume is harmful.

### Exploratory correlations

These are Spearman correlations between `V(2)` and **absolute NRMSE gain**, defined as `reference NRMSE - hybrid NRMSE`, for alpha 0.75. They use five held-out outlets, after seed averaging, and keep horizons separate.

| Inputs | Reference | 1 day | 3 days | 7 days |
| --- | --- | ---: | ---: | ---: |
| Weather | Message passing | -0.70 | -0.70 | -0.60 |
| Weather | Attention | +0.90 | +0.90 | +0.90 |
| Weather | Pooling | 0.00 | +0.30 | +0.30 |
| Weather + flow | Message passing | -0.80 | -0.30 | -0.40 |
| Weather + flow | Attention | +0.50 | +0.10 | +0.60 |
| Weather + flow | Pooling | +0.50 | +0.30 | +0.50 |

The sign changes with the reference model. Leaving out one dependency group gives weather-versus-message-passing correlation ranges of **[-1.0, +0.5]**, **[-1.0, +0.5]**, and **[-0.8, +1.0]** at the three horizons. These are sensitivity ranges, not confidence intervals.

Across all 32 graphs, size and depth are strongly correlated (Spearman approximately **0.83**), as are size and drainage area (approximately **0.77**). Large-radius volumes eventually equal graph size. The analysis also explored many related radii and metrics. These issues, the limited independent groups, and the baseline dependence prevent a robust general claim from the observed correlations. Training seeds measure optimization variability; they do not create additional river networks.

## 6. Controlled experiment built next

The separately delivered `tree_growth_lab.zip` generates controlled rooted trees and trains fresh weights using the original river spatial architecture. `river_models.py` is byte-identical to the supplied river model source. The synthetic input/output adapter changes: one time step, four dynamic channels, no static features, zero edge/pair attributes, and one scalar output.

### Declared main design

| Factor | Setting |
| --- | --- |
| Nodes | 63 or 127 |
| Maximum root distance | 4 or 8 hops |
| Growth profile | Early, uniform, late |
| Replicates per size/depth/profile cell | 2 train, 2 validation, 4 test |
| Total unique rooted trees | **96: 24 train, 24 validation, 48 test** |
| Root children / maximum children | Exactly 4 / at most 4 |
| Spatial architecture | Width 64, 8 layers, 4 heads, dropout 0.1 |
| Source distances | 1 hop and 4 hops |
| Relevant values per query | 4 |
| Marked distractors per query | 4 |
| Models | Five alpha values, pooling, outlet-only |
| Training seeds | 42, 43, 44 |
| Total training fits | **21**, each jointly trained on both query types |

Matched early/late graphs have the same size, depth, and construction seed. Generation requires realized early-growth `V(2)` to exceed late-growth `V(2)`. Root degree remains fixed, so `V(1)=5`. Node indices, including the root, are permuted, and canonical rooted-tree hashes reject isomorphic duplicates within and across splits.

Each example gives every node an independent standard-normal value. Four nodes at one hop and four at four hops are marked. A root query selects the nearby or distant marked set. The target is `sum(four requested values) / sqrt(4)`, with population mean zero and variance one for both tasks. Paired queries share values and marks; only the query changes. The model receives value, marked-node flag, root flag, and root query, with no node-ID or node-depth input features.

Exact reference controls are included: a graph-aware oracle has MSE zero; predicting zero has population MSE one; a topology-blind predictor using half the normalized sum of all eight marked values has population MSE 0.5. Finite samples fluctuate around these population values.

Both query distances are inside the eight-layer message-passing receptive field. Attention retains its distance bias. This controls a simple reachability explanation, but does not isolate oversquashing from expressivity, optimization, aggregation, or parameter-budget effects.

### Training and analysis safeguards

Training uses AdamW with learning rate 0.001, gradient clipping at one, a 30-epoch cap, and patience six. The best checkpoint minimizes balanced validation MSE. Models receive matched signals, and comparisons pair graph, task, and seed. Epoch checkpoints include optimizer and random-number state, allowing interrupted work to resume; changed code or configuration requires a new work directory.

The primary structural contrast is **hybrid gain on an early-growth tree minus hybrid gain on its matched late-growth tree**, separately for size, depth, query type, alpha, and baseline. Report comparisons with message passing, attention, and pooling. Alpha selection uses validation only; report all declared test curves.

Seed averaging precedes resampling of matched graph pairs. Four test pairs per size/depth stratum remain a small exploratory sample. Early/late generation changes several aspects of shape, and four-hop sources are leaves at depth four but can be internal at depth eight. Interpret contrasts within depth; this is not a pure intervention on one scalar volume statistic.

The main graph seed is **20261015**. Default smoke, pilot, and main topology sets were checked for rooted-isomorphic non-overlap. The main seed was chosen using this structural check, without inspecting model outcomes.

### What was actually tested

CPU checks passed for topology constraints, paired targets, exact oracles, permutation invariance, finite forward/backward computation, and message-passing reach. Interrupted/resumed training reproduced uninterrupted CPU metrics. A complete five-fit, two-epoch smoke run generated metrics, plots, and a report ZIP. Analysis fixtures checked known contrasts, validation-only selection, graph-pair counts, and rejection of incomplete metrics. The main-width model also passed a 127-node forward/backward benchmark.

**No 21-fit main sweep or GPU execution of this new package was performed here. Smoke results and fabricated test fixtures are software-validation evidence, not scientific findings.**

## 7. Reproduction and next steps

The commands below require the separately supplied `tree_growth_lab.zip`; this Markdown commit does not add that package or the raw data to the repository. Use the laptop environment with working CUDA:

```bash
cd ~/Downloads/river_hybrid_lab
source .venv/bin/activate
python -m zipfile -e ~/Downloads/tree_growth_lab.zip .
python -m pip install -r tree_growth_lab/requirements.txt
python -m tree_growth_lab check --device cuda
python -m tree_growth_lab smoke --work tree_growth_smoke --device cuda
python -m tree_growth_lab benchmark --work tree_growth_experiment --preset main --device cuda

systemd-inhibit --what=sleep:idle --mode=block \
  --why="Controlled tree GNN experiment" \
  python -u -m tree_growth_lab run \
  --work tree_growth_experiment --preset main --device cuda
```

The requirements preserve the existing PyTorch installation. Repeat the same main command to resume. Keep the laptop powered and the terminal/session alive; screen locking does not suspend computation, while system sleep does. The completed run produces a `tree_growth_review_<timestamp>.zip` for analysis.

A separate census can locate structurally eligible, previously unseen river dependency groups:

```bash
python -m tree_growth_lab census \
  --raw river_experiment/raw \
  --seen river_experiment/processed/manifest.json \
  --out tree_fresh_river_census
```

This uses the existing river reader, excludes entire previously inspected transfer-connected groups, and reads structural attributes rather than discharge targets. It identifies candidates, not a qualified test set. Actual candidate counts still require the local raw data, and training-period availability must be checked.

The next research steps are to complete the controlled sweep, inspect all predeclared contrasts and baselines, and then freeze a model/selection rule before evaluating fresh real-data groups. More independent graphs and hydrological groups are needed for confirmation; adding training seeds alone does not resolve the current independence limitation. No literature-novelty or state-of-the-art claim is established by this report.

## 8. Evidence and artifact provenance

The numerical findings were checked against the supplied result archives, especially `model_summary.csv`, `paired_outlet_gains.csv`, `exploratory_correlations.csv`, `graph_growth_features.csv`, `analysis_status.json`, and the original selector/context/runtime outputs. Controlled-package status was checked against its source, configuration, and `VALIDATION.txt`.

| Artifact supplied separately | SHA-256 |
| --- | --- |
| `river_results_review.zip` | `1fca8cef55b25ae177c655e6ebd5c84294dfebafd95d3915a5c6c2163404f503` |
| `river_growth_review_20261004T200140_464098Z.zip` — forecast diagnostics and original graph-loader failure | `71fed390affd5e54bcad0562e7f4ad791843ec4bf91d72ce5f80fcdc7a389ba4` |
| `river_growth_review_20261004T201159_073126Z.zip` — corrected actual-graph analysis | `bb6997f8ae6942f29c750899912aedcc620fd9cae122082642d5cbeab54c847d` |
| `tree_growth_lab.zip` — implemented, tested controlled experiment | `34e433b77d2556d7d935c0344e8b215bd60e7b83f3def9bae467cf0422eb389f` |

The unchanged copied river model source has SHA-256 `9f39362a4cec2c117f52834cbfb04f2c4d8b30fded9a28175b1653e19ae34e0a`. Archive hashes identify the reviewed versions; the archives themselves are not included in this Markdown-only upload.
