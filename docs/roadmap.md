# Thesis roadmap

Thesis: *Mechanistic Analysis of Over-Squashing and Attention Dilution in Hybrid
Graph Neural Networks* (UChicago MS Applied Data Science capstone).

Central claim being built: graph structure (effective resistance) plus task
properties (task range) predict where on the local-to-global spectrum a model
should sit. The spectrum is a blend weight α (0 = pure MPNN, 1 = pure attention),
and the mechanistic measurements (Jacobian sensitivity, attention entropy, range)
explain why. The final deliverable is a phase diagram: effective resistance on
one axis, task range / dilution exposure on the other, coloured by the best α.

## Phases

| Phase | What | Gate to move on | Status |
|---|---|---|---|
| 0 | Thesis text fixes needing no new runs | n/a | Deferred by the author: results will change anyway; folded into Track C of each phase |
| 1 | Shared training + measurement package (`gnn_mech/`), pilot bug fixes | GPU check: 4-layer α=0.5 ≥ 0.652 test AP; α=0 and α=1 train | Code done (83 tests pass). **GPU check pending on Colab** |
| 2 | Re-baseline Peptides-func (α=0/0.5/1 + GCN, 3 seeds), range measure, Reddit fixes | G1: α=0.5 ≥ 0.63, α=1 ≥ 0.60, GCN ≥ 0.66 (or gap documented); G2: Jacobian slope difference vs transformer reported with CI | Code done (range measure, predictions, sweep, analysis, GCN alignment, Reddit track; CPU tests pass). **GPU sweep and real-data Reddit runs pending on Colab**; see `docs/phase2_plan.md` § Status |
| 3 | α sweep on Peptides-func (0, .25, .5, .75, 1 × 3 seeds); best α per resistance bin; Jacobian slope and entropy vs α; gated-attention baseline | Performance-vs-α curve with CIs | Not started |
| 4 | Task-range axis: range on Peptides-func vs Peptides-struct (same graphs, different task) | Range differs or is shown equal, with CIs | Range *measure* moved into Phase 2 |
| 5 | Synthetic suite: controlled effective resistance and long-range dependence, distractor injection; α sweep over the grid → phase diagram; optional known long-range real dataset | Phase diagram with clear regions | Not started |
| 6 | Predictor (structure + range → best α), held-out check, final writeup | — | Not started |

Deferred: depth ablation (4/6/8 layers), after Phase 3. Cut order if time is
short: per-graph learned gate → extra real datasets → held-out prediction. Never
cut Phases 1–3 or the synthetic suite.

## Literature-review items and where they land

| Item | Source | Phase |
|---|---|---|
| Range measure as the task axis next to effective resistance | Bamberger et al., *On Measuring Long-Range Interactions in GNNs*, ICML 2025 (arXiv 2506.05971) | 2 (measure), 4 (analysis) |
| Peptides tasks are reportedly short-range → expect a low best α, say so | same | 2–3 |
| Tuned MPNNs close the LRGB gap → tuned baselines at both α ends | Tönshoff et al., TMLR 2024 | 2 |
| Gated / capacity-controlled global attention as a baseline | *Capacity-Controlled Global Attention for Graph Transformers*, arXiv 2604.17324 | 3 |
| Dilution framing: over-globalizing, over-aggregating | Xing et al. ICML 2024 (arXiv 2405.01102); Sun et al. 2025 (arXiv 2510.21267) | Writeup (Track C) |
| Effective resistance ↔ over-squashing citation | Black et al. 2023 (arXiv 2302.06835) | Writeup |
| Short-range over-squashing (bottleneck vs vanishing gradients) | Mishayev et al., LoG 2025 (arXiv 2511.20406) | Writeup / discussion |
| Known long-range benchmark | *Can You Hear Me Now?* (arXiv 2512.17762) | 5 (optional) |

## Thesis-draft corrections owed (from the review of the proposal PDF)

To apply in Track C once Phase 2 numbers exist:
1. Entropy claim: in the pilot, normalized entropy rose with size only for GPS
   (r=+0.60), not the transformer (r=−0.09, p=0.07).
2. Report AP of every model whose mechanism is measured.
3. §5.1 claims RWSE for all models; pilot Experiment 3 had none.
4. Transformer Jacobian slope is a size-confound baseline, not "bypassing".
5. "Replication" of Tönshoff was partial (pilot GCN 0.626 vs published ~0.686).
6. Reddit data description: SNAP hyperlink network, 35,776 nodes, 286,561
   directed edges, 86 post properties (not all LIWC). Target was feature 0
   (character count), nearly a function of the node's own inputs.
7. Add the citations in the table above.
