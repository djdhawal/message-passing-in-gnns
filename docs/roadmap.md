# Thesis roadmap

Thesis: *Mechanistic Analysis of Over-Squashing and Attention Dilution in Hybrid
Graph Neural Networks* (UChicago MS Applied Data Science capstone).

**Direction (decided 2026-11-02): large social graphs.** All main experiments run on
SNAP social graphs (Reddit, Twitch, Twitter) and on subgraphs sampled from them. The
Peptides-func and river experiments are kept as background in the appendix.

Central claim: graph structure (bottlenecks, measured by effective resistance) and
task range predict where on the local-to-global spectrum a model should sit. The
spectrum is a blend weight α (0 = pure MPNN, 1 = pure linear-cost Performer attention).
The Jacobian, attention-entropy and range measurements explain *why*. The deliverable is
a phase diagram, with structure on one axis, task range on the other, and cells coloured
by the best α. Then it asks the same question of graph foundation models (do they reach
far enough?) and releases the planted-range social tasks as a benchmark. Neither LRGB
nor ECHO contains social graphs.

Detailed plan, interfaces and gates: `docs/social_plan.md`.

## Phases

| Phase | What | Gate to move on | Status |
|---|---|---|---|
| 0–2 (old) | `gnn_mech/` package, Peptides re-baseline, range measure, Reddit rebuild | — | Code done. Peptides G0 ran once (0.6259 vs 0.652). **No further Peptides GPU work**; results go to the appendix as they stand |
| S0 | Social data layer: loaders (Reddit, Twitch, Twitter), subgraph sampler, structure stats (degree-corrected resistance) | Stats table per dataset; resistance-vs-degree collapse quantified | Not started |
| S1 | Performer attention branch + node-level hybrid model | Performer matches softmax attention on small graphs; α=0 and α=1 beat MLP on one real target | Not started |
| S2 | α sweep on real targets (α ∈ {0, .25, .5, .75, 1} × 3 seeds × datasets) with Jacobian / entropy / range | Performance-vs-α curves with CIs per dataset | Not started |
| S3 | Planted semi-synthetic targets with known range k on the same subgraphs; measured range of trained models | Measured range tracks planted k (validates the measure); α sweep on planted tasks | Not started |
| S4 | Phase diagram: structure bin × task range → best α, with entropy overlay | Diagram with CIs over subgraphs; regions stable across seeds | Not started |
| S5 | GFM benchmark: GraphPFN (+ GraphAny if feasible) on the planted tasks; measure their range | Range of each GFM vs required k; accuracy vs the α-sweep models | Not started |
| S6 | Benchmark release (generator + indices + datasheet) and final writeup | Reproducible from a clean clone | Not started |

Cut order if time is short: GraphAny → Higgs multilayer analysis → benchmark packaging
polish. Never cut S1–S4.

## Literature and where it lands

| Item | Source | Phase |
|---|---|---|
| Range measure (task axis) | Bamberger et al., ICML 2025 (arXiv 2506.05971) | S2–S4 |
| Fixed-backbone GFMs under-reach on tasks of differing range; architecture adaptivity | Gutteridge, Bronstein & Dong, ICLR 2026 GRaM workshop (arXiv 2603.22984) | S5, framing |
| Node-level GFM to benchmark | GraphPFN, Eremeev et al., ICML 2026 (arXiv 2509.21489; github.com/yandex-research/graphpfn) | S5 |
| GFMs vs tuned GNNs on node classification (no long-range analysis) | NodeGround (arXiv 2609.39673); Fair Evaluation of GFMs (arXiv 2606.24509) | S5, related work |
| Long-range benchmark without social graphs | ECHO, Miglior et al., ICLR 2026 (arXiv 2512.17762) | Related work; motivates S6 |
| Linear attention | Performer / FAVOR+, Choromanski et al., ICLR 2021 | S1 |
| Effective resistance ↔ over-squashing | Black et al. 2023 (arXiv 2302.06835) | S0, writeup |
| Resistance collapses to 1/d_u + 1/d_v on large well-connected graphs | von Luxburg, Radl & Hein, JMLR 2014 | S0 (degree correction) |
| Dilution framing | Xing et al. ICML 2024 (arXiv 2405.01102); Sun et al. 2025 (arXiv 2510.21267) | Writeup |
| Tuned MPNN baselines | Tönshoff et al., TMLR 2024 | S2 |
| Gated / capacity-controlled attention | arXiv 2604.17324 | Optional baseline in S2 |

## Thesis-draft corrections still owed

From the review of the proposal PDF. Items 1–5 now apply to the appendix (Peptides pilot):
1. Entropy claim: normalized entropy rose with size only for GPS (r=+0.60), not the transformer (r=−0.09, p=0.07).
2. Report AP of every model whose mechanism is measured.
3. §5.1 claims RWSE for all models; pilot Experiment 3 had none.
4. Transformer Jacobian slope is a size-confound baseline, not "bypassing".
5. "Replication" of Tönshoff was partial (pilot GCN 0.626 vs published ~0.686).
6. Reddit data description: SNAP hyperlink network, 35,776 nodes, 286,561 directed
   edges, 86 post properties (not all LIWC). Pilot target was feature 0 (character count).
7. Add the citations in the table above.
8. New: rewrite the Introduction, Data and Methodology sections around the social-graph design; move Peptides
   and river findings to the appendix.
