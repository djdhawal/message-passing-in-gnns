# Testing when graph growth favors message passing, attention, or a hybrid

**Research plan — 2026-10-05**  
**Project:** [message-passing-in-gnns](https://github.com/djdhawal/message-passing-in-gnns)  
**Status:** proposed protocol, independently critiqued; no new model training or large dataset download was performed for this plan.

## 1. Recommendation and intended contribution

Build a new, deliberately conventional **residual sum-message-passing + attention model**. Use it to test a narrower, falsifiable question:

> At fixed graph size, source distance, and model width, does changing a tree's growth profile change the advantage of global communication, and does that relationship depend on the information demanded by the task?

The intended contribution is an experimental account of **when different communication mechanisms help**. Combining message passing and attention is already established, including in GraphGPS [1]. Neither an architectural novelty claim nor a universal graph-to-alpha rule is justified yet.

Run three levels of evidence in order:

1. **Controlled synthetic trees:** establish whether a structural intervention changes relative performance while targets and other controlled factors are matched.
2. **Real trees with controlled targets:** repeat the communication tests on extracted river or syntax trees. This is explicitly semi-synthetic.
3. **Real trees with real targets:** forecast river discharge or predict method names from syntax trees. These test practical relevance; their structure–performance associations remain observational.

The first implementation should retain a simple fixed coefficient, $\alpha$, throughout each trained model. A learned graph-conditioned gate can follow once the endpoints and measurements are trustworthy. **Alpha is a coefficient in the model equation, not the percentage of useful information carried by attention.**

The central design change is to make marked summation a **sanity test**, and addressed retrieval the **main communication test**. Summing four known relevant values can be compressed into one scalar; preserving associations for a query that becomes available only at the receiving node is a different demand. Retrieval-style tree bottleneck tasks have precedent [2]; this is a controlled extension, not a claim to have invented them.

## 2. What the completed experiments tell us

This section summarizes the previously reviewed project artifacts. It is evidence motivating this proposal, not a result of the proposed new model.

### 2.1 Real river experiment

The previous LamaH-CE study used a per-node GRU, 90-day histories, eight spatial layers, width 64, and forecasts at 1, 3, and 7 days. It used seeds 42, 43, and 44. Its completed records comprise 72 trained fits and one persistence baseline.

| Held-out aggregate NRMSE; lower is better | Weather | Weather + historical flow |
|---|---:|---:|
| Message passing, alpha 0 | 0.266572 | 0.169096 |
| Hybrid, alpha 0.25 | 0.262828 | 0.172186 |
| Hybrid, alpha 0.50 | 0.248851 | 0.170592 |
| Hybrid, alpha 0.75 | 0.240945 | 0.171644 |
| Attention, alpha 1 | 0.242612 | 0.171251 |
| Upstream pooling | 0.243693 | 0.168917 |

Weather-only results favor attention/global context, but the best hybrid's advantage over attention is small. With historical flow, message passing and pooling are competitive. This supports including **input information and task demands**, rather than graph structure alone, in the hypothesis. These are project-specific normalized test metrics; they are not directly comparable to arbitrary published MSE or NSE values.

Only five held-out outlets in four dependency groups were available. This is insufficient for a convincing general graph-to-architecture predictor. The previously inspected outlets and their dependency groups must now be treated as development evidence.

### 2.2 Completed controlled-tree experiment

The supplied main archive contained all 21 planned fits: five alpha values plus two baselines, each with three seeds. The validation-selected alpha was 0.25.

| Configuration | Near-task test MSE | Far-task test MSE | Balanced test MSE |
|---|---:|---:|---:|
| Message passing | 0.003580 | 1.009409 | 0.506495 |
| Hybrid, alpha 0.25 | 0.002439 | 0.012930 | 0.007684 |
| Hybrid, alpha 0.50 | 0.007362 | 0.014666 | 0.011014 |
| Hybrid, alpha 0.75 | 0.005025 | 0.012205 | 0.008615 |
| Attention | 0.009334 | 0.018948 | 0.014141 |

The hybrid improves on the tested attention endpoint, and the MP endpoint essentially predicts zero on the distant task. However, the following issues prevent a clean oversquashing or growth conclusion:

- **Target draws differed between growth profiles.** For one size/depth stratum, the early-minus-late contrast in hybrid-over-MP gain was about 0.1115, while the corresponding zero-predictor target-energy contrast was about 0.1140. A large apparent structural effect can therefore arise from unequal realized targets. Same-graph model comparisons did share examples and remain informative.
- **Aggregation did not match the task cleanly.** The MP branch used mean aggregation without an explicit count input; the label involved summation. This is a plausible disadvantage, not proof that mean aggregation cannot learn the task.
- **Structural access differed.** Attention had a learned exact hop-distance bias relevant to the label rule; MP did not receive an equivalent precomputed feature.
- **Branch normalization could discard amplitude.** LayerNorm directly after aggregation can remove a scale needed for summation.
- **Active parameter counts differed:** approximately 355k for MP, 491k for the hybrid, and 291k for attention.
- **Only four matched test pairs per size/depth stratum** were available. Training seeds do not create additional graph interventions.
- **Nominal reach was sufficient**, but this alone does not distinguish compression, optimization, positional reasoning, or vanishing sensitivity. A small untrained Jacobian is not a diagnosis of the trained model.

The review archive did not contain the trained checkpoints needed for a new paired-target evaluation. If checkpoints remain on the laptop, re-evaluate them diagnostically without retraining. That follow-up uses already inspected trees and is not fresh confirmation.

## 3. Definitions: distinguish structure, demand, and performance

For a receiver/root $v$, orient tree edges from upstream child to downstream parent. Define

$$
B_v(r)=\{u:d(u,v)\le r,\ u\text{ can reach }v\},\qquad V_v(r)=|B_v(r)|.
$$

The receiver is included: $V_v(0)=1$. Define shell size $S_v(r)=V_v(r)-V_v(r-1)$, and log growth $g_v(r)=\log V_v(r)-\log V_v(r-1)$.

Record the **full profile** $V(0),\ldots,V(D)$. A value such as $V(2)$ is cumulative volume, not a growth rate, water volume, or a complete bottleneck measure.

| Quantity | Meaning | Important distinction |
|---|---|---|
| $N,D$ | Node count and maximum root distance | Larger graphs need not have greater task-relevant demand |
| $V(r),S(r),g(r)$ | Cumulative, shell, and incremental growth | Growth depends on the graph representation |
| $R$ | Required source-to-root distance | Known from the synthetic target construction |
| $K$ | Number of candidate associations | More candidates can increase routing demand and attention competition |
| $p$ | Payload dimension per candidate | Affects information load; targets retain unit variance per coordinate |
| $C$ | Candidate concentration behind a specified cut/branch | Distinct from total graph volume |
| $d,L$ | Hidden width and communication layers | Both affect the available communication computation |
| $\alpha$ | Fixed mixing coefficient | Not an attribution or compute percentage |

On an **unweighted undirected tree with unit edge resistances**, pairwise effective resistance equals path length: there is only one conducting path. Therefore resistance to sources at fixed distance four cannot independently distinguish growth profiles. Weighted resistance can contain other information, but requires a justified edge-weight definition. Do not add a resistance feature and assume it measures branching independently.

A trained Jacobian measures what a fitted model uses; it is not automatically the task's intrinsic range or a feature available before model training. Formal range measures such as Bamberger et al.'s require their actual definitions and estimation procedure [3]. For the synthetic study, the generator supplies the true dependency set directly.

## 4. New model: an explicit implementation contract

Call this model **Residual Sum-MP/Attention** in experiment files. Use familiar components so an improvement can be interpreted. Aggregation expressiveness and topology/sensitivity have substantial prior literature [4,5].

### 4.1 Common inputs and output

- Width $d=64$, four attention heads, eight layers in the main experiment.
- Separate models for summation and retrieval; each model jointly trains across that task's declared conditions.
- Every alpha receives the same node features, graph, source examples, encoder, and output-head design.
- Common features include root flag, task-role flag, payload, and log in/out degree. In summation the flag identifies a relevant summand; in retrieval it identifies an eligible candidate, **never the queried candidate**. Retrieval additionally has a key and a query channel. Non-root query channels are zero in the root-only condition.
- Omit root-distance features in the primary retrieval test, where all candidates have $R=4$. If added in a diagnostic, give them to every architecture. There is no attention-only hop or pair bias in the main study.
- Encode with a two-layer GELU MLP. Predict from the final root state and its original features, using another MLP. Do not apply a final LayerNorm that removes the root state's amplitude.
- Padding nodes cannot send messages, receive attention mass, or enter losses. No graph identifier or stable node index is an input.

### 4.2 Local branch

Let $C(v)$ be the immediate upstream children. At layer $\ell$:

$$
s_v^{\ell}=\frac{1}{\sqrt{\Delta}}\sum_{u\in C(v)}\phi_\ell(h_u^\ell,e_{uv}),\qquad
m_v^\ell=\psi_\ell([h_v^\ell,s_v^\ell,\log(1+\deg_{in}v),\log(1+\deg_{out}v)]).
$$

Here $\Delta=4$ is the fixed maximum-child bound of the synthetic family. This constant rescales a sum; it is not a node-dependent mean and does not erase count information. Use two-layer GELU MLPs for $\phi$ and $\psi$. Empty child sets produce a zero aggregate. Edge features are zero in the base synthetic experiment.

### 4.3 Attention branch

Let $U(v)$ contain all upstream nodes that can reach $v$, including $v$. Use ordinary multihead attention over $U(v)$:

$$
a_v^\ell=W_O\operatorname{concat}_j\left[\sum_{u\in U(v)}
\operatorname{softmax}_{u\in U(v)}\left(\frac{q_{vj}^{\mathsf T}k_{uj}}{\sqrt{d/4}}\right)W_{V,j}h_u^\ell\right].
$$

In the declared main model, compute Q/K from LayerNorm of the input states; use **raw states for values**. The root can directly access all candidates. That shortcut is an intended difference in communication paths. Supplying the same features does not make the two branches computationally identical.

### 4.4 Residual update and alpha

$$
z_v^\ell=h_v^\ell+\beta[(1-\alpha)m_v^\ell+\alpha a_v^\ell],\qquad
h_v^{\ell+1}=z_v^\ell+\beta\operatorname{FFN}_\ell(\operatorname{LN}(z_v^\ell)),
\quad\beta=(2L)^{-1/2}.
$$

Use a two-layer FFN with hidden width $2d$. Branch outputs are not normalized before mixing. The residual path preserves their magnitude. Start with zero dropout for these noise-free synthetic targets, and freeze that choice after development checks.

The primary uses branch scale constants equal to one. Log MP/attention update RMS, combined-update norms, and gradients by layer, profile, and K. A secondary scale-control experiment may use fixed inverse-RMS constants estimated from training inputs at initialization, shared across all alphas within a seed. It must not use labels or validation/test scores. Learned weights can still rescale branches, so even this calibration does not make alpha an information fraction. A future growth-to-alpha claim must survive scale-control analysis: local sums and global softmax averages can have different structure-dependent amplitudes.

Use $\alpha\in\{0,.25,.5,.75,1\}$. Initialize common modules identically within each seed, with independent RNG streams for parameters, examples, and graph selection. Inactive modules must not affect data order. Count **active** parameters; merely registering unused modules does not equalize model capacity.

## 5. Synthetic data: exact pairing is the highest-priority repair

### 5.1 Proposed graph corpus

| Factor | Planned specification |
|---|---|
| Node counts | 63, 127 |
| Exact maximum depths | 4, 8 |
| Root child count | Exactly 4 |
| Maximum children | 4 |
| Source support | At least four distance-4 nodes in each root branch |
| Growth profiles | Early, intermediate, late growth |
| Training | 8 matched triplets per $(N,D)$: 96 trees |
| Validation | 8 matched triplets per $(N,D)$: 96 trees |
| Fresh test | 32 matched triplets per $(N,D)$: 384 trees |
| Structural test units | 128 matched triplets, not 384 independent treatments |

These are proposed counts, **not a claim that this new constrained corpus has already been generated**. Build and inspect its topology-only feasibility census before training.

An implementable proposal generator starts with four root branches extending to the required depth, then attaches new leaves to eligible nodes with a breadth-biased, neutral, or depth-biased sampling weight. Reject violations of size, depth, maximum degree, or source support. Use a separate topology-only calibration pool to fix low/middle/high $V(2)$ bins in each $(N,D)$, with a minimum early–late gap of four nodes. Freeze those bins and generator settings without observing model errors.

For each prespecified split and block ID, generate a fresh candidate pool with an independent graph RNG. Select one eligible tree per bin, uniformly from candidates satisfying the fixed constraints. A block contains three distinct trees with the same N/D; trees are never reused across blocks or splits. Reject globally duplicate rooted-tree shapes. Save candidate seeds, bin thresholds, selection probabilities/rules, rejection counts, and block membership. If fixed bins have insufficient support, revise at the feasibility stage and freeze again before training. These are **matched blocks of separately generated trees**, not three counterfactual versions of one latent tree.

The profile names describe achieved geometry, not a guarantee that only $V(2)$ changes. Leaf fraction, degree sequence, source-path degrees, and higher-radius volumes can also change. Report them, along with candidate counts below every edge and candidate lowest-common-ancestor/path-sharing statistics. This experiment identifies the effect of the declared **tree-construction intervention**, not a surgical causal effect of one scalar statistic.

The smallest deep cell is constrained: four length-eight anchor chains plus three additional distance-four siblings per branch occupy 45 nodes, leaving only 18 additional nodes when N=63. This construction demonstrates one feasible support pattern, not that all desired growth profiles are feasible. Do not hide weak separation in that cell behind a pooled estimate.

Randomly permute node IDs after construction. Deduplicate using rooted-tree isomorphism, including against previously inspected trees. Split whole matched triplets; no profile from a triplet may cross a split. Source-role placement must be randomized among eligible nodes and recorded. At depth four, distance-four nodes are leaves; at depth eight they can be internal. Keep depth-specific estimates visible.

### 5.2 Common semantic random numbers

Generate the latent task example **before** assigning it to nodes. Pair semantic roles, not numerical RNG seeds:

```python
# Protocol pseudocode, not a supplied executable implementation.
latent = draw_role_bank(split_seed, block_id, example_id)
for condition in task_conditions:
    target = target_from_roles(latent, condition)
    for profile in ("early", "intermediate", "late"):
        graph = matched_block[profile]
        role_map = eligible_role_assignment(graph, latent, condition)
        x = scatter_roles_and_nuisance(graph, latent, role_map, condition)
        save_or_yield(graph, x, target, block_id, condition, example_id)
```

The target must be exactly equal across matched growth profiles. Where specified below, it is also identical across task conditions. Use an independent role bank for each construction block. Within a fixed condition and node count, retain the same multiset of nuisance values across profiles, scattered by independent permutations. With different candidate counts, nuisance/candidate-role counts necessarily change; record that intervention rather than claiming all inputs remain identical.

Training draws can be generated online. Validation and test banks are fixed and disjoint from training. Use 128 validation draws and 512 test draws per construction block and task condition initially. Every model and training seed evaluates the same banks.

### 5.3 Assertions required before spending GPU time

1. Connected rooted tree, $N-1$ edges, correct orientation, exact size/depth, and legal degree counts.
2. Every source is at its declared radius and can reach the root in the appropriate number of layers.
3. Source keys/payloads/relevance roles and targets match exactly across paired profiles.
4. Targets and the zero-predictor errors match exactly across the paired conditions that promise common targets.
5. No isomorphic tree, matched block, or signal-bank overlap across splits; old inspected trees are excluded from confirmation.
6. Node relabeling preserves targets and model predictions within numerical tolerance, with all masks/features permuted consistently.
7. Analytic target oracles reproduce the labels; padding and non-upstream nodes cannot alter a prediction through a forbidden path.
8. A tiny fixed batch can be overfit by each intended task-capable model. Failure is diagnosed before interpreting an architecture gap.

Preserve the current experiment as version 1. A repaired generator, features, or model creates version 2; do not mix their metrics in one sweep.

## 6. Experiments and hypotheses

### Phase A — signal-transport sanity test: 9 fits

Train three alphas $\{0,.5,1\}$ with three seeds $\{42,43,44\}$. Each fit jointly sees balanced near/far conditions.

Select four relevant sources, one in each root branch, at distance $R=1$ or $R=4$. Each relevant source has a relevance flag visible locally. Use the same four scalar values in both radius conditions and across all three profiles:

$$
y=\frac{x_1+x_2+x_3+x_4}{\sqrt4},\qquad x_i\sim\mathcal N(0,1).
$$

Remove the previous root-only near/far query. All other nodes carry independent nuisance values with relevance flag zero. The selected sources know which values matter. The target variance is one and matched target energy is exact.

Include a zero predictor, an exact marked-sum pooling oracle, and a deterministic local propagation oracle using separate hop channels. These require no learned fits. They demonstrate that local communication can solve this task without attention.

**Development gate:** MP should attain validation MSE below 0.02 separately on both radii, and below $10^{-4}$ on a tiny fixed training batch. These are engineering acceptance thresholds, not a theorem. If it fails, audit reach, counting, repeated-message accumulation, gradients, normalization, and optimization. Do not interpret its failure as an oversquashing result or continue a confirmatory comparison with a knowingly weak endpoint.

A null growth effect here is expected to be plausible: this task admits a small sufficient statistic. Record the near/far growth interaction as a secondary result, not the main compression claim.

### Phase B — addressed retrieval: 25 main fits

Train all five alphas with five seeds $\{42,43,44,45,46\}$. Each model jointly trains across two balanced candidate-count conditions.

| Item | Main specification |
|---|---|
| Candidate distance | $R=4$ |
| Candidate count | $K=4$ or $16$ |
| Payload | Independent $v_i\in\mathbb R^8$, standard-normal coordinates |
| Keys | Fresh distinct 16-dimensional Gaussian vectors, normalized to unit length then multiplied by 4 |
| Query | Exact key of one candidate, supplied only at the root |
| Target | That candidate's eight-dimensional payload |
| Source distribution | One candidate per root branch for $K=4$, four for $K=16$ |
| Loss | MSE averaged over eight coordinates and examples |

Construct the paired K conditions exactly as follows:

1. Draw 16 semantic records, four assigned to each root branch, with independent keys and payloads.
2. Uniformly choose one record in each branch to form the four-candidate subset.
3. Uniformly choose the queried record from those four. Across examples, every one of the 16 records is equally likely to be queried.
4. K=16 activates all 16 records; K=4 activates the selected four. For the other 12, change only the candidate flag to zero: retain their key/payload channels as matched nuisance information.
5. Randomly map the 16 records to eligible distance-four nodes within their branch, independently of their values, keeping that placement identical across K. Across profiles, preserve semantic branch/record identity even though physical node IDs differ.
6. Copy the selected key into the root query and its payload into the target for both K values and every profile.

Save and audit selection frequencies. No record is permanently special and no high-K candidate is permanently unqueryable. Keys are regenerated, so the model cannot memorize a finite dictionary. Multiplication by 4 gives key/query coordinates variance approximately one, comparable to payload coordinates; this scale is identical for all models. Audit exact key uniqueness and the query-to-nearest-other-eligible-key cosine margin. Increasing K can change this margin as well as association count. An orthogonal-key control could isolate that distinction, but would be a separately frozen key distribution, not a post-result replacement.

Non-candidates also have random payload/key channels, with candidate flag zero. The query reveals no payload. An exact key-match oracle has zero error. A zero predictor has expected MSE one. A learned query-conditioned set model is required in Phase C.

**Hypothesis:** structural growth can interact with the number of eligible associations that must be preserved. No direction is guaranteed. K changes eligibility and the opportunity to filter records locally, while attention still sees the full node set in both conditions. It is not a pure measurement of compression capacity or an isolated attention-dilution intervention. These are conditional scores of models trained jointly on the balanced K mixture, not separately optimized models for each K.

This is a **moderate-load test**. Four candidate payloads per root branch and width 64 may be easy, especially with eight layers for distance-four sources. A null result is scientifically acceptable. $Kp/d$ is a descriptive proxy, not a proven capacity bound: messages can cross a cut on several rounds, and receiver states can accumulate information.

### Phase C — explanation controls: 15 fits

| Control | Fits | What it distinguishes |
|---|---:|---|
| Query broadcast to every node: 3 alphas × 3 seeds | 9 | Whether local filtering removes the root-only association burden |
| Bidirectional MP, $L=8=2R$, 3 seeds | 3 | Whether sending the query outward and a response inward changes the result |
| Query-conditioned set cross-attention, 3 seeds | 3 | Whether unconstrained query access solves retrieval without using topology |

Use the same retrieval examples and keys, with control seeds 42/43/44 paired to the corresponding main seeds. Broadcast changes the location of query values, not input dimensionality or encoder size. The bidirectional control includes explicit direction features and a count-preserving sum over incoming relations; its changed topology and any parameter differences must be reported. Compare it with the already trained inward MP at the same eight-layer depth. Check the actual outward-query/inward-response paths; eight layers create the nominal opportunity to learn them, not a guarantee of learning.

The set model encodes every node record independently, uses the root query as its readout query, and attends directly to **all real node records**, including nuisance nodes. It receives the same key/payload/candidate information and no node IDs. It does not receive a privileged candidate-only mask. Tune it sufficiently to establish a credible global-access control.

**Crucial interpretation:** unique-key retrieval does not intrinsically require a tree to determine the target. It tests communication constrained to a graph. Broadcasting changes the available communication protocol; it is not a free repair to the original root-only task. These controls make that limitation explicit.

### Prespecified extensions, separately budgeted

- **Communication rounds:** repeat MP, alpha .5, and attention at $L=R=4$, three seeds: 9 fits. This is necessary before a strong compression-versus-extra-rounds interpretation.
- **Capacity control:** choose endpoint widths that match the hybrid's active parameter count within 5%, then run MP and attention with five seeds: 10 fits. If an interior hybrid advantage is a headline result, this is required before claiming the mix itself explains it. Equal parameters do not imply equal FLOPs; report both.
- **Higher demand:** a separately preregistered width/payload/candidate-concentration factorial can use width 32 versus 64 and payload dimension 8 versus 16. Candidate concentration behind one root branch requires its own feasible graph corpus. Run it regardless of whether the moderate-load result is null, if it was committed in advance; do not selectively add difficult cases to manufacture a desired effect.
- **Dilution intervention:** fix candidate records, query, payload, and relevant routes, then add irrelevant nodes in a declared location. This changes graph size and is a different experiment from the fixed-N growth comparison. Increasing K alone entangles association load with distractor competition.
- **Exact parameter-parity operator control:** share Q/K/V projections between local-mask and global-mask attention and mix their outputs. This tests attention-support mixing, not the same sum-MP architecture.
- **Learned gate:** only after fixed-alpha behavior is understood; compare with the best validation-selected fixed coefficient, include gate overhead, and assess held-out regret.

## 7. Training protocol, fair comparisons, and compute budget

The 49 core fits are **9 + 25 + 15**. They exclude hyperparameter trials, capacity/depth extensions, and all real-data training. Do not describe 49 as the complete publication budget.

Proposed synthetic training defaults, finalized using development data only:

- AdamW; learning-rate candidates $3\times10^{-4},10^{-3},3\times10^{-3}$; weight decay $10^{-4}$; gradient clipping at norm 1.
- Batch size 32; maximum 6,000 optimizer updates; evaluate fixed validation banks every 500 updates; minimum 2,000 updates before stopping; patience four evaluations. An improvement means an absolute validation-MSE decrease greater than $10^{-5}$; ties retain the earlier checkpoint. Count patience in validation checks, and log K-specific trajectories even though stopping uses the balanced metric.
- Sample strata, growth profiles, and task conditions uniformly, then sample a training graph within the cell. Use fresh signals online and the same logical example stream across compared models.
- Select checkpoints by validation loss with equal weighting over size/depth, profile, and task condition. Test once after model/protocol choices are frozen. Keep the shared confirmation graph bank sealed across phases A–C: Phase A validity gates use validation only, and its test scores are released with the final evaluation. If test outcomes cause a design revision, reserve a new confirmation bank for that revision.
- Record gradients before clipping, branch RMS, learning rate, valid example count, and losses by condition. Save every best checkpoint and the complete training history.
- Keep precision mode identical. Validate full precision first; if mixed precision is adopted for speed, make it a shared development decision and check for numerical failures.

For the retrieval main sweep, give each alpha the same three-learning-rate screening budget on a separate development seed, for example 2,000 updates per candidate: **15 short pilot fits**. Select each alpha's learning rate on the same validation criterion, then freeze it for the five confirmation seeds. This estimates performance under a bounded equal search allowance, not globally optimized performance. Do not discard an unfavorable seed.

Controls need their own recorded tuning allowance. A failed set or bidirectional baseline cannot be dismissed after only borrowing the hybrid's settings. Capacity-matched endpoints also require new tuning; count those trials separately.

The following illustrates the budget honestly:

| Work package | Full fits | Short pilot fits |
|---|---:|---:|
| Core phases A–C | 49 | Not included |
| Main retrieval learning-rate screen | 0 | 15 |
| Parameter-matched endpoints | 10 | At least 6 for the same 3-rate screen |
| Four-layer comparison | 9 | Additional only if its recipe needs selection |
| Real-data pilots and other controls | Separate budget | Separate budget |

The prior 21-fit tree sweep recorded about 49 minutes, but its training/evaluation workload was different. It does **not** imply this enlarged study takes $49/21$ times as long. Each main validation pass has 96×2×128 = **24,576 examples**; each main test has 384×2×512 = **393,216 examples**. Across 25 main fits, twelve validation checks and one test amount to approximately 7.37 million validation and 9.83 million test predictions. Benchmark 200 representative updates and a complete validation pass on the 127-node case before scheduling the sweep:

$$
T_{run}\approx n_{updates}t_{update}+n_{val}t_{val}+t_{test}+t_{checkpoint}.
$$

Report measured median and slow-case times, multiply by the planned run inventory, and add an explicit contingency. At the proposed cap, 25 main fits alone represent 150,000 optimizer updates before validation and pilots. Keep the Ubuntu laptop powered, ventilated, and awake; screen locking is different from system suspend. Reuse the working CUDA environment rather than replacing it during the experiment.

If the measured budget exceeds the week, complete implementation, validity gates, and development pilots first. Do not quietly reduce seeds, alter the test bank, or drop controls after seeing test results.

## 8. Statistical analysis and decision rules

### 8.1 One primary contrast

Use fixed alpha .5 for the main mechanism test. It is prespecified rather than selected to win on test data. For profile $p$ and candidate count $K$, define

$$
G_{p,K}=\operatorname{MSE}_{MP,p,K}-\operatorname{MSE}_{H(.5),p,K}.
$$

Compute the following contrast **within each matched block and training seed**, then average blocks equally within each $(N,D)$ stratum and average the four strata equally:

$$
\Theta=\operatorname{mean}_{N,D}\left[
(G_{early,16}-G_{late,16})-(G_{early,4}-G_{late,4})\right].
$$

Use a two-sided interval/test. Positive $\Theta$ means the early-versus-late intervention increases hybrid-over-MP gain more under the higher candidate count. Negative $\Theta$, or an interval close to zero, is a valid outcome. This is **not a primary test of source distance**: R is fixed in retrieval. The near/far transport contrast addresses distance separately.

A gain over MP alone does not establish hybrid superiority. Also report the same contrasts against attention and all absolute losses. If attention matches or exceeds the hybrid, the finding concerns global access rather than a benefit from combining both branches.

### 8.2 Alpha selection and practical value

Report all five alpha curves and every seed. Separately choose one global alpha using balanced validation error averaged over confirmation seeds. Evaluate that fixed selection on test; keep it distinct from the fixed-.5 primary contrast. If an endpoint is selected, report it.

Do not select a separate test-optimal alpha for each test graph and call that a predictive rule. An oracle curve can be shown explicitly as a retrospective upper bound, accompanied by uncertainty in the apparent minimizer.

Predeclare a practically meaningful effect, for example **0.01 MSE** on the unit-variance synthetic target, before confirmation. This is a proposed engineering threshold requiring agreement during development, not a universal scientific cutoff. Report absolute differences; relative percentage improvements can look enormous when a baseline is near chance or an oracle is near zero.

### 8.3 Independent units, intervals, and power

- Resample matched construction blocks within size/depth strata, preserving all profiles, K conditions, common examples, and paired model evaluations.
- Report a block bootstrap conditional on fitted models and a crossed block-by-training-seed bootstrap. Five seeds give limited information about training variability; show the five individual contrasts.
- Never treat node rows, repeated targets, 512 examples, or five seeds as 512×5 independent graph interventions.
- The shared training corpus is itself one draw. These intervals do not establish robustness across independently drawn training corpora; a later generator/training-corpus replication is needed.
- Use development pilot contrasts to estimate precision and choose the graph count before opening the fresh test. More evaluation examples reduce Monte Carlo noise; more blocks address structural variation; more seeds address training variation. None substitutes automatically for the others.
- If several secondary contrasts receive significance tests, label them exploratory and use a declared multiplicity correction such as Holm. Do not treat every radius, seed, and alpha as an independent confirmatory discovery.

The proposed 128 test blocks and five seeds are pragmatic starting numbers, **not a power calculation or guarantee**. If the development pilot cannot support the desired precision, change and freeze the sample plan before confirmation.

### 8.4 Mechanism diagnostics

| Competing explanation | Required evidence/check |
|---|---|
| Insufficient reach | Actual paths, masks, layer count; deliberate $L<R$ negative control |
| Count or aggregation mismatch | Sum sanity test, mean-plus-degree diagnostic, exact local oracle |
| Optimization failure | Tiny-batch fit, train/validation curves, equal tuning allowance |
| Positional advantage | Remove attention-only distances; shared-feature ablation |
| Root-only query restriction | Broadcast and bidirectional controls |
| Finite representation/communication burden | K/p/width/round controls; multiple-query responses |
| Vanishing sensitivity | Trained source derivatives compared with known oracle derivatives |
| Oversmoothing | Role-aware representation variance/similarity across layers |
| Attention dilution | Controlled irrelevant-node intervention, target attention mass, off-source sensitivity |
| Extra capacity | Active-parameter-matched endpoints and measured compute |

For summation the oracle derivative for each relevant scalar is $1/2$. For retrieval the selected payload derivative is an identity matrix and non-selected payload derivatives are zero. Report derivative error relative to these oracles, plus finite perturbation responses.

For a fixed key/payload bank, evaluate every possible query and stack the resulting input–output Jacobians. This examines whether all associations are preserved. A single scalar-output Jacobian is rank at most one by construction, so its low rank alone cannot diagnose compression. Sensitivity may also be small because the model is untrained, saturated, or using a proxy.

Attention entropy is descriptive, not proof of information use. Post-training branch ablation changes the distribution seen by the remaining computation and is not equivalent to retraining an endpoint. Bottleneck and vanishing-gradient explanations should remain separate [6].

## 9. Real-world datasets: prioritized, verified options

The factual descriptions below were checked against original papers, official records, or authors' code. Acquisition protocols and proposed uses are our design recommendations. No new dataset census, completeness check, or large download has yet been performed.

| Priority | Dataset | Natural representation and real target | Best role |
|---|---|---|---|
| 1 | LamaH-CE | Connected gauge/catchment network; measured discharge | Immediate continuation, subject to fresh-group availability |
| 2 | ogbg-code2 | Python syntax trees; developer method-name tokens | Independent real-label domain with many graph shapes |
| 3 | LamaH-Ice | Icelandic gauge/catchment networks; measured discharge | Geographic replication after topology census |
| 4 | Stanford Sentiment Treebank | Sentence parse trees; human sentiment labels | Small laptop development pilot |
| Later | USGS streamgages + hydrography | Constructed gauge network; measured discharge | Larger hydrology program with more engineering |

### 9.1 LamaH-CE: reuse the data pipeline, replace the weak evaluation sample

**Verified data/access:** the original paper describes 859 gauged catchments. Delineation B supplies intermediate catchments and explicit topology fields including `NEXTUPID`/`NEXTDOWNID`. The versioned daily archive is approximately 1.5 GB compressed and 5 GB unpacked. The record states CC BY-SA 4.0 and additional source-specific conditions, including a restriction on operational warning use of Czech runoff. Obtain data from [Zenodo record 5153305](https://zenodo.org/records/5153305); consult the [original paper](https://essd.copernicus.org/articles/13/4529/2021/) [7,8].

**Proposed acquisition and evaluation:**

1. Reuse the existing raw version and record checksums. Read gauge/topology/transfer metadata before time-series downloads or model selection.
2. Build hydrological dependency groups including nested networks and documented transfers. Exclude every previously inspected group's membership from the fresh confirmation pool. Disjoint node sets alone are insufficient.
3. Produce a census of complete upstream trees: N, D, full V profile, source-path degrees, represented area, river lengths, gauge density, transfer status, and concurrent history coverage.
4. Select eligible graphs with rules based on topology and data coverage, without looking at architecture errors. If too few independent groups or too little growth variation remain, stop the proposed confirmatory river test and use it as development only.
5. Split independent groups and chronological periods explicitly. All input observations precede the forecast issue time; all scored targets belong to the declared target period. Fit feature transforms and imputation on training data only.
6. Compare weather-only and weather-plus-flow separately at 1/3/7 days, retaining a common temporal encoder/history allowance across spatial architectures. Add outlet-only temporal, persistence, and upstream pooling baselines. Record the depth of every actual graph: if sources lie beyond an eight-layer MP receptive field, a performance difference can reflect insufficient reach. A depth/reach control is required before calling it oversquashing.
7. Report per-outlet RMSE in physical units, declared normalized RMSE, NSE, and optionally KGE. Predeclare low-variance exclusions and normalization. For continuity, the previous global-training specific-discharge scale can be retained; do not fit input normalization to held-out labels.
8. Aggregate equally by outlet or dependency group according to the preregistered question. Use dependency groups and time blocks for uncertainty, not prediction rows as independent samples. Calendar blocks must remain aligned across groups/outlets to preserve shared-storm dependence; choose block length using development data. Hydrological grouping does not eliminate regional weather dependence.

Growth here counts gauges. It can reflect station spacing rather than physical branching. Candidate adjustment variables are N, D, drainage area, gauge density, climate, and input regime; compare hop growth with physical-distance/represented-area descriptors. The independent-group census determines whether such an adjusted model is estimable. With only a few groups, report descriptive/stratified associations rather than fit a many-covariate regression and claim confounding is removed. Exclude runoff-derived catalog signatures from pre-training predictors unless recomputed using permitted training years. Correlated weather and local flow history can substitute for distant information, so architecture gains alone do not demonstrate upstream causal transport.

### 9.2 ogbg-code2: strongest second domain

**Verified data/access:** OGB describes 452,741 Python ASTs, averaging 125.2 nodes, from 13,587 repositories, with project-separated splits and method-name subtoken F1. OGB lists MIT as the dataset license. Use `ogbg-code2`; it masks leakage present in the deprecated `ogbg-code` version [11]. The official example adds reverse AST and bidirectional next-token edges, so its processed graph is not a strict tree [12].

Official dataset: [OGB graph-property documentation](https://ogb.stanford.edu/docs/graphprop/#ogbg-code2). Acquisition starts with the official loader in a separately checked environment:

```python
from ogb.graphproppred import GraphPropPredDataset, Evaluator
dataset = GraphPropPredDataset(name="ogbg-code2", root="data/ogb")
split = dataset.get_idx_split()
evaluator = Evaluator(name="ogbg-code2")
```

This is an intended acquisition snippet, not a download executed for this report. Preserve the working CUDA installation and verify dependency compatibility before installing additional packages.

**Proposed protocol:** preserve official splits, and maintain separate configurations for raw-AST trees and the standard augmented graph. Preserve child order/edge roles; unordered aggregation can remove program semantics. Use the same token vocabulary, masking, features, head, and supervision across models.

Start with a reproducible label-blind development subset and node-count bucketing for laptop memory. Report size cutoffs and excluded graph counts. Subset/raw-tree scores are not full OGB leaderboard results. Include a tuned MP/virtual-node baseline and token-pooling baseline. No full code2 sweep is promised in the first week; acquisition, project metadata, and subset throughput are feasibility gates.

Standard graph pooling provides direct access from every node to the output, which can bypass a single-root bottleneck. A root-readout experiment is a useful separate mechanistic variant, but a nonstandard benchmark protocol. Report growth on the adjacency actually used, and raw-tree growth separately if augmentation is enabled.

Analyze paired model performance differences against growth with controls for N, depth, token count, node-type composition, token rarity/OOV rate, and target-name length. Target-name length is a retrospective diagnostic involving the label; it must not enter a deployment or before-training selector. Official project separation is verified; a convenient graph-to-project mapping in the downloaded package is **not yet verified**. Recover it before claiming project-clustered intervals. If unavailable, disclose the limitation rather than treating all methods as independent projects.

### 9.3 LamaH-Ice: new geography, not guaranteed structural sample size

**Verified data/access:** the original paper describes 107 basins, 51 nested. The v1.5 record adds gauges, so 107 is not asserted as its current total. Daily meteorology/discharge and intermediate-catchment topology are available; daily files occupy roughly 2 GB unpacked. Streamflow is specifically CC BY-NC 4.0; other data are CC BY 4.0, despite the record's general CC-BY badge. Access [v1.5 on HydroShare](https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/) and the [original paper](https://essd.copernicus.org/articles/16/2741/2024/) [9,10].

**Proposed use:** run a topology-only census, followed by a time-series availability and quality-mask audit. Count independent terminal groups, depth/growth support, and overlapping valid histories. Static metadata alone may not establish concurrent usable observations. Many gauges can belong to shallow or nested networks; basin count is not an independent graph count. Freeze model choices using other development data, then declare whether Icelandic data support external testing alone or a separate adaptation/validation/test split.

If structural diversity is weak, use Ice to test forecast transfer rather than the primary growth claim. Snow, glaciers, ice-related missing observations, and different meteorological errors can explain geographic differences. Keep those limitations in the result interpretation.

### 9.4 Stanford Sentiment Treebank: inexpensive implementation pilot

The original dataset contains 11,855 sentences and 215,154 labeled phrases, represented with binary parse trees [13]. Obtain the official train/dev/test trees from [Stanford's dataset page](https://nlp.stanford.edu/sentiment/code.html). The inspected official material did not establish an explicit dataset redistribution license; retain and inspect the supplied terms before redistributing it.

Our proposed task predicts root sentiment from leaf tokens and the parse structure. **Remove all sentiment labels from input features**, including internal and leaf labels. Keep all phrases of a sentence within its existing split. Start with root-only supervision; if phrase supervision is added, give every architecture the same labels.

Preserve left/right child order and include an ordered recursive/Tree-LSTM baseline plus a sequence or token-pooling baseline. Binary branching restricts small-radius volume variation, so inspect the full growth profile and adjust for sentence length, depth, vocabulary, negation, and target class. True target class is for retrospective stratification only, never a selector input. Tree shape and language content are coupled; this is observational validation, not evidence that physical signals traverse a parse tree.

### 9.5 USGS + NLDI: expansion if packaged river data are insufficient

Official NLDI documentation supports upstream/downstream navigation of indexed hydrological features; USGS water-data APIs supply station metadata and daily/continuous measurements [14,15]. Start with [NLDI navigation](https://api.water.usgs.gov/docs/nldi/navigation/) and [water-data API documentation](https://api.waterdata.usgs.gov/docs/ogcapi/).

Our proposed pipeline selects separate terminal drainage systems, discovers upstream gauges, retrieves flowlines, builds verified adjacency, resolves gauge/reach placement, and downloads quality-flagged historical discharge. Navigation membership alone is not an adjacency matrix. Begin with flow-history forecasting; spatially matched weather requires a separate catchment/forcing pipeline.

Diversions can produce non-tree systems; reservoirs alter storage and travel times even when topology remains a tree. Retain only verified arborescences for a strict tree study, or declare a DAG study; do not silently delete inconvenient edges. Station counts, download completeness, and usable independent systems have not yet been established. This is a later engineering project, not a one-day substitute for LamaH.

### 9.6 Semi-synthetic bridge on real trees

Before a large real-label sweep, extract verified river trees or raw ASTs and place the same sum/retrieval tasks on eligible shapes. Match role values and target vectors, record source distance/concentration, and split by hydrological group or software project. Natural graphs may not support exactly matched N/D/growth cells; report the resulting observational structural comparison rather than forcing a match by undocumented pruning.

This checks robustness to realistic topology while preserving known task dependencies. It does **not** establish real flood forecasting or real code-understanding performance.

## 10. Predicting a useful mix before training: a later, separate experiment

There are two distinct goals:

1. **Choose among already trained alpha models for a new graph.** A bank selector can use graph features and training/validation evidence to predict which fitted model to use at inference.
2. **Predict an architecture before training on a new graph/task dataset.** This requires many separate datasets, each with its own independently trained alpha sweep, and dataset-level splits for the selector.

The 25-fit main sweep studies the first ingredients, not the second goal in full. It trains one model per alpha/seed across a corpus; hundreds of test trees do not create hundreds of independent architecture-training experiments.

A later selector should use features available at its decision time: growth profile, N/D, cut descriptors, known synthetic range/load, and permitted task metadata. A task-range estimate fitted using a pilot model has a training cost and must use training-only data; it cannot be described as a free purely structural pre-training feature.

Predict the loss of each alpha or a set of near-optimal choices rather than a brittle single best-alpha class. Evaluate held-out **selection regret**, computation cost, and calibration against global fixed alpha, task-specific fixed alpha, and simple N/D rules. Permit the conclusion that graph growth adds no predictive value beyond those controls.

## 11. Independent agent critique and how it changed this plan

Four agents reviewed separate parts of the problem. Their judgments were advisory; the final choices below reconcile their disagreements rather than count votes.

| Reviewer role | Criticism or finding | Decision in this plan | Remaining limitation |
|---|---|---|---|
| Model/mechanism reviewer | Mean aggregation, normalization, hop-bias asymmetry, and unequal active capacity weaken the old comparison | Use sum MP, preserve amplitude, remove attention-only bias, budget matched endpoints | Sum aggregation still does not guarantee successful optimization |
| Model/mechanism reviewer | A four-value sum is compressible and weak evidence for oversquashing | Move summation to a sanity phase; make retrieval the main test | Retrieval is topology-invariant given global access |
| Statistics reviewer | Different role-value draws can masquerade as a growth effect | Enforce semantic pairing and identical target-energy assertions | Other shape properties change with growth intervention |
| Statistics reviewer | Proposed a near/far transport interaction as primary | Retain that contrast as secondary; use growth-by-candidate-load as the main retrieval contrast | The primary now fixes range; it cannot alone establish range dependence |
| Statistics reviewer | Few graph blocks, crossed seeds, and test-selected alpha create unstable claims | Increase matched blocks, use five main seeds, prespecify alpha .5, separate validation selection | One training corpus and five seeds still limit generalization |
| River-data reviewer | Gauge volume is affected by instrument placement; prior held-out groups were few | Require fresh-group census, physical-distance/area descriptors, input-regime separation | Independent eligible groups may remain insufficient |
| Real-tree-data reviewer | Official code2 preprocessing is not a tree; graph pooling bypasses a root bottleneck | Separate raw-tree and augmented benchmark protocols and readout variants | Real-label structure correlations remain confounded |
| Model/mechanism follow-up | Balanced K16/p8/width64 with eight layers may not impose a strong bottleneck | Label it moderate load; preregister width/round/concentration extensions separately | A null moderate-load result cannot rule out harder regimes |
| Final statistics review | Triplet formation and query-key geometry need an auditable specification | Generate disjoint matched blocks with fixed topology bins; log key margins and exact query-role frequencies | The growth treatment changes multiple shape properties and K changes eligible-key competition |
| Final model review | A candidate-only set mask would give the control privileged filtering | Make learned set attention see all real node records and the same role flag | Global access remains an intentional computational advantage |
| Final data review | Metadata cannot establish simultaneous valid histories, and storms create dependence across basins | Add availability/quality audit and synchronized time-block resampling | Small group counts may still prevent an adjusted structural analysis |

One model reviewer proposed initialization-RMS branch calibration as the main treatment. This plan uses unit branch constants for the simplest primary model and makes calibration a declared robustness check. It logs branch RMS because neither choice makes alpha a causal attribution.

No reviewer supplied evidence that a hybrid must win. The experiment is designed to make an MP win, attention win, hybrid win, or no structural relationship equally reportable.

## 12. One-week execution plan and stopping rules

| Day | Concrete work | Deliverable/gate |
|---|---|---|
| 1 | Freeze question and estimand; implement graph/role pairing and new layer; inspect topology feasibility | Protocol/config hash, graph census, oracle and permutation checks |
| 2 | Run tiny-batch checks, transport development fits, and throughput benchmark | Credible MP endpoint; measured run-time budget; no fresh test inspection |
| 3 | Complete transport sanity phase and bounded retrieval tuning | Frozen optimizer/checkpoint policy; decision whether compute budget supports the core sweep |
| 4–5 | Run main retrieval sweep and explanation controls as budget allows; metadata-only real-data census in parallel | Checkpoints, histories, complete planned-run manifest, dataset feasibility table |
| 6 | Evaluate fresh test once; compute paired contrasts, intervals, and diagnostics | Full alpha curves, per-seed/per-stratum effects, failure analyses |
| 7 | Independent review of results; initiate one feasible semi-synthetic or real-data pilot | Report supported/null findings and a specifically budgeted next stage |

This is a dependency schedule, not a promise that all 49 full fits plus tuning finish in seven days on the laptop. If validation or compute gates fail, the week's successful outcome can be a validated implementation and a corrected budget.

Stop or revise before confirmation if the topology constraints lack support, pairing assertions fail, MP cannot solve the transport sanity task, a credible global retrieval baseline cannot learn, or the real-data census cannot support independent evaluation. Freeze and version any revision. A null hypothesis result after valid execution is not a reason to change the protocol retroactively.

## 13. Required saved outputs and acceptance criteria

Save these with the new experiment; the filenames are an implementation checklist, not claims that these files already exist:

```text
protocol.md
config.yaml
environment.txt
code_hashes.json
graphs.json
graph_census.csv
split_manifest.json
role_bank_manifest.json
pairing_assertions.json
sweep_plan.json
runs/<run_id>/checkpoint_best.pt
runs/<run_id>/training_history.csv
runs/<run_id>/result.json
analysis/absolute_scores.csv
analysis/paired_block_seed_contrasts.csv
analysis/primary_interval.json
analysis/alpha_curves.csv
analysis/mechanism_diagnostics.csv
analysis/capacity_and_runtime.csv
data_sources.md
final_report.md
```

Record exact software/CUDA versions, dataset versions, graph-generation seeds, role-bank seeds, training seeds, accepted/rejected graph counts, early-stopping choices, and all attempted tuning configurations. Preserve enough information to recreate every prediction's graph, semantic roles, and target. Avoid deleting unfavorable runs.

The next report should be able to answer:

1. Did the new MP endpoint pass the signal-transport sanity test?
2. Is the prespecified growth-by-load interaction distinguishable from zero and practically meaningful?
3. Does any hybrid improvement survive comparison with attention, credible global pooling, and matched capacity?
4. Do query delivery, width, or communication rounds explain the effect?
5. Does the finding transfer to real tree shapes, then to real labels under independent splits?
6. Which conclusions remain unsupported, including a universal pre-training alpha predictor?

## 14. Sources and provenance

External sources were checked on 2026-10-05. Descriptions above distinguish verified dataset facts from proposed experimental choices. This is a focused source review, not an exhaustive novelty search.

1. Rampášek et al. **Recipe for a General, Powerful, Scalable Graph Transformer (GraphGPS).** https://arxiv.org/abs/2205.12454
2. Alon and Yahav. **On the Bottleneck of Graph Neural Networks and its Practical Implications.** https://arxiv.org/abs/2006.05205 ; authors' code: https://github.com/tech-srl/bottleneck
3. Bamberger et al. **On Measuring Long-Range Interactions in Graph Neural Networks.** https://proceedings.mlr.press/v267/bamberger25a.html
4. Xu et al. **How Powerful Are Graph Neural Networks?** https://arxiv.org/abs/1810.00826
5. Di Giovanni et al. **On Over-Squashing in Message Passing Neural Networks: The Impact of Width, Depth, and Topology.** https://proceedings.mlr.press/v202/di-giovanni23a.html
6. Mishayev et al. **Short-Range Oversquashing.** https://arxiv.org/abs/2511.20406
7. Klingler et al. **LamaH-CE dataset paper.** https://essd.copernicus.org/articles/13/4529/2021/
8. **LamaH-CE versioned data record.** https://zenodo.org/records/5153305
9. Helgason and Nijssen. **LamaH-Ice dataset paper.** https://essd.copernicus.org/articles/16/2741/2024/
10. **LamaH-Ice v1.5 data record.** https://www.hydroshare.org/resource/705d69c0f77c48538d83cf383f8c63d6/ ; authors' preprocessing: https://github.com/hhelgason/LamaH-Ice
11. **Official OGB graph-property prediction documentation: ogbg-code2.** https://ogb.stanford.edu/docs/graphprop/#ogbg-code2
12. **Official code2 preprocessing and training example.** https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/code2/utils.py ; https://github.com/snap-stanford/ogb/blob/master/examples/graphproppred/code2/main_pyg.py
13. Socher et al. **Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.** https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf ; data: https://nlp.stanford.edu/sentiment/code.html
14. **USGS NLDI navigation documentation.** https://api.water.usgs.gov/docs/nldi/navigation/
15. **USGS water-data OGC API documentation.** https://api.waterdata.usgs.gov/docs/ogcapi/

Project evidence: the previously saved `RIVER_GROWTH_EXPERIMENT_REPORT_2026-10-04.md`, reviewed experiment code, and supplied controlled-tree archive `tree_growth_review_20261004T213627_943969Z.zip` (SHA-256 `24847e4d59235f1793c3c6cd4190e70402f0e06829d2b842838e346a4d1ede8d`). The new design deliberately does not reuse its already inspected test trees as fresh confirmation.
