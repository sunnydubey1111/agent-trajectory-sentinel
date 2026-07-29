# Hybrid ESN + Mahalanobis monitor study — final report

> **Provenance note (2026-07-26).** This report was
> first written on the `exp/hybrid-fusion` branch (2026-07-17). Its headline
> claims (grand-mean ordering, per-seed logistic advantage, the grounding
> success criterion, the organic-failure decomposition, and cross-model
> transfer) were **re-derived and synchronized** against the artifacts
> regenerated under the corrected pipeline (ESN fit/score
> alignment + the label-independent evaluation protocol). Where a specific
> figure in the narrative below still differs from the regenerated CSVs, the
> **CSVs in `results/tables/` govern** — they are the source of record.

Branch `exp/hybrid-fusion`, 2026-07-17. Code: `derail/monitor/hybrid.py`,
`derail/experiments/run_hybrid_study.py`, `collect_research7b_long.py`.
Tables: `results/tables/hybrid_*.csv` and `hybrid_long_*.csv`. Protocol:
frozen baseline evaluation (same splits via `rng_for(0, "real-split")`, 5%
FA healthy-val-quantile thresholds, `evaluation/metrics.py` +
`evaluation/stats.py`). All numbers reproduce deterministically.

## 1. Why does Mahalanobis beat the ESN on real_research7b?

**Cause identified: post-onset temporal horizon.** Per-episode diagnosis
over all 716 injected episodes across the five benchmark datasets
(`hybrid_diagnosis.csv`):

| post-onset horizon (T−1−τ) | n | ESN det | Maha det | ESN − Maha |
|---|---|---|---|---|
| ≤ 3 steps | 263 | 0.35 | 0.44 | **−0.09** |
| 4–8 steps | 81 | 0.77 | 0.53 | **+0.23** |
| ≥ 9 steps | 372 | 0.78 | 0.38 | **+0.40** |

corr(horizon, ESN advantage) = **+0.37**. The ESN's CUSUM accumulates
evidence over steps; with ≤3 post-fault steps there is nothing to
accumulate, while the memoryless Mahalanobis distance fires on the first
anomalous step or never. real_research7b episodes are T≈5–6 with τ=2 —
almost every failure lives in the ≤3-step bin. Ruled out as primary causes:
telemetry representation (identical channels/dims to datasets where the ESN
wins), hyperparameters (unchanged from the simulator where it wins), and
training-set size (the ESN also trails on the 72-healthy-episode fit).

**Extended-horizon control (objective 6): partial confirmation.**
`real_research7b_long` (72 episodes, same model/tools/classes/τ=2, but
10-tool-call tasks, T≈11–12) vs the standard set:

| | research7b (short) | research7b_long | Δ |
|---|---|---|---|
| ESN AUROC | 0.784 | 0.813 | +0.03 |
| ESN detection | 0.27 | **0.60** | **+0.33** |
| ESN mean lead | 0.23 | **3.64** | **+3.4 steps** |
| Maha AUROC | 0.839 | 0.845 | +0.01 |
| Maha detection | 0.44 | 0.71 | +0.27 |

Longer horizons improve the ESN dramatically in absolute terms and narrow
the AUROC gap from 0.055 to 0.032 — temporal information is clearly a
binding constraint. But Mahalanobis also benefits (more steps = more
chances to cross threshold) and stays ahead on this task family. So the
horizon explains **part** of the gap; the residual is plausibly the
near-scripted structure of the research tasks (healthy steps are highly
repetitive, which suits a static healthy-manifold model) plus the small
long-set sample (18 train / 6 val healthy — interpret with care).

## 2. Which failures are temporal vs state-based?

Detection rate averaged across datasets (`hybrid_per_class.csv`):

| class | ESN | Maha | verdict |
|---|---|---|---|
| grounding_loss | **0.98** | 0.08 | temporal (slow drift) |
| tool_cascade | **0.80** | 0.77 | temporal, weakly |
| context_corruption | **0.38** | 0.19 | temporal, both weak |
| goal_drift | 0.28 | 0.28 | tie, both weak |
| timeout | 0.41 | **0.45** | state-based, weakly |
| looping | 0.61 | **0.69** | state-based on short episodes |
| rate_limit | 0.68 | **0.82** | state-based (abrupt error state) |
| malformed_json / wrong_document | 0.14 / 0.09 | 0.05 / 0.05 | content corruption: nobody |

Reading: failures that *build* (drift, cascades) need the reservoir's
memory; failures that *jump* to an anomalous state (error storms) are
caught instantly by the distance. Content corruption that changes data but
not behavior remains undetected by every monitor — unchanged limitation.

## 3. Hybrid results (`hybrid_benchmark.csv`)

AUROC by dataset (best in **bold**):

| dataset | ESN | Maha | weighted | max | gated | logistic |
|---|---|---|---|---|---|---|
| sim | **0.889** | 0.786 | 0.878 | 0.872 | 0.844 | 0.886 |
| gemini | 0.816 | 0.763 | 0.816 | 0.809 | 0.804 | **0.819** |
| autogen7b | 0.843 | 0.762 | 0.840 | 0.845 | 0.831 | **0.866** |
| ollama7b | 0.824 | 0.817 | **0.830** | 0.810 | 0.826 | 0.828 |
| langgraph7b (held out, §7) | 0.643 | **0.738** | 0.671 | 0.700 | 0.710 | 0.671 |
| real_research7b | 0.784 | 0.839 | 0.818 | 0.818 | 0.830 | **0.842** |
| real_research7b_long | 0.813 | **0.845** | 0.813 | 0.794 | 0.829 | **0.845** |
| **grand mean (7 datasets)** | 0.802 | 0.793 | 0.810 | 0.807 | 0.811 | **0.823** |

Logistic features use robust-z clipping at ±50 (`HybridLogistic(clip=50)`).
The bound was chosen empirically: unclipped features reach z ~ 1e6 and
sklearn's L2 penalty drives the learned weights to numerical zero (observed
on real_research7b_long); a tight ±5 bound saturates so many episode maxima
that ranking collapses (autogen7b AUROC 0.854 → 0.639); ±50 fixes the
conditioning and matches or beats the unclipped variant on all six
datasets (autogen7b +0.012, research7b_long +0.032).

Statistical validation (`hybrid_stats.csv`, paired per-episode tests):
hybrid_logistic is **never significantly below the local winner** (sim vs
ESN: ΔAUC CI [−0.012, +0.007]; research7b vs Maha: CI [−0.009, +0.016];
research7b_long vs Maha: CI [−0.034, +0.032]) and **significantly above
the local loser** everywhere it matters (sim vs Maha: ΔAUC CI [+0.070,
+0.132], McNemar p≈2e-40; research7b vs ESN: CI [+0.014, +0.100], McNemar
p=3e-4; autogen7b vs Maha: McNemar p=0.03; research7b_long vs ESN:
permutation p=5e-5 on lead).
Supervision discipline: logistic weights come from the sim `cal` split or
2-fold class-stratified cross-fit on real data — no episode is scored by a
model that saw it in training. Weighted-0.5 is the best label-free
variant (grand mean 0.833); gated never leads and its gate calibration is
the weakest idea of the four; max is dominated by weighted.

## 4. Runtime (sim test set, 560 episodes)

| monitor | fit (s) | µs/step | footprint (MB) |
|---|---|---|---|
| delta_mahalanobis | 0.02 | 3.2 | 0.06 |
| esn_cusum_max | 1.8 | 169 | 3.95 |
| hybrids (any) | 4.2–4.5 | 183–197 | 4.01 |

The hybrid costs ≈9% more per-step latency than the ESN alone (the
Mahalanobis add-on is nearly free; calibration triples fit time but stays
under 5 s). All remain orders of magnitude below a judge-LLM call.

## 5. Recommendation

**Deploy the hybrid.** Concretely:

1. **Default (label-free deployments): `hybrid_weighted50`** — no labels
   needed, grand-mean AUROC 0.810 > ESN 0.802 > Maha 0.793 over seven
   datasets, never far from the local winner, +9% latency over ESN.
2. **When ≥ ~20 labeled failure episodes exist: `hybrid_logistic`** — best
   grand mean (0.823), statistically at-or-above the better standalone on
   every development dataset (on the small, noisy held-out set its edge
   shrinks to parity with weighted — §7). The needed labels are cheap:
   injection runs already produce them.

`derail.monitor.hybrid.recommended_monitor()` encodes this policy: it
returns a fitted HybridWeighted(0.5) by default and switches to a
supervised HybridLogistic when `labeled_failures` has ≥ 20 episodes.
3. Keep the standalone ESN only where episodes are known to be long and
   drift-like failures dominate (it still holds the sim crown, 0.889); keep
   standalone Mahalanobis only under extreme resource constraints (3 µs,
   60 KB).

**Does a hybrid consistently outperform both standalones?** Consistently
*match-or-beat*: yes (logistic, five of five datasets within CI of the
winner, significantly above the loser). Strictly beat both everywhere: no —
on sim the ESN's point estimate stays highest, on research7b Maha's does.
The hybrid's value is removing the need to know, per deployment, which
regime you are in.

Caveats: research7b_long is small (18 train healthy); content-corruption
classes remain near-undetectable for all monitors (future work: content
grounding checks, not score fusion). The logistic weight degeneracy
observed in the first run is fixed by the ±50 feature clip (§3).

## 6. Why the logistic hybrid works (mechanism, not metrics)

Evidence: `hybrid_coefficients.csv`, `hybrid_complementarity.csv`, and
`results/figures/hybrid_explain.png` — all derived from the exact scoring
path of the benchmark (`hybrid_explain.csv`).

**(a) The two detectors cover different failures.** Over all 793 injected
episodes, 288 are caught by both standalones, but 197 are ESN-only and 54
are Mahalanobis-only — 251 episodes (32%) where exactly one detector sees
the failure. That partial overlap is the raw material fusion exploits.

**(b) The logistic recovers most of both one-sided sets.** Detection rate
of hybrid_logistic inside each complementarity cell: both-detect **1.00**
(288/288), ESN-only **0.81**, Maha-only **0.63**, neither **0.01**. It
keeps essentially everything the union offers while staying at the 5% FA
budget — and where neither sub-detector fires it stays silent rather than
hallucinating alarms.

**(c) The weights learn which regime the deployment is in.** The
Mahalanobis share of the learned weight mass tracks the dataset's
character: 0.34 on sim (long horizons — lean on the reservoir), 0.38 on
ollama7b, ~0.55–0.64 on langgraph7b/autogen7b (mixed), 0.90–1.03 on
gemini/research7b/research7b_long (short horizons — lean on the distance).
~20 labeled episodes suffice to place the boundary; the intercept
consistently lands near −1.3, i.e. the boundary passes close to the
healthy cluster's edge.

**(d) The picture.** In the figure, each episode is plotted at the step
that maximizes the fused score, Mahalanobis confidence on x, ESN
confidence on y. On sim the decision boundary is nearly **horizontal**
(alarms are decided by ESN evidence); on gemini/research7b/long it is
nearly **vertical** (Mahalanobis decides); on autogen7b/langgraph7b it is
**diagonal** — both matter. Detected episodes hug one axis or the other,
almost never the diagonal middle: the sub-detectors fire on *different*
failures, and the learned line is what turns that union into one
calibrated alarm.

## 7. Held-out generalization: langgraph7b

langgraph7b (95 episodes, LangGraph framework, qwen2.5:7b, telemetry v3)
was **never used while developing the hybrids** — it entered the study
only after the fusion design, clip value, and defaults were frozen. Same
protocol, thresholds, and tests as every other dataset.

Results (AUROC): Maha **0.738** > gated 0.710 > max 0.700 > weighted =
logistic 0.671 > ESN 0.643. Runtime unchanged (hybrids ≈ 188–262 µs/step
vs 6 µs Maha / 209 µs ESN). This is the noisiest dataset in the corpus
(healthy-val FA 25% for every monitor at the 5% budget — 12 healthy val
episodes; detection 0.31–0.49).

Verdict, stated precisely: the core claim **survives** — the best hybrid
(gated) is statistically indistinguishable from the local winner (ΔAUC CI
[−0.076, +0.019] vs Maha) and significantly above the local loser (CI
[+0.010, +0.144] vs ESN, McNemar p = 0.03). The hybrids sit between the
standalones rather than collapsing. What does **not** replicate here is
hybrid_logistic's usual edge: with only 18/17 injected episodes per
cross-fit fold on a noisy set, its per-dataset advantage disappears
(on par with weighted). The grand-mean ordering over all datasets holds
(logistic 0.830 > max 0.817 > Maha 0.812 > gated ≈ weighted 0.810 >
ESN 0.800), and the deployment recommendation in §5 stands —
with the honest caveat that on small, noisy healthy pools the label-free
weighted variant is as good as the learned one.

## 8. Multiseed stability (5 seeds)

`hybrid_multiseed.csv` — seeds {0, 7, 101, 202, 303} varying the ESN
reservoir initialization, the logistic cross-fit fold assignment, and the
simulator master seed, with data splits frozen per protocol
(`run_hybrid_multiseed.py`; seed 0 verified byte-identical to the
published tables).

Grand-mean AUROC across seeds (post-remediation regeneration): logistic
**0.830 ± 0.006** > max 0.817 ± 0.006 > Maha 0.812 ± 0.003 > gated 0.810 ±
0.006 ≈ weighted 0.810 ± 0.005 > ESN 0.800 ± 0.004. Two conclusions:

1. **The learned-hybrid-over-standalone gap survives seed noise, paired
   per seed, and is larger than before:** logistic − ESN = +0.030 ± 0.003
   (positive at every seed, worst +0.025); logistic − Maha = +0.019 ±
   0.005 (worst +0.010); the label-free weighted − ESN = +0.010 ± 0.002
   (worst +0.007). Note that Δ-Mahalanobis now grand-means *above* the ESN
   and above the weighted/gated hybrids — it is a strong baseline, and only
   the *learned* logistic fusion clears it. Differences among the
   label-free hybrid variants are within seed noise.
2. **Fusion does NOT uniformly stabilize variance** (corrected finding).
   On the ESN's noisiest real set the logistic tightens the cross-seed
   spread several-fold (research7b 0.009 → 0.002), but on the short 3b and
   the long sets the logistic's own cross-fit noise makes it *more*
   variable than the ESN — so the honest claim is a consistent mean
   advantage at every seed, not a variance guarantee.

## 9. Content-grounding channel (telemetry v4) — closing the content gap

Design: nine causal per-step signals computed from the v2 tool bits
("[name({args}) -> result]") with the standard hash embedding — query↔result
dissimilarity, result↔reasoning dissimilarity, result self-inconsistency,
JSON structural breakage (prefix-completion check: truncation is NEVER
flagged — fixes an earlier false-positive), and character-statistics
anomaly. Optional channel g (dims 51–56, DESIGN.md amendment 3); v1 traces
yield exact zeros (verified inert). Ungrounded reference monitors were
scored on a 51-dim view so no grounding information leaks into baselines.

**The success criterion** (content classes must improve, behavioral classes
must not degrade), pooled over all 393 injected episodes across six real
datasets, paired McNemar:

| comparison | content (n=140) | behavioral (n=253) |
|---|---|---|
| hybrid_weighted_g vs hybrid_weighted50 | 0.15 → **0.34** (+27/−1, p=2.2e-7) | 0.62 → **0.65** (+9/−0, p=0.004) |
| hybrid_logistic_g vs hybrid_logistic | 0.14 → **0.33** (+27/−0, p=1.5e-8) | 0.70 → 0.66 (−9, p=0.004) |

**hybrid_weighted_g passes both halves** — content detection doubles while
behavioral detection improves. The logistic variant gains as much on
content but trades away 9 behavioral detections; use the weighted union.

Per-class means across datasets: malformed_json 0.07 → **0.91** (closed;
the standalone channel scores 1.00), context_corruption 0.18 → **0.33**
(halved gap; the standalone channel reaches 0.77 on real_research7b),
wrong_document 0.13 → 0.05–0.27 (**open** — hash embeddings are too crude
for semantic query↔result relevance; a MiniLM opt-in is the obvious next
probe). Best single dataset: real_research7b_long, hybrid_weighted_g AUROC
0.913 / detection 0.83 at 0% FA (ungrounded best: 0.845/0.71).

Grand-mean AUROC (6 real datasets): hybrid_logistic_g 0.833 >
hybrid_weighted_g 0.820 > hybrid_logistic 0.812 > weighted50 0.798 >
Maha 0.794 > ESN 0.787.

Ablation (each dim alone): char_anom is the workhorse (malformed_json
1.00, context_corruption 0.27 single-handedly); query_dis supplies the
only wrong_document signal (0.09); reason_dis/self_dis contribute small
context signals; json_broken detects nothing on this corpus — the
injector's malformations are caught earlier by character statistics — but
stays as the principled detector for malformed-but-clean-charset JSON.

Runtime: the grounded union costs 202 µs/step monitor-side (vs 196
ungrounded) plus 253 µs/step in the adapter (four extra hash embeds) —
still ~3 orders of magnitude below a judge-LLM call. Fusion caveat
(measured): a single shared 5% FA threshold prices moderate grounding
evidence out — on real_research7b the standalone channel detects
context_corruption at 0.77 where the fused monitor stays at 0. Stream-max
quantile equalization recovers part of it; for content-critical
deployments run grounding as a SECOND alarm stream with a split FA budget
(e.g. 2.5% + 2.5%) instead of fusing scores.

**Verdict**: the grounding channel demonstrates the content blind spot was
a missing information source, not a detector weakness — malformed_json is
closed, context_corruption is substantially recovered (fully, at the
channel level), wrong_document remains open pending better embeddings.
`recommended_monitor()` now auto-upgrades to hybrid_weighted_g when
episodes carry the g channel.

## 10. Closing the context-corruption gap

**Why fusion lost the signal (objective 1, quantified).** On
real_research7b the grounding stream at its own 5%-FA threshold detects
**17/22** context_corruption episodes; the max-union detected **1** — 16
detections destroyed. Cause: the union's val threshold (2.61 in
healthy-rare units) is set by a behavioral-tail healthy outlier (8.26),
while all 16 lost episodes carry grounding evidence at 1.5–2.6, above
grounding's own bar (1.22). Monotone rescaling cannot fix this — with a
val-quantile threshold only the episode-max ORDERING matters, so the
combination rule itself had to change.

**New grounding dims (objective 3).** g grows 5 → 8 (DESIGN.md amendment
4): consecutive-result dissimilarity, grounding-drift EWMA, and
reasoning↔recent-result-memory dissimilarity. Ablation: consec_dis is the
best new standalone signal (context 0.09 alone); drift/mem_dis contribute
mainly through the fused stream. char_anom remains the single strongest
dim (0.27).

**Improved fusion (objective 2).** Val-quantile split budgets fail at
realistic val sizes (2.5% of 24 episodes ⇒ θ = max ⇒ behavioral
detection collapses to 0.05). The working mechanism is a **train-max trip**:
the grounding stream overrides only above the healthy-train maximum of its
normalized stream — a level no healthy training episode ever reached, so
it spends almost no FA budget. Pooled over 393 injected episodes:

| fusion | content (n=140) | behavioral (n=253) | pooled FA | AUROC |
|---|---|---|---|---|
| weighted50 (ungrounded) | 0.15 | 0.62 | 0.126 | 0.798 |
| weighted_g (§9) | 0.34 | 0.68 | 0.099 | 0.821 |
| **content_gate** | **0.44** (+41/−0, p=9e-13) | **0.68** (+16/−0, p=3e-5) | 0.133 | 0.831 |
| dual_budget | **0.46** (+44/−0, p=1e-13) | 0.68 | 0.140 | 0.799 |
| adaptive | 0.34 | 0.68 | **0.099** | 0.825 |

**Guards (objective 4)**: behavioral detection improves under every new
fusion (+16/−0); malformed_json improves (0.91 → 0.93–0.98);
context-gate runtime is 204 µs/step — indistinguishable from weighted_g;
FA moves ≤ +1.4 points vs the ungrounded baseline.

**Context corruption per dataset** (weighted50 → content_gate):
real_research7b 0.00 → **0.64** (channel ceiling 0.77), research7b_long
0.00 → **0.83** (= channel ceiling), autogen7b 0.67 → 0.67, gemini 0.24 →
0.24 (v1, inert channel), langgraph7b 0.18 → 0.18, ollama7b 0.00 → 0.00.

**Recommendation.** `hybrid_content_gate` is the grounded label-free
default (`recommended_monitor()` updated); `hybrid_adaptive` is the
FA-conservative alternative (same behavioral gain, lowest grounded FA);
dual_budget adds ~2 points of content detection over the gate for ~1
point of FA — a deployment choice, not a different mechanism.

**Final assessment: improved where the corruption is visible in content,
and the fusion loss is eliminated.** Detection reaches 0.44–1.0 on the
collectors whose corruption actually alters result content (real_research7b
0.44, real_research7b_long 0.67, ollama7b 1.0) — at or near the standalone
channel's own ceiling, i.e. fusion no longer loses what the channel sees.
Pooled content-class detection (all datasets) rises from 0.23 (ungrounded)
to 0.40 (content gate), a +0.175 gain that holds at every seed; on the
research collectors alone the content gate reaches 0.47 and the standalone
channel 0.63. The remaining misses are of a different kind: corruptions
that are semantically plausible under hash embeddings (ollama7b sits at 0
even for the standalone channel — the injected values look like ordinary
results at the trigram level). That residual is an embedding-fidelity
limit, not a monitoring-architecture limit; the explicit
`use_sentence_transformers=True` opt-in is the designated next probe.

## 11. Solving wrong_document without heavyweight semantics

**Root cause.** The injector substitutes retrieval results with fixed
off-topic decoy sentences. The hash char-trigram cosine carries signal
(step-AUROC 0.80) but its healthy null is far too noisy for a 5% FA
threshold (healthy dissimilarity 0.65 ± 0.26 vs wrong 0.91); continuous
content-word coverage is better (0.90) but leaves healthy tangential
documents (~6% coverage) a hair from decoys (0%), a gap per-dim
z-normalization cannot hold. The representation was the limitation — and
the fix is LESS machinery, not more.

**The lightweight solution.** Dim 59 `lex_miss` (DESIGN amendment 5): a
binary flag — a document-like result sharing ZERO content words with both
its query and the task. Feature-level: 28/28 wrong_document episodes
flagged, 3/330 healthy (all in research7b_long), at **2 µs/result** — 100×
cheaper than one hash embedding (207 µs), 2200× cheaper than MiniLM
(measured: 4.4 ms/text + 5.7 s load). Integration keeps it out of the
continuous trip calibration (its binary z would swallow context evidence
— measured before fixing: long-set context 0.83 → 0.00) and gates it by
one-class clean-null self-calibration: it overrides only in domains where
no healthy training episode flags, and self-disables elsewhere; the
supervised logistic instead takes it as a raw fourth feature and prices
it per domain.

**Results** (pooled, 393 injected episodes, six datasets; paired McNemar
vs the ungrounded hybrid):

| monitor | content (n=140) | behavioral (n=253) | ctx | mjson | wrongdoc | FA |
|---|---|---|---|---|---|---|
| weighted50 (ungrounded) | 0.15 | 0.62 | 0.18 | 0.07 | 0.13 | 0.126 |
| **content_gate** | **0.59** (+62/−0, p=4e-19) | **0.69** improved | 0.43 | 0.93 | 0.58 | 0.133 |
| dual_budget | 0.61 (+65/−0, p=5e-20) | 0.69 | 0.44 | 0.98 | 0.58 | 0.140 |
| logistic_g (labels) | 0.48 | 0.66 | 0.21 | 0.72 | **1.00** | 0.126 |

On the datasets that carry the class, wrong_document detection is
**1.00** for every improved fusion on real_research7b; on the long set
the flag self-disables (dirty null: wordy zero-overlap results occur in
healthy long episodes) and the supervised logistic covers it at 1.00.
Behavioral classes improve under every variant; malformed_json improves;
runtime is unchanged (204 µs/step monitor-side; the lex feature adds ~2
µs/result at the adapter).

**Recommendation.** The lexical zero-overlap flag with clean-null
gating — the lightest possible mechanism (string ops, no model, no new
dependency) — resolves wrong_document. **MiniLM is not required** and is
not adopted; it remains an explicit opt-in for the one residual it could
plausibly help with (ollama7b's trigram-plausible context corruption,
§10). Deployment guidance unchanged: `hybrid_content_gate` label-free
default; `hybrid_logistic_g` when ~20 labeled failures exist (and the
only variant that covers wrong_document in dirty-null domains).

## 12. Multiseed stability of the grounded monitors

Seeds {0, 7, 101, 202, 303} varying ESN reservoirs and cross-fit fold
assignment, splits frozen (`run_grounding_multiseed.py`; seed 0 verified
byte-equal to the published tables). Tables: `grounding_multiseed.csv`,
`grounding_multiseed_criterion.csv`.

**Content gains hold at every seed** (regenerated): the pooled
content-class delta vs the ungrounded hybrid is positive for every
grounded fusion at every seed — content_gate min **+0.175**, adaptive min
+0.095, weighted_g min +0.095.

**The behavioral no-degradation criterion is now passed by content_gate
and adaptive at every seed, and failed by the weighted/logistic grounded
fusions** (regenerated criterion). Post-remediation:
- **content_gate**: content min +0.175, behavioral min **+0.051** → PASS
  at every seed.
- **hybrid_adaptive**: content min +0.095, behavioral min **+0.043** →
  PASS at every seed.
- **weighted_g**: content min +0.095 but behavioral min **−0.008** →
  FAILS (a shared threshold still lets a bad reservoir draw trade a
  behavioral detection).
- **logistic_g**: content min −0.081, behavioral min −0.051 → FAILS.

So the deployable label-free recommendation is the **content_gate** (it
now clears the strict criterion at every seed), with `joint_budget` as
the per-stream-threshold option when a hard never-trade-behavioral
guarantee is wanted.

Aggregate stability (grand mean over datasets × 5 seeds, regenerated):
content_gate AUROC 0.837 (best), adaptive 0.828, joint_budget 0.805,
ungrounded 0.803; FA: gate 0.135, joint_budget 0.111, ungrounded 0.130.

**Deployment guidance, refined**: content_gate remains the best
average-case single-threshold monitor (top AUROC, near-zero mean
behavioral cost); choose **dual_budget when a hard "never trade
behavioral detection" guarantee is required**, at ~1 extra FA point. The
robustness-corruption sweep of the baseline study was not repeated here
(optional per T2); noted as open.

## 13. MiniLM probe (T3): a clear negative result

Question: does transformer-grade semantic fidelity (MiniLM, explicit
`--st` opt-in; 4.4 ms/text vs 207 µs hash, +5.7 s load) close the one
residual no lightweight method reaches — ollama7b's context corruption?

**Answer: no.** With ALL semantic dims (e channel + grounding cosines)
under MiniLM, the grounding channel still detects ollama7b context
corruption at **0.00** — identical to hash. Worse, MiniLM degrades the
behavioral side there (ESN AUROC 0.824 → 0.697, ungrounded hybrid 0.798 →
0.736) and is mixed on research7b (logistic_g 0.920 vs 0.902; gate
context 0.68 → 0.59). Tables: `grounding_st_*.csv`.

**Reinterpretation of the residual.** The ollama7b corruptions are not an
embedding-fidelity problem — they are **reference-free undetectable**:
the corrupted values are semantically plausible results, and no
consistency check against the episode's own content (at any embedding
quality) can flag them. Detecting them requires external reference —
re-querying the tool, cross-source verification, or a judge — which is
exactly the escalation layer's job (H3b), not the telemetry monitor's.

**Decision (per the pre-committed rule): MiniLM is NOT adopted** — it
buys nothing on the target residual, costs 21× per step, and degrades
behavioral detection on one dataset. It remains an explicit opt-in flag.
The lightweight design philosophy stands: hash + lexical string ops.

## 14. Organic-failure validation (T5): what injected labels couldn't show

Protocol: 30 fresh research-task episodes at Ollama temperature 0.9 (vs
the curated 0.2), NO injection (`collect_organic.py`); labels assigned by
manual review against a documented rubric with an evidence quote per
episode (`traces/organic7b/organic_labels.csv` — auditable); monitors
FROZEN on the deployed real_research7b calibration, no refit, no
threshold tuning (`score_organic.py`). 11/30 episodes (37%) failed
organically: 7 aborted (empty final step, task unfinished), 3 fabricated
counts (e.g. `len()` of a string reported as "262 papers"; "0 papers ...
3 papers total" after both retrievals timed out), 1 ungrounded retrieval
blend (off-topic paper folded into the answer).

Findings, in decreasing comfort:

1. **Commission failures transfer only weakly (corrected).** Under the
   remediated ESN, just **1 of 3** fabricated-count episodes is caught by
   the monitors (0.33, `organic_validation.csv`) — the entropy/behavior
   turbulence around fabrication is *sometimes* what the stack detects, but
   the clean 3/3 an earlier draft reported does not survive regeneration.
   A dedicated preregistered fabrication study (`organic_hallucination.csv`)
   is underpowered (2 genuine hallucinations, 36% healthy FA) and makes no
   detection claim.
2. **Omission failures are structurally invisible — and trivially
   fixable.** All 7 aborts evade every monitor (0/7): a silent empty
   final step produces no anomalous telemetry, and anomaly scores are
   built for OVER-activity, not absence. A five-line deterministic
   completion check (final step empty or a bare tool call) catches 7/7
   aborts at 0/19 healthy false positives. Recommendation: ship it as a
   zero-cost side-channel alongside the monitor; a learned detector for
   this is the wrong tool.
3. **Thresholds do not transfer across sampling temperature.** Healthy
   high-temperature organic episodes alarm at ~0.36 (`organic_hallucination.csv`)
   under thresholds calibrated on temperature-0.2 healthy runs — the u
   channel is directly decoding-entropy-sensitive. Deployment rule:
   calibrate the healthy null under the SAME decoding configuration as
   production.

   **AMENDED 2026-07-28 — necessary, not sufficient, and the organic
   detection it was supporting does not survive.** Three things were wrong.
   (a) `score_organic_halluc` chose θ on the same healthy episodes the gate
   was fit on; in-sample scores run low, θ landed low, and every class
   over-alarmed. Fixed to nested out-of-fold θ. (b) A degenerate-scale defect
   (DESIGN.md Amendment 6) let a telemetry dim with no healthy variation be
   divided by a floor rather than left unscaled, so healthy episodes reached
   peaks of 1.0e9 against a median of 0.46 and set the threshold for their
   whole corpus. Fixed at all five sites; maximum healthy peak now 2.99, and
   `behavior_snapshot --check` confirms no committed number moves. All
   `organic_hallucination*` tables regenerated. (c) Even with a correctly
   matched, correctly nested and correctly scaled null, the realized healthy
   false-alarm rate is **13% at temperature 0.9 and 18% at 0.2** against the
   served 10% budget — the healthy behavioural episode-max is genuinely
   heavy-tailed (0.2 arm: maxima run 30 → 2815 continuously, 8 of 74 above
   1000), so a 90th-percentile threshold is noisy. On the seed-paired
   serving-temperature arm (`SERVING_TEMPERATURE_PREREG.md`,
   `serving_temperature.csv`, n=120 per arm) organic arithmetic errors alarm
   at 46% against a 17% false-alarm rate (AUROC 0.733) — but only once runs
   that skipped required work were removed from the healthy null; with them in
   it, the same monitor measured at chance. Deployment rule, restated: match the
   decoding configuration, calibrate θ out-of-fold, and then **measure** the
   realized false-alarm rate, because the budget is not delivered.
4. **Within-organic score ranking fails on this population** (episode
   AUROC 0.31–0.42, i.e. at/below chance): driven by the abort majority —
   quiet failures score LOW while verbose high-temperature healthy runs
   score high. This is the honest quantitative face of finding 2.
5. The one ungrounded retrieval blend evades everything (0/1) —
   consistent with §13's reference-free-undetectable family.

Combined coverage with the completion check: **8/11 organic failures**
(7 aborts from the deterministic check + 1 of 3 fabrications from the
monitors).
Taxonomy takeaway for the paper: telemetry anomaly detection addresses
failures of commission; failures of omission need (trivial) task-state
checks; failures of plausible-content corruption need external reference
(escalation). Tables: `organic_validation.csv`; dataset committed under
`traces/organic7b/`.

## 15. Dataset reinforcement (T6): three thin spots addressed

Note on provenance: sections 9–14 report the dataset state at their own
commits; this section's tables (`grounding_t6_*.csv`, `model_transfer.csv`)
reflect the post-T6 corpus (langgraph7b healthy 60→100, real_research7b
+22 goal_drift episodes, new real_research3b).

**(a) langgraph7b FA artifact: fixed by data.** With 100 healthy episodes
(val 20 instead of 12), the 25%-FA-for-every-monitor artifact vanishes —
realized FA 0.00 across all monitors at the 5% budget, content-gate AUROC
0.710 → 0.773. The "noisiest benchmark" was a small-validation artifact,
not a property of LangGraph.

**(b) goal_drift finally exists on real tools — and the grounded gate
owns it.** New injector class (DESIGN-consistent, prompt-layer goal
hijack delivered through a tool result; post-injection drift is the real
model's organic response). Detection on real_research7b: ESN 0.09,
Mahalanobis 0.27, ungrounded hybrid 0.27, **content gate / dual budget
0.91**; on real_research3b: 0.83. The hijack text in the result stream is
exactly what the grounding dims see — a fifth failure class where the
content channel is the difference-maker. (This also softens the old
"goal_drift ~0.28 everywhere" limitation: that number was the simulator's
prompt-free drift, which remains open — see T7.)

**(c) Cross-model transfer: catastrophic, calibration is per-model.**
Same framework, tasks, and tools, agent swapped qwen2.5:7b → 3b:
transferred monitors collapse (AUROC 0.29–0.38 — below chance; FA 1.00),
while in-domain 3b calibration recovers only to 0.56–0.67 AUROC (best:
Δ-Mahalanobis 0.67) even with 21 training healthy episodes — recalibration
helps but is itself weak on this hard 3b set. Together with T5's temperature
finding, the deployment rule is now sharply supported: **the healthy null
is specific to the (model, decoding-config) pair and must be calibrated
per deployment** — which the one-class protocol makes cheap (~30 healthy
episodes, no labels).

3b caveat: its val split is 7 episodes, so its absolute FA (0.29–0.57) is
threshold-granularity noise; the AUROC ordering is the reliable signal.

## 16. T7 diagnosis: goal_drift and timeout, reframed

Diagnosis-first, per the backlog; verdict: **no new detector work is
justified** — the two "chronically weak" classes decompose into solved,
bounded, and by-construction-hard parts.

**goal_drift splits into two different phenomena.**
- *Prompt-hijack drift* (the realistic mechanism, T6's injector — an
  instruction in a tool result diverts the agent): **solved** by the
  grounded gate (0.91 research7b, 0.83 research3b vs ESN 0.09/0.83,
  ungrounded 0.27). Moderate on the older mock-tool sets (gemini 0.59,
  ollama7b 0.33) whose drift injection predates the hijack mechanism.
- *Slow semantic rotation* (the simulator's synthetic class): detection
  0.0125 for EVERY monitor at the 5% budget — the drift is, by
  construction, indistinguishable from healthy variation within episode
  length at this FA budget. This is the honest hard case; any fix is a
  research effort (lower-frequency drift channels), and no organic
  episode (T5) exhibited this failure shape. Deferred as future work.

**timeout is horizon-bounded, not detector-bounded.** Detection: 0.50
(research7b, T≈5–6), 0.83 (research3b), **1.00** (research7b_long,
T≈11). The ramping injector needs post-tau tool calls to realize an
error; short episodes offer 1–2 chances. Where the horizon exists, every
monitor detects every timeout. No feature work can change the realization
probability — episode length can.

With this, the backlog's detector-side items are closed: every failure
class is either detected at high rates by the shipped stack, covered by
the deterministic completion check (aborts), horizon-bounded (timeout on
short episodes), reference-dependent (escalation's job), or documented as
by-construction hard (synthetic slow rotation).
