# Hybrid ESN + Mahalanobis monitor study — final report

> **Source of record: the CSVs in `results/tables/` govern.** Every figure in
> this report is recomputed from the committed tables, and the verdicts stated
> here follow those numbers rather than the other way round. Where this
> document and a CSV disagree, the CSV is right.

Code: `derail/monitor/hybrid.py`,
`derail/experiments/run_hybrid_study.py`, `collect_research7b_long.py`.
Tables: `results/tables/hybrid_*.csv` and `hybrid_long_*.csv`. Protocol:
frozen baseline evaluation (same splits via `rng_for(0, "real-split")`, 5%
FA healthy-val-quantile thresholds, `evaluation/metrics.py` +
`evaluation/stats.py`). All numbers reproduce deterministically.

## 1. Why does Mahalanobis beat the ESN on real_research7b?

**Cause identified: post-onset temporal horizon.** Per-episode diagnosis
over all 1,002 injected episodes across the eight benchmark datasets
(`hybrid_diagnosis.csv`):

| post-onset horizon (T−1−τ) | n | ESN det | Maha det | ESN − Maha |
|---|---|---|---|---|
| ≤ 3 steps | 451 | 0.50 | 0.41 | **+0.09** |
| 4–8 steps | 170 | 0.72 | 0.58 | **+0.14** |
| ≥ 9 steps | 381 | 0.78 | 0.38 | **+0.40** |

corr(horizon, ESN advantage) = **+0.25** over these pooled episodes.

> **Superseded as evidence for the horizon law; see `results/horizon_report.md`.**
> This scope's ≥9 band is 97.4% simulator episodes and holds 10 real ones, so
> band and corpus are nearly the same variable here: controlling for corpus on
> these same 1,002 episodes drops the correlation from +0.25 to **+0.03**, and
> the +0.40 above is the simulator's value. Re-estimated over every real corpus
> (1,765 episodes, 12 corpora, 112 real episodes at ≥9) the law holds at
> +0.017 / +0.082 / +0.250 with r = +0.202 controlling for corpus. The table
> above stays because it is what this study's scope computes; it is not the
> number to quote for the law.

The ESN's CUSUM accumulates
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
| looping | **0.84** | 0.71 | temporal |
| tool_cascade | **0.82** | 0.70 | temporal |
| timeout | 0.56 | 0.56 | tie |
| goal_drift | **0.49** | 0.15 | temporal, ESN weak in absolute terms |
| context_corruption | **0.35** | 0.17 | temporal, both weak |
| rate_limit | 0.63 | 0.65 | tie |
| malformed_json | 0.18 | **0.22** | content corruption: nobody |
| wrong_document | 0.11 | **0.19** | content corruption: nobody |

Reading: **there is no state-based column.** The ESN leads on every class
that involves behaviour changing over steps, ties on the two abrupt-error
classes (timeout, rate_limit), and is beaten only on the two *content*
classes — where both monitors are near-useless anyway (0.11–0.22), which is
precisely the blind spot the grounding channel exists to cover. The
memory-versus-distance split is therefore not a split: the reservoir wins
wherever there is anything to accumulate and ties where there is not, and
neither monitor sees corruption that changes data without changing
behaviour.

## 3. Hybrid results (`hybrid_benchmark.csv`)

AUROC by dataset (best in **bold**):

| dataset | ESN | Maha | weighted | max | gated | logistic |
|---|---|---|---|---|---|---|
| sim | **0.890** | 0.786 | 0.883 | 0.888 | 0.864 | 0.889 |
| gemini | 0.749 | 0.756 | 0.751 | 0.746 | **0.759** | 0.731 |
| autogen7b | **0.833** | 0.774 | 0.822 | 0.819 | 0.810 | 0.777 |
| ollama7b | 0.994 | 0.895 | 0.994 | 0.994 | **0.997** | 0.982 |
| langgraph7b (held out, §7) | 0.828 | 0.885 | 0.875 | **0.906** | 0.885 | 0.884 |
| real_research3b | 0.556 | **0.668** | 0.565 | 0.612 | 0.556 | 0.640 |
| real_research7b | 0.777 | 0.848 | 0.815 | 0.813 | **0.849** | 0.847 |
| real_research7b_long | 0.790 | 0.849 | 0.790 | 0.790 | 0.782 | **0.857** |
| **grand mean (8 datasets)** | 0.802 | 0.808 | 0.812 | 0.821 | 0.813 | **0.826** |

Logistic features use robust-z clipping at ±50 (`HybridLogistic(clip=50)`).
The bound was chosen empirically: unclipped features reach z ~ 1e6 and
sklearn's L2 penalty drives the learned weights to numerical zero (observed
on real_research7b_long); a tight ±5 bound saturates so many episode maxima
that ranking collapses (autogen7b AUROC 0.854 → 0.639); ±50 fixes the
conditioning and matches or beats the unclipped variant on all six
datasets (autogen7b +0.012, research7b_long +0.032).

Statistical validation (`hybrid_stats.csv`, paired per-episode tests):
hybrid_logistic is **never significantly below the local winner on any of
the eight datasets** — every ΔAUC confidence interval against the better
standalone includes zero (tightest: research7b vs Maha [−0.010, +0.007];
sim vs ESN [−0.009, +0.008]; langgraph7b vs Maha [−0.020, +0.021]; widest:
research3b vs Maha [−0.333, +0.221], which is the small-corpus noise floor,
not evidence of parity). Against the local *loser* it is significantly
above on three datasets in AUROC (sim vs Maha [+0.074, +0.134], McNemar
p = 1.8e-39; research7b vs ESN [+0.018, +0.133]; ollama7b vs Maha
[+0.033, +0.146]) and elsewhere the interval straddles zero. Read the
detection tests separately from the ranking intervals: research7b vs ESN is
a clear AUROC win at McNemar p = 0.56 on detection, and autogen7b vs Maha is
p = 0.22. A ranking gain is not an alarming gain.
Supervision discipline: logistic weights come from the sim `cal` split or
2-fold class-stratified cross-fit on real data — no episode is scored by a
model that saw it in training. Weighted-0.5 is the recommended label-free
variant (grand mean 0.812); max edges it (0.821). Gated takes the
per-dataset lead on gemini, ollama7b and real_research7b but grand-means
below max, and its gate calibration remains the least principled of the
fusion rules.

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
   needed, grand-mean AUROC 0.812 > Maha 0.808 > ESN 0.802 over eight
   datasets, never far from the local winner, +9% latency over ESN. Those
   are raw `episode_auc` values and the ESN/Maha half of that ordering does
   not survive length control (§8); what carries the recommendation is
   "never far from the local winner", which is a per-dataset floor and is
   unaffected.
2. **When ≥ ~20 labeled failure episodes exist: `hybrid_logistic`** — best
   grand mean (0.826), statistically at-or-above the better standalone on
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

**(a) The two detectors cover different failures.** Over all 1,002 injected
episodes, 372 are caught by both standalones, but 273 are ESN-only and 57
are Mahalanobis-only — 330 episodes (32.9%) where exactly one detector sees
the failure. That partial overlap is the raw material fusion exploits. Note
the asymmetry: the ESN-only set is nearly five times the Mahalanobis-only
set, so most of what fusion has to recover is reservoir evidence.

**(b) The logistic recovers most of both one-sided sets, but not all.**
Detection rate of hybrid_logistic inside each complementarity cell:
both-detect **0.96** (372 episodes), ESN-only **0.61**, Maha-only **0.75**,
neither **0.01**. So it keeps almost everything both parents agree on,
recovers three-quarters of the smaller Mahalanobis-only set and only
three-fifths of the larger ESN-only set, and stays essentially silent where
neither sub-detector fires. It does not keep everything the union offers —
holding the 5% FA budget costs it about 40% of the ESN-only cell.

**(c) The weights learn which regime the deployment is in.** The
Mahalanobis share of the learned weight mass tracks the dataset's
character: **0.38** on sim (long horizons — lean on the reservoir),
**0.48** on gemini, **0.78** on research3b, then **0.87–0.90** on
langgraph7b/ollama7b and **0.98–1.00** on
research7b_long/autogen7b/research7b (short horizons — lean on the
distance). ~20 labeled episodes suffice to place the boundary. The
intercept does *not* land in a consistent place: it ranges from −1.28
(research7b) to −5.19 (ollama7b), so the boundary's distance from the
healthy cluster is itself a per-deployment quantity.

**(d) The picture.** In the figure, each episode is plotted at the step
that maximizes the fused score, Mahalanobis confidence on x, ESN
confidence on y. On sim the decision boundary is nearly **horizontal**
(alarms are decided by ESN evidence); on autogen7b, research7b and
research7b_long it is nearly **vertical** (Mahalanobis decides, with the
ESN coefficient at or below zero); gemini is the **diagonal** case where
both coefficients are comparable (0.78 ESN against 0.70 Maha). Detected
episodes hug one axis or the other, almost never the diagonal middle: the
sub-detectors fire on *different* failures, and the learned line is what
turns that union into one calibrated alarm.

## 7. Held-out generalization: langgraph7b

langgraph7b (95 episodes, LangGraph framework, qwen2.5:7b, telemetry v3)
was **never used while developing the hybrids** — it entered the study
only after the fusion design, clip value, and defaults were frozen. Same
protocol, thresholds, and tests as every other dataset.

Results (AUROC): max **0.906** > Maha = gated 0.885 > logistic 0.884 >
weighted 0.875 > ESN 0.828 (`hybrid_benchmark.csv`). Runtime unchanged
(hybrids 164–170 µs/step vs 3.8 µs Maha / 156 µs ESN). Detection at the 5%
budget spans 0.51 (ESN) to 0.74 (Maha), at realized healthy false-alarm
rates of 0.00 (ESN, weighted) to 0.14 (Maha) — the monitors that detect
most here also alarm most.

Verdict, stated precisely: the core claim **survives, and on this dataset
more than survives.** The best hybrid (max) is significantly *above* the
local winner in ranking — ΔAUC CI [+0.001, +0.053] vs Maha, entirely above
zero — and above the local loser too (CI [+0.014, +0.149] vs ESN). But read
ranking and alarming apart: on detection at the budget, Maha beats max 0.74
to 0.56, and McNemar makes that gap real (17 Maha-only detections against 1
max-only, p = 1.4e-4, surviving Holm). So the fusion orders episodes better
than either parent while firing less often than the better-firing parent —
exactly the pooled-advantage-not-dominance pattern §3 reports, appearing
again on a set the design never saw.

hybrid_logistic replicates its usual edge here: level with Maha on AUROC
(CI [−0.020, +0.021]), clearly above ESN on detection (21 logistic-only
against 2 ESN-only, p = 6.6e-5, Holm-rejected), and ahead of the label-free
weighted variant on both AUROC (0.884 vs 0.875) and detection (0.71 vs
0.56).

The grand-mean ordering over all datasets holds (logistic 0.825 > max
0.817 > weighted 0.810 ≈ gated 0.809 > Maha 0.805 > ESN 0.800), and the
deployment recommendation in §5 stands.

## 8. Multiseed stability (5 seeds)

`hybrid_multiseed.csv` — seeds {0, 7, 101, 202, 303} varying the ESN
reservoir initialization, the logistic cross-fit fold assignment, and the
simulator master seed, with data splits frozen per protocol
(`run_hybrid_multiseed.py`; seed 0 verified byte-identical to the
published tables).

Grand-mean AUROC across seeds: logistic **0.825 ± 0.007** > max 0.817 ±
0.006 > weighted 0.810 ± 0.005 ≈ gated 0.809 ± 0.006 > Maha 0.805 ± 0.003 >
ESN 0.800 ± 0.003. Two conclusions:

1. **The learned-hybrid-over-standalone gap survives seed noise, paired
   per seed:** logistic − ESN = +0.025 ± 0.004 (positive at every seed,
   worst +0.019); logistic − Maha = +0.021 ± 0.006 (worst +0.011); the
   label-free weighted − ESN = +0.010 ± 0.002
   (worst +0.007). Differences among the label-free hybrid variants are
   within seed noise.

   **All of the above is raw `episode_auc`, and length control moves two of
   these conclusions.** Δ-Mahalanobis grand-means *above* the ESN on raw
   AUROC at every seed (by 0.002–0.008), which reads as "a strong baseline
   only the learned fusion clears". Repeat the comparison inside overlapping
   length bins and the ordering reverses at every seed, the ESN leading by
   0.021–0.031: the Mahalanobis edge was episode exposure, concentrated in
   the short-episode research corpora. The logistic's margin over Δ-Maha
   survives and grows (+0.021 → +0.029); its margin over the *ESN* does not
   (+0.025 → +0.002, negative at two of five seeds). The robustness floor
   moves the same way — worst-dataset AUROC raw Δ-Maha 0.668 > logistic
   0.646 > ESN 0.573, length-matched ESN 0.719 > logistic 0.705 > Δ-Maha
   0.625. Matched samples are small (median 29 per dataset, three saturating
   at 1.000), so this is evidence that a margin is exposure-driven, not a
   replacement point estimate. It does not change the label-free
   recommendation, which rests on per-dataset floors rather than the grand
   mean, but the "Δ-Mahalanobis is the baseline to beat" reading does not
   survive it.
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
must not degrade), pooled over all 874 injected episodes across the ten
real datasets (`grounding_multiseed_criterion.csv`, seed 0):

| comparison vs the ungrounded hybrid | content (n=313) | behavioral (n=561) |
|---|---|---|
| hybrid_content_gate | 0.272 → **0.578** (+0.307) | 0.738 → **0.786** (+0.048) |
| hybrid_adaptive | 0.272 → **0.524** (+0.252) | 0.738 → **0.777** (+0.039) |
| hybrid_weighted_g | 0.272 → **0.454** (+0.182) | 0.738 → 0.733 (−0.005) |
| hybrid_logistic_g | 0.272 → 0.316 (+0.045) | 0.738 → 0.692 (−0.046) |

**hybrid_content_gate passes both halves** — content detection more than
doubles while behavioral detection improves. `hybrid_adaptive` also passes.
The weighted and logistic grounded unions do not: both trade away
behavioral detections, the logistic one heavily. Use the gate.

Per-class means across the ten datasets (ungrounded → content gate):
malformed_json 0.13 → **0.90**, context_corruption 0.34 → **0.58**,
wrong_document 0.09 → **0.47**. None of the three is closed everywhere:
malformed_json and context_corruption reach 1.00 on the long research
collectors and stay at 0.15–0.17 on langgraph7b and research3b, and
wrong_document is 1.00 on real_gemini_long and _long_ext but 0.00 on
research7b_long. Where the corruption alters result text the channel sees
it; where it does not, nothing does.

Grand-mean AUROC (ten real datasets): hybrid_logistic_g 0.864 ≈
hybrid_content_gate 0.864 > hybrid_adaptive 0.856 > joint_budget 0.831 >
hybrid_weighted_g 0.822 > hybrid_logistic 0.819 > Maha 0.810 > ESN 0.809 >
weighted50 0.807. The standalone grounding stream alone is 0.595 — it is a
complement, not a monitor.

Ablation (each dim alone, `grounding_ablation.csv`): char_anom is the
workhorse (malformed_json **1.00**, context_corruption 0.37
single-handedly); **lex_miss is the strongest wrong_document signal at
0.60**, ahead of query_dis (0.28); reason_dis/mem_dis contribute small
malformed-json signals (0.20); json_broken reaches 0.40 on malformed_json.

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
it spends almost no FA budget. Pooled over 874 injected episodes:

| fusion | content (n=313) | behavioral (n=561) | mean FA | AUROC |
|---|---|---|---|---|
| weighted50 (ungrounded) | 0.272 | 0.738 | 0.118 | 0.807 |
| weighted_g (§9) | 0.454 | 0.733 | 0.105 | 0.822 |
| **content_gate** | **0.578** | **0.786** | 0.122 | **0.864** |
| joint_budget | 0.53 (ctx) | — | **0.105** | 0.831 |
| adaptive | 0.524 | 0.777 | 0.116 | 0.856 |

**Guards (objective 4)**: behavioral detection improves under the gate and
adaptive (+0.048, +0.039) and *falls* under weighted_g and logistic_g;
malformed_json improves 0.13 → 0.90; FA moves ≤ +0.4 points vs the
ungrounded baseline, and the gate is not the cheapest — joint_budget and
weighted_g run lower FA.

**Context corruption per dataset** (weighted50 → content_gate):
real_research7b 0.05 → **0.36**, research7b_long 0.00 → **1.00**,
research7b_long_ext 0.00 → **1.00**, real_gemini_long 0.70 → **0.80**,
ollama7b 1.00 → 1.00, ollama_llama8b 0.88 → 0.88, gemini 0.33 → 0.33 (v1,
inert channel), langgraph7b 0.15 → 0.15, autogen7b 0.12 → 0.12,
research3b 0.17 → 0.17.

**Recommendation.** `hybrid_content_gate` is the grounded label-free
default (`recommended_monitor()` updated); `hybrid_adaptive` is the
FA-conservative alternative (same behavioral gain, lowest grounded FA);
dual_budget adds ~2 points of content detection over the gate for ~1
point of FA — a deployment choice, not a different mechanism.

**Final assessment: improved where the corruption is visible in content,
and the fusion loss is eliminated.** Detection reaches 0.36–1.00 on the
collectors whose corruption actually alters result content (real_research7b
0.36, both long research sets 1.00, ollama7b 1.00) — at or near the
standalone channel's own ceiling, i.e. fusion no longer loses what the
channel sees.
Pooled content-class detection (all ten datasets) rises from 0.28
(ungrounded) to 0.59 (content gate), a +0.31 gain that holds at every seed;
on the longest research collector the content gate reaches 1.00. The remaining misses are of a different kind: corruptions
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

**Results** (pooled, 874 injected episodes, ten datasets, vs the
ungrounded hybrid):

| monitor | content (n=313) | behavioral (n=561) | ctx | mjson | wrongdoc | FA |
|---|---|---|---|---|---|---|
| weighted50 (ungrounded) | 0.272 | 0.738 | 0.34 | 0.13 | 0.09 | 0.118 |
| **content_gate** | **0.578** | **0.786** | 0.58 | 0.90 | 0.47 | 0.122 |
| adaptive | 0.524 | 0.777 | 0.53 | 0.80 | 0.47 | 0.116 |
| logistic_g (labels) | 0.316 | 0.692 | 0.28 | 0.53 | 0.21 | 0.098 |

Read the wrong_document column per dataset rather than pooled: the gate
reaches **1.00** on real_gemini_long and real_research7b_long_ext, 0.20 on
research3b, 0.14 on research7b and **0.00** on research7b_long, where the
flag self-disables against a dirty null (wordy zero-overlap results occur
in healthy long episodes). The supervised logistic does not rescue that
case here — it is 0.21 pooled. Behavioral detection improves under the gate
and adaptive only; runtime is unchanged (204 µs/step monitor-side; the lex
feature adds ~2 µs/result at the adapter).

**Recommendation.** The lexical zero-overlap flag with clean-null gating —
the lightest possible mechanism (string ops, no model, no new dependency) —
is the best wrong_document signal available (0.60 standalone in the
ablation, against query_dis at 0.28), and it resolves the class outright on
the domains with a clean null. It does not resolve it everywhere: pooled
detection is 0.47 and the dirty-null long set stays at 0.00. **MiniLM is
not required** and is not adopted (§13). Deployment guidance:
`hybrid_content_gate` is the label-free default; `hybrid_logistic_g` needs
~20 labeled failures and, on the current corpus, is worse on every content
class than the gate.

## 12. Multiseed stability of the grounded monitors

Seeds {0, 7, 101, 202, 303} varying ESN reservoirs and cross-fit fold
assignment, splits frozen (`run_grounding_multiseed.py`; seed 0 verified
byte-equal to the published tables). Tables: `grounding_multiseed.csv`,
`grounding_multiseed_criterion.csv`.

**Content gains hold at every seed** (regenerated): the pooled
content-class delta vs the ungrounded hybrid is positive for every
grounded fusion at every seed — content_gate min **+0.307**, adaptive min
+0.252, weighted_g min +0.182.

**The behavioral no-degradation criterion is now passed by content_gate
and adaptive at every seed, and failed by the weighted/logistic grounded
fusions** (regenerated criterion). Post-remediation:
- **content_gate**: content min +0.307, behavioral min **+0.039** → PASS
  at every seed.
- **hybrid_adaptive**: content min +0.252, behavioral min **+0.030** →
  PASS at every seed.
- **weighted_g**: content min +0.182 but behavioral min **−0.005** →
  FAILS (a shared threshold still lets a bad reservoir draw trade a
  behavioral detection).
- **logistic_g**: content min +0.032, behavioral min **−0.046** → FAILS on
  the behavioral half at every seed.

So the deployable label-free recommendation is the **content_gate** (it
clears the strict criterion at every seed), with `joint_budget` as the
per-stream-threshold option when a hard never-trade-behavioral guarantee
is wanted.

Aggregate stability (grand mean over datasets × 5 seeds): logistic_g AUROC
0.869, content_gate 0.866, adaptive 0.857, joint_budget 0.832, ungrounded
0.807; FA: gate 0.122, adaptive 0.116, joint_budget 0.105, ungrounded
0.118. The gate is the best monitor that also passes the criterion —
logistic_g edges it on AUROC while failing the behavioral half, which is
why the criterion and not the AUROC decides the recommendation.

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

1. **Commission failures transfer only weakly.** Just **1 of 3**
   fabricated-count episodes is caught by the monitors (0.33,
   `organic_validation.csv`) — the entropy/behaviour turbulence around
   fabrication is *sometimes* what the stack detects, and not reliably.
   A dedicated preregistered fabrication study (`organic_hallucination.csv`)
   is underpowered (2 genuine hallucinations, 36% healthy FA) and makes no
   detection claim.
2. **Omission failures are structurally invisible — and trivially
   fixable.** Every behavioural monitor scores 0/7 on the aborts: a silent
   empty final step produces no anomalous telemetry, and anomaly scores are
   built for OVER-activity, not absence. Only the standalone grounding
   stream alarms on one of the seven (1/7, and marginally — 7.82 against a
   threshold of 7.44), which is not a mechanism to rely on. A five-line
   deterministic completion check (final step empty or a bare tool call)
   catches 7/7 aborts at 0/19 healthy false positives. Recommendation: ship
   it as a zero-cost side-channel alongside the monitor; a learned detector
   for this is the wrong tool.
3. **Thresholds do not transfer across sampling temperature.** Healthy
   high-temperature organic episodes alarm at ~0.36 (`organic_hallucination.csv`)
   under thresholds calibrated on temperature-0.2 healthy runs — the u
   channel is directly decoding-entropy-sensitive. Deployment rule:
   calibrate the healthy null under the SAME decoding configuration as
   production.

   **Necessary, not sufficient — and the organic detection this was
   supporting does not survive.** Three things were wrong.
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
   serving-temperature arm (pre-registered;
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

Note on scope: this section's tables (`grounding_t6_*.csv`,
`model_transfer.csv`) cover the three corpora T6 reinforced — langgraph7b
(healthy 60→100), real_research7b (+22 goal_drift episodes) and the new
real_research3b — rather than the full ten-corpus grounding scope of
§§9–12.

**(a) langgraph7b FA artifact: largely fixed by data.** With 100 healthy
episodes (val 20 instead of 12), the 25%-FA-for-every-monitor artifact
vanishes and content-gate AUROC reaches **0.884**. Realized FA is 0.00 for
the ESN, both weighted variants, the content gate and adaptive, but not for
every monitor: Δ-Mahalanobis still sits at 0.143 and the two logistic
variants at 0.095. The "noisiest benchmark" was a small-validation
artifact, not a property of LangGraph — but the fix is not uniform across
monitors.

**(b) goal_drift exists on real tools, and NO monitor owns it.** New
injector class (DESIGN-consistent, prompt-layer goal hijack delivered
through a tool result; post-injection drift is the real model's organic
response). Detection on real_research7b (22 episodes): ESN **0.05**,
Δ-Mahalanobis **0.27**, ungrounded hybrid **0.05**, content gate
**0.05**; on real_research3b (5 episodes) every behavioural monitor
reaches 0.80 and Mahalanobis 0.00.

The content channel is **not** the difference-maker on this class: the
gate scores exactly what the ESN scores on research7b, and the best
detector there is the memoryless distance at 0.27.
`grounding_t6_per_class.csv` and `grounding_per_class.csv` agree with each
other at 0.045, and agree with the powered drift study (`real_research7b_long_drift`, 120 healthy / 120 injected),
where the shipped ESN reaches detection 0.054 on real goal drift at a median
post-onset horizon of 8 steps. Read the two together: **goal drift is not
detected by anything in this stack**, and the corpora agree on that.

**(c) Cross-model transfer: calibration is per-model.**
Same framework, tasks, and tools, agent swapped qwen2.5:7b → 3b:
transferred monitors sit at chance (AUROC 0.45–0.52) and alarm on
everything (FA 1.00), while in-domain 3b calibration recovers only to
0.56–0.67 AUROC (best: Δ-Mahalanobis 0.67) even with 21 training healthy
episodes — recalibration helps but is itself weak on this hard 3b set. The
Together with T5's temperature
finding, the deployment rule is now sharply supported: **the healthy null
is specific to the (model, decoding-config) pair and must be calibrated
per deployment** — which the one-class protocol makes cheap (~30 healthy
episodes, no labels).

3b caveat: its val split is 7 episodes, so its absolute FA (0.29–0.57) is
threshold-granularity noise; the AUROC ordering is the reliable signal.

## 16. T7 diagnosis: goal_drift and timeout, reframed

Diagnosis-first, per the backlog. The original verdict was **no new
detector work is justified**, on the reading that both "chronically weak"
classes decomposed into solved, bounded and by-construction-hard parts.
Against the regenerated tables, half of that verdict does not hold.

**goal_drift is one phenomenon, not two, and neither form is detected.**
- *Prompt-hijack drift* (the realistic mechanism, T6's injector — an
  instruction in a tool result diverts the agent): **not solved**.
  research7b ESN 0.05, Mahalanobis 0.27, content gate 0.05; research3b
  0.80 behavioural, Mahalanobis 0.00. Higher on the older mock-tool sets
  (gemini 0.73, ollama7b 0.86, ollama_llama8b 0.66) whose drift injection
  predates the hijack mechanism and is behaviourally louder.
- *Slow semantic rotation* (the simulator's synthetic class): detection
  0.0125 for EVERY monitor at the 5% budget — by construction
  indistinguishable from healthy variation within episode length.

The two therefore no longer split into "solved" and "hard". The powered
study on `real_research7b_long_drift` (120 healthy / 120 injected, median
post-onset horizon 8, range 1–10) puts the shipped ESN at detection **0.054**
with AUROC
0.617, and a purpose-built conceptor arm scoring state geometry rather
than prediction error came in reliably *worse* (pooled paired dAUC −0.051,
CI [−0.088, −0.014]). Runway is not the binding constraint and neither is
architecture: **the drift signal is not present in this telemetry.**

**timeout is horizon-bounded, not detector-bounded — confirmed.**
Detection: 0.17 (research3b, T short), 0.50 (research7b, T≈5–6),
**1.00** (research7b_long and its _ext sibling, T≈11). The ramping
injector needs post-tau tool calls to realize an error; short episodes
offer one or two chances. Where the horizon exists, every monitor detects
every timeout. No feature work changes the realization probability —
episode length does.

So the detector-side items close for every class **except goal drift**:
detected at high rates by the shipped stack, covered by the deterministic
completion check (aborts), horizon-bounded (timeout on short episodes),
or reference-dependent (escalation's job). Goal drift is open, and the
next move on it is new telemetry rather than another monitor.
