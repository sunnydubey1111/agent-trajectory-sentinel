# Contributions

What this project establishes, with the artifact each result is computed from.
Every number here is either in [`CLAIMS.md`](CLAIMS.md) — where it is recomputed
from its committed file on every run — or is derivable from a table named
beside it. Nothing in this list is an aspiration; where a result is bounded or
negative, that is stated rather than omitted.

---

## 1. Laws and mechanisms

**The horizon law.** Within a deployment, the temporal monitor's advantage over
a memoryless baseline grows with post-onset runway. Over **1,765 real injected
episodes from 12 corpora** the ESN-minus-Mahalanobis detection gap runs
**+0.017** at horizon ≤3 steps, **+0.082** at 4–8 and **+0.250** at ≥9, and the
horizon/advantage correlation is r = **+0.202** controlling for corpus and
**+0.226** controlling for corpus and failure class (p < 1e-4, horizon shuffled
inside each corpus). Leave-one-corpus-out moves it between +0.176 and +0.221.
The law was stated before the corpus that tested it existed.
→ `results/horizon_report.md`, `results/tables/horizon_*.csv`

**What the law does not claim.** It governs the *slope*, not the *level*. The
sign of the advantage is deployment-specific: `langgraph7b` is negative in
every band, `ollama7b` runs backwards. The slope is positive in 8 of 9 corpora
for the ≤3→4–8 step, but with 3–9 corpora the deployment-level effect is not
statistically resolvable (p = 0.14 to 0.50) — a statement about how many
deployments exist, not evidence against the law.
→ `results/tables/horizon_by_dataset.csv`, `horizon_contrasts.csv`

**It predicted its own failure region out of sample.** On AFTraj-2K, a corpus
built by another group, detection is **0.509** on failures with ≥9 steps of
runway against 0.170 for the memoryless baseline — but only 53 of 771 AFTraj
failures have that much room, so pooled detection is **0.048**, which is what
the law forecasts. A law that says where it will fail, and then fails there on
someone else's data, is stronger evidence than a higher pooled number.
→ `results/tables/aftraj_benchmark.csv`, `aftraj_diagnosis.csv`

**The mechanism resolves an apparent contradiction.** Mahalanobis wins outright
on almost-entirely-short-horizon corpora (`real_research7b` 0.848 against
0.777), and pooled across corpora the ESN's band means stay positive. Horizon
explains both facts without needing the ESN to win everywhere — and it does
not win everywhere, which is why the per-corpus table is published beside the
pooled one.

**Complementarity, exactly counted.** Of 1,002 injected episodes: **372**
caught by both parents, **273** by the ESN alone, **57** by Mahalanobis alone,
**300** by neither — **330 (32.9%) visible to exactly one**. This is the
measurement that justifies combining streams rather than picking a winner.
→ `results/tables/hybrid_diagnosis.csv`

**Channel specialisation is near-disjoint.** The `e` channel alone reaches
looping 0.988, tool_cascade 0.975 and context_corruption 0.913 while scoring
**0.000** on goal_drift and grounding_loss; the `u` channel alone reaches
grounding_loss 0.938 and **0.000** on looping. Multi-channel coverage is a
mechanistic requirement here, not a hedge.
→ `results/tables/h2_channels.csv`

**One-class monitors are per-deployment by construction.** Transferred from
qwen2.5:7b to llama3.1:8b over a matched task, tool and injector plan, the
monitor sits at chance (AUROC **0.527**) while the *same* target recalibrated
on itself reaches **0.885**; within-family 7b→3b sits at **0.522**. The
failure is the calibration, not a hard target.
→ `results/tables/model_transfer.csv`

**Provider differences are horizon effects, not telemetry effects.** At matched
horizon ≥4 the Gemini and qwen corpora agree (**0.532** against **0.561**).

**Token logprobs are not required.** Ablating the `u` channel inside the
corpora that carry it moves ESN AUROC by **+0.000** on average — 0.790→0.794,
0.994→0.998, 0.885→0.895 — so no corpus loses. The deployment requirement is
structured tool results, not logprob access, which is what makes the approach
usable against providers that withhold them.
→ `results/tables/telemetry_dependence.csv`

---

## 2. Boundaries, measured rather than asserted

**The observability boundary.** A wrong-but-well-formed value perturbs no
behavioural channel, so no reference-free monitor can reject it. The floor is
measured, not argued: on a plausible-corruption set a 2,200×-more-expensive
MiniLM embedding leaves detection at **0.00** while behavioural AUROC falls
0.82→0.70. The failure is observability, not feature capacity.

**Channel redundancy and its end, from one table.** Evading any single channel
costs 0.055–0.157 detection; evading `e+u+m` together drives detection from
**0.705 to exactly 0.000**. The same measurement gives both the redundancy and
the floor.
→ `results/tables/adversarial_evasion.csv`

**The adversarial limit is followed until it breaks.** A white-box adversary
shaping all three channels reaches the floor; a tamper check closes that; an
adaptive replay defeats the tamper check. The defence is pursued to its own
failure rather than stopped at the first win.

**Budget unreachability as an order statistic.** With 16 healthy validation
episodes the achievable false-alarm floor is 1/(n+1) = **5.9%**, so a 5% budget
is *unreachable*, not merely missed; realized FA is 20%. `pick_threshold`
warns rather than silently missing the budget, and the README states that no
monitor here should be described as respecting it.

**The weakest class is named.** Context corruption at **0.29**, because
corruption that keeps a legal shape needs an external reference.

**Slow goal drift evades every per-step-surprise monitor tested.** The limit is
the *rate* of change: abrupt goal changes are caught at 0.66–0.86.

---

## 3. Deterministic verification

**Precision by construction, on one shared healthy population.** On the same
episodes and the same objective labels, the checks catch 60% of failures at
**0 of 63 false positives** against the monitor's 54% at **11 of 63 (17%)**.
The `tool_contract` check scores **0 of those same 63** too, so all three
layers are comparable here without changing population. At temperature 0.9 —
a *different* corpus, `organic_demo7b_ext` — it is **0 of 38** against 6 of 38.
The 63 and the 38 are different healthy arms and are never summed.
→ `results/tables/verification_vs_monitor.csv`

**False-positive denominators, kept apart.** Three zeros are reported and they
are not one result: the recomputation checks see **0 of 177 healthy** across
the five organic demo corpora where an answer exists to recompute; the contract
check sees **0 of 2,080** across every labelled corpus of ours, because it needs
no answer; and the head-to-head above is **0 of 63** on one corpus. The largest
denominator belongs to the check with the narrowest job, so quoting it for the
other layer overstates that layer's evidence roughly twelvefold.
→ `results/tables/tool_contract_denominators.csv`

**Calibration-free verification transfers where the monitor cannot.** A
llama3.1:8b arm on the same task seeds, nothing retuned: **110 of 110**
failures caught at **0 of 10** false positives. Pooled over the transfer arms,
**217 of 223** failures at **0** false positives on 137 healthy episodes. The
checks carry no calibration, so they have nothing to transfer.
→ `results/tables/verification_organic_llama8b_cold.csv`

**A genuine held-out arm.** 120 episodes at disjoint task seeds, checks frozen:
93% caught with coverage, **0 of 64** false positives, arithmetic errors 36/36.
→ `results/tables/verification_holdout.csv`

**Contract violations are detected at the step the result arrives.** Across
every labelled corpus of ours `tool_contract` trips on **0 of 2,080 healthy** episodes,
and of 218 flagged episodes **215 fire within one step of onset** — 198 at the
onset step itself, 17 one step later.
→ `results/tables/tool_contract_coverage.csv`

**Three checks, none subsuming another.** Totals, coverage and contract each
catch cases the others miss, shown rather than claimed.

---

## 4. The judge, measured against its own stipulation

**The escalation parameter was assumed; we measured it.** A real
gemini-2.5-flash judge scores p_detect **0.548** (95% CI 0.44–0.65) and
p_false **0.052**, against the 0.90 / 0.02 the analysis had assumed — both
stipulated values fall outside their measured intervals.
→ `results/tables/judge_calibration_summary.json`

**The correction was propagated until it reversed our own result.** Substituting
the measured rates across all five master seeds leaves the call saving intact
and drops detection recovery from 82% to **43%**, supported at **0 of 5 seeds**.
The published claim is the lower number.

**Judge and monitor fail on disjoint classes.** The judge is perfect on goal
drift (21/21) and nearly blind on context corruption (4/22, 0.18); the monitor
is the reverse (1.00). This is the empirical case for escalation as a
complementary layer rather than a cheaper approximation.

---

## 5. Repair — detection closed into an outcome

**A measured end-to-end lift.** Task success rises **52% → 73%** with **zero
correct runs broken**, at roughly 1.3 amortised extra model calls per run, over
n=55 genuinely-wrong episodes with three independent repeats each.
→ `results/tables/repair_policies.csv`

**Retry luck is controlled for.** Plain resampling recovers 16%; only the
margin above that is credited to the repair.

**Naming the failing check does at least as well as supplying the answer.**
The value-free `located` rung recovers **45%** (p=0.0001) against `specific` at
36% (p=0.0023) — and 26 of 55 `specific` hints contained the correct total
outright. The recovery does not come from handing over the answer.

**Two rungs are reported as failures.** `recompute` 28% (p=0.13) and `adaptive`
21% (p=0.48) do not beat retry luck. They are kept as comparison arms.

**Rollback is real.** A committed trace plus its seed rebuilds the conversation
at step *k*; every step after is a fresh model call, and the monitor, grounding
and verifier state rewind with the agent.

**Coverage is partial and the shape of the gap is measured.** Every behavioural
alarm is followed by a repair attempt (21 of 21), but `goal_drift` is the only
class a retry fixes (4 of 5). Where the tool layer is broken a retry fetches the
same broken result, and the value of the intervention is ending the episode
fast.

---

## 6. Grounding and content

**The content gate closes the content gap without trading behaviour**, reported
at the *worst* seed rather than the mean: content **+0.307**, behavioural
**+0.039**. Across five seeds content rises 0.278→0.592 and behavioural
0.746→0.791. That `hybrid_logistic_g` *degrades* behaviour (−0.033) is the
evidence that the gating, not the extra signal, does the work.
→ `results/tables/grounding_multiseed_criterion.csv`

**The content gain is population-dependent, and the figure to compare against
the behavioural layer is the matched one.** The grounding study covers 874
episodes over 10 corpora; the behavioural study covers 1,002 over 8, of which
400 are simulator. They share **602 episodes over 7 corpora**, and on that
matched population the content gain is **+0.171**, against **+0.297** on the
grounding study's full population and **+0.559** on the three corpora only it
covers. The behavioural half moves the other way — **+0.072** matched against
+0.054 pooled — so the no-degradation result is stronger where the layers are
comparable. The gain requires two corpus properties, both measured: the
grounding stream must see the corruption (gain tracks it at r = +0.71; the four
corpora where that stream detects nothing average +0.008) and the behavioural
monitor must leave headroom (r = +0.50).
→ `results/layer_alignment_report.md`, `results/tables/layer_alignment_*.csv`

**The fusion-ordering result.** Under a single shared threshold, one
heavy-tailed healthy episode destroyed **16 of 17** grounding-only detections
of context corruption. Under a val-quantile threshold only episode-max
*ordering* matters, so no monotone rescaling repairs it; per-stream thresholds
do. This generalises to any multi-stream one-class detector, and is the most
transferable finding here.

**A null result redirected the architecture.** Across 91 real episodes and
three elicitation methods, qwen2.5:7b and :3b produced **zero** genuine numeric
fabrications — they abstain or mis-add rather than invent. Hallucination onset
therefore moved from a statistical monitor to a deterministic numeric-grounding
verifier, because that is the mechanism the failure actually has.

**Provocation as experimental design.** Making a fraction of price-bearing tool
calls fail *transiently* raises the fabrication base rate without licensing the
model to invent: retrying still yields a grounded total, so inventing is the
model's own choice. That makes a rare class testable, and the verifier catches
**0.55** of the 11 ungrounded-input fabrications it produces at 0 false
positives — specific, and only half sensitive.

---

## 7. Method and cost

**Reservoir computing applied to agent monitoring**, with no prior art found in
the LLM/agent setting. Reservoir computing for time-series anomaly detection is
an established field; the application is what is new here.

**The credit is assigned honestly.** The per-channel max-fusion wrapper, not
the reservoir, carries most of the margin: giving a GRU the same wrapper lifts
it to detection **0.735** / AUROC 0.862, past the ESN's 0.693 / 0.847 measured
in the same run. (Both figures come from `fairness.csv`; the ESN's five-seed
0.71 / 0.872 is a different population and is not the right comparator for
this sentence.) The wrapper is the transferable contribution; the ESN is kept
for a lower healthy false-alarm rate (0.069 against 0.100) and a ~100× faster
fit, which is what makes per-deployment recalibration affordable.

**The cost profile.** **219 µs** median per step (p95 266 µs), **3.95 MB** of
state, **1.7 s** to fit on 240 healthy episodes, timed over 4,316 steps. A GRU
fit takes 68 s. Three orders of magnitude below one agent step, and no second
model, no GPU, no fine-tuning.
→ `results/tables/runtime.csv`

**A 43–60 dimensional causal feature vector, built from observable telemetry
alone.** No model internals are read, because the premise is the deployment
where the weights are not yours to read — not a claim that telemetry beats
activations. Where the weights *are* available the paper says the opposite
(§2: with the weights in hand, activations beat telemetry), and no experiment
here compares the two. The embedding is a deterministic character-trigram hash
— no encoder, no network, no download — and MiniLM at 2,200× the cost adds
nothing measurable.

**Robust to its own hyperparameters.** Reservoir size 32→256 moves AUROC only
0.839–0.853; spectral radius 0.6–1.05 moves it 0.843–0.851. The result is not a
tuning artifact.

**An operating envelope established and then confirmed causally.** Re-collecting
with the requirements met lifted every source by +0.15–0.24 AUC with nothing
else changed — an intervention, not a correlation.

**`safe_scale`.** A degenerate healthy scale clamped to an epsilon turns an
*uninformative* channel into the most sensitive one in the system; measured, a
constant-zero dim blew up to 1.015e9 against a corpus median of 0.458. A
general failure mode in one-class monitoring, not a local bug.

**Healthy-null purity, and its corollary.** A null must contain only runs that
did the task *and got it right*; a systematic error present in ~40% of the
healthy reference becomes undetectable by construction.

**Detection at realized false-alarm rate as the reporting standard**, with the
argument for why AUROC alone is insufficient for a verifier permitted to halt.

---

## 8. The artifact

- **3,226 episodes across 28 corpora**, 1,010 using real tools, four agent models
  and three frameworks — public, loadable, and checksummed.
- **A self-checking claim ledger**: every headline number carries its source
  artifact and regeneration command, and is recomputed from that file on every
  run, gated in CI. Every published rate also carries the denominator it was
  computed over, labelled with what it counts and drift-checked like the value
  — a rate with no `n` cannot be sanity-checked by a reader or by CI, which is
  how an AUC measured on a held-out split of 94 was published as being on a
  corpus of 187.
- **A SHA-256 manifest over every file**, a behavioural-snapshot tripwire that
  re-runs the study and diffs every value, and one master seed that reproduces
  bit-for-bit within an environment.
- Episode AUROC recomputed independently from `results/scores/*.npz` against
  `h1_main.csv` agrees exactly (diff 0.0000).

The evidence discipline is itself a contribution: a reader can verify the chain
from claim to artifact to command in about ten minutes, without running the
study.
