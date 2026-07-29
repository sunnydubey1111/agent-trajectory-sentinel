# Online derailment detection for LLM agents from step telemetry

Can a near-zero-cost temporal model, trained only on healthy runs, watch the
observable telemetry of an LLM agent — semantic trajectory *and* token-level
uncertainty — and raise a calibrated alarm at derailment onset, steps before
the task fails or the budget burns?

This repository implements the full study: a controlled telemetry testbed
with a failure injector (ground-truth onset τ), a family of one-class causal
online monitors built on echo-state-network (ESN) ensembles, six baselines,
and the complete H1/H2/H3 evaluation harness (matched false-alarm budgets,
survivorship-free lead metrics, label-free confidence calibration, and a
cost-accounted escalation policy against a modeled judge-LLM).

**Design docs:** [`DESIGN.md`](DESIGN.md) is the per-module low-level
contract — what each component guarantees, and the telemetry schema every
collector writes.

## Quick start

```
py -m pip install -r requirements.txt
py -m derail.experiments.run_experiment          # full study, ~2 min CPU
py -m derail.experiments.plots                   # figures from results/
py -m derail.experiments.run_experiment --seed 7 # replication -> results/seed7/
py -m derail.experiments.run_experiment --quick  # quarter-size integration run
py -m derail.experiments.run_multiseed           # 5-seed stability, ~35 min
py -m derail.experiments.run_ablation            # ESN hyperparameter sweep
py -m derail.experiments.run_benchmark           # per-step latency / footprint
py -m derail.experiments.run_fairness            # GRU/LSTM fairness diagnostics
py -m derail.experiments.collect_traces --mock-llm   # real-trace pipeline dry run
py -m derail.experiments.run_real_traces         # evaluate collected real traces
py -m derail.experiments.run_hybrid_study        # hybrid ESN+Mahalanobis benchmark
py -m derail.experiments.demo                    # LIVE injection demo -> localhost:8765
py -m derail.verify.run_verification_study       # checks vs monitor, real traces
py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout
py -m derail.intervene.evaluate_repair_policies --parallel 4   # offline repair study
py -m derail.intervene.evaluate_repair_policies --from-csv     # re-analyse, no Ollama
py -m verification.l3_serving_temperature        # serving vs provoking temperature
```

**Reproducing a published number** requires the pinned environment, not the
loose bounds above: `pip install -r requirements.lock.txt` (add
`torch==2.12.0+cpu` from the CPU wheel index for the GRU/LSTM/TCN baselines).
For a hermetic CPU-only, network-free repro there is a **Dockerfile**:

```
docker build -t agentwatch-repro .          # full fidelity (incl. torch baselines)
docker run  --rm agentwatch-repro           # deterministic gate: fast tests + snapshot --check
docker build --build-arg REPRO_MODE=lean -t agentwatch-repro:lean .   # torch-free, reduced snapshot
```

> **Status: statically validated, never built.** Docker is not available in the
> environment this repository was developed in, so the image has not been built
> even once — we say so rather than imply a verified path. What *has* been
> checked, and is enforced by `test_docker_repro_lock_covers_the_gate`:
> `requirements-core.lock.txt` covers **every module-level import** the
> container's gate command reaches, the pytest markers it filters on are
> registered (`--strict-markers` would otherwise fail the run), and
> `.dockerignore` does not exclude `results/` or `traces/`, which the behaviour
> snapshot needs. Two things cannot be checked offline and are what a first
> build would prove: that the `python:3.14-slim` base image and the
> `torch==2.12.0+cpu` wheel both resolve.

`requirements-core.lock.txt` is the synthetic-study subset the image installs;
the real-trace/framework/Gemini stack is intentionally excluded (it collects new
data, it does not reproduce a committed number).

**Recommended production monitor (hybrid study, 2026-07).** The ESN wins
when failures have temporal room to develop (its detection advantage over
a delta-Mahalanobis baseline grows monotonically with the post-onset
horizon: −0.09 at ≤3 steps, +0.23 at 4–8, +0.40 at ≥9; n=716 episodes,
six datasets), while the memoryless distance wins on very short episodes.
`derail.monitor.hybrid.recommended_monitor()` therefore returns the
**hybrid_weighted50** fusion (label-free default, grand-mean AUROC 0.833
vs 0.828 ESN / 0.802 Mahalanobis) and upgrades to **hybrid_logistic**
(0.848, statistically at-or-above the better standalone on every dataset)
when ≥ 20 labeled failure episodes are supplied. Full analysis:
`results/hybrid_report.md`.

**Telemetry v2 (tool results in the semantic channel).** The live demo
exposed a measured blind spot: context corruption poisons *data values*
without changing the agent's *behavior*, and v1 telemetry recorded only the
agent's text and tool arguments — the corrupted results never passed
through the monitor. Tool results are observable step output (no model
internals), so collectors now append each result (truncated) to the step
text: `[lookup_hotel({"city": "Osaka"}) -> $215/night]`. Corrupted results
then shift the semantic embedding directly. The earlier real-trace tables
were produced under v1 telemetry (noted here for provenance); new
collections use v2.

## AgentWatch Live — the failure-injection demo (the presentation climax)

`py -m derail.experiments.demo` (Ollama running, qwen2.5:7b pulled) serves
**AgentWatch Live** at `http://localhost:8765` (rebuilt from scratch
2026-07-18; light professional UI with a plain-English layer, measured
healthy-range band, verbatim injected-payload display, and ground-truth
answer verification):
a real qwen2.5:7b agent works a long booking task while the SHIPPED
grounded content-gate monitor (v4 telemetry, 60 dims: per-channel
ESN-CUSUM max + delta-Mahalanobis + content-grounding stream, dual-budget
serving) scores every step live. Every value the UI shows is normalized so
**1.0 = alarm**; the default chart is one line against that alarm line
(linear axis), with streams and detector channels behind a "monitor
internals" toggle. Nothing is mocked or hardcoded: θ is recomputed from
data at every startup, and all post-injection behavior is real model
output.

**Task-scoped toolset (2026-07-19).** The demo agent gets only the tools
its task needs — flights, hotels, weather, calculator. The shared generic
suite also carries `search_catalog`, which has nothing to do with pricing
a trip and measurably poisoned both the demo and its calibration corpus:
46/113 runs of the previous corpus called it, and a catalog price
contaminated the final total in **15 of the 26 wrong bills** (the agent
variously *added* it or even *multiplied* the whole trip subtotal by it).
Its own spec steered the model — `"Item id, e.g. item-3"` — which is why
`item-3` dominated. Scoping an agent's tools to its task is ordinary
engineering, not demo rigging, and the effect is measured, not asserted:
correct bills rose **51% → 70%** with zero catalog calls (63% on the
corpus as since extended to 120 runs and graded by the stricter total
parser), while the
remaining wrong answers are genuine agent arithmetic errors, so the
ground-truth answer check keeps real work to do. The generic
`TOOL_SPECS` used by every study collector is unchanged. Because the
healthy null must always be collected under the tools actually served,
this required a fresh corpus (`traces/demo7b_scoped/`; the previous
`traces/demo7b/` is retained as historical data, never deleted), and
`fit_monitor()` now **refuses to calibrate** on a corpus containing calls
to a retired tool.

Calibration is one-class **cross-fit**: every healthy demo-task trace is
scored out-of-fold and θ comes from those maxima. Three declared policies
exclude traces from the null: glitched runs (unicode bursts, empty steps),
**task-incomplete runs** that price the trip but never check the weather the
task asks for, and runs that **stated the wrong total**. On the current
120-trace corpus that retains **58**. The last two matter more than they
look — an incomplete run is strongly anomalous to the monitor and a wrong-total
run is simply a failure, so leaving either in the null inflates its spread and
lifts the alarm threshold above where real failures live (DESIGN.md
Amendment 7).

Correctness here is recomputed from the task seed, not read from the
manifest's `success` field, which is a substring test. It is also the one
filter a production deployment cannot copy for free: the demo knows the true
total because its world is seeded, whereas a real deployment would have to
verify its calibration runs by hand. That is a genuine cost of calibration,
not a step that can be skipped. The healthy corpus deliberately spans operating conditions — both
machine-speed regimes (a transient 2.4× inference slowdown once drove
3/3 healthy runs to latency-factor false alarms against a stale null; the
distribution-shift lesson, live) and **probe-extended runs** (benign
follow-up questioning). The probed runs matter: a falsification control
(no-op hijack — the task "rewritten" to identical text so only the probes
differ from healthy) measured that a null WITHOUT probed runs alarms on
the probe pattern alone ~9 steps into probing, which had silently
inflated earlier hallucination/corruption "detections". With probed runs
in the null, that control scores 0.32 (clean) and alarms during probing
are attributable to the failure. Refresh workflow when conditions change:
`py -m derail.experiments.demo --collect-healthy N [--probed]`
(resumable, additive — existing traces are never deleted or pruned).
Machine invariance (2026-07-18 evening): wall-clock latency features are
NEUTRALIZED in the demo monitor, symmetrically at calibration and
serving — on a shared local box they measure the machine (GPU load,
thermals), not the agent, and were the sole driver of every recalibration
churn above. Cloud/API deployments keep them (stationary infrastructure =
real signal); published study tables are unaffected. Post-change
rehearsal: healthy 0.61/0.55 clean under active machine drift, and every
detection margin IMPROVED (hallucination peak 1.48→4.04, hijack
2.09→4.27) because healthy maxima are no longer inflated by machine
noise.

Five buttons inject a real failure mid-run (Loop Trap, Goal Hijack, Tool
Failures, Data Corruption, and the Hallucination button described below);
on alarm the agent is halted (toggleable) and
an **explainability panel** answers "why" from the monitor's own
attribution (per-channel CUSUM shares and per-dim accumulated prediction
error), not canned text.

### Hallucination detection — the honest story and the working detector

The project's core goal is hallucination-onset detection. Two hard,
measured facts shaped how it is delivered:

1. **The behavioural (statistical) monitor does not catch numeric
   hallucination.** Measured, not assumed: the content-grounding channel
   caught 0/4 real organic hallucinations (T5), and injected "answer from
   memory" hallucination fires ~0/10 on qwen2.5:7b (the model ignores the
   note and keeps using tools).

2. **These models barely hallucinate at all.** Across **91 real episodes**
   and three elicitation methods — injection (~0/10), withhold one tool
   result (0/16), withhold all (0/16) — qwen2.5:7b and :3b produced **zero
   genuine numeric fabrications**. When they lack a number they *abstain*
   ("cannot proceed without hotel prices") or make an arithmetic error;
   they do not invent specific false figures. That is good alignment, and
   it is why hallucination cannot be *elicited* from the model on demand.

So hallucination onset is detected by **the right mechanism, not the
statistical monitor**: a deterministic per-step **numeric-grounding
verifier** ([`derail/monitor/grounding_verify.py`](derail/monitor/grounding_verify.py)).
At each step it checks whether every monetary figure the agent asserts
traces to a tool result it actually received (or a legitimate arithmetic
combination). It flags a fabrication the moment it appears, needs **no
ground-truth answer** (deployable online), and cleanly separates a
*fabricated input* (hallucination) from a *wrong total* (arithmetic error,
handled by the answer-check). Verified: catches fabrication in unit tests,
**0 false positives on 25 real healthy runs**, and across both organic
corpora (175 episodes) the objective labeller flags **9 as
hallucinated — but only 2 of those are fabricated *inputs*** (an ungrounded
item figure, the class this verifier targets); the other 7 are *totals* that
match neither the truth nor any arithmetic derivation of grounded inputs.
The verifier itself confirms **neither** of those 2, so unprovoked
input-fabrication is ~0 in 175 runs — far below the pre-registered minimum
of 10, and no claim is made from it.

**Provoked fabrication finally powers the class** (`traces/organic_demo7b_provoked`,
`py -m verification.score_provoked_fabrication`). Making 20% of price-bearing
tool calls fail *transiently* — the retry succeeds, so reporting a grounded
total stays available and inventing the figure is the model's own choice —
raises the rate to **11 ungrounded-input fabrications in 120 episodes**. On
that class the verifier catches **0.55**, with **0 false positives on healthy
episodes** (0/9 here, 0/25 and 0/55 on the two unprovoked corpora). So the
honest reading is: the verifier is specific but only about half sensitive, and
that number exists at all only under provocation.
It is wired live into the demo: on these aligned models it passes
almost always (and says so), and it headlines "Hallucination Caught" the
instant an ungrounded figure appears.

Bottom line, stated plainly: **the detector works; qwen2.5 simply does not
fabricate.** The demo's **Hallucination button** therefore injects the
fault itself, exactly as the other buttons inject theirs (a hijack rewrites
the task, corruption garbles tool results): it inserts a fabricated fee
line — a figure **no tool returned** — into the agent's final answer,
disclosed verbatim in the "Injected Failure" panel. Detection stays fully
live and unrigged: the grounding check is never told which figure was
injected; it verifies every figure in the answer against the tool results
the agent actually received and finds the fabrication on its own (verified
on a live run: injected $171.91 flagged, confirmed absent from every tool
output, the agent's real figures untouched). When it does *not* catch —
the run ends without an answer, or the drawn figure coincidentally matches
a real combination of tool values (the check's one documented
false-negative) — the UI says exactly which case occurred, and why. An
earlier variant that instead rewrote the system prompt to force *real*
fabrication was measured to detect the intervention, not a failure, and is
forbidden in the code. (A brief cross-model mode — running a fabrication-
prone qwen2.5:0.5b as the live agent — was also removed: genuinely
erratic, but too confusing to present.) A pre-registered organic study
(n=55, temperature 0.9) is the evidence base for the "models do not
fabricate" claim; it also motivated the temperature-matched-null principle
(vs 42–47% healthy false alarms for a mismatched null). Full record:
[`verification/ORGANIC_HALLUCINATION_PREREG.md`](verification/ORGANIC_HALLUCINATION_PREREG.md).

**A one-class monitor is only as good as its definition of "healthy"**
(2026-07-28). Three corrections landed together here, and the third reversed
the conclusion of the other two.

1. **In-sample θ.** `score_organic_halluc` selected the alarm threshold on the
   same healthy episodes the monitor was *fit* on; in-sample scores run low, so
   θ landed low and every class over-alarmed. It now uses nested out-of-fold θ.
2. **Degenerate-scale amplification** (DESIGN.md Amendment 6). A telemetry dim
   with no healthy variation was divided by a floor instead of left unscaled,
   so an uninformative channel became the most sensitive in the system —
   healthy episodes peaked at **1.0e9 against a median of 0.46**. Fixed at all
   five sites; the maximum healthy peak is now **2.99**. No committed study
   number moves, and `behavior_snapshot --check` confirms it.
3. **A contaminated healthy null.** Roughly one run in six states the correct
   grand total but never performs the weather lookups the task explicitly asks
   for. The labeller graded only the total, so those runs counted as `healthy`
   and entered the null the monitor calibrates against. They are not healthy —
   the task was not done — and the monitor separates them from genuinely
   healthy runs almost perfectly (AUROC 0.95–0.98). Carrying that many
   strongly-anomalous episodes inside the healthy reference inflated its
   spread, pushed the threshold far above where it belonged, and buried the
   real signal underneath it.

With those runs given their own `incomplete` label, on the seed-paired arm at
the temperature the demo actually serves
([`verification/SERVING_TEMPERATURE_PREREG.md`](verification/SERVING_TEMPERATURE_PREREG.md),
120 episodes per arm):

| | T=0.9 (provoking) | T=0.2 (served) |
|---|---|---|
| healthy false alarms (10% budget) | 16% | 17% |
| arithmetic_error detected | 32% | **46%** (p=0.0025) |
| incomplete detected | 5/18 | **11/13** |
| arithmetic_error AUROC | 0.686 | **0.733** [0.622, 0.835] |

Scored with the earlier label set the same monitor looked like it was at
**chance** (AUROC 0.508). It was not: the null was contaminated. The corpus was
already temperature-matched, toolset-matched, cross-fit and out-of-fold
calibrated — every precaution previously identified — and one over-permissive
label still hid the signal completely. **A null must be built from runs that
did the task, not merely from runs that got the answer.** The `incomplete`
class is derived from the task's own structure, never from the checks below, so
a coverage check catching those runs stays a measurement rather than a
tautology. The label change is a disclosed post-hoc deviation from the
pre-registration; see that document for what it does and does not license.

**The baseline calibrates itself from here on.** A corpus is needed once, not
forever. `derail/monitor/baseline.py` carries the healthy reference as a
rolling window over the deployment's own completed runs, seeded at startup from
the corpus above so the demo begins `trusted` rather than blind. Three
properties matter:

* **It knows which system it belongs to.** `ServingConfig.fingerprint()` covers
  model, temperature, serving prompt, tool roster and telemetry width. When any
  of them moves the null is *retired*, not aged — it describes a different
  system, and keeping it would make the threshold confidently wrong rather than
  merely absent.
* **It cannot be poisoned.** A run joins the window only if it passed the
  deterministic checks and did not itself alarm. That is precisely the failure
  mode that made this corpus unusable until task-incomplete and wrong-total
  runs were removed from it.
* **It says when it cannot judge.** State is explicit — `warming_up`,
  `trusted`, `drifting`, `recalibrating` — and `can_act()` is false while no
  usable threshold exists. How long that lasts is arithmetic, not a guess:
  below 1/(n+1) runs the requested false-alarm budget is unreachable, so at a
  10% budget the monitor stays blind for 9 runs. The deterministic checks need
  no baseline and run from the very first one, which is what makes a blind
  period acceptable at all.

Seeded on the current corpus it reports `trusted` at n=58 with a realized
false-alarm rate of **8.6%** against the 10% budget — the operating point
measured rather than assumed.

## Deterministic verification — checking the answer instead of the trajectory

```
py -m derail.verify.run_verification_study     # head-to-head, real traces
py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout
```

[`derail/verify/checks.py`](derail/verify/checks.py) recomputes a run's stated
total from **the tool results that run actually received**, and confirms every
call the task requires was made. No healthy null, no threshold, no
calibration, and nothing to recollect when the model, temperature, toolset or
framework changes — the recalibration treadmill the behavioural monitor is on
does not apply to a check. It never reads the hidden world the task was
generated from (enforced by `test_checks_never_read_the_hidden_world`), so it
is strictly weaker than the study's oracle and is deployable as-is.

**When each layer fires, and why they differ.** The two layers answer
different questions, so they necessarily report at different times. The
behavioural monitor is causal and scores every step as the episode unfolds: an
anomaly is visible while the run is still going. Correctness is not — there is
nothing to verify until the agent commits to an answer, so `total_consistency`
reports when one exists, and a required call is only *missing* once the run has
stopped. That is a property of the questions, not a limitation of the checks.

In deployment the two compose as one gate: the monitor watches the trajectory
live, and the verification layer stands between the agent and the user, so a
run that fails its checks is rolled back and re-run before anything is
delivered. The user never receives the unverified answer, which is why the
end-to-end number that matters is task success (52% -> 72%) rather than
detection latency.

**AgentWatch Live serves the checks.** The demo reports
the deterministic verdict beside the behavioural monitor: whether the stated
total equals the sum of the figures the tools returned, and whether every
required call was made. Unlike the demo's "Verified Correct" badge — which
compares against the ground-truth total and is therefore possible only in a
demo — the consistency check uses nothing but the run's own tool results, so
what the audience sees is the deployable artefact.

**The demo repairs, too.** When the checks reject an answer the demo rewinds
the agent to its last fact-gathering step and asks again with the finding,
capped at one attempt. The agent's conversation is
rewound; the display keeps every step, so the run, the rejected answer, the
finding and the retry are all visible in order — nothing is hidden to make the
repair look clean.

Measured live over six seeds: three runs were rejected and one of those was
repaired to a correct answer, in line with the 47% the policy comparison
reports.
The two that stayed wrong are shown as `Repair Did Not Fix It`, not quietly
dropped.

Head-to-head on the same episodes and the same objective labels
(`results/tables/verification_vs_monitor.csv`):

| | checks | monitor |
|---|---|---|
| **T=0.2 (served)** failures caught | 60% (**96%** with coverage) | 54% |
| **T=0.2** false positives | **0/63 = 0%** | 11/63 = 17% |
| **T=0.9** failures caught | 65% (**96%** with coverage) | 40% |
| **T=0.9** false positives | **0/38 = 0%** | 6/38 = 16% |

On recall the two are comparable at the served temperature; the difference is
precision. The checks reach 97% with the coverage check and raise **no false
alarm at all**, because a deterministic check has nothing to be uncertain
about. The monitor buys its 61% at 13% false alarms — one healthy run in eight
interrupted — and needs a calibration corpus per configuration to get even
that.

**A second model family** (`results/tables/verification_organic_llama8b_cold.csv`).
The checks were written against qwen2.5:7b. A llama3.1:8b arm was collected at
the same serving temperature on the **same 120 task seeds**, so both models
price identical worlds, and scored with the checks frozen:

| | qwen2.5:7b | llama3.1:8b |
|---|---|---|
| failures caught (with coverage) | 96% | **100%** (110/110) |
| false positives | 0/63 | **0/10** |
| failure rate | 48/120 | 110/120 |

llama fails far more often on this task — 53 of its runs never produce a
parsable answer — but every failure it does produce is caught, and the checks
raise no false alarm on either model. Nothing in them was tuned for llama; the
same `BOOKING_SPEC` and the same code ran on both.

The arm needed `AGENTWATCH_TOOL_NUDGE=1` (103 of 120 runs used it): without it
llama answers the affordance gap between a per-night hotel price and a
two-night stay by inventing tool names, and dies on every episode, so the
comparison would measure that rather than the models. The nudge is recorded
per episode in the manifest.

**One defect this cross-model test found and fixed.** `total_consistency`
originally required the stated total to equal the sum of *every* observed
price. llama priced six flights for a four-leg tour and correctly totalled the
right four — a correct run the check called wrong. It now asks whether some
selection of the declared size reproduces the total, so an unused lookup is
allowed while a dropped or double-counted one still fails. False positives on
the llama arm went 1/10 to 0/10 with detection unchanged on every arm.

**Fabrication, now powered** (`results/tables/verification_provoked.csv`).
Every earlier study called the hallucination class underpowered — it never
reached the pre-registered floor of 10 events. The provoked corpus reaches 26
without injecting anything (a fraction of priced tool calls fail transiently
the first time, so the model may retry or invent). The checks catch **26/26**.
That corpus cannot score the behavioural monitor at all — provoking enough
fabrication leaves 2 healthy episodes, far below the 15 a null needs — which is
the asymmetry in a sentence: a check can be evaluated exactly where a one-class
monitor structurally cannot.

**Held out (`results/tables/verification_holdout.csv`).** The checks were
written by inspecting failures in the serving arm, so that arm cannot also be
their test set. A further 120 episodes were collected afterwards at disjoint
task seeds (40000+, zero overlap) and scored with the checks frozen:

| | design corpus | **held out** |
|---|---|---|
| failures caught (totals check) | 60% | **54%** |
| failures caught (+ coverage) | 96% | **93%** |
| arithmetic errors (+ coverage) | 37/37 = 100% | **36/36 = 100%** |
| false positives | 0/63 = 0% | **0/64 = 0%** |
| arithmetic AUROC (monitor) | 0.733 | **0.824** [0.730, 0.905] |

Arithmetic errors are caught outright on both, and the zero false-positive rate
holds exactly. The overall figure drops by the margin a genuine held-out test
should cost, driven by the small `hallucinated` class (4/8).

Three checks, deliberately complementary and none subsuming the others:
`total_consistency` catches a wrong *combination* of what was looked up (a
dropped line item, a spurious `*3` on the flight subtotal); `required_coverage`
catches work never done, which the totals check cannot see because an agent
that prices three of four legs and then totals those three consistently is
self-consistent. Coverage is what catches the `incomplete` class — runs that
price everything correctly but skip the weather lookups the task asks for —
which it flags 13/13 at 0.2 and 12/12 on the held-out corpus.

**`tool_contract` checks the evidence instead of the answer.** The other two
ask what the agent did with its tool results; this one asks whether those
results were ever valid. Each tool declares the shapes a successful result may
take — `lookup_flight` returns `$361` or `No route found`, nothing else — so a
result matching none of them was malformed at the boundary and the agent
should never have been handed it. Because it needs no answer to compare
against, it reports at the step the result *arrives*, which makes it the
earliest verdict anywhere in the system.

Scored across every labelled corpus in the repository
(`py -m derail.verify.run_verification_study --contract-coverage` →
`results/tables/tool_contract_coverage.csv`): **0 of 1825 healthy episodes**
trip it, which is what lets it ship with no null; it flags **46%** of
`context_corruption` and 44% of `looping`, and **0%** of every other injected
class — it is a contract check, not a general alarm. Where it fires it is
immediate: **215 of 218** flagged episodes are caught within one step of
onset. It closes the demo's Data Corruption button, which the behavioural
monitor sees only weakly because this world's tool results are too terse to
carry statistical mass when garbled.

It is deliberately silent on corruption that keeps a legal shape. A price
altered from `$361` to `$605` is a well-formed price, and telling it from a
real one needs an external reference this layer does not have — the same
boundary drawn at the tool rather than at the answer.

In the live demo this is the **Tool Contract** card. It appears *during* the
run, not with the final answer, because that is the whole point: it names the
step the malformed result arrived on and how many steps ahead of the
behavioural alarm it was. Measured end-to-end on an injected run, the contract
fired at step 6 and the Watchdog at step 12 — a six-step lead, on screen. The
card stays hidden on healthy runs.

**The demo repairs on an alarm too, when halting is off.** The halt toggle
picks which response an alarm gets: halting stops the run for inspection,
while with halting off the alarm is recovered from — one rollback and retry,
capped, with its own budget so it cannot consume the one the checks would use.

Measured over five injection classes x five task seeds, halting off
(`results/tables/alarm_repair.csv`, n=25 live episodes): **every one of the 18
behavioural alarms was followed by a repair attempt**, and no run that did not
alarm was interrupted. What the retry then achieves depends entirely on whether
the tool layer still works:

| class | alarms | repaired | escalated | steps | peak score |
|---|---|---|---|---|---|
| `goal_drift` | 5/5 | **2** | 0 | 15–18 | 3.28–5.87 |
| `context_corruption` | 3/5 | 0 | 0 | 16–18 | 0.62–6.63 |
| `tool_cascade` | 5/5 | 0 | 3 | 11–14 | 3.12–9.53 |
| `looping` | 5/5 | 0 | 5 | **10–10** | **2.95–3.08** |
| `grounding_loss` | 0/5 | — | — | 12–14 | 0.18–0.80 |

Goal drift is the class a retry can fix, and it does so in **2 of 5** runs —
the agent reasoned badly over evidence that was still sound. Where the tools
themselves are broken the retry cannot win, and the value of the intervention
is that it **ends the episode fast**: a loop trap escalates in 5 of 5 runs at
exactly 10 steps with a peak of ~3, against 30 steps and a peak of 97 before
the circuit breaker and the `escalated_tool_layer_down` exit were added.
`grounding_loss` never alarms behaviourally at all (0/5) — it is the
grounding check's class, by design, and the alarm path correctly stays out
of it.

The score does not return under 1.0 for the tool-failure classes (0/5 on both),
and that is stated rather than engineered away: the remaining steps are refused
tool calls, which are genuinely anomalous, and hiding them from the monitor
would be rigging it. The recovery for those classes is a fast clean exit at
about one extra model call, not a healthy trajectory.

A contract violation is **escalated, not repaired**, and the card says so.
Rollback-and-retry works when the agent reasoned badly over sound evidence; if
the tool itself returns something invalid, a retry fetches the same broken
result. Repair in this system is therefore driven by the answer checks, which
is where the 52% → 73% result comes from — not by the contract check.

## Does detection actually improve the agent?

```
py -m derail.intervene.evaluate_repair_policies --parallel 4   # offline; real model calls
py -m derail.intervene.evaluate_repair_policies --from-csv     # re-analyse only
```

Every flagged episode is rolled back to the same checkpoint and re-run under
each repair rung, paired on the identical prefix and task. The rollback is
real: a committed trace plus its seed rebuilds the agent's conversation at
step *k*, and every step after that is a fresh qwen2.5:7b call. Success is
graded by the study oracle (`expected_total`), which the repair prompt never
sees.

Each cell is the mean over **three independent repeats** of every retry, with
the observed range, so a stochastic result is not reported as a point estimate
(n=55 genuinely-wrong episodes):

| rung | rate | range | vs `resample` | calls per recovery |
|---|---|---|---|---|
| `none` — untouched | 0% | — | — | — |
| `resample` — rollback + fresh sample | 16% | 15–18% | *the control* | 14.7 |
| **`located` — + which check failed, no values** | **45%** | 44–47% | **p=0.0005** | 6.4 |
| `generic` — + "re-check your work" | 36% | 35–38% | p=0.0347 | **5.8** |
| `specific` — + the check's finding, with values | 36% | 29–42% | p=0.0192 | 8.1 |
| `recompute` — + "add them with the calculator" | 28% | 25–31% | p=0.17 (n.s.) | 7.2 |
| `adaptive` — specific only when the answer is wrong | 21% | 16–24% | p=0.61 (n.s.) | 10.6 |

Net over all 120 episodes, charging each policy for any correct run it broke:

| policy | correct | rate | recovered | broken |
|---|---|---|---|---|
| none | 63 | **52%** | — | — |
| **located** | 88 | **73%** | 25.0 | 0 |
| generic | 83 | 69% | 19.7 | 0 |
| specific | 83 | 69% | 20.0 | 0 |
| recompute | 78 | 65% | 15.3 | 0 |
| resample | 72 | 60% | 9.0 | 0 |

**What it costs.** The repair fires on 55 of 120 runs (46%), and every figure
below is measured, not assumed — extra model calls from the study rows, step
latency from the retried traces themselves:

| rung | extra calls | s/step | added wall-clock | calls per recovery |
|---|---|---|---|---|
| resample | 2.41 | 2.68 | 6.5 s | 14.7 |
| generic | 2.07 | 2.68 | 5.6 s | **5.8** |
| **located** | 2.89 | 2.69 | **7.8 s** | 6.4 |
| specific | 2.96 | 2.69 | 7.9 s | 8.1 |
| recompute | 2.00 | 2.68 | 5.4 s | 7.2 |

So the recommended policy adds about **7.8 seconds to a flagged run** and buys
one recovered failure per ~6 model calls. Amortised over every run, including
the 65 never flagged, that is ~1.3 extra calls and ~3.6 s per run. `generic` is
marginally cheaper per recovery but recovers nine fewer failures in absolute
terms.

**52% → 73% task success, for about one extra model call per run.** Retry luck is
real and is controlled for — plain resampling alone recovers 16%, so only the
margin above that is credited to the repair. **No policy broke a correct run**
(0 across every rung), because the checks flagged no already-correct episode.

**Asking for a re-check is what works, and naming the failing check works
best.** `located` (45%) leads, with `generic` and `specific` together at 36%;
all three clearly beat the resampling control. Two rungs do not: `adaptive`
(21%, p=0.61), which withholds the prompt when only completeness is at fault,
and `recompute` (28%, p=0.17).

**A rung that should have helped, and did not.** The dominant failure is
arithmetic over figures the agent looked up correctly, and the agent is holding
a calculator it did not use, so `recompute` directs it to add the figures with
that tool instead of in its head. It recovers 28% — better than doing nothing,
but it does not beat retry luck at this sample size and it loses to simply
naming the failing check. Routing the step to a tool that cannot make the error
is a reasonable idea that this measurement does not support; it is kept as a
comparison arm rather than dropped.

**Supplying the recomputed answer buys nothing.** `total_consistency` derives
the total from the agent's own figures, so for a run that merely mis-added,
that value *is* the correct answer, and 26 of 55 `specific` hints contain it.
`located` states which check failed and no value at all (0 of 55). It recovers
at least as much, so the recovery is not coming from being handed the answer —
a result worth having, because it is the objection a reader would raise first.

**Recommended: `located`, and it is what the live demo serves.** It recovers
most (45%, p=0.0005), states no computed value, and gives the operator a reason
for the retry, at essentially the same cost as `specific`. `generic` remains
the cheapest per recovery and a reasonable fallback where the finding cannot be
surfaced. `specific` is not recommended: no better than `generic`, more
expensive, and it hands over an answer it does not need to.

*(An earlier reading of this experiment, under a label set that counted a run
with the right total but missing required work as healthy, found the opposite —
that `specific` damaged correct runs. What it actually did was send the agent
back to finish work it had skipped. See DESIGN.md Module 9.)*

**Rehearsal of record** (2026-07-21, all five classes, task-scoped
toolset, machine-invariant monitor, probed-inclusive null, `--rehearse`;
80-run corpus, θ_b10 = 11.85, θ_b5 = 13.35, alarm line 1.0): healthy ×2
clean (peaks 0.69 / 0.78) — **0 false alarms**; looping **+0**;
hallucination **caught by the grounding check** at the answer (the
Watchdog correctly stayed silent — behaviour was normal, the fabrication
lives in the content); goal hijack **+1** (peak 13.80); tool cascade
**+1**; data corruption **+7** (peak 1.14). **0 missed, 0 false alarms.**
(An immediately preceding gate run had one healthy false alarm at peak
1.01 — the ~1-in-10 the 10% budget allows — plus an Ollama ReadTimeout
that aborted one scenario; both disclosed here, clean on rerun.)

Removing the contaminating catalog tool tightened the healthy null
(θ_b10 16.12 → 11.85), which lifted data corruption over the line in this
rehearsal (+5, peak 1.10). **This does not make corruption a solved
class:** it cleared the line by only 0.10, and the 26-episode E2E matrix
measured it at 2/4 across injection timings. Treat corruption as
**run-dependent**, not guaranteed; the honest-miss banner still exists
because misses still happen. Corruption remains bounded by telemetry
completeness — terse tool results carry little statistical mass for
garbling to move the content dims (both injector flavors, word-shuffle
and mojibake, were measured).

Broader live evidence — a 26-episode E2E matrix across three injection
timings (arm@4 / @7 / @10, plus halt-off playouts) — is in
an internal end-to-end audit: **18/21
injections detected**, with two mechanistic findings: hallucination
injected *late* can be undetectable by construction (the agent already
gathered the real data, so "answer from memory" yields a grounded,
verified-correct answer), and corruption is borderline run-dependent
(2/4). (At a 10% FA budget, ~1 in 10 healthy runs may still legitimately
alarm — by design.)

Every module has a self-contained smoke test: `py -m derail.telemetry.generator`,
`py -m derail.monitor.esn`, `py -m derail.monitor.baselines`,
`py -m derail.monitor.calibration`, `py -m derail.monitor.escalation`,
`py -m derail.evaluation.metrics`.

## The problem (one paragraph)

An agent episode is a sequence of steps t = 1..T; each step emits an
observable signal x_t = [e_t; u_t; m_t]: a semantic embedding of the step's
output, per-step aggregates of the token-logprob stream (mean/max entropy,
entropy slope, high-entropy fraction), and action metadata (type, latency,
output length, error flag). Episodes are healthy or contain a derailment
onset at unknown step τ, after which the trajectory distribution shifts and
the episode ends in failure at T_fail. From **healthy episodes only**, learn
a **causal online** monitor emitting a score s_t and alarming at
τ̂ = min{t : s_t > θ} — maximizing detection lead (T_fail − τ̂) at a fixed
false-alarm budget, with calibrated alarm confidence driving cost-optimal
escalation to an expensive judge-LLM. Per-step monitor compute ≪ one LLM
call (a few small matrix-vector products).

## Layout

```
DESIGN.md                        implementation contract + amendments
derail/common.py                 channel layout, Episode, OnlineMonitor ABC
derail/config.py                 secure API-key resolution (OS vault > env > .env)
derail/telemetry/generator.py    healthy simulator + 5-class failure injector
derail/telemetry/adapter.py      real-trace JSONL -> Episode converter
derail/monitor/esn.py            ESN ensemble (EWMA + CUSUM streams),
                                 per-channel max fusion (primary monitor)
derail/monitor/baselines.py      cosine/self drift, entropy, Mahalanobis,
                                 delta-Mahalanobis, isolation forest
derail/monitor/seq_baselines.py  trained sequence baselines: VAR-ridge,
                                 GRU, LSTM, TCN (same one-class protocol)
derail/monitor/calibration.py    label-free (healthy-ECDF) + oracle isotonic
derail/monitor/escalation.py     modeled judge, 4 policies, cost accounting
derail/evaluation/metrics.py     alarms, thresholds, lead/delay, ECE, bootstrap
derail/evaluation/stats.py       paired permutation + McNemar tests
derail/experiments/run_experiment.py   end-to-end study -> results/
derail/experiments/run_multiseed.py    5-seed stability (mean +/- std)
derail/experiments/run_ablation.py     ESN hyperparameter sensitivity
derail/experiments/run_benchmark.py    fit time, per-step latency, footprint
derail/experiments/run_fairness.py     GRU/LSTM convergence + wrapper parity
derail/experiments/collect_traces.py   live Gemini agent -> traces/*.jsonl
derail/experiments/run_real_traces.py  fit + evaluate on real traces
derail/experiments/plots.py      figures -> results/figures/
```

## Failure classes (injector fixes ground-truth τ, severity, onset ramp)

| class | designed signature |
|---|---|
| goal_drift | gradual semantic rotation toward a distractor goal; confident |
| looping | cycling among recent states; repeated actions; entropy drops |
| tool_cascade | error flags + latency inflation + retry ping-pong |
| grounding_loss | semantics stay plausible; uncertainty channel shifts |
| context_corruption | AR coherence collapses; dynamics unpredictable |

Low-severity onsets ramp in slowly and are genuinely hard; injected pre-τ
steps are statistically indistinguishable from healthy ones.

## Headline results (H1/H3b SUPPORTED at 4 of 5 seeds; H2/H3a at all 5)

Numbers below come from the current regeneration (five full-size seeds
under the corrected ESN fit/score alignment and the label-independent
evaluation protocol). H1 and H3b are honestly **not supported / mixed at one
of the five seeds** — the survivorship-free picture — and supported at the
other four; H2 and H3a hold at every seed.

- **H1 (temporal advantage) — SUPPORTED at 4 of 5 seeds.** The primary
  monitor (`esn_cusum_max`: one ESN-CUSUM detector per channel, alarm on
  the max) reaches detection 0.71 ± 0.07, 4.6 ± 1.0 steps of budget saved
  per failure episode, AUC 0.872 ± 0.015 (mean ± std over five dataset
  seeds) at a 5% false-alarm budget, vs 0.37 ± 0.03 detection and
  3.6 ± 0.4 steps for the best memoryless baseline (delta-Mahalanobis).
  Paired permutation and McNemar tests vs every memoryless baseline are
  significant (delta-Mahalanobis: 130-vs-4 discordant detections, McNemar
  p ≈ 1e-33, paired-lead perm p = 2e-4).
- **Trained sequence baselines (GRU / LSTM / TCN / linear VAR), same
  one-class protocol.** The ESN channel-max beats monolithic GRU (det
  0.60 ± 0.01, AUC 0.82), LSTM (det 0.61 ± 0.02, AUC 0.82 —
  indistinguishable from the GRU, as expected at this scale) and TCN
  (det 0.39 ± 0.04, AUC 0.74), while fitting in seconds (ridge readout)
  instead of minutes of backprop. The fairness study (below) shows the
  per-channel max-fusion wrapper — not the reservoir per se — carries most
  of that margin: it is the transferable architectural contribution. The
  honest surprise: **linear vector-autoregression is a strong baseline** —
  best expected budget saved (6.1 ± 0.2), and it actually *leads* the ESN
  by ~1.9 steps (that lead advantage is significant, Holm p = 0.001),
  though it is far behind on detection (0.61 vs 0.71; McNemar 60-vs-24
  discordant, p ≈ 1e-4). Much of the temporal signal is linear; the ESN's
  edge is catching *more* failures, not catching them earlier.
- **H2 (channel complementarity) — SUPPORTED, with two twists.** The
  uncertainty channel alone detects grounding loss perfectly (det 1.00)
  where the semantic channel is blind (det 0.00); the semantic channel owns
  looping and context corruption; metadata owns tool cascades. Twist 1: a
  monolithic ESN averaging surprise over all 43 dims *dilutes* the
  4-dim uncertainty signal (grounding-loss det 0.24) — per-channel detectors
  fused by max fix this, and that fusion is the primary monitor. Twist 2:
  slow goal drift evades *every* per-step-surprise channel (even with CUSUM
  accumulation) because each step stays locally predictable; it is caught
  only by a trajectory self-consistency statistic (`self_drift`,
  1 − cos(e_t, running centroid), det ≈ 0.4). Complementarity holds across
  monitor *families*, not just channels.
- **H3a (calibration) — SUPPORTED at all 5 seeds.** The label-free null
  calibrator (1 − p-value of the running-max score under the
  healthy-validation ECDF) is validated by the *uniformity* of its healthy
  scores rather than a category-mismatched ECE: KS distance to uniform
  ≈ 0.06 for the best component stream (surprise), 0.12 fused, with a
  realized false-alarm ≈ 0.05–0.07 at the 0.95 confidence gate. The labeled
  isotonic oracle posterior reaches ECE ≈ 0.02–0.03. The label-free readout
  is usable for ranking and escalation gating, not yet for calibrated
  probability readouts.
- **H3b (escalation) — SUPPORTED at 4 of 5 seeds (MIXED at one).**
  Escalating to the (modeled) judge only while the monitor is confident —
  operating point selected on the cal split — recovers 83% of
  judge-every-step detection (0.82 vs 0.99) at 8% of its judge calls
  (2.2 vs 29.1 per episode; total-cost ratio 61%) with zero wrongful halts
  of healthy episodes.

Figures in `results/figures/`: score traces per class, H1 lead comparison
(95% bootstrap CIs), H2 channel×class heatmap, reliability diagram,
escalation frontier.

## Robustness and statistics

- **Hyperparameter ablation** (`results/tables/esn_ablation.csv`):
  one-at-a-time sweep over reservoir size (32–256), spectral radius
  (0.6–1.05), leak rate (0.15–0.7), ensemble size K (1–16), CUSUM drift
  allowance (0.25–1.0), and disagreement weight. Detection stays in
  0.62–0.72 and AUC in 0.84–0.87 across the grid, dropping to ≈0.56 only at
  the degenerate ends (a single reservoir K=1, or zero ensemble-disagreement
  weight) — no tuning cliff in the main hyperparameters, and the defaults
  were not cherry-picked (several off-default cells — β_disagreement=1,
  reservoir 256, spectral radius 0.6 — are marginally better).
- **Paired significance tests** (`results/tables/h1_significance.csv`):
  sign-flip permutation test on per-episode budget-saved differences and
  exact McNemar on detection outcomes, primary vs every other monitor,
  paired by test episode.
- **Multi-seed stability** (`results/tables/multiseed_summary.csv`,
  `results/multiseed.json`): mean ± std of detection / lead / AUC over five
  fresh dataset seeds, plus per-seed hypothesis verdicts.
- **Training-fairness study** (`results/tables/fairness.csv`, single
  diagnostic seed): the backprop baselines are properly trained — training
  loss falls < 8% over the final quarter of the default 40 epochs, and
  neither 3× the epochs (GRU AUC 0.824 → 0.827) nor 2× the hidden width
  (0.825) closes the monolithic-GRU gap. The decisive test gives the GRU the
  ESN primary's own per-channel max-fusion wrapper: `gru_cusum_max` reaches
  det 0.76 / AUC 0.873 — so most of the wrapped monitor's margin over the
  **monolithic** GRU comes from the wrapper, which transfers across predictor
  families and is the reusable architectural contribution. Honest nuance:
  on this single-seed diagnostic the wrapped GRU actually *edges the ESN ref*
  on detection and AUC (0.76 / 0.873 vs 0.69 / 0.847); the ESN keeps the
  false-alarm advantage (FA 0.069 vs 0.113 — it respects the 5% budget where
  the wrapped GRU does not) and fits ~100× faster (closed-form ridge vs three
  backprop runs). It stays the primary for that cost/false-alarm profile and
  for its multiseed detection lead, not for a raw-detection win on every
  dataset.
- **Runtime** (`results/tables/runtime.csv`, `score_step` with nothing else
  running): the primary monitor scores a step in ~219 µs median (p95
  266 µs), fits on 240 healthy episodes in 1.7 s, and holds 4.0 MB of state
  — roughly three orders of magnitude below one LLM agent step. (Wall-clock
  latencies are machine-sensitive; the artifact is the source of record.)
  Single ESNs score in 30–89 µs and fit in ~0.1–0.7 s; the GRU needs 68 s of
  backprop to fit (vs seconds for ridge readouts), and the TCN's windowed
  streaming costs 642 µs/step.

## Methodological guarantees

- **One-class discipline.** Monitors fit on healthy train only; thresholds
  from healthy val only; the labeled cal split feeds only the isotonic
  oracle and the escalation operating point; every reported number is test.
- **Causality.** Every score/confidence stream at step t is a function of
  x_1..x_t plus fit-time quantities — no lookahead, no full-episode stats.
- **Survivorship-free comparison.** Monitors are compared by expected steps
  saved per failure episode (misses count 0), not lead-among-detected.
- **Determinism.** One master seed reproduces results bit-for-bit;
  `--seed N` writes a replication to `results/seedN/`.
- Two independent review findings were fixed: escalation operating point
  was previously selected on test (winner's curse), and the oracle
  calibrator was previously fit at a different class prevalence than it was
  evaluated at. See DESIGN.md "Post-contract amendments".

## Plugging in real traces

External validation is an adapter problem, not a rewrite: everything above
the telemetry layer consumes only `Episode` objects. `derail/telemetry/
adapter.py` converts JSONL traces (one step object per line: `text`,
`token_logprobs`, `action`, `latency_s`, `error`) from any framework
(LangGraph, AutoGen, CrewAI, a bespoke loop) into Episodes:

```python
from derail.telemetry.adapter import load_trace_jsonl
healthy = [load_trace_jsonl(p) for p in Path("traces/healthy").glob("*.jsonl")]
monitor.fit(healthy)                       # one-class, as in the study
```

Semantic embeddings use sentence-transformers when installed (deterministic
hashing embedding otherwise). Monitors must be re-fit on healthy runs from
the same source — do not mix simulator-trained monitors with real traces.

### Collecting real Gemini traces (built in)

The repo ships a live trace collector: a tool-using Gemini agent on a
deterministic local task suite (mock flights/hotels/catalog/calculator
tools), with **live failure injection at a known step τ** — tool cascades,
looping, goal drift (the task text is silently rewritten mid-run), and
context corruption (earlier tool results are garbled). The derailment that
follows is real model behavior; only the trigger is controlled. Gemini's
`response_logprobs` populates the uncertainty channel with real per-token
surprisal, so all three channels are live on real traces.

```
py -m pip install google-genai keyring
py -m derail.config set-key GEMINI_API_KEY              # one-time, hidden input
py -m derail.experiments.collect_traces --mock-llm      # offline dry run (free)
py -m derail.experiments.collect_traces --estimate      # cost preview
py -m derail.experiments.collect_traces --yes           # real collection (~$0.72 default)
py -m derail.experiments.run_real_traces                # fit + evaluate -> results/tables/real_traces.csv
```

**Key handling** (`derail/config.py`): the key is stored in the OS
credential vault (Windows Credential Manager) via `keyring`, entered with
hidden input — never typed into a shell command, never echoed, never
logged; `check` prints only a masked suffix. Fallbacks: environment
variable, then a gitignored `.env`. For a server deployment use a real
secret manager instead. Real collection refuses to run without `--yes`
after printing the cost estimate.

Honest constraints: grounding-loss remains simulator-only (it is natural
hallucination — not injectable); if the model/tier rejects
`response_logprobs`, the collector degrades to e+m and says so; and the
mock-LLM dry run validates the *pipeline only* — its traces are scripted,
so its near-chance detection numbers are expected.

**How much does that logprob degradation cost? Almost nothing.** Ablating the
token-surprisal channel *within* four corpora that carry it — same episodes,
splits, seeds and thresholds — moves the ESN by **AUROC +0.000 / detection
+0.002** on average (`py -m experimental.telemetry_dependence` →
`results/tables/telemetry_dependence.csv`). The real deployment requirement is
**structured tool results**, not logprob access, so a provider that refuses
logprobs is still fully supported.

**Cross-framework local validation** (Ollama + qwen2.5:3b, zero API cost;
`results/tables/real_traces_{ollama,langgraph,autogen}.csv`): the same
monitor stack was run on traces from three additional sources — our native
loop on Ollama (74 eps, and this Ollama version DOES return per-token
logprobs, so the u channel is live locally), a genuine LangGraph StateGraph
agent (55 eps), and AutoGen's AssistantAgent (59 eps). Results are honest
and mixed: AutoGen shows real signal (linear VAR AUC 0.785, GRU 0.701,
channel-max 0.653), LangGraph and native-Ollama sit near chance. The
boundary condition this exposes: detection degrades sharply when healthy
training sets are tiny (13–20 episodes vs 47 for Gemini), episodes are
barely longer than the monitor's 3-step washout (T≈4 leaves ~1 scored
step), and the agent model itself is small and erratic (a 3B model's
healthy behavior is noisy enough to drown failure shifts). This is an
operating-envelope finding, not a pipeline failure — and it was
**confirmed causally**: re-collecting with the requirements met
(qwen2.5:7b, 60 healthy + 12/class per source, episodes T=5–8;
`real_traces_{ollama7b,langgraph7b,autogen7b}.csv`) lifted every source by
+0.15–0.24 AUC with nothing else changed. 7B results: AutoGen channel-max
**AUC 0.856** (det 0.61 at 8% FA — budget-respecting), Ollama-native
channel-max **0.800 with the live e+u+m channels** (delta-Mahalanobis 0.812
close behind), LangGraph 0.605 for channel-max with the Mahalanobis family
leading there (0.745) — no single monitor dominates every source, echoing
the complementarity theme. The 3B tables are retained as the envelope
evidence.

**Telemetry v3 — derived behaviour channel x** (dims 43–50, opt-in via
`extended=True` in the adapter; DESIGN.md item 13): eight portable features
computed causally from the raw steps — cosine embedding drift vs the
previous step, similarity to the task anchor (step-0 embedding), per-step
tool success rate, retry count (exact repeats of earlier calls), log
per-tool latency, approximate context-utilization ratio, reasoning depth
(# tool calls per step), and self-consistency vs the run's own running-mean
embedding. The simulator and every 43-dim table above are untouched.
`py -m derail.experiments.run_real_traces --extended` compares the primary
channel-max with and without x on identical splits and seeds
(`results/tables/real_traces_ext*.csv`):

| source (channels) | channel-max det / AUC | + x channel det / AUC |
|---|---|---|
| Gemini, 187 eps (e+m) | 0.63 / 0.787 | **0.65 / 0.802** |
| Ollama-native 7B (e+u+m) | 0.38 / 0.800 | **0.43 / 0.807** |
| AutoGen 7B (e+m) | 0.61 / 0.856 | **0.67** / 0.854 |
| LangGraph 7B (e+m) | 0.46 / 0.605 | **0.54 / 0.631** |

Detection improves on **all four sources at the same FA budget** (+2 to +8
points); AUC rises on three and is flat on AutoGen. The x channel alone is
weak-to-moderate (AUC 0.45–0.80, or over the FA budget) — it complements
the core channels rather than replacing them, echoing the complementarity
theme. Caveat: these trace sets predate the v2 text format, so tool
*results* are absent from their step texts and the success-rate feature is
inert there; retry counts, call depth, per-tool latency, context ratio, and
the three embedding-derived features carry the gain.

**Real-trace results** (187 episodes, gemini-2.5-flash, free tier —
logprobs tier-gated, so e+m channels; `results/tables/real_traces.csv`):
the e+m channel-max monitor is the strongest on real agent behavior —
episode AUC 0.787 with detection 0.63 at 6.7% healthy false alarms (the
only strong monitor that actually respects the 5% budget; LSTM reaches
det 0.68 only by blowing FA to 27%). Full roster on real traces: GRU
AUC 0.76, LSTM 0.75, linear VAR 0.75, delta-Mahalanobis 0.73,
Mahalanobis 0.67, isolation forest 0.65, self-drift 0.42 (real episodes
are too short for centroid drift). Per class (channel-max): looping 0.94
(injected loops visibly run to T≈13 vs T≈5 healthy), tool cascade 0.83,
goal drift 0.59, context corruption 0.24. Nuance vs the simulator: the
live goal-drift injection is an *abrupt* task rewrite, which per-step
surprise does catch; the simulator's *slow* drift remains the documented
hard case. Small-sample caveats still apply (62 usable healthy episodes);
a paid-tier key adds the u channel with zero code changes.

## Honest limitations

- The main testbed is synthetic: failure-class channel signatures are
  designed in, so H2's *direction* is partly by construction; the
  non-obvious findings (dilution under naive fusion, CUSUM's slow-drift
  blindness, the family-complementarity axis, the wrapper-vs-reservoir
  attribution) were not designed and constitute the actual contribution.
  The real-trace validation (187 live Gemini episodes) confirms the
  per-class structure, but at small scale, without the uncertainty channel
  (tier-gated logprobs), with injected rather than naturally occurring
  failures, and on one agent/task suite — broader real-world coverage
  remains future work.
- **Deployment requirement (coverage is conditional, not unconditional).**
  The monitors consume a specific telemetry contract, and each channel is
  only as available as its input: the semantic (`e`) and content-grounding
  (`g`) channels need **structured tool-call/result events** in the trace
  (name, args, and the actual result text); the uncertainty (`u`) channel
  needs **token-level logprobs**. Where a provider tier-gates or omits
  logprobs the collector degrades to `e+m` and *says so* (it never
  fabricates a `u` channel), and where tool results are unstructured the
  `g` channel is inert (all-zero). Reported detection is therefore
  conditional on this contract being met; a deployment that exposes neither
  logprobs nor structured results gets only the metadata channel and
  correspondingly reduced coverage. See `derail/telemetry/adapter.py`.
- The judge-LLM was a modeled noisy oracle; it is now **measured** — a real
  gemini-2.5-flash judge scores p_detect 0.548 / p_false 0.057 against the
  stipulated 0.90 / 0.02, which halves the H3b detection-recovery claim
  (`py -m derail.experiments.run_judge_calibration`, and
  `experimental/judge_sensitivity.py` for the 5-seed consequence).
- Robustness to an adversary who deliberately shapes the telemetry is a
  **measured limit, not a defended capability** (`experimental/adversarial_evasion.py`):
  single-channel evasion is blunted by max-fusion (detection 0.70 → 0.55–0.65
  when one of e/u/m is healthy-ised), but a full white-box adversary shaping
  all three behavioural channels drops detection to the false-alarm floor
  (0.70 → 0.00). A **cross-channel tamper check**
  (`experimental/tamper_check.py`) closes that specific attack completely —
  pinning a channel to a constant destroys its temporal variability, which no
  genuine episode does, so all pinned episodes are flagged (0.00 → 1.00) for
  +3pp healthy false alarms. But an **adaptive** adversary who replays a real
  healthy episode's channel trace keeps the variability and defeats the check
  (flag rate 0.05 = no signal; combined detection only 0.35). So adversarial
  robustness remains **future work**: the naive evasion is cheap to close, the
  informed one is not. Repairing/steering the agent post-alarm is likewise out
  of scope.
- The primary monitor (channel-max CUSUM) was chosen after test-set
  diagnostics on the headline seed; the four untouched-seed replications
  (`results/seed{7,101,202,303}/`) are the guard against that selection
  overfitting — and are reported honestly: H1/H3b hold at four of the five
  seeds and are not supported / mixed at seed 7, H2/H3a at all five.
