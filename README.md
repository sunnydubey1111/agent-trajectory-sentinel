# AgentWatch — Real-Time Detection and Repair of LLM Agent Failures

Can a near-zero-cost temporal model, trained only on healthy runs, watch an
agent's step telemetry and raise a calibrated alarm at derailment onset —
steps before the task fails or the budget burns?

AgentWatch answers that with two layers that compose into a single gate:

1. **A behavioural monitor.** Echo-state-network CUSUM ensembles fit on
   healthy episodes only, scoring every step causally. It sees trajectory
   failures live, while the run is still going.
2. **Deterministic verification.** Checks that recompute a run's answer from
   the tool results that run actually received. No healthy null, no
   threshold, no calibration — and no false alarms.

On a real qwen2.5:7b booking agent, the two layers plus rollback-and-retry
take **task success from 52% to 73%** for about one extra model call per run.

## Install

```
pip install -r requirements.txt          # development
pip install -r requirements.lock.txt     # pinned — required to reproduce a published number
```

Python 3.13+. The GRU/LSTM/TCN baselines additionally need
`torch==2.12.0+cpu` from the CPU wheel index; everything else runs without it.

## Quick start

```
py -m derail.experiments.run_experiment        # full synthetic study, ~2 min CPU
py -m derail.experiments.plots                 # figures from results/
py -m derail.verify.run_verification_study     # deterministic checks vs monitor, real traces
py -m derail.intervene.evaluate_repair_policies --from-csv   # repair study, re-analysis only
py -m derail.experiments.demo                  # live demo -> localhost:8765
```

The demo needs a local Ollama with `qwen2.5:7b` pulled. Everything else is
CPU-only and offline.

<details>
<summary>Full command list</summary>

```
py -m derail.experiments.run_experiment --seed 7   # replication -> results/seed7/
py -m derail.experiments.run_experiment --quick    # quarter-size integration run
py -m derail.experiments.run_multiseed             # 5-seed stability, ~35 min
py -m derail.experiments.run_ablation              # ESN hyperparameter sweep
py -m derail.experiments.run_benchmark             # per-step latency / footprint
py -m derail.experiments.run_fairness              # GRU/LSTM fairness diagnostics
py -m derail.experiments.run_hybrid_study          # hybrid ESN + Mahalanobis
py -m derail.experiments.collect_traces --mock-llm # real-trace pipeline dry run
py -m derail.experiments.run_real_traces           # evaluate collected real traces
py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout
py -m derail.intervene.evaluate_repair_policies --parallel 4   # offline; real model calls
py -m verification.l3_serving_temperature          # serving vs provoking temperature
```

Every module also carries a self-contained smoke test:
`py -m derail.telemetry.generator`, `py -m derail.monitor.esn`,
`py -m derail.monitor.baselines`, `py -m derail.monitor.calibration`,
`py -m derail.monitor.escalation`, `py -m derail.evaluation.metrics`.

</details>

## The problem

An agent episode is a sequence of steps `t = 1..T`. Each step emits an
observable signal `x_t = [e_t; u_t; m_t]`: a semantic embedding of the step's
output, per-step aggregates of the token-logprob stream, and action metadata
(type, latency, output length, error flag). Episodes are healthy, or contain a
derailment onset at unknown step τ after which the trajectory distribution
shifts and the run ends in failure.

From **healthy episodes only**, learn a **causal online** monitor emitting a
score `s_t` and alarming at `τ̂ = min{t : s_t > θ}` — maximising detection lead
at a fixed false-alarm budget. Per-step compute must stay far below one LLM
call.

## Results

### Behavioural monitor — synthetic testbed, five seeds

The primary monitor is `esn_cusum_max`: one ESN-CUSUM detector per channel,
alarm on the max.

| | detection | steps saved | AUC |
|---|---|---|---|
| **`esn_cusum_max`** (primary) | **0.71 ± 0.07** | 4.6 ± 1.0 | **0.872 ± 0.015** |
| delta-Mahalanobis (best memoryless) | 0.37 ± 0.03 | 3.6 ± 0.4 | — |
| GRU (monolithic) | 0.60 ± 0.01 | — | 0.82 |
| LSTM (monolithic) | 0.61 ± 0.02 | — | 0.82 |
| TCN | 0.39 ± 0.04 | — | 0.74 |
| linear VAR-ridge | 0.61 | **6.1 ± 0.2** | — |

Mean ± std over five dataset seeds at a 5% false-alarm budget. Paired
permutation and exact McNemar tests against every memoryless baseline are
significant (vs delta-Mahalanobis: 130-vs-4 discordant detections,
McNemar p ≈ 1e-33).

Two findings worth the space. **Linear vector-autoregression is a strong
baseline** — it saves the most budget and actually leads the ESN by ~1.9 steps
— but it is far behind on detection. Much of the temporal signal is linear;
the ESN's edge is catching *more* failures, not catching them earlier. And
**the per-channel max-fusion wrapper, not the reservoir, carries most of the
margin**: giving a GRU that same wrapper lifts it to det 0.76 / AUC 0.873. The
wrapper is the transferable contribution; the ESN stays primary for its
false-alarm discipline (0.069 vs 0.113) and ~100× faster fit.

Hypothesis verdicts, reported per seed rather than pooled: **H1** (temporal
advantage) and **H3b** (cost-accounted escalation) are supported at 4 of 5
seeds and not supported at seed 7; **H2** (channel complementarity) and
**H3a** (calibration) hold at all five.

**Cost.** ~219 µs median per step (p95 266 µs), 1.7 s to fit on 240 healthy
episodes, 4.0 MB of state — roughly three orders of magnitude below one agent
step.

### Deterministic checks vs the behavioural monitor — real traces

Same episodes, same objective labels
(`results/tables/verification_vs_monitor.csv`):

| | checks | monitor |
|---|---|---|
| failures caught, T=0.2 (served) | 60% (**96%** with coverage) | 54% |
| false positives, T=0.2 | **0/63 = 0%** | 11/63 = 17% |
| failures caught, T=0.9 | 65% (**96%** with coverage) | 40% |
| false positives, T=0.9 | **0/38 = 0%** | 6/38 = 16% |

Recall is comparable at the served temperature; the difference is precision. A
deterministic check has nothing to be uncertain about, and needs no
recalibration when the model, temperature, toolset or framework changes.

The checks were written by inspecting failures in the serving arm, so that arm
cannot also be their test set. A further **120 episodes at disjoint task seeds**
were collected afterwards and scored with the checks frozen: 93% caught with
coverage, **0/64 false positives**, arithmetic errors 36/36. A **llama3.1:8b**
arm on the same 120 task seeds, nothing retuned: **110/110** failures caught,
0/10 false positives.

Three complementary checks, none subsuming the others — `total_consistency`
(wrong combination of what was looked up), `required_coverage` (work never
done), and `tool_contract`, which asks whether a tool result was ever valid
and therefore reports at the step the result *arrives*: **0 of 1825 healthy
episodes** trip it, and 215 of 218 flagged episodes are caught within one step
of onset.

### Does detection improve the agent?

Every flagged episode is rolled back to the same checkpoint and re-run under
each repair rung, paired on the identical prefix and task. Rollback is real: a
committed trace plus its seed rebuilds the conversation at step *k*, and every
step after is a fresh model call. Each rate is the mean of three independent
repeats (n=55 genuinely-wrong episodes):

| rung | recovery rate | vs `resample` | calls per recovery |
|---|---|---|---|
| `none` — untouched | 0% | — | — |
| `resample` — rollback + fresh sample | 16% | *the control* | 14.7 |
| **`located` — + which check failed, no values** | **45%** | **p=0.0005** | 6.4 |
| `generic` — + "re-check your work" | 36% | p=0.0347 | **5.8** |
| `specific` — + the finding, with values | 36% | p=0.0192 | 8.1 |
| `recompute` — + "use the calculator" | 28% | p=0.17 (n.s.) | 7.2 |
| `adaptive` | 21% | p=0.61 (n.s.) | 10.6 |

Net over all 120 episodes, charging each policy for any correct run it broke:
`located` takes success **52% → 73%**, recovering 25 failures and breaking
**zero** correct runs. Retry luck is controlled for — plain resampling alone
recovers 16%, so only the margin above that is credited to the repair.

Asking for a re-check is what works, and *naming the failing check* works
best. Supplying the recomputed answer buys nothing: `located` states no
computed value at all and still recovers at least as much as `specific`.

Cost: ~7.8 s added to a flagged run; the repair fires on 46% of runs, so
amortised that is ~1.3 extra calls and ~3.6 s per run.

### Hallucination

The project's core goal was hallucination-onset detection, and the measurement
reshaped the delivery. Across **91 real episodes** and three elicitation
methods, qwen2.5:7b and :3b produced **zero genuine numeric fabrications** —
lacking a number, they abstain or mis-add rather than invent. That is good
alignment, and it means hallucination cannot be elicited on demand.

So hallucination onset is caught by the right mechanism rather than the
statistical monitor: a deterministic per-step **numeric-grounding verifier**
([`derail/monitor/grounding_verify.py`](derail/monitor/grounding_verify.py))
that checks whether every monetary figure the agent asserts traces to a tool
result it actually received. It needs no ground-truth answer, so it is
deployable online. Measured: **0 false positives across 89 real healthy runs**
(0/25, 0/55, 0/9), and on a provoked-fabrication corpus that finally powers the
class (11 ungrounded-input fabrications in 120 episodes) it catches **0.55**.
Specific, about half sensitive — stated as measured.

### Real traces and other frameworks

187 live gemini-2.5-flash episodes (`results/tables/real_traces.csv`): the
channel-max monitor leads every other monitor on real agent behaviour at
**AUC 0.840 / detection 0.71**. Per class: looping 1.00, tool cascade 0.83,
goal drift 0.77, context corruption 0.29.

Its realized false-alarm rate is **20%**, and that number should be read for
what it is. The test split holds 15 healthy episodes, so a single episode moves
the rate by 6.7 points; the validation split holds 16, and an empirical
quantile over 16 episodes *cannot* deliver a 5% budget at all — the
order-statistic floor is 1/(n+1) = 5.9%, and reaching 5% needs n >= 19.
`pick_threshold` warns rather than silently missing the budget. On this corpus
the budget is unreachable, not merely missed, and no monitor here should be
described as respecting it.

The same stack runs on LangGraph, AutoGen and native-Ollama traces. Those runs
established an **operating envelope**, then confirmed it causally: with tiny
healthy sets and episodes barely longer than the monitor's 3-step washout,
detection sits near chance; re-collecting with the requirements met
(qwen2.5:7b, 60 healthy episodes per source, T=5–8) lifted every source by
+0.15–0.24 AUC with nothing else changed.

## Live demo

`py -m derail.experiments.demo` serves **AgentWatch Live** at
`http://localhost:8765`. A real qwen2.5:7b agent works a long booking task
while the shipped monitor scores every step. Five buttons inject a real
failure mid-run — loop trap, goal hijack, tool failures, data corruption,
hallucination — and on alarm the run is either halted for inspection or
repaired, your choice via a toggle. An explainability panel answers "why" from
the monitor's own attribution, not canned text.

Nothing is mocked or hardcoded: θ is recomputed from data at startup, and all
post-injection behaviour is real model output. Injected payloads are displayed
verbatim, and when a check *misses*, the UI says which case occurred and why.

Measured over five injection classes × five task seeds with halting off
(`results/tables/alarm_repair.csv`, n=25 live episodes): every one of the 18
behavioural alarms was followed by a repair attempt, and no run that did not
alarm was interrupted. `goal_drift` is the class a retry fixes (2 of 5); where
the tool layer itself is broken the retry cannot win, and the value of the
intervention is ending the episode fast — a loop trap escalates 5 of 5 at
exactly 10 steps, against 30 steps before the circuit breaker existed.

## Layout

```
DESIGN.md                        implementation contract + amendments
derail/common.py                 channel layout, Episode, OnlineMonitor ABC
derail/config.py                 secure API-key resolution (OS vault > env > .env)
derail/telemetry/generator.py    healthy simulator + 5-class failure injector
derail/telemetry/adapter.py      real-trace JSONL -> Episode converter
derail/monitor/esn.py            ESN ensemble, per-channel max fusion (primary)
derail/monitor/baselines.py      cosine/self drift, entropy, Mahalanobis, iForest
derail/monitor/seq_baselines.py  VAR-ridge, GRU, LSTM, TCN (same one-class protocol)
derail/monitor/baseline.py       self-calibrating rolling healthy reference
derail/monitor/grounding_verify.py  per-step numeric-grounding verifier
derail/monitor/calibration.py    label-free (healthy-ECDF) + oracle isotonic
derail/monitor/escalation.py     modeled judge, 4 policies, cost accounting
derail/verify/checks.py          deterministic answer + coverage + contract checks
derail/intervene/                rollback and repair-policy evaluation
derail/evaluation/               metrics, protocol, paired significance tests
derail/experiments/              study runners, collectors, plots, live demo
```

## Failure classes

The injector fixes ground-truth τ, severity and onset ramp.

| class | designed signature |
|---|---|
| `goal_drift` | gradual semantic rotation toward a distractor goal; confident |
| `looping` | cycling among recent states; repeated actions; entropy drops |
| `tool_cascade` | error flags + latency inflation + retry ping-pong |
| `grounding_loss` | semantics stay plausible; uncertainty channel shifts |
| `context_corruption` | AR coherence collapses; dynamics unpredictable |

Low-severity onsets ramp in slowly and are genuinely hard; injected pre-τ steps
are statistically indistinguishable from healthy ones.

## Reproducing

Published numbers need the pinned environment, not the loose bounds in
`requirements.txt`. For a hermetic CPU-only, network-free repro there is a
Dockerfile:

```
docker build -t agentwatch-repro .        # full fidelity (incl. torch baselines)
docker run  --rm agentwatch-repro         # deterministic gate: fast tests + snapshot --check
```

> **Status: statically validated, never built.** Docker was not available in
> the development environment, so the image has not been built even once — we
> say so rather than imply a verified path. What *is* enforced by
> `test_docker_repro_lock_covers_the_gate`: the lockfile covers every
> module-level import the gate command reaches, the pytest markers it filters
> on are registered, and `.dockerignore` excludes neither `results/` nor
> `traces/`. What a first build would prove, and nothing offline can: that the
> `python:3.14-slim` base image and the `torch==2.12.0+cpu` wheel resolve.

**Guarantees.** Monitors fit on healthy train only; thresholds from healthy val
only; the labeled cal split feeds only the isotonic oracle and the escalation
operating point; every reported number is test. Every score at step *t* is a
function of `x_1..x_t` plus fit-time quantities — no lookahead. Monitors are
compared by expected steps saved per failure episode, so misses count zero and
the comparison is survivorship-free. One master seed reproduces results
bit-for-bit.

**Plugging in your own traces** is an adapter problem, not a rewrite —
everything above the telemetry layer consumes only `Episode` objects:

```python
from derail.telemetry.adapter import load_trace_jsonl
healthy = [load_trace_jsonl(p) for p in Path("traces/healthy").glob("*.jsonl")]
monitor.fit(healthy)                       # one-class, as in the study
```

Re-fit on healthy runs from your own source; do not mix a simulator-trained
monitor with real traces.

## Limitations and negative results

Stated as measured. Each of these is a place the system does not yet work, and
each has a number attached rather than a hedge.

- **Repair coverage is partial, and the gap has a shape.** `located` recovers
  45% of flagged failures — which leaves 55% unrecovered. Broken out by class
  over 25 live episodes, `goal_drift` is the only class a retry actually fixes
  (2 of 5). Where the tool layer itself is broken a retry fetches the same
  broken result: `tool_cascade` and `looping` escalate rather than recover, and
  the value of the intervention there is ending the episode fast, not producing
  a correct answer. `grounding_loss` never alarms behaviourally at all (0 of 5).
  A contract violation is escalated, never repaired.
- **Hallucination detection is specific but only half sensitive.** On the
  provoked corpus the numeric-grounding verifier catches **0.55** of ungrounded
  input fabrications. It raises no false positives (0 across 89 healthy runs),
  so what it flags is real — but roughly half of fabrications pass. It has one
  documented false-negative mode: a fabricated figure that coincidentally equals
  some real combination of tool values. And that 0.55 exists only under
  provocation; unprovoked fabrication runs at 2 in 175 episodes, far below the
  pre-registered minimum of 10, so no unprovoked detection claim is made in
  either direction.
- **Two repair rungs did not work, and are kept as measured.** `recompute`
  routes the failing step to a calculator the agent is already holding, which
  should fix the dominant arithmetic failure — it recovers 28% and does not beat
  retry luck at this sample size (p=0.17). `adaptive` recovers 21% (p=0.61).
  Both are retained as comparison arms rather than dropped.
- **Monitors do not transfer across deployments.** Calibrated on qwen2.5:7b and
  deployed on llama3.1:8b over a matched task and tool plan, the primary monitor
  sits at chance (AUROC 0.527, healthy FA 0.75) while the same target
  recalibrated on itself reaches 0.885. Within a family (7b → 3b) it collapses
  below chance, 0.29–0.38. A healthy null must be collected under the exact
  serving distribution.
- **The judge-LLM claim is halved by measurement.** A real gemini-2.5-flash
  judge scores p_detect 0.548 / p_false 0.057 against the stipulated 0.90 /
  0.02. The call-saving survives; the detection-recovery claim does not.
- **Context corruption remains the weakest detected class** (0.29 on real
  traces). Corruption that keeps a legal shape — a price altered from `$361` to
  `$605` — is undetectable from telemetry by construction and needs an external
  reference.
- **On the live-trace corpus the false-alarm budget is unreachable, not just
  missed.** 16 healthy validation episodes put the order-statistic floor at
  1/(n+1) = 5.9%, so an empirical 5% quantile does not exist; realized FA is 20%
  over a 15-episode test split, where one episode is worth 6.7 points.
  `pick_threshold` warns instead of silently missing the budget. Any real-trace
  false-alarm figure in this repository should be read as an order-of-magnitude
  statement, not a rate.
- **The main testbed is synthetic.** Failure-class channel signatures are
  designed in, so H2's *direction* is partly by construction. The non-obvious
  findings — dilution under naive fusion, CUSUM's slow-drift blindness, the
  wrapper-vs-reservoir attribution — were not designed and are the actual
  contribution. Real-trace validation confirms the per-class structure at
  smaller scale, on one agent and task suite.
- **Coverage is conditional on a telemetry contract.** The semantic and
  content-grounding channels need structured tool-call/result events; the
  uncertainty channel needs token-level logprobs. Where a provider omits
  logprobs the collector degrades to `e+m` and says so — it never fabricates a
  channel. Ablating the logprob channel costs almost nothing in practice
  (AUROC +0.000, detection +0.002 across four corpora), so the real
  requirement is structured tool results.
- **The judge-LLM claim is halved by measurement.** A real gemini-2.5-flash
  judge scores p_detect 0.548 / p_false 0.057 against the stipulated 0.90 /
  0.02.
- **Adversarial robustness is a measured limit, not a defended capability.**
  Max-fusion blunts single-channel evasion (0.70 → 0.55–0.65), but a white-box
  adversary shaping all three channels drops detection to the false-alarm
  floor. A cross-channel tamper check closes that attack completely (0.00 →
  1.00 for +3pp false alarms) — and an adaptive adversary replaying a real
  healthy channel trace defeats the check. Closing the informed attack is open
  work.
- **The primary monitor was chosen after test-set diagnostics on the headline
  seed.** The four untouched-seed replications are the guard against that
  selection overfitting, and are reported per seed rather than pooled.

## Documentation

- [`CLAIMS.md`](CLAIMS.md) — claim-to-evidence ledger: every headline number
  above, the artifact it is read from, and the command that regenerates it.
  `py -m devtools.claims_ledger --check` recomputes all 28 and fails on drift.
- [`REPRODUCE.md`](REPRODUCE.md) — models, seeds, hardware, package versions,
  settings, and the exact command behind each result.
- [`DATA_CARD.md`](DATA_CARD.md) — all 25 corpora: sizes, models, injected vs
  organic, episode lengths, channel availability.
- [`CHECKSUMS.md`](CHECKSUMS.md) — SHA-256 coverage and the root digest.
- [`DESIGN.md`](DESIGN.md) — per-module low-level contract, the telemetry
  schema every collector writes, and the numbered amendments.
- Papers — [`paper/main.pdf`](paper/main.pdf) (conference format) and
  [`paper/paper.pdf`](paper/paper.pdf) (full length, source in
  [`paper/paper.md`](paper/paper.md)).
- `results/` — every table and figure the claims above cite.
- [`LICENSE`](LICENSE) (MIT) and [`CITATION.cff`](CITATION.cff).
