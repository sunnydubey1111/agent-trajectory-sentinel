# Pre-registration — serving-temperature organic detection (2026-07-28)

Written and committed **after the temperature-0.2 arm began collecting but
before any episode of it was labelled or scored.** No cold-arm monitor score
existed when the analysis below was fixed.

## The confound being tested

Every organic detection number in this repository was measured at sampling
temperature **0.9** (`organic_demo7b`, `organic_demo7b_ext`,
`organic_llama8b`, `organic_demo7b_provoked`). The demo and the shipped
monitor serve at **0.2**. Temperature 0.9 is a failure-*provoking* setting:
the repo's own collector documents that it makes small models "emit
junk-token bursts and leak raw tool-call syntax in a sizable fraction of
runs."

So two explanations currently fit the organic evidence equally well:

- **(A) The monitor detects organic task failure.** It would then work at
  0.2 too.
- **(B) The monitor detects high-temperature behavioural degradation**,
  which merely *co-occurs* with failure at 0.9. It would then collapse at
  0.2.

Nothing collected so far separates them, because temperature was never
varied under a fixed protocol.

## Design

- **Arms:** temperature **0.9** (existing, frozen `organic_demo7b_ext`,
  n=120) versus temperature **0.2** (new `organic_demo7b_cold`, n=120).
- **Seed-paired:** the cold arm uses seed base 20000 — the *same task seeds*
  as the 0.9 arm — so both arms price the identical 120 worlds with the
  identical ground-truth totals. Task difficulty is therefore held fixed by
  construction, not by assumption.
- **Everything else identical:** same model (qwen2.5:7b), same demo booking
  task, same task-scoped toolset, same collector, no injection, no
  withholding (`WITHHOLD_RATE=0`), no tool nudge (`TOOL_NUDGE` off), same
  objective labeller, same shipped content-gate monitor, same served 10% FA
  budget.
- **Only temperature differs.**
- **Null is per-arm and temperature-matched**, cross-fit 5-fold from that
  arm's own healthy-labelled subset — the methodology the earlier study
  established as the fix for a mismatched null. A shared null would
  reintroduce the very confound under test.

## Analysis, fixed in advance

Per arm, per label (`healthy`, `arithmetic_error`, `hallucinated`, `other`):
n, alarm count, alarm rate. Then:

1. **Within-arm**: detection vs that arm's healthy FA (Fisher exact,
   one-sided).
2. **Across-arm**: all-failure detection at 0.2 vs at 0.9 (Fisher exact,
   two-sided) — the primary comparison.
3. **Failure-mix shift**: the label distribution at 0.2 vs 0.9
   (chi-square). Reported *before* any detection claim, because if the cold
   arm produces a different mix of failures — in particular fewer `other`
   (aborted, no parsable answer) episodes — then a drop in overall
   detection is partly a change in what failed, not only in what was
   detected. Detection is therefore also reported **per class**, and the
   per-class arithmetic_error comparison is the confound-free one.
4. **Base rate**: organic failure rate per arm.

## Pre-declared verdicts

- **(B) SUPPORTED — detection is temperature-carried** if all-failure
  detection at 0.2 is materially below 0.9 at comparable FA (Fisher
  p < 0.05), **and** the drop persists within the `arithmetic_error` class
  alone, so it cannot be explained by the failure mix.
- **(A) SUPPORTED — detection is real at serving temperature** if
  arithmetic_error detection at 0.2 is above the cold arm's own healthy FA
  with Fisher p < 0.05.
- **INCONCLUSIVE** if the cold arm yields fewer than 10 failures in a class,
  that class is reported underpowered and no claim is made for it. If the
  cold arm's total failures number fewer than 10, no verdict is issued at
  all.

Both outcomes are publishable and will be published.

## Commitments

- No threshold, monitor, calibration, labeller or corpus definition changes
  after scores are seen.
- The 0.9 arm is frozen and is not recollected or re-scored.
- The result lands in README/findings whether positive, negative, or
  inconclusive.

---

## RESULTS (2026-07-28, both arms n=120, seed-paired)

`results/tables/serving_temperature.csv`, from
`organic_hallucination_ext.csv` (0.9) and `organic_hallucination_cold.csv`
(0.2). The cold arm collected 120/120 clean episodes in 120 attempts (zero
rejections), serially, on the same 120 task seeds as the 0.9 arm.

| label | T=0.9 | rate | T=0.2 | rate |
|---|---|---|---|---|
| healthy (false alarms) | 6/38 | 16% | 11/63 | 17% |
| arithmetic_error | 12/37 | 32% | 17/37 | 46% |
| hallucinated | 3/7 | 43% | 3/7 | 43% |
| incomplete | 5/18 | 28% | 11/13 | 85% |
| other (no parsable answer) | 13/20 | 65% | — | (none occurred) |
| **all failures** | 33/82 | 40% | 31/57 | 54% |

**VERDICT: (A) SUPPORTED — detection is real at the serving temperature.**
`arithmetic_error` alarms at 46% against the cold arm's own 17% healthy
false-alarm rate (Fisher p = 0.0025). All-failure detection does not differ
significantly between the arms (40% vs 54%, p = 0.12), so (B) — that detection
is carried by high-temperature degradation — is rejected.

Threshold-free confirmation (episode peak, failures vs healthy):

| | T=0.9 | T=0.2 (served) |
|---|---|---|
| arithmetic_error AUROC | 0.686 | **0.733** [0.622, 0.835] |
| held-out arithmetic AUROC | — | **0.824** [0.730, 0.905] |

**The failure mix shifts** (chi-square p = 0.0006): the 0.2 arm produced zero
`other` (aborted) episodes against 20 at 0.9. This is why the pre-registration
fixed `arithmetic_error` — matched at 39 vs 38 episodes — as the confound-free
comparison.

### DEVIATION FROM PRE-REGISTRATION (disclosed)

The analysis above was pre-registered over four labels — `healthy`,
`arithmetic_error`, `hallucinated`, `other`. A fifth, `incomplete`, was added
**after** the arms had been scored, and it changes the headline: under the
pre-registered label set the verdict was INCONCLUSIVE (arithmetic 18% against
an 18% false-alarm rate), and under the amended set it is (A) SUPPORTED.

This is a post-hoc change and is reported as one. Three things bear on how much
weight it should carry:

- The change was **not selected for its effect on the verdict**. It was made
  because the task text asks the agent to check the weather in three cities and
  roughly one run in six does not, which is a failure by any reading of the
  task; the consequence for the monitor was discovered afterwards.
- The rule is **not tunable**. A run is `incomplete` if it omits work the task
  specifies, decided from the task's own structure. There is no threshold or
  free parameter to move.
- The pre-registered comparison itself is **unchanged**: `arithmetic_error`
  versus that arm's own healthy false-alarm rate, at the served budget.

A reader who prefers the pre-registered label set should read the verdict as
INCONCLUSIVE and treat this section as the exploratory finding it is. The
independent replication below on a corpus collected afterwards
(`organic_demo7b_holdout`, arithmetic AUROC 0.824) is the stronger evidence,
and it was scored under the amended labels from the start.

### The result depended on a label definition, and that is the finding

An earlier scoring of these same two arms found the monitor at **chance** at
the serving temperature (all-failure AUROC 0.508, arithmetic detection 18%
against an 18% false-alarm rate). The difference is not the monitor and not
the threshold: it is which runs were counted as healthy.

Roughly one run in six states the correct grand total but never performs the
weather lookups the task explicitly asks for. The original labeller graded only
the stated total, so those runs were labelled `healthy` and entered the null
the monitor calibrates against. They are not healthy — the task was not done —
and the monitor separates them from genuinely healthy runs almost perfectly
(AUROC 0.948 at 0.2, 0.982 on the held-out corpus, detected 13/13). Carrying
that many strongly-anomalous episodes inside the healthy reference inflated its
spread, pushed the budgeted threshold far above where it belonged, and left
real failures indistinguishable underneath it.

Those runs now carry their own `incomplete` label, derived from the task's own
structure (seed → world) rather than from `derail.verify.checks`, so a coverage
check detecting them remains a measurement rather than a tautology.

**The general lesson is sharper than the specific number.** A one-class monitor
is only as good as the definition of "healthy" it is given. This corpus was
temperature-matched, toolset-matched, cross-fit and out-of-fold calibrated —
every precaution the study had previously identified — and a single
over-permissive label still hid the signal completely. A null must be built
from runs that did the task, not merely from runs that got the answer.

### Two calibration defects fixed along the way

1. **In-sample θ.** `score_organic_halluc` selected θ on the same healthy
   episodes the gate had been fit on, so θ landed low and every class
   over-alarmed. Fold *k*'s θ now comes from the out-of-fold scores of the
   other folds.
2. **Degenerate-scale amplification** (DESIGN.md Amendment 6). A telemetry dim
   with no healthy variation was divided by a floor rather than left unscaled,
   sending healthy episodes to peaks of ~1e9 against a corpus median of 0.46.
   Fixed at all five sites; the maximum healthy peak is now 2.99.

Per-fold θ variation that remains is estimator variance on a heavy-tailed
healthy distribution, not amplification.

### Why `traces/demo7b_scoped` is not a substitute for this arm

That corpus is also temperature 0.2 on the same task, so it looks like a
cheaper source of serving-temperature evidence. It is not, for two reasons that
matter to anyone reusing it:

1. **Its `success` field is a substring test** (`str(expected) in text`) — the
   method `demo._stated_total` was written to replace, because a wrong answer
   containing the expected digits in a line item reads as correct.
2. **It mixes run shapes.** Probe-extended episodes (`demo-healthy-p-*`, with
   extra `PROBE_MSG` turns appended) sit alongside plain ones, so a healthy
   reference built from it spans two different structures.
