# The post-onset horizon law, re-estimated

Regenerate: `py -m derail.experiments.run_horizon_study`
Tables: `results/tables/horizon_{pooled,by_dataset,contrasts,within,robustness}.csv`

The claim under test is that a monitor with memory beats a memoryless one by a
margin that grows with the number of steps available after a failure begins
(the post-onset horizon, `H = T - 1 - tau`). This report re-estimates it,
because the way it was previously measured could not separate the horizon from
the corpus the horizon came from.

## 1. What was wrong with the previous estimate

The published figures — a detection gap of `+0.087` at `H <= 3`, `+0.135` at
4–8 and `+0.404` at `>= 9`, with `r = +0.251` over 1,002 injected episodes —
are arithmetically correct for the table they were read from. The problem is
what that table contains.

Each corpus is a different model, framework, task and injector, and each one
occupies a different part of the horizon range. In the published scope the two
variables are almost the same variable:

| band | episodes | share from the simulator |
|---|---|---|
| `<= 3` | 451 | 0% |
| 4–8 | 170 | 17% |
| `>= 9` | 381 | **97.4%** |

Ten real episodes sit in the top band. The simulator supplies the other 371,
and the simulator is also the corpus where the ESN's margin is largest for
reasons that have nothing to do with horizon. So "the gap at `>= 9`" and "the
gap on the simulator" were, in that scope, the same measurement.

The size of the error is measurable. Controlling for corpus — centring horizon
and gap inside each corpus before correlating, so that corpora differing in
level contribute nothing — on **the same 1,002 episodes**:

| estimator | r |
|---|---|
| pooled, as published | **+0.251** |
| within corpus | **+0.032** |

On the published scope the relationship was seven-eighths composition. A
pooled correlation over episodes clustered by corpus also treats those episodes
as independent, which is what produced `p = 7.3e-16`.

## 2. The re-estimate

Two changes: score every real corpus the repository holds rather than the seven
in the published scope, and estimate within corpus rather than pooled.

Widening the scope is what makes the top band answerable. It adds five real
corpora the horizon claim never covered — `ollama_llama8b`, `real_gemini_long`,
`real_research7b_long_ext`, the 120 real `goal_drift` episodes collected for
the conceptor study, and the external AFTraj-2K — taking the real `>= 9` band
from 10 episodes to **112**, drawn from nine corpora.

**Real-corpus band means** (1,765 injected episodes, 12 corpora, no simulator):

| band | n | ESN | Mahalanobis | gap |
|---|---|---|---|---|
| `<= 3` | 1,027 | 0.277 | 0.260 | **+0.017** |
| 4–8 | 626 | 0.312 | 0.230 | **+0.082** |
| `>= 9` | 112 | 0.509 | 0.259 | **+0.250** |

Monotone, and roughly 60% of the published magnitude at the top band. The
simulator's own top band is `+0.404` — the figure that was published as the
pooled value is the simulator's value.

**Correlation, real corpora, at three levels of control:**

| control | r | p |
|---|---|---|
| none (pooled) | +0.163 | < 1e-4 |
| corpus | **+0.202** | < 1e-4 |
| corpus and failure class | **+0.226** | < 1e-4 |

The relationship *strengthens* under control here, which is the opposite of
what happened on the published scope. Both facts have the same cause: the
simulator was carrying the pooled estimate, and once real corpora populate the
top band the effect can be seen inside them instead of across them. p-values
come from shuffling horizon inside each corpus, so they test the within-corpus
relationship rather than assuming independence across corpora.

**Stability.** Dropping one corpus at a time moves the within-corpus estimate
between `+0.176` and `+0.221`, and the top-band gap between `+0.170` and
`+0.301`. Dropping AFTraj — the largest single contributor, and the only corpus
built by another group — leaves `+0.176` and `+0.170`. No corpus carries the
result.

## 3. The gap widens for two different reasons

The stated mechanism — a CUSUM integrates evidence, so the ESN improves as it
is given more post-onset steps — is only half of what the real data shows, and
which half depends on the corpus:

| | `<= 3` | 4–8 | `>= 9` |
|---|---|---|---|
| our corpora, ESN | 0.508 | 0.492 | 0.508 |
| our corpora, Mahalanobis | 0.476 | 0.366 | 0.339 |
| AFTraj, ESN | 0.006 | 0.029 | **0.509** |
| AFTraj, Mahalanobis | 0.008 | 0.016 | 0.170 |

On our corpora the ESN is **flat** across bands and the gap opens because the
memoryless baseline *degrades* with horizon. On AFTraj the ESN rises steeply
and the baseline barely moves. Both produce a widening gap, and the law is
about the gap, so both support it — but only AFTraj matches the accumulation
story as usually told. The likely reason for our corpora is that longer
horizons carry the slower, subtler classes, which a per-step distance cannot
see; the effect surviving the corpus-and-class control (r = +0.226) says class
composition is not the whole of it, but this is worth stating rather than
asserting the tidy mechanism for both.

## 4. Level and slope are different claims

Per-corpus direction is where the previous framing goes wrong in a way the
numbers alone do not show. The statement "averaged over episodes the ESN never
loses a band" is true of the pooled means and false of most deployments taken
one at a time:

- `langgraph7b` has a **negative** gap in every band it occupies (−0.250,
  −0.167): the memoryless baseline is simply better there.
- `ollama7b` runs **backwards** — its gap falls from +0.718 at `<= 3` to
  +0.429 at 4–8.
- `real_research7b` and `ollama_llama8b` are negative in their lowest band and
  positive above it.

What generalises is the **slope**, not the level. Measured inside each corpus,
the `<= 3` to 4–8 step is positive in 8 of 9 corpora (mean `+0.148`), and the
4–8 to `>= 9` step in 3 of 4 (mean `+0.068`).

But the slope is not resolvable at the deployment level with the corpora that
exist. Treating each corpus as one observation:

| contrast | corpora | mean delta | positive | p |
|---|---|---|---|---|
| `<= 3` -> 4–8 | 9 | +0.148 | 8 | 0.14 |
| 4–8 -> `>= 9` | 5 | +0.068 | 3 | 0.50 |
| `<= 3` -> `>= 9` | 3 | +0.422 | 3 | 0.25 |

This is a statement about how many deployments exist, not evidence against the
law: the within-corpus effect is significant over 1,765 episodes, while the
between-corpus effect has three to nine units and cannot reach significance
whatever its size. Both belong in any honest statement of the result.

## 5. What the law now says

Within a fixed deployment, the ESN's detection advantage over the memoryless
baseline increases with post-onset horizon: `r = +0.202` controlling for
corpus, `+0.226` controlling for corpus and failure class, `p < 1e-4` over
1,765 real injected episodes from 12 corpora, stable under
leave-one-corpus-out. Real band means run `+0.017` / `+0.082` / `+0.250`.

The *level* of the advantage is deployment-specific and changes sign across
corpora, so the law predicts how a deployment's margin moves with runway, not
whether that margin is positive. Its practical content is unchanged and now
rests on real data: a deployment whose failures are caught within a few steps
of onset has little to gain from a temporal monitor over a memoryless
distance, and one whose failures run long has a lot.

## 6. The law against the live system

The law was measured on offline corpora, and the live serving path had no
standalone memoryless arm at all — its shipped monitor fuses the two streams
(`zb = 0.5*z_esn + 0.5*z_maha`), so no ESN-minus-Mahalanobis gap existed for
the deployment. `traces/demo_real_varied_ext` closes that: 102 healthy and 16
injected episodes collected through the same collector, task, model and serving
path as the demo, sized so its validation split can support the 5% budget the
offline corpora are scored at. The frozen `demo_real_varied` is untouched.

Scored under the identical protocol (`live_ext_*` tables):

| monitor | AUROC | detection | false alarms | lead |
|---|---|---|---|---|
| `esn_cusum_max` | **0.976** | 0.81 | **0.05** | 1.88 |
| `delta_mahalanobis` | 0.836 | 0.81 | **0.19** | 2.69 |

**The advantage on this deployment is not a detection gap.** Both arms detect
the same 0.81; the ESN does it inside the 5% budget while the memoryless
baseline needs 19% false alarms to match, and ranks worse (0.976 vs 0.836).
The detection gap the law is stated in is **+0.00** in both bands here.

**And this deployment cannot test the law.** Its post-onset horizons run 3–8,
median 5: 14 of 16 injected episodes sit in the middle band and none reach
`>= 9`. So the `+0.250` top-band figure describes a regime this deployment
never enters, and with one band populated there is no within-corpus slope to
measure. That is the answer to "does the law describe the live system": not
directly — it describes where the deployment sits (the middle band, where the
cross-corpus mean gap is `+0.082`), and the deployment's own gap of `+0.00`
falls inside the per-deployment spread the law explicitly does not constrain,
since only the slope generalises and the level does not.

**Sizing the null is what made this measurable.** At 33 healthy episodes the
corpus yields 7 validation episodes, whose order-statistic floor is a 12.5%
false-alarm budget, and the two arms then land at 17% and 33% false alarms —
different operating points, so their detection rates were not comparable. The
same comparison at an intermediate 80-healthy snapshot put both arms at 12%
false alarms and reported ESN 1.00 against Mahalanobis 0.40. Neither is the
result: at the budget every other corpus is scored at, detection is equal and
the difference moves into the false-alarm rate. A gap read off an unmatched
operating point is not a gap.

## 7. Two corrections found while doing this

**The band edges were applied to two different definitions of horizon.** The
study defines `H = T - 1 - tau`. The `goal_drift` corpus collected to supply
long-horizon episodes was assessed with `T - tau`, one step larger, and
recorded as "min 9 / median 9 / max 11 — 117 of 120 clear the `>= 9` bar".
Under the study's own definition it is min 1 / median 8 / max 10, and **19 of
120** clear it. The same off-by-one appears in `paper/paper.tex`, which
describes those episodes as having "9–11 post-onset steps".

**All three long-form research corpora sit at median horizon 8**, one step
below the threshold the law's headline band uses — `real_research7b_long`,
`..._long_ext` and `..._long_drift` alike. The repository's real long-horizon
evidence is therefore concentrated exactly at the band edge, which is worth
knowing before quoting a `>= 9` figure from any of them.
