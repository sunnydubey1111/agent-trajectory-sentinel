# Demo walkthrough

A 10-minute script for driving the live demo, and what to look for at each
step. Every number on screen is computed from the run in front of you — nothing
is replayed or scripted.

To watch someone else drive it first, the
[recorded walkthrough](https://youtu.be/a05n_000klE?t=0) reaches the live demo at
18:12 and runs it to the end.

## Before you start

```
ollama serve                     # in another terminal
ollama pull qwen2.5:7b
py -m derail.experiments.demo    # then open http://localhost:8765
```

Startup takes a few seconds: the monitor calibrates its alarm threshold from
the healthy corpus before the page will run anything. Watch the top chips —
**Model** should read `qwen2.5:7b` and **Monitor** should name the shipped
grounded gate. If the server chip is red, Ollama is not reachable.

**The one rule for reading the chart:** every score is normalised so that
**1.0 is the alarm line**. Below 1.0 the agent looks like the healthy runs the
monitor was fitted on; above it, it does not. You never need to know the raw
threshold.

---

## 1. Healthy run — establish the baseline

Leave every injection button alone. Press **Run Agent**.

The agent works a multi-city trip-pricing task: it looks up flights, hotels and
weather, then states a total.

**What to look for**

| Where | What you should see | What it means |
|---|---|---|
| Chart | A line that stays well under 1.0 (typically 0.1–0.3) | The trajectory looks like the healthy null |
| **Verdict** | `Idle` → finishes without an alarm | No behavioural anomaly |
| **Final Answer** | A total, with a **Verified Correct** badge | The demo knows the true total because its world is seeded |
| **Tool Contract** card | Stays hidden | No tool returned a malformed result |
| **Why** panel | Small factor bars, no dominant one | Nothing is driving the score |

**What to take away.** The alarm line is not arbitrary — it was fitted to runs
like this one, and a healthy run does not come close to it. Run it twice: the
line moves a little, the verdict does not. That variation is why the threshold
is set at a false-alarm *budget* (10%) rather than at a fixed number.

> Note the **Verified Correct** badge is demo-only: it compares against the
> seeded ground truth. The **Answer Checks** card next to it is the deployable
> one — it recomputes the total from the tool results this run actually
> received, and needs no ground truth at all.

---

## 2. Halt **ON** — catching a failure and stopping the bleeding

Make sure **Halt Agent on Alarm** is ticked (it is by default).

### 2a. Loop Trap

Press **Run Agent**, then press **Loop Trap** while it is working.

**What to look for**

1. **Chart** — the line climbs steeply within a couple of steps and crosses
   1.0. The crossing point is the alarm.
2. **Verdict** — flips to `halted`. The run stops there rather than burning its
   remaining budget.
3. **Why** panel — this is the point of the demo. It is the monitor's own
   attribution, not a caption: expect **Unusual action pattern** or repeated
   tool-call factors to dominate, because a loop is a repetition anomaly.
4. **Savings banner** (appears under Why on a halt) — reads
   *"Saved by halting: N of M budget steps (X%), ≈ T output tokens"* with a
   dollar estimate beneath.

**How to read the savings banner honestly.** The *steps* are real — that many
budget steps genuinely remained. The *tokens* are estimated from this run's own
average tokens/step, and the *dollars* apply a reference cloud price. Running
locally on Ollama the true cost is $0; the banner is showing you the waste an
alarm prevents at scale, which is the actual argument for a monitor that costs
~200 µs per step.

### 2b. Data Corruption

Clear, press **Run Agent**, then press **Data Corruption**.

This one is different, and the difference is the lesson.

1. **Tool Contract card appears** — during the run, not at the end. It names
   the step the malformed result arrived on. That is the earliest verdict
   anywhere in the system, because it inspects the evidence rather than the
   trajectory.
2. **Chart** — often stays *below* 1.0. The behavioural monitor frequently does
   **not** alarm on corruption, and that is expected: this world's tool results
   are short, so garbling them moves too little statistical mass.
3. **Verdict** — the run is stopped by the contract check.

**What to take away.** Two independent detectors, and the cheaper deterministic
one wins here. Across every labelled corpus the contract check flags **0 of
2080** healthy episodes and catches 46% of injected corruption, 215 of 218 of
them within one step of onset. If the chart does not cross the line, the demo
has not failed — the right layer caught it.

---

## 3. Halt **OFF** — repair instead of stopping

Untick **Halt Agent on Alarm**. Press **Run Agent**, then **Goal Hijack**.

A hijack silently rewrites the task mid-run — the agent starts pricing
something the user never asked for. Watch the **Mission** card: it shows the
task actually in force.

**What to look for, in order**

1. **Chart** crosses 1.0 — the alarm still fires; only the *response* changed.
2. **Repair card** appears. With halting off, an alarm is recovered from: the
   agent is rewound to its last fact-gathering step and asked again.
3. **The feed keeps every step.** The rejected answer, the finding, and the
   retry all stay visible in order. Nothing is hidden to make the repair look
   clean.

**How to tell the repair succeeded**

| Signal | Success | Failure |
|---|---|---|
| Repair card | `Repaired` | `Repair Did Not Fix It` |
| Final Answer | New answer, **Verified Correct** | Still wrong, and says so |
| Answer Checks | Pass after the retry | Still failing |
| Chart | Settles back under 1.0 | Stays elevated |

**What to take away.** `goal_drift` is the class a retry actually fixes — the
agent reasoned badly over evidence that was still sound, so re-asking works. It
succeeds in about 2 of 5 attempts live, and offline across 120 episodes the
same policy lifts task success from **52% to 73%** for roughly one extra model
call per run. Detection alone changes nothing; this is the step that does.

---

## 4. Future work

Shown honestly, because the demo will show you these too.

- **Repair only fixes reasoning, not broken tools.** Try **Tool Failures** with
  halting off: the alarm fires and a repair is attempted, but the retry refetches
  the same broken result. The card reports `escalated`, not repaired. The value
  there is a *fast exit* — a loop trap ends at ~10 steps instead of 30 — not a
  correct answer. Automatic recovery from a broken tool layer is future work.
- **Corruption that keeps a legal shape is invisible.** The contract check
  catches malformed results. A price quietly changed from `$361` to `$605` is a
  perfectly well-formed price; telling it from a real one needs an external
  reference the demo does not have.
- **The Hallucination button injects the fabrication.** These models are
  well-aligned and essentially never invent figures on their own — 0 in 91
  episodes across three elicitation methods — so the button inserts an
  ungrounded fee line, disclosed verbatim in the Injected Failure panel. The
  *detection* is still live and unrigged: the grounding check is never told
  which figure was injected. It catches about 55% of fabrications under
  provocation, so roughly half slip through; raising that is future work.
- **About 1 healthy run in 10 may alarm.** The threshold is set to a 10%
  false-alarm budget, so an occasional clean run crossing the line is the
  system working as specified, not a bug.
- **Calibration is per-deployment.** Change the model, the sampling
  temperature or the tool roster and the healthy null is retired rather than
  reused — monitors do not transfer across deployments without recalibration.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Server chip red | Ollama not running | `ollama serve`, then reload |
| Startup fails on calibration | `qwen2.5:7b` not pulled | `ollama pull qwen2.5:7b` |
| Run stalls mid-step | Model still loading on first call | Wait, or re-run once warm |
| Injection button does nothing | Pressed before the run started | Press **Run Agent** first, then inject |

For a headless version of scenarios 1 and 2 that prints a table instead of
serving a page:

```
py -m derail.experiments.demo --rehearse
```
