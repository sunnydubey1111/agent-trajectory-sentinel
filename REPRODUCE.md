# Reproduction record

Exactly what produced the committed artifacts: which models, which data, which
seeds, which machine, which package versions, and which command per result.

Everything on this page is either regenerable from the repository or recorded in
an artifact you can read. Where a number cannot be regenerated offline — because
it needed a live model or a paid API — that is stated on the line, not buried.

## 1. Environment

The artifacts in `results/` were produced on this machine:

| | |
|---|---|
| OS | Windows 11 (`Windows-11-10.0.26200-SP0`), 64-bit |
| CPU | Intel64 Family 6 Model 198 Stepping 2, 24 logical cores |
| GPU | not used — every monitor is CPU-only |
| Python | 3.14.5 (CPython, MSC v.1944, 64-bit) |
| numpy / scipy | 2.4.6 / 1.17.1 |
| scikit-learn / pandas | 1.9.0 / 3.0.3 |
| matplotlib | 3.11.0 |
| torch (optional) | 2.12.0+cpu — GRU/LSTM/TCN baselines only |
| Ollama | local server, `qwen2.5:7b` and `llama3.1:8b` pulled |
| LaTeX | MiKTeX, `latexmk` + `pdflatex` |

The same provenance block is written into `results/run_meta.json` by every
study run, so an artifact always carries the environment that produced it.

**Installing that environment.** `requirements.txt` holds loose bounds for
casual installs and is *not* what a published number reproduces against.

```
pip install -r requirements.lock.txt                       # exact pins
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`requirements-core.lock.txt` is a smaller subset sufficient for the synthetic
study alone; it deliberately excludes the real-trace, framework and Gemini
stack, because those collect new data rather than reproduce a committed number.

## 2. Models

| role | model | served by | temperature |
|---|---|---|---|
| primary agent | `qwen2.5:7b` | Ollama, local | 0.2 serving, 0.9 provoking |
| cross-family agent | `llama3.1:8b` | Ollama, local | 0.2 serving, 0.9 provoking |
| small-model arm | `qwen2.5:3b` | Ollama, local | 0.9 |
| API agent | `gemini-2.5-flash` | Google, free tier | provider default |
| judge (measured) | `gemini-2.5-flash` | Google, free tier | provider default |
| embeddings | deterministic hashing embedding | in-process | — |

Two notes that change how results read. `qwen2.5:3b` is no longer pulled on the
collection machine, so its corpora are frozen historical data and its cells
cannot be re-collected as themselves; the collector preflight refuses rather
than quietly producing a 7b corpus under a 3b name. And embeddings are the
deterministic hashing embedding *unless a caller explicitly opts in* to
sentence-transformers — installing that package must never change a result.

## 3. Data

`DATA_CARD.md` is the full per-corpus card, generated from the manifests.
Summary: **2,823 episodes across 25 corpora**, of which 770 use real tools.

Splits and calibration follow one discipline throughout: monitors fit on
healthy-train only, thresholds come from healthy-validation only, the labelled
calibration split feeds only the isotonic oracle and the escalation operating
point, and **every reported number is test**. Organic corpora are scored
cross-fit, 5-fold, so no episode is ever scored by a monitor that saw it.

The held-out corpus uses task seeds 40000+, disjoint from every corpus the
checks were designed against. That separation is the point: the checks were
written by inspecting failures in the serving arm, so the serving arm cannot
also be their test set.

## 4. Seeds

| purpose | seed(s) |
|---|---|
| master seed, synthetic study | `20260713` |
| seed replications | `7`, `101`, `202`, `303` (→ `results/seed{N}/`) |
| behavioural snapshot tripwire | `424242` (disposable; never touches published artifacts) |
| live alarm/repair matrix | `21`–`25` |
| held-out task seeds | `40000+` |

All randomness flows through `rng_for(seed, *tags)`. One master seed reproduces
the study bit-for-bit; `devtools/behavior_snapshot.py` verified 1107/1107
identical leaf values across two consecutive runs.

## 5. Commands, by result

### Regenerable offline, deterministically

These need nothing but the pinned environment. Run them and the committed
artifact comes back byte-identical.

```
py -m derail.experiments.run_experiment                  # results/results.json, h1/h2/h3 tables  (~3 min)
py -m derail.experiments.plots                           # results/figures/*.png
py -m derail.experiments.run_experiment --seed 7         # replication -> results/seed7/
py -m derail.experiments.run_multiseed                   # multiseed_summary.csv               (~35 min)
py -m derail.experiments.run_ablation                    # esn_ablation.csv
py -m derail.experiments.run_benchmark                   # runtime.csv (timings are machine-specific)
py -m derail.experiments.run_fairness                    # fairness.csv
py -m derail.experiments.run_real_traces                 # real_traces.csv, from committed traces
py -m derail.experiments.run_hybrid_study                # hybrid_*.csv
py -m derail.verify.run_verification_study               # verification_vs_monitor.csv
py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout
py -m derail.verify.run_verification_study --contract-coverage
py -m verification.l3_serving_temperature                # serving_temperature.csv
py -m derail.intervene.evaluate_repair_policies --from-csv   # re-analyse the repair study
```

**The AFTraj-2K tables need one extra step**, because that corpus is not ours
and is not committed. `results/tables/aftraj_*.csv` regenerate only after the
corpus is fetched:

```bash
py -m derail.experiments.import_aftraj                   # download + convert -> traces/_aftraj/
py -m derail.experiments.run_hybrid_study --datasets aftraj --out-prefix aftraj
```

`results/tables/atbench_*.csv` work the same way, in one command:

```bash
py -m derail.experiments.run_atbench_study               # download + score
```

The import needs network access to Hugging Face; `--from` converts an
already-downloaded copy instead. The corpus is CC-BY-4.0 and is redistributed
by its authors, not by this repository, so a checkout will not contain it and
`run_hybrid_study` skips the dataset with a note rather than failing when it is
absent.

`run_benchmark` is the one exception to byte-identity: wall-clock latency is a
property of the machine, so `runtime.csv` is the source of record for the
figures quoted, not a value you should expect to match.

### Needs a served model (Ollama, local, free)

```
py -m derail.experiments.demo                            # live demo -> localhost:8765
py -m derail.experiments.demo --rehearse                 # headless: all injections + controls
py -m derail.experiments.demo --alarm-repair-matrix      # -> results/tables/alarm_repair.csv
py -m derail.experiments.demo --collect-healthy N        # extend the demo healthy null
py -m derail.intervene.evaluate_repair_policies --parallel 4   # re-runs real model calls
py -m derail.experiments.collect_framework_traces        # LangGraph / AutoGen corpora
py -m verification.organic_hallucination                 # collect organic episodes
py -m verification.score_provoked_fabrication            # score the provoked corpus
```

### Costs money (Gemini API)

```
py -m derail.config set-key GEMINI_API_KEY               # one-time, hidden input
py -m derail.experiments.collect_traces --estimate       # cost preview
py -m derail.experiments.collect_traces --yes            # real collection (~$0.72 default)
py -m derail.experiments.run_judge_calibration           # measured judge
```

Real collection refuses to run without `--yes` after printing the estimate. The
API key is stored in the OS credential vault via `keyring`, entered with hidden
input, never echoed and never logged; fallbacks are an environment variable and
then a gitignored `.env`.

### Verification gates

```
py -m pytest -m "not network and not ollama"     # 312 tests, the default gate
py -m devtools.behavior_snapshot --check          # end-to-end behavioural tripwire
py -m devtools.artifact_manifest --check          # SHA-256 over every committed file
py -m devtools.claims_ledger --check              # every headline number vs its artifact
py -m devtools.data_card --check                  # data card vs the committed corpora
```

### Papers

```
cd paper && latexmk -pdf main.tex                 # conference format  -> paper/main.pdf
py -m devtools.md_to_latex --build                # full length        -> paper/paper.pdf
```

## 6. Settings that change results

Recorded here because each one silently invalidates a calibration if it moves.

- **False-alarm budget.** 5% for the synthetic study, 10% for the live demo.
  Thresholds are selected on healthy-validation to hit the budget; the
  *realized* rate is then measured and reported, because the budget is not
  always delivered.
- **ESN washout.** 3 steps. An episode needs `T >= 4` to produce any score, and
  the acceptance gate enforces it. This is why short-episode corpora sit near
  chance — an operating-envelope property, not a pipeline failure.
- **Sampling temperature.** A null calibrated at 0.9 does not transfer to 0.2.
  The two organic arms are seed-paired for exactly this reason.
- **Toolset.** The demo agent is scoped to the tools its task needs.
  `fit_monitor()` refuses to calibrate on a corpus containing calls to a retired
  tool, because a healthy null must be collected under the tools actually
  served.
- **Latency features.** Neutralized in the local demo monitor, symmetrically at
  calibration and serving: on a shared local box wall-clock latency measures the
  machine, not the agent. Cloud and API deployments keep them. Published study
  tables are unaffected.
- **Telemetry width.** v1 (agent text and tool args) through v4 (60 dims,
  per-channel CUSUM + delta-Mahalanobis + content grounding). `ServingConfig.fingerprint()`
  covers model, temperature, serving prompt, tool roster and telemetry width; when
  any of them moves the healthy null is *retired* rather than aged.

## 7. What a fresh reader should run first

```
pip install -r requirements.lock.txt
py -m pytest -m "not network and not ollama"
py -m devtools.claims_ledger --check
py -m derail.experiments.run_experiment
```

If the suite is green and the ledger reports all claims matching, the checkout
reproduces the published numbers. `CLAIMS.md` then maps each headline figure to
the artifact it came from and the command that regenerates it.
