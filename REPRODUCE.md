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

**The `torch==2.12.0+cpu` pin is deliberate and is not upgraded on a security
advisory.** Dependabot flags it for CVE-affected versions `<= 2.12.1` of
`torch.jit.script` (memory corruption, local host, patched in 2.13.0). It does
not reach this project, and the check is mechanical rather than a judgement
call:

```
grep -rn "torch.jit" --include=*.py .     # no hits: the function is never called
grep -rln "import torch" --include=*.py . # one file: derail/monitor/seq_baselines.py
```

`torch` is an optional dependency behind a `try/except ImportError`, used only
for the GRU/LSTM/TCN comparison baselines; the study runs without it. Against
that, the pin is load-bearing: the committed behaviour snapshot and the
published GRU/LSTM/TCN numbers were produced at 2.12.0, so moving it can move a
number the paper reports. Upgrading would trade a vulnerability this code does
not exercise for a reproducibility break that would have to be re-verified
across the whole study.

Re-check the two greps above before accepting this reasoning — if a future
change starts calling `torch.jit`, the pin has to go and the affected baselines
have to be re-run and re-published together.

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
Summary: **3,294 episodes across 31 corpora**, of which 1,319 use real tools.
The arXiv v1 submission describes the tree at commit `00c0673`, which held
**2,823 episodes across 25 corpora** (770 real-tool); that state is preserved
by its tag, and the ledger carries both figures as `corpus.*` and `corpus.*_v1`.

**Totals that look like they should add up, and why they do not.** Several
counts circulate and only some may be summed. `py -m devtools.episode_accounting`
derives all of them from the manifests and prints the identities that hold:

| identity | holds |
|---|---|
| committed = healthy + injected (3,294 = 2,108 + 1,186) | yes |
| v1 = v1 healthy + v1 injected (2,823 = 1,825 + 998) | yes |
| committed = v1 + added since (3,294 = 2,823 + 471) | yes |
| all committed = glob scope + root corpus (3,481 = 3,294 + 187) | yes |
| attempted = accepted + rejected, where recorded (2,248 = 1,707 + 541) | yes |
| v1 healthy + behavioural study = v1 total (1,825 + 1,002 = 2,823) | **no** |

The last one is the trap: it lands at 2,827, four away from 2,823, and looks
like a rounding slip. It is not. It adds a *label* count to a *study
population* that contains 400 generated simulator episodes which are not
committed traces at all, and whose 602 real episodes are a subset of the 998
committed injected ones. Three incommensurable quantities that happen to land
near a fourth.

Study populations are never summed for the same reason: the behavioural
study's real half is a strict subset of the grounding study's, so adding
602 + 874 double-counts every one of the 602.

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
py -m derail.verify.run_verification_study --contract-coverage   # tool_contract_coverage.csv
py -m verification.serving_temperature                # serving_temperature.csv
py -m derail.intervene.evaluate_repair_policies --from-csv   # re-analyse the repair study
py -m derail.experiments.score_organic                   # organic_validation.csv
py -m derail.experiments.run_judge_calibration --replay --n-per-stratum 120
```

**The four `verification_*.csv` tables.** One study, four corpora. The table is
named after the corpus, and the corpus name is in each row's `dataset` column —
without it the four are indistinguishable, since every corpus numbers its
episodes `organic-demo-000` upward.

```
py -m derail.verify.run_verification_study --holdout organic_demo7b_cold      # verification_cold.csv
py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout   # verification_holdout.csv
py -m derail.verify.run_verification_study --holdout organic_demo7b_provoked  # verification_provoked.csv
py -m derail.verify.run_verification_study --holdout organic_llama8b_cold     # verification_organic_llama8b_cold.csv
```

**The organic scoring tables.** Both modules read one corpus at a time from
`AGENTWATCH_ORGANIC_DIR`, so the corpus is chosen by environment rather than by
flag. `score_organic_halluc` also needs an explicit output path, or a second
corpus overwrites the first corpus's published table.

```
AGENTWATCH_ORGANIC_DIR=traces/organic_demo7b \
AGENTWATCH_ORGANIC_OUT_CSV=results/tables/organic_hallucination.csv \
py -m verification.score_organic_halluc
AGENTWATCH_ORGANIC_DIR=traces/organic_demo7b_cold \
AGENTWATCH_ORGANIC_OUT_CSV=results/tables/organic_hallucination_cold.csv \
py -m verification.score_organic_halluc
AGENTWATCH_ORGANIC_DIR=traces/organic_demo7b_ext \
AGENTWATCH_ORGANIC_OUT_CSV=results/tables/organic_hallucination_ext.csv \
py -m verification.score_organic_halluc
AGENTWATCH_ORGANIC_DIR=traces/organic_demo7b_holdout \
AGENTWATCH_ORGANIC_OUT_CSV=results/tables/organic_hallucination_holdout.csv \
py -m verification.score_organic_halluc

AGENTWATCH_ORGANIC_DIR=traces/organic_demo7b py -m verification.score_provoked_fabrication
AGENTWATCH_ORGANIC_DIR=traces/organic_demo7b_ext py -m verification.score_provoked_fabrication
AGENTWATCH_ORGANIC_DIR=traces/organic_demo7b_provoked py -m verification.score_provoked_fabrication
```

**The rest of `results/tables/`.** Both studies pin their published dataset
scope (`PUBLISHED_DATASETS`, `GROUNDING_PUBLISHED_DATASETS`), so a bare run
reproduces the published scope rather than picking up corpora added since.

```
py -m derail.experiments.run_grounding_study             # grounding_*.csv
py -m derail.experiments.explain_hybrid                  # hybrid_coefficients/_complementarity
py -m derail.experiments.run_hybrid_multiseed            # hybrid_seed*/_multiseed   (5 x hybrid)
py -m derail.experiments.run_grounding_multiseed         # grounding_seed*/_multiseed (5 x grounding)
py -m derail.experiments.run_model_transfer              # model_transfer.csv
py -m experimental.power_analysis                        # power_analysis.csv (reads hybrid_diagnosis)

# The published scope plus the two later corpora, reported separately so the
# eight-dataset tables above keep their scope.
py -m derail.experiments.run_hybrid_study --out-prefix l7b --datasets \
    sim gemini autogen7b ollama7b langgraph7b real_research7b \
    real_research7b_long real_research3b ollama_llama8b real_gemini_long
py -m experimental.power_analysis --diagnosis l7b_diagnosis --out power_analysis_l7b

# The three corpora the dataset-reinforcement section reports on.
py -m derail.experiments.run_grounding_study --out-prefix grounding_t6 \
    --datasets langgraph7b real_research7b real_research3b

# The two real corpora the horizon study needs that no other study scores:
# the long-form goal_drift corpus and the corpus the live demo serves.
py -m derail.experiments.run_hybrid_study \
    --datasets real_research7b_long_drift --out-prefix drift
py -m derail.experiments.run_hybrid_study \
    --datasets demo_real_varied --out-prefix live
```

**The horizon law** is estimated by its own study, which reads the
episode-level diagnosis tables above rather than the traces, so run it after
them. It pools every corpus it finds, so a missing `aftraj_diagnosis.csv` or
`drift_diagnosis.csv` narrows the scope silently rather than failing:

```bash
py -m derail.experiments.run_horizon_study               # horizon_*.csv
```

**The layer comparison** reads the behavioural and grounding diagnosis tables
and reports each quantity on both studies' own populations and on the 602
episodes they share, so run it after both:

```bash
py -m derail.experiments.run_layer_alignment             # layer_alignment_*.csv
```

`power_analysis` reads a diagnosis table rather than the traces, so run it
after the study whose diagnosis it names.

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

# Framework x real-tool validation and live rollback/retry
py -m derail.experiments.framework_monitor_freeze                 # ONE TIME before any episode is scored; refuses to overwrite
py -m derail.experiments.collect_framework_real_traces \
    --framework langgraph                                         # -> traces/langgraph7b_real
py -m derail.experiments.collect_framework_real_traces \
    --framework autogen                                           # -> traces/autogen7b_real
py -m derail.experiments.run_framework_real_tool_analysis         # -> results/framework_real_tool_report.md

# Per-deployment healthy-only calibration, on a corpus disjoint from the above.
# --shuffle-order interleaves healthy and injected so host-load drift cannot
# align with the label; --seed-base keeps the episodes disjoint.
py -m derail.experiments.collect_framework_real_traces \
    --framework langgraph --out-dir traces/langgraph7b_real2 \
    --seed-base 91177 --n-healthy 120 --n-per-class 6 --shuffle-order 7
py -m derail.experiments.collect_framework_real_traces \
    --framework autogen --out-dir traces/autogen7b_real2 \
    --seed-base 91177 --n-healthy 120 --n-per-class 6 --shuffle-order 7
py -m derail.experiments.run_framework_generalized_monitor_eval   # -> results/framework_generalized_monitor_report.md

py -m derail.experiments.collect_real_task_rollback_source         # -> traces/real_task_rollback
py -m derail.experiments.run_real_task_rollback                   # -> results/real_task_rollback_report.md
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
py -m pytest -m "not network and not ollama"     # Run the default test gate
py -m devtools.behavior_snapshot --check          # end-to-end behavioural tripwire
py -m devtools.artifact_manifest --check          # SHA-256 over every committed file
py -m devtools.claims_ledger --check              # every headline number vs its artifact
py -m devtools.data_card --check                  # data card vs the committed corpora
py -m devtools.social_card --check                # link-preview card vs its generator
```

`social_card --check` compares bytes and so only holds on the machine that drew
the card; text rasterisation varies by font file and freetype version. The
suite checks the card's dimensions and its numbers, which do travel.

### Papers

**The manuscripts are not in this repository.** `paper/` is local to the
author; the preprint itself is public at
[arXiv:2608.02464](https://arxiv.org/abs/2608.02464). Nothing else
here depends on them — every number they state is recomputed from a committed
table by `py -m devtools.claims_ledger --check`, which is the artifact that
makes the results checkable. The commands below apply to a tree that has the
directory, and the tests covering them skip where it is absent.

```
cd paper && latexmk -pdf main.tex                 # conference format  -> paper/main.pdf
py -m devtools.md_to_latex --build                # full length        -> paper/paper.pdf
py -m devtools.arxiv_package --build --check      # arXiv upload       -> build/arxiv/
```

`paper/main.tex` is the arXiv submission — announced as
[arXiv:2608.02464](https://arxiv.org/abs/2608.02464) (cs.AI, CC BY 4.0) — and
`py -m devtools.arxiv_package` flattens it into a self-contained upload.

**arXiv:2608.02464 corresponds to commit `00c0673`.** That is the tree the
announced preprint describes, recorded here because it is the one fact a reader
of the paper cannot recover from the repository itself. It was tagged `v1.3.0`
until the tags predating the Apache-2.0 relicensing were removed; the commit is
an ancestor of `main` and stays reachable, so `git show 00c0673` still resolves.

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
