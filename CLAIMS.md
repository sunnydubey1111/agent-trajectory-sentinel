# Claim-to-evidence ledger

Every headline number in the README, `DESIGN.md` and both papers, with
the artifact it is read from and the command that regenerates that
artifact. This file is generated -- edit the study, not the ledger:

`n` is the denominator the value was computed over, recomputed and
drift-checked like the value itself, and labelled with what it counts:
these are not all episodes. A rate shown without one is a rate nobody
can sanity-check, which is how an AUC computed on a held-out split of
94 was published as being on a corpus of 187.

```
py -m devtools.claims_ledger --check    # recompute and verify
py -m devtools.claims_ledger --write    # regenerate this file
```

Status at generation: **all claims verified**
 (109 claims checked).

## Corpus

| claim | value | n | source artifact | regenerate with |
|---|---|---|---|---|
| Committed agent episodes (current) | `3581` | — | `traces/*/manifest.json` | `py -m devtools.claims_ledger --check` |
| Committed corpora (current) | `33` | — | `traces/*/manifest.json` | `py -m devtools.claims_ledger --check` |
| Episodes using real tools (current) | `1606` | — | `traces/*/manifest.json (content-derived: see real_tools.episode_used_real_tools)` | `py -m devtools.claims_ledger --check` |
| Committed episodes outside the traces/*/ glob every total uses | `187` | — | `results/tables/episode_accounting.csv` | `py -m devtools.episode_accounting --check --write` |
| Committed episodes of ours including the root corpus | `3768` | — | `results/tables/episode_accounting.csv` | `py -m devtools.episode_accounting --check --write` |
| Episodes scored by both the behavioural and grounding studies | `602` | — | `results/tables/episode_accounting.csv` | `py -m devtools.episode_accounting --check --write` |
| Scored episodes with no committed episode behind them | `0` | — | `results/tables/episode_accounting.csv` | `py -m devtools.episode_accounting --check --write` |
| Committed agent episodes as of arXiv v1 (commit 00c0673) | `2823` | — | `traces/*/manifest.json` | `py -m devtools.claims_ledger --check` |
| Committed corpora as of arXiv v1 (commit 00c0673) | `25` | — | `traces/*/manifest.json` | `py -m devtools.claims_ledger --check` |
| Episodes using real tools as of arXiv v1 (commit 00c0673) | `770` | — | `traces/real*/manifest.json` | `py -m devtools.claims_ledger --check` |

## Monitor

| claim | value | n | source artifact | regenerate with |
|---|---|---|---|---|
| esn_cusum_max detection (5 seeds) | `0.707` | — | `results/tables/multiseed_summary.csv` | `py -m derail.experiments.run_multiseed` |
| esn_cusum_max episode AUC (5 seeds) | `0.872` | `560` episodes/seed, 5 seeds | `results/tables/multiseed_summary.csv` | `py -m derail.experiments.run_multiseed` |
| esn_cusum_max mean budget saved (5 seeds) | `4.613` | — | `results/tables/multiseed_summary.csv` | `py -m derail.experiments.run_multiseed` |
| delta-Mahalanobis detection (5 seeds) | `0.374` | — | `results/tables/multiseed_summary.csv` | `py -m derail.experiments.run_multiseed` |
| Primary monitor step latency is 100-999 us | `100-999 us` | — | `results/tables/runtime.csv` | `py -m derail.experiments.run_benchmark (timings are machine-specific)` |
| Primary monitor state footprint (MB) | `3.95` | — | `results/tables/runtime.csv` | `py -m derail.experiments.run_benchmark` |
| Primary monitor step latency, median us (machine-specific) | `219` | `4316` timed steps | `results/tables/runtime.csv` | `py -m derail.experiments.run_benchmark (timings are machine-specific)` |
| delta-Mahalanobis step latency, median us -- the baseline the reservoir is ~50x more expensive than | `4` | `4316` timed steps | `results/tables/runtime.csv` | `py -m derail.experiments.run_benchmark (timings are machine-specific)` |
| Full v4 telemetry construction cost at the adapter, median us | `673.7` | `491` timed steps | `results/tables/telemetry_runtime.csv` | `py -m experimental.telemetry_runtime (timings are machine-specific)` |
| Full v4 telemetry construction cost, p95 us | `1045` | — | `results/tables/telemetry_runtime.csv` | `py -m experimental.telemetry_runtime (timings are machine-specific)` |
| Channel-max ESN step latency on AFTraj-2K, median us (NOT the hybrid's 172.6 -- the two were once conflated) | `162.8` | — | `results/tables/aftraj_benchmark.csv` | `py -m derail.experiments.run_hybrid_study --datasets aftraj --out-prefix aftraj` |
| Channel-max AUC, held-out split of the 187-episode Gemini corpus (79 injected + 15 healthy) | `0.84` | `94` episodes | `results/tables/real_traces.csv` | `py -m derail.experiments.run_real_traces` |
| Channel-max realized false-alarm rate, 15 healthy test episodes (real traces) | `0.2` | `15` episodes | `results/tables/real_traces.csv` | `py -m derail.experiments.run_real_traces` |
| Channel-max detection on context corruption (real traces) | `0.286` | — | `results/tables/real_traces.csv` | `py -m derail.experiments.run_real_traces` |
| hybrid_weighted50 grand-mean AUROC (label-free default) | `0.812` | `8` datasets | `results/tables/hybrid_benchmark.csv` | `py -m derail.experiments.run_hybrid_study` |
| hybrid_logistic grand-mean AUROC (with labels) | `0.826` | `8` datasets | `results/tables/hybrid_benchmark.csv` | `py -m derail.experiments.run_hybrid_study` |
| esn_cusum_max grand-mean AUROC on the same eight datasets | `0.802` | `8` datasets | `results/tables/hybrid_benchmark.csv` | `py -m derail.experiments.run_hybrid_study` |
| esn_cusum_max grand-mean AUROC, healthy/injected matched on length | `0.887` | `8` datasets | `results/tables/hybrid_length_confound.csv` | `py -m derail.experiments.run_hybrid_study` |
| delta_mahalanobis grand-mean AUROC, matched on length | `0.866` | `8` datasets | `results/tables/hybrid_length_confound.csv` | `py -m derail.experiments.run_hybrid_study` |
| hybrid_logistic grand-mean AUROC, matched on length | `0.891` | `8` datasets | `results/tables/hybrid_length_confound.csv` | `py -m derail.experiments.run_hybrid_study` |
| esn_cusum_max episode AUROC on AFTraj-2K (external) | `0.745` | `771` episodes | `results/tables/aftraj_benchmark.csv` | `py -m derail.experiments.import_aftraj && py -m derail.experiments.run_hybrid_study --datasets aftraj --out-prefix aftraj` |
| esn_cusum_max detection on AFTraj-2K at the 5% budget | `0.048` | — | `results/tables/aftraj_benchmark.csv` | `py -m derail.experiments.run_hybrid_study --datasets aftraj --out-prefix aftraj` |
| esn_cusum_max detection on AFTraj-2K failures with >= 9 steps of post-onset horizon | `0.509` | — | `results/tables/aftraj_diagnosis.csv` | `py -m derail.experiments.run_hybrid_study --datasets aftraj --out-prefix aftraj` |
| AFTraj-2K failures with >= 9 steps of post-onset horizon | `53` | — | `results/tables/aftraj_diagnosis.csv` | `py -m derail.experiments.run_hybrid_study --datasets aftraj --out-prefix aftraj` |
| esn_cusum_max episode AUROC on ATBench (external) | `0.779` | `381` episodes | `results/tables/atbench_benchmark.csv` | `py -m derail.experiments.run_atbench_study` |
| esn_cusum_max detection on ATBench at the 5% budget | `0.311` | — | `results/tables/atbench_benchmark.csv` | `py -m derail.experiments.run_atbench_study` |
| hybrid_weighted50 episode AUROC on ATBench (fusion collapses to chance when a parent does) | `0.463` | `381` episodes | `results/tables/atbench_benchmark.csv` | `py -m derail.experiments.run_atbench_study` |
| esn_cusum_max detection on unconfirmed/over-privileged actions (ATBench) | `0.508` | — | `results/tables/atbench_per_mode.csv` | `py -m derail.experiments.run_atbench_study` |
| esn_cusum_max detection on inaccurate/misleading information (ATBench, the known content blind spot) | `0.038` | `26` episodes | `results/tables/atbench_per_mode.csv` | `py -m derail.experiments.run_atbench_study` |
| ESN advantage at post-onset horizon <= 3 steps, real corpora | `0.017` | `1027` episodes | `results/tables/horizon_pooled.csv` | `py -m derail.experiments.run_horizon_study` |
| ESN advantage at post-onset horizon 4-8 steps, real corpora | `0.082` | `626` episodes | `results/tables/horizon_pooled.csv` | `py -m derail.experiments.run_horizon_study` |
| ESN advantage at post-onset horizon >= 9 steps, real corpora | `0.25` | `112` episodes | `results/tables/horizon_pooled.csv` | `py -m derail.experiments.run_horizon_study` |
| ESN advantage at post-onset horizon >= 9 steps, simulator | `0.404` | `371` episodes | `results/tables/horizon_pooled.csv` | `py -m derail.experiments.run_horizon_study` |
| Horizon/advantage correlation within corpus, real corpora | `0.202` | — | `results/tables/horizon_within.csv` | `py -m derail.experiments.run_horizon_study` |
| Healthy false-alarm rate of the ESN on the live corpus, 5% budget | `0.048` | `21` healthy episodes | `results/tables/live_ext_benchmark.csv` | `py -m derail.experiments.run_hybrid_study --datasets demo_real_varied_ext --out-prefix live_ext` |
| Healthy false-alarm rate of the memoryless baseline at the same detection | `0.191` | `21` healthy episodes | `results/tables/live_ext_benchmark.csv` | `py -m derail.experiments.run_hybrid_study --datasets demo_real_varied_ext --out-prefix live_ext` |
| Episode AUROC of the ESN on the live corpus | `0.976` | `37` test episodes | `results/tables/live_ext_benchmark.csv` | `py -m derail.experiments.run_hybrid_study --datasets demo_real_varied_ext --out-prefix live_ext` |
| Horizon/advantage correlation within corpus and failure class | `0.226` | — | `results/tables/horizon_within.csv` | `py -m derail.experiments.run_horizon_study` |
| Content-gate detection gain on the content classes, worst seed | `0.307` | — | `results/tables/grounding_multiseed_criterion.csv` | `py -m derail.experiments.run_grounding_multiseed` |
| Content gate does not degrade behavioural detection, worst seed | `0.039` | — | `results/tables/grounding_multiseed_criterion.csv` | `py -m derail.experiments.run_grounding_multiseed` |
| Best within-family transfer AUROC (qwen2.5:7b -> 3b), uncalibrated | `0.522` | `53` episodes | `results/tables/model_transfer.csv` | `py -m derail.experiments.run_model_transfer` |
| Measured gemini-2.5-flash judge detection rate | `0.548` | `84` positives | `results/tables/judge_calibration_summary.json` | `py -m derail.experiments.run_judge_calibration --replay --n-per-stratum 120` |
| Measured gemini-2.5-flash judge false-alarm rate | `0.052` | `77` negatives | `results/tables/judge_calibration_summary.json` | `py -m derail.experiments.run_judge_calibration --replay --n-per-stratum 120` |
| Episodes both the behavioural and grounding studies scored | `602` | — | `results/tables/layer_alignment_summary.csv` | `py -m derail.experiments.run_layer_alignment` |
| Content-gate gain on the matched population | `0.171` | `211` content episodes | `results/tables/layer_alignment_summary.csv` | `py -m derail.experiments.run_layer_alignment` |
| Content-gate gain on the grounding study's own population | `0.297` | `313` content episodes | `results/tables/layer_alignment_summary.csv` | `py -m derail.experiments.run_layer_alignment` |
| Content-gate gain on corpora only the grounding study scores | `0.559` | `102` content episodes | `results/tables/layer_alignment_summary.csv` | `py -m derail.experiments.run_layer_alignment` |
| Behavioural detection change under the gate, matched population | `0.072` | `391` behavioural episodes | `results/tables/layer_alignment_summary.csv` | `py -m derail.experiments.run_layer_alignment` |
| Pooled injected episodes in the grounding table | `874` | — | `results/tables/grounding_diagnosis.csv` | `py -m derail.experiments.run_grounding_study` |
| Content-class episodes in the grounding table | `313` | — | `results/tables/grounding_diagnosis.csv` | `py -m derail.experiments.run_grounding_study` |
| Behavioural-class episodes in the grounding table | `561` | — | `results/tables/grounding_diagnosis.csv` | `py -m derail.experiments.run_grounding_study` |
| Ungrounded parent detection on the content classes | `0.272` | `313` episodes | `results/tables/grounding_diagnosis.csv` | `py -m derail.experiments.run_grounding_study` |
| Content-gate detection on the content classes | `0.578` | `313` episodes | `results/tables/grounding_diagnosis.csv` | `py -m derail.experiments.run_grounding_study` |
| Ungrounded parent detection on the behavioural classes | `0.738` | `561` episodes | `results/tables/grounding_diagnosis.csv` | `py -m derail.experiments.run_grounding_study` |
| Content-gate detection on the behavioural classes -- the gate must not cost behavioural detection | `0.786` | `561` episodes | `results/tables/grounding_diagnosis.csv` | `py -m derail.experiments.run_grounding_study` |
| Joint-budget fusion detection on the content classes | `0.454` | `313` episodes | `results/tables/grounding_diagnosis.csv` | `py -m derail.experiments.run_grounding_study` |

## Verification

| claim | value | n | source artifact | regenerate with |
|---|---|---|---|---|
| Held-out failures caught by totals check | `0.536` | — | `results/tables/verification_holdout.csv` | `py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout` |
| Held-out failures caught with coverage | `0.929` | — | `results/tables/verification_holdout.csv` | `py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout` |
| Held-out false positives | `0` | — | `results/tables/verification_holdout.csv` | `py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout` |
| llama3.1:8b failures caught (all checks) | `1` | — | `results/tables/verification_organic_llama8b_cold.csv` | `py -m derail.verify.run_verification_study --holdout organic_llama8b_cold` |
| llama3.1:8b false positives | `0` | — | `results/tables/verification_organic_llama8b_cold.csv` | `py -m derail.verify.run_verification_study --holdout organic_llama8b_cold` |
| Provoked fabrications caught | `26` | — | `results/tables/verification_provoked.csv` | `py -m derail.verify.run_verification_study --holdout organic_demo7b_provoked` |
| Grounding-verifier false positives on label-healthy runs | `0` | `55` episodes | `results/tables/fabrication_organic_demo7b.csv` | `AGENTWATCH_ORGANIC_DIR=traces/organic_demo7b py -m verification.score_provoked_fabrication` |
| tool_contract false positives, every labelled corpus of ours | `0` | `2080` healthy episodes | `results/tables/tool_contract_denominators.csv` | `py -m derail.verify.run_verification_study --contract-coverage` |
| Recomputation-check false positives, organic demo corpora | `0` | `177` healthy episodes | `results/tables/verification_*.csv` | `py -m derail.verify.run_verification_study` |
| Episodes flagged by tool_contract | `218` | — | `results/tables/tool_contract_coverage.csv` | `py -m derail.verify.run_verification_study --contract-coverage` |
| Flagged episodes caught within one step of onset | `215` | — | `results/tables/tool_contract_coverage.csv` | `py -m derail.verify.run_verification_study --contract-coverage` |
| Checks: failures caught at T=0.2 (totals only) | `0.597` | `57` failures | `results/tables/verification_vs_monitor.csv` | `py -m derail.verify.run_verification_study` |
| Checks: failures caught at T=0.2 with coverage | `0.965` | `57` failures | `results/tables/verification_vs_monitor.csv` | `py -m derail.verify.run_verification_study` |
| Monitor: failures caught at T=0.2 | `0.544` | `57` failures | `results/tables/verification_vs_monitor.csv` | `py -m derail.verify.run_verification_study` |
| Monitor false-alarm rate at T=0.2, against the checks' 0 | `0.175` | `63` healthy episodes | `results/tables/verification_vs_monitor.csv` | `py -m derail.verify.run_verification_study` |
| Checks: failures caught at T=0.9 (totals only) | `0.646` | `82` failures | `results/tables/verification_vs_monitor.csv` | `py -m derail.verify.run_verification_study` |
| Monitor: failures caught at T=0.9 | `0.402` | `82` failures | `results/tables/verification_vs_monitor.csv` | `py -m derail.verify.run_verification_study` |

## Repair

| claim | value | n | source artifact | regenerate with |
|---|---|---|---|---|
| `located` recovery rate | `0.455` | `55` episodes | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| `generic` recovery rate | `0.358` | `55` episodes | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| `specific` recovery rate | `0.364` | `55` episodes | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| `recompute` recovery rate (not significant) | `0.279` | `55` episodes | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| `adaptive` recovery rate (not significant) | `0.212` | `55` episodes | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| `resample` control recovery rate | `0.164` | `55` episodes | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| Correct runs broken by any repair policy | `0` | — | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| Genuinely-wrong episodes in the repair study | `55` | — | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| Every behavioural alarm is followed by a repair attempt | `all alarms attempted` | — | `results/tables/alarm_repair.csv` | `py -m derail.experiments.demo --alarm-repair-matrix (live)` |
| Behavioural alarms in the live matrix | `21` | `25` live episodes | `results/tables/alarm_repair.csv` | `py -m derail.experiments.demo --alarm-repair-matrix (live)` |
| Live episodes ended without emitting an answer | `9` | `25` live episodes | `results/tables/alarm_repair.csv` | `py -m derail.experiments.demo --alarm-repair-matrix (live)` |
| Live episodes a retry turned into a correct answer | `3` | `25` live episodes | `results/tables/alarm_repair.csv` | `py -m derail.experiments.demo --alarm-repair-matrix (live)` |
| Live episodes that answered and were still wrong | `10` | `25` live episodes | `results/tables/alarm_repair.csv` | `py -m derail.experiments.demo --alarm-repair-matrix (live)` |
| goal_drift episodes repaired in the live matrix | `4` | — | `results/tables/alarm_repair.csv` | `py -m derail.experiments.demo --alarm-repair-matrix (live)` |
| Net task success with no intervention | `0.525` | `120` episodes | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| Net task success under `located` | `0.733` | `120` episodes | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| Failures `located` recovers, mean of 3 repeats | `25` | — | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |

## Real-tool validation

| claim | value | n | source artifact | regenerate with |
|---|---|---|---|---|
| LangGraph x real-tool healthy false-alarm rate | `1` | `12` healthy episodes | `results/tables/framework_real_tool_alarms.csv` | `py -m derail.experiments.collect_framework_real_traces --framework {langgraph,autogen} && py -m derail.experiments.run_framework_real_tool_analysis (live)` |
| LangGraph x real-tool detection rate (uninformative alongside a 1.00 false-alarm rate) | `0.833` | `12` injected episodes | `results/tables/framework_real_tool_alarms.csv` | `py -m derail.experiments.collect_framework_real_traces --framework {langgraph,autogen} && py -m derail.experiments.run_framework_real_tool_analysis (live)` |
| AutoGen x real-tool healthy false-alarm rate | `0.667` | `12` healthy episodes | `results/tables/framework_real_tool_alarms.csv` | `py -m derail.experiments.collect_framework_real_traces --framework {langgraph,autogen} && py -m derail.experiments.run_framework_real_tool_analysis (live)` |
| AutoGen x real-tool detection rate | `0.833` | `12` injected episodes | `results/tables/framework_real_tool_alarms.csv` | `py -m derail.experiments.collect_framework_real_traces --framework {langgraph,autogen} && py -m derail.experiments.run_framework_real_tool_analysis (live)` |
| Episodes collected but excluded from monitor scoring (the injector never applied, or applied with no following step -- see inject.replay_against_trace); the collector enforces this at admission time, so none reach the corpus | `0` | — | `results/tables/framework_real_tool_alarms.csv` | `py -m derail.experiments.collect_framework_real_traces --framework {langgraph,autogen} && py -m derail.experiments.run_framework_real_tool_analysis (live)` |
| Primary-arm (causal) trigger rate | `0.625` | `16` injected episodes | `results/tables/real_task_rollback_outcomes.csv` | `py -m derail.experiments.collect_real_task_rollback_source && py -m derail.experiments.run_real_task_rollback (live)` |
| Primary-arm recovery given a triggered, reconstructable checkpoint | `0.8` | `10` triggered episodes | `results/tables/real_task_rollback_outcomes.csv` | `py -m derail.experiments.collect_real_task_rollback_source && py -m derail.experiments.run_real_task_rollback (live)` |
| Primary-arm end-to-end recovery | `0.5` | `16` injected episodes | `results/tables/real_task_rollback_outcomes.csv` | `py -m derail.experiments.collect_real_task_rollback_source && py -m derail.experiments.run_real_task_rollback (live)` |
| Oracle-tau upper bound end-to-end recovery (NOT the deployable result) | `0.875` | `16` injected episodes | `results/tables/real_task_rollback_outcomes.csv` | `py -m derail.experiments.collect_real_task_rollback_source && py -m derail.experiments.run_real_task_rollback (live)` |

## What this ledger does not cover

Numbers that are properties of a *statistical test* rather than of a
stored table -- p-values, bootstrap intervals, and the per-seed
hypothesis verdicts -- are regenerated by the study runners and checked
by `tests/test_evaluation_validity.py`, not here. The same is true of
the live-demo rehearsal figures, which are measured per run and are
reported as ranges rather than as fixed values.

One row above has no offline regenerator, and says so in its command
column: `alarm_repair.csv` records 25 live episodes driven through the
demo with halting off. Re-running it needs a served model and yields a
fresh sample rather than that one, so the committed CSV is itself the
evidence. Every other row regenerates from committed code and data.
