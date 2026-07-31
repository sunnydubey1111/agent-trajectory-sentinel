# Claim-to-evidence ledger

Every headline number in the README, `DESIGN.md` and both papers, with
the artifact it is read from and the command that regenerates that
artifact. This file is generated -- edit the study, not the ledger:

```
py -m devtools.claims_ledger --check    # recompute and verify
py -m devtools.claims_ledger --write    # regenerate this file
```

Status at generation: **all claims verified**
 (29 claims checked).

## Corpus

| claim | value | source artifact | regenerate with |
|---|---|---|---|
| Committed agent episodes | `2823` | `traces/*/manifest.json` | `py -m devtools.claims_ledger --check` |
| Committed corpora | `25` | `traces/*/manifest.json` | `py -m devtools.claims_ledger --check` |
| Episodes using real tools | `770` | `traces/real*/manifest.json` | `py -m devtools.claims_ledger --check` |

## Monitor

| claim | value | source artifact | regenerate with |
|---|---|---|---|
| esn_cusum_max detection (5 seeds) | `0.707` | `results/tables/multiseed_summary.csv` | `py -m derail.experiments.run_multiseed` |
| esn_cusum_max episode AUC (5 seeds) | `0.872` | `results/tables/multiseed_summary.csv` | `py -m derail.experiments.run_multiseed` |
| esn_cusum_max mean budget saved (5 seeds) | `4.613` | `results/tables/multiseed_summary.csv` | `py -m derail.experiments.run_multiseed` |
| delta-Mahalanobis detection (5 seeds) | `0.374` | `results/tables/multiseed_summary.csv` | `py -m derail.experiments.run_multiseed` |
| Primary monitor median step latency (us) | `219` | `results/tables/runtime.csv` | `py -m derail.experiments.run_benchmark` |
| Primary monitor state footprint (MB) | `3.95` | `results/tables/runtime.csv` | `py -m derail.experiments.run_benchmark` |
| Channel-max AUC on 187 live Gemini episodes | `0.84` | `results/tables/real_traces.csv` | `py -m derail.experiments.run_real_traces` |
| Channel-max realized false-alarm rate (real traces) | `0.2` | `results/tables/real_traces.csv` | `py -m derail.experiments.run_real_traces` |
| Channel-max detection on context corruption (real traces) | `0.286` | `results/tables/real_traces.csv` | `py -m derail.experiments.run_real_traces` |

## Verification

| claim | value | source artifact | regenerate with |
|---|---|---|---|
| Held-out failures caught by totals check | `0.536` | `results/tables/verification_holdout.csv` | `py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout` |
| Held-out failures caught with coverage | `0.929` | `results/tables/verification_holdout.csv` | `py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout` |
| Held-out false positives | `0` | `results/tables/verification_holdout.csv` | `py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout` |
| llama3.1:8b failures caught (all checks) | `1` | `results/tables/verification_organic_llama8b_cold.csv` | `py -m derail.verify.run_verification_study` |
| llama3.1:8b false positives | `0` | `results/tables/verification_organic_llama8b_cold.csv` | `py -m derail.verify.run_verification_study` |
| Provoked fabrications caught | `26` | `results/tables/verification_provoked.csv` | `py -m verification.score_provoked_fabrication` |
| Episodes flagged by tool_contract | `218` | `results/tables/tool_contract_coverage.csv` | `py -m derail.verify.run_verification_study --contract-coverage` |
| Flagged episodes caught within one step of onset | `215` | `results/tables/tool_contract_coverage.csv` | `py -m derail.verify.run_verification_study --contract-coverage` |

## Repair

| claim | value | source artifact | regenerate with |
|---|---|---|---|
| `located` recovery rate | `0.455` | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| `generic` recovery rate | `0.358` | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| `specific` recovery rate | `0.364` | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| `recompute` recovery rate (not significant) | `0.279` | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| `adaptive` recovery rate (not significant) | `0.212` | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| `resample` control recovery rate | `0.164` | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| Correct runs broken by any repair policy | `0` | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| Genuinely-wrong episodes in the repair study | `55` | `results/tables/repair_policies.csv` | `py -m derail.intervene.evaluate_repair_policies --from-csv` |
| Behavioural alarms followed by a repair attempt | `18` | `results/tables/alarm_repair.csv` | `py -m derail.experiments.demo --alarm-repair-matrix` |

## What this ledger does not cover

Numbers that are properties of a *statistical test* rather than of a
stored table -- p-values, bootstrap intervals, and the per-seed
hypothesis verdicts -- are regenerated by the study runners and checked
by `tests/test_evaluation_validity.py`, not here. The same is true of
the live-demo rehearsal figures, which are measured per run and are
reported as ranges rather than as fixed values.
