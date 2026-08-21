# Live rollback/retry recovery on real-tool episodes

Corpus: `traces\real_task_rollback` (16 injected episodes, classes: ['context_corruption', 'looping', 'tool_cascade', 'wrong_document']). Monitor freeze: `results/framework_monitor_freeze.json` (theta_b5=40.08187009017411).

## Primary (frozen monitor's own causal alarm)

- trigger_rate: 10/16 = 0.625 (95% CI 0.354-0.848)
- conditional_recovery: 8/10 = 0.800 (95% CI 0.444-0.975)
- end_to_end_recovery: 8/16 = 0.500 (95% CI 0.247-0.753)
- not_triggered: 6
- checkpoint_at_start: 0
- reconstruction_failed: 0
- recovered: 8
- still_wrong: 1
- halted: 1
- n_selected: 16

## Oracle upper bound (ground-truth tau -- NOT the deployable result)

- trigger_rate: 16/16 = 1.000 (95% CI 0.794-1.000)
- conditional_recovery: 14/16 = 0.875 (95% CI 0.617-0.984)
- end_to_end_recovery: 14/16 = 0.875 (95% CI 0.617-0.984)
- not_triggered: 0
- checkpoint_at_start: 0
- reconstruction_failed: 0
- recovered: 14
- still_wrong: 2
- halted: 0
- n_selected: 16
