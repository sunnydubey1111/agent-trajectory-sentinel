# Post-fix generalized monitor evaluation: LangGraph and AutoGen

Per-deployment healthy-only calibration (`derail.monitor.deployment_calibration.calibrate`), scored on a NEW disjoint corpus collected with a different seed base than the zero-shot baseline -- no episode is shared between the two. FA budget: 0.05.

Both arms score the SAME held-out episodes, so the only difference between them is the monitor: `frozen` is the native-harness `esn_cusum_max[e,m]` at its published `theta_b5`, `calibrated` is refit on this deployment's own healthy episodes. The frozen arm is the control that separates the calibration change from the telemetry-contract fixes this corpus was collected under.

| framework | monitor | calib train/val | n healthy (test) | n injected | theta | detection (95% CI) | healthy FA (95% CI) | AUC | zero-shot baseline det/fa/auc |
|---|---|---|---|---|---|---|---|---|---|
| langgraph | frozen | -- | 24 | 24 | 40.082 | 0.917 (0.730-0.990) | 0.958 (0.789-0.999) | 0.639 | 0.833/1.000/0.549 |
| langgraph | calibrated | 72/24 | 24 | 24 | 191.860 | 0.625 (0.406-0.812) | 0.000 (0.000-0.142) | 0.878 | 0.833/1.000/0.549 |
| autogen | frozen | -- | 24 | 23 | 40.082 | 0.696 (0.471-0.868) | 0.792 (0.578-0.929) | 0.576 | 0.833/0.667/0.653 |
| autogen | calibrated | 72/24 | 24 | 23 | 5.974 | 0.913 (0.720-0.989) | 0.125 (0.027-0.324) | 0.899 | 0.833/0.667/0.653 |

The last column reads `results/framework_real_tool_report.md` as already published -- the original 48-episode zero-shot result on a DIFFERENT corpus; never recomputed here.