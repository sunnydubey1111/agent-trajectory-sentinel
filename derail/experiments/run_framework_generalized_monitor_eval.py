"""Post-fix evaluation: per-deployment healthy-only calibrated monitors,
scored on a NEW, disjoint LangGraph/AutoGen real-tool corpus.

The zero-shot baseline (`traces/{langgraph,autogen}7b_real`,
`results/framework_monitor_freeze.json`,
`results/framework_real_tool_report.md`) is immutable and untouched by
this module -- it is read back only for side-by-side comparison, never
recomputed. This module's own corpus
(`traces/{langgraph,autogen}7b_real2`) was collected with a different
seed base specifically so it shares no episode with the baseline.

Calibration (`derail.monitor.deployment_calibration.calibrate`) uses
ONLY this corpus's own healthy episodes (60/20/20 train/val/test,
matching `framework_monitor_freeze.py`'s own split convention) -- no
failure labels, no other deployment's data, no framework-name branch.
Injected episodes are scoring-only, never part of calibration.

Writes:
  results/tables/framework_generalized_monitor_alarms.csv
  results/tables/framework_generalized_monitor_summary.csv
  results/framework_generalized_monitor_report.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats as sps

from derail.evaluation.metrics import episode_auc, evaluate_alarms, summarize
from derail.experiments.framework_monitor_freeze import load_frozen_monitor
from derail.monitor.deployment_calibration import calibrate
from derail.telemetry.adapter import episode_from_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES_ROOT = REPO_ROOT / "traces"
OUT_CSV = REPO_ROOT / "results" / "tables" / "framework_generalized_monitor_alarms.csv"
#: Per-arm rates, so the published figures have a machine-readable source.
#: `OUT_CSV` is per-episode alarm outcomes: detection and FA are derivable
#: from it, episode AUC is not (it ranks per-step score streams, which the
#: alarm table does not carry), and it holds only the calibrated arm. Every
#: number in the report table below is written here instead, both arms, so
#: `devtools/claims_ledger.py` can check the README against an artifact
#: rather than against prose.
OUT_SUMMARY = REPO_ROOT / "results" / "tables" / "framework_generalized_monitor_summary.csv"
OUT_REPORT = REPO_ROOT / "results" / "framework_generalized_monitor_report.md"
BASELINE_REPORT = REPO_ROOT / "results" / "framework_real_tool_report.md"
FRAMEWORKS = {"langgraph7b_real2": "langgraph", "autogen7b_real2": "autogen"}
FA_BUDGET = 0.05


def _load_episodes(corpus_dir: Path) -> list:
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    episodes = []
    for entry in manifest:
        steps = [json.loads(l) for l in
                (corpus_dir / entry["file"]).read_text("utf-8").splitlines()
                if l.strip()]
        tau = entry["tau"] if entry["failure_class"] else None
        episodes.append(episode_from_trace(
            steps, entry["episode_id"], tau=tau,
            failure_class=entry["failure_class"],
            use_sentence_transformers=False, extended=True))
    return episodes


def _rate_ci(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    ci = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return k / n, float(ci.low), float(ci.high)


def _score(monitor, episodes: list) -> dict:
    scores = {}
    for ep in episodes:
        monitor.start_episode()
        scores[ep.episode_id] = [monitor.score_step(ep.X[t])
                                 for t in range(ep.X.shape[0])]
    return scores


def _arm_metrics(scored: list, scores: dict, theta: float, n_healthy: int,
                 n_injected: int) -> tuple[dict, "pd.DataFrame"]:
    df = evaluate_alarms(scored, scores, theta)
    summ = summarize(df)
    fa_rate, fa_lo, fa_hi = _rate_ci(
        int((df["outcome"] == "false_alarm").sum()), n_healthy)
    det_rate, det_lo, det_hi = _rate_ci(
        int((df["outcome"] == "true_alarm").sum()), n_injected)
    return {"theta": theta, "detection_rate": summ["detection_rate"],
            "healthy_fa_rate": summ["healthy_fa_rate"],
            "det_ci": (det_lo, det_hi), "fa_ci": (fa_lo, fa_hi),
            "auc": episode_auc(scored, scores)}, df


def _baseline_numbers() -> dict:
    """Read-only: the frozen zero-shot report's own printed numbers, for
    side-by-side display. Never recomputed, never regenerated."""
    if not BASELINE_REPORT.exists():
        return {}
    text = BASELINE_REPORT.read_text("utf-8")
    out = {}
    section = None
    for line in text.splitlines():
        if line.startswith("## langgraph7b_real"):
            section = "langgraph"
        elif line.startswith("## autogen7b_real"):
            section = "autogen"
        elif line.startswith("## Pooled") or line.startswith("## Diagnosis"):
            section = None
        elif section and line.startswith("- "):
            for key in ("episode_auc", "healthy_fa_rate", "detection_rate"):
                if key in line:
                    out.setdefault(section, {}).setdefault(
                        key, line.split(":")[-1].strip())
    return out


def main() -> None:
    baseline = _baseline_numbers()
    frozen, theta_b5, _ = load_frozen_monitor()
    print(f"[gen-monitor-eval] frozen monitor loaded: theta_b5={theta_b5}")
    all_rows = []
    per_framework = {}
    for corpus_name, framework in FRAMEWORKS.items():
        corpus_dir = TRACES_ROOT / corpus_name
        if not (corpus_dir / "manifest.json").exists():
            raise SystemExit(f"[gen-monitor-eval] {corpus_dir}/manifest.json "
                             f"missing -- run the disjoint collector first")
        episodes = _load_episodes(corpus_dir)
        healthy = [ep for ep in episodes if ep.is_healthy]
        injected = [ep for ep in episodes if not ep.is_healthy]

        cal = calibrate(healthy, channels=("e", "m"), fa_budget=FA_BUDGET,
                        seed=0)
        test_healthy = [ep for ep in healthy if ep.episode_id in cal.test_ids]
        scored = test_healthy + injected
        nh, ni = len(test_healthy), len(injected)

        arms = {}
        arms["frozen"], _ = _arm_metrics(
            scored, _score(frozen, scored), theta_b5, nh, ni)
        arms["calibrated"], df = _arm_metrics(
            scored, _score(cal.monitor, scored), cal.theta, nh, ni)

        df.insert(0, "framework", framework)
        df.insert(1, "dataset", corpus_name)
        all_rows.append(df)
        per_framework[framework] = {
            "n_healthy_test": nh, "n_injected": ni,
            "n_calib_train": len(cal.train_ids),
            "n_calib_val": len(cal.val_ids), "arms": arms}
        for name, m in arms.items():
            print(f"[gen-monitor-eval] {framework} [{name}]: "
                 f"theta={m['theta']:.3f} n_healthy_test={nh} n_injected={ni} "
                 f"det={m['detection_rate']:.3f} "
                 f"(95% CI {m['det_ci'][0]:.3f}-{m['det_ci'][1]:.3f}) "
                 f"fa={m['healthy_fa_rate']:.3f} "
                 f"(95% CI {m['fa_ci'][0]:.3f}-{m['fa_ci'][1]:.3f}) "
                 f"auc={m['auc']:.3f}")

    combined = pd.concat(all_rows, ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_CSV, index=False)
    print(f"[gen-monitor-eval] wrote {OUT_CSV}")

    summary = pd.DataFrame([
        {"framework": fw, "dataset": corpus, "monitor": arm,
         "n_healthy_test": m["n_healthy_test"], "n_injected": m["n_injected"],
         "n_calib_fit": m["n_calib_train"] if arm == "calibrated" else None,
         "n_calib_cal": m["n_calib_val"] if arm == "calibrated" else None,
         "theta": a["theta"], "detection_rate": a["detection_rate"],
         "detection_ci_lo": a["det_ci"][0], "detection_ci_hi": a["det_ci"][1],
         "healthy_fa_rate": a["healthy_fa_rate"],
         "healthy_fa_ci_lo": a["fa_ci"][0], "healthy_fa_ci_hi": a["fa_ci"][1],
         "episode_auc": a["auc"]}
        for corpus, fw in FRAMEWORKS.items()
        for arm, a in per_framework[fw]["arms"].items()
        for m in (per_framework[fw],)])
    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"[gen-monitor-eval] wrote {OUT_SUMMARY}")

    lines = ["# Post-fix generalized monitor evaluation: LangGraph and AutoGen",
             "",
             "Per-deployment healthy-only calibration "
             "(`derail.monitor.deployment_calibration.calibrate`), scored on a "
             "NEW disjoint corpus collected with a different seed base than "
             "the zero-shot baseline -- no episode is shared between the two. "
             f"FA budget: {FA_BUDGET}.",
             "",
             "`calibrate` splits each deployment's healthy episodes "
             "60/20/20: the fit fold sets the standardizer and the reservoir, "
             "the calibration fold picks `theta` at the FA budget, and the "
             "test fold is scored once and is the ONLY healthy population in "
             "the FA and AUC columns below. No held-out episode reaches any "
             "fitting, scaling or threshold path. Detection is over the "
             "injected episodes, which are scoring-only throughout. Episode "
             "AUC ranks exactly those two groups -- held-out healthy against "
             "injected -- and nothing else. Intervals are Clopper-Pearson at "
             "95%.",
             "",
             "`healthy FA` is the REALIZED held-out rate. `theta` targets the "
             f"{FA_BUDGET:.0%} budget on the calibration fold; that is an "
             "in-sample guarantee and does not transfer, so a realized rate "
             "above the budget means the threshold missed it out of sample, "
             "not that the budget was unreachable. Reachability is a separate "
             "question, answered by the calibration fold's order-statistic "
             "floor of 1/(n+1) (`derail.evaluation.metrics.pick_threshold`).",
             "",
             "Both arms score the SAME held-out episodes, so the only "
             "difference between them is the monitor: `frozen` is the "
             "native-harness `esn_cusum_max[e,m]` at its published "
             "`theta_b5`, `calibrated` is refit on this deployment's own "
             "healthy episodes. The frozen arm is the control that separates "
             "the calibration change from the telemetry-contract fixes this "
             "corpus was collected under.",
             "",
             "| framework | monitor | calib fit/cal | n healthy (held-out test) | "
             "n injected | theta | detection (95% CI) | healthy FA (95% CI) | "
             "AUC | zero-shot baseline det/fa/auc |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for fw, m in per_framework.items():
        b = baseline.get(fw, {})
        b_str = (f"{b.get('detection_rate','?')}/{b.get('healthy_fa_rate','?')}/"
                f"{b.get('episode_auc','?')}") if b else "n/a"
        for name, a in m["arms"].items():
            calib = (f"{m['n_calib_train']}/{m['n_calib_val']}"
                    if name == "calibrated" else "--")
            lines.append(
                f"| {fw} | {name} | {calib} | {m['n_healthy_test']} | "
                f"{m['n_injected']} | {a['theta']:.3f} | "
                f"{a['detection_rate']:.3f} "
                f"({a['det_ci'][0]:.3f}-{a['det_ci'][1]:.3f}) | "
                f"{a['healthy_fa_rate']:.3f} "
                f"({a['fa_ci'][0]:.3f}-{a['fa_ci'][1]:.3f}) | {a['auc']:.3f} | "
                f"{b_str} |")
    lines.append("")
    lines.append("The last column reads `results/framework_real_tool_report.md` "
                "as already published -- the original 48-episode zero-shot "
                "result on a DIFFERENT corpus; never recomputed here.")
    OUT_REPORT.write_text("\n".join(lines), "utf-8")
    print(f"[gen-monitor-eval] wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
