"""Score live rollback/retry recovery on the real-task rollback corpus.

Two arms, kept structurally separate (real_tool_rollback.py's own point):

  primary  -- the frozen monitor's own causal alarm decides whether/where
              to check-point. This is the deployable result.
  oracle   -- ground-truth tau (the injector's observed onset, recorded at
              collection time) decides the checkpoint. Reported ONLY as an
              upper bound, never averaged into the primary result.

Recovery is scored by the episode's own `RealTask.success_fn` against the
retried continuation's final response -- the same objective criterion
REAL_TASKS already uses, unmodified. No repair hint, no subjective verifier.

Every episode in the corpus with `failure_class` set was already accepted by
`accept_episode` at collection time
(collect_real_task_rollback_source.py), i.e. the injection demonstrably
landed -- this script does not re-check that.

Writes:
  results/tables/real_task_rollback_outcomes.csv  -- one row per
                                                       (episode, arm)
  results/real_task_rollback_report.md            -- metrics + CIs +
                                                       denominators
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from derail.experiments.framework_monitor_freeze import load_frozen_monitor
from derail.evaluation.rollback_metrics import compute_metrics
from derail.harness.real_tools import _ensure_tls, build_registry
from derail.harness.tasks import REAL_TASKS
from derail.intervene.real_tool_rollback import (
    HALTED, NOT_TRIGGERED, ORACLE_UPPER_BOUND, PRIMARY, RECONSTRUCTION_FAILED,
    RECOVERED, STILL_WRONG, compute_checkpoint_oracle,
    compute_checkpoint_primary, retry_real_task_from_checkpoint)
from derail.telemetry.adapter import episode_from_trace

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = REPO_ROOT / "traces" / "real_task_rollback"
OUT_CSV = REPO_ROOT / "results" / "tables" / "real_task_rollback_outcomes.csv"
OUT_REPORT = REPO_ROOT / "results" / "real_task_rollback_report.md"
MODEL = "qwen2.5:7b"
TEMPERATURE = 0.2
MAX_STEPS = 12

TASK_BY_NAME = {t.name: t for t in REAL_TASKS}


def _ollama_backend_factory(**kw):
    from derail.experiments.collect_traces import OllamaBackend
    return OllamaBackend(**kw)


def _load_steps(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text("utf-8").splitlines()
           if l.strip()]


def _score_retry(entry: dict, task, checkpoint, arm: str, registry,
                 corpus_dir: Path, dataset: str) -> dict:
    steps = _load_steps(corpus_dir / entry["file"])
    row = {"dataset": dataset, "episode_id": entry["episode_id"],
          "task_name": entry["task_name"],
          "failure_class": entry["failure_class"], "tau": entry["tau"],
          "arm": arm, "checkpoint_k": checkpoint.k,
          "checkpoint_outcome": checkpoint.outcome,
          "alarm_step": checkpoint.alarm_step,
          "checkpoint_at_start": checkpoint.outcome == "checkpoint_at_start"}

    if checkpoint.outcome == NOT_TRIGGERED:
        row.update(outcome=NOT_TRIGGERED, success=None)
        return row

    out = retry_real_task_from_checkpoint(
        steps, task.prompt, checkpoint, arm, model=MODEL,
        temperature=TEMPERATURE, max_steps=MAX_STEPS, registry=registry,
        backend_factory=_ollama_backend_factory)

    if out.outcome == RECONSTRUCTION_FAILED:
        row.update(outcome=RECONSTRUCTION_FAILED, success=None, error=out.error)
        return row
    if out.outcome == HALTED:
        row.update(outcome=HALTED, success=False)
        return row

    final_text = out.steps[-1].get("text", "") if out.steps else ""
    success = bool(task.verify(final_text, out.steps))
    row.update(outcome=(RECOVERED if success else STILL_WRONG), success=success)
    return row


def build_report(rows: list[dict], *, corpus_label: str, n_selected: int,
                 classes: list[str], theta_b5: float) -> str:
    """Render the metrics report from already-scored rows.

    Pure function of `rows` -- takes no live dependency, so the report can be
    regenerated from an already-collected `real_task_rollback_outcomes.csv`
    without re-running any retry.
    """
    lines = ["# Live rollback/retry recovery on real-tool episodes",
             "",
             f"Corpus: `{corpus_label}` ({n_selected} injected episodes, "
             f"classes: {classes}). Monitor freeze: "
             f"`results/framework_monitor_freeze.json` (theta_b5={theta_b5}).",
             ""]
    for arm in (PRIMARY, ORACLE_UPPER_BOUND):
        arm_rows = [r for r in rows if r["arm"] == arm]
        outcomes = [r["outcome"] for r in arm_rows]
        m = compute_metrics(outcomes)
        n_at_start = sum(1 for r in arm_rows if r["checkpoint_at_start"])
        title = ("Primary (frozen monitor's own causal alarm)" if arm == PRIMARY
                 else "Oracle upper bound (ground-truth tau -- NOT the "
                      "deployable result)")
        lines += [
            f"## {title}",
            "",
            f"- trigger_rate: {m.trigger_rate.k}/{m.trigger_rate.n} = "
            f"{m.trigger_rate.rate:.3f} "
            f"(95% CI {m.trigger_rate.ci_low:.3f}-{m.trigger_rate.ci_high:.3f})",
            f"- conditional_recovery: {m.conditional_recovery.k}/"
            f"{m.conditional_recovery.n} = {m.conditional_recovery.rate:.3f} "
            f"(95% CI {m.conditional_recovery.ci_low:.3f}-"
            f"{m.conditional_recovery.ci_high:.3f})",
            f"- end_to_end_recovery: {m.end_to_end_recovery.k}/"
            f"{m.end_to_end_recovery.n} = {m.end_to_end_recovery.rate:.3f} "
            f"(95% CI {m.end_to_end_recovery.ci_low:.3f}-"
            f"{m.end_to_end_recovery.ci_high:.3f})",
            f"- not_triggered: {m.n_not_triggered}",
            f"- checkpoint_at_start: {n_at_start}",
            f"- reconstruction_failed: {m.n_reconstruction_failed}",
            f"- recovered: {m.n_recovered}",
            f"- still_wrong: {m.n_still_wrong}",
            f"- halted: {m.n_halted}",
            f"- n_selected: {len(arm_rows)}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="py -m derail.experiments.run_real_task_rollback")
    ap.add_argument("--corpus-dir", default=None)
    args = ap.parse_args(argv)

    _ensure_tls()
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else CORPUS_DIR
    manifest = json.loads((corpus_dir / "manifest.json").read_text("utf-8"))
    selected = [e for e in manifest if e["failure_class"] is not None]
    if not selected:
        raise SystemExit(f"[rollback-recovery] no injected episodes in "
                         f"{corpus_dir}/manifest.json -- run the collector first")
    print(f"[rollback-recovery] {len(selected)} injected episodes selected "
         f"from {corpus_dir}")

    monitor, theta_b5, freeze_artifact = load_frozen_monitor()
    print(f"[rollback-recovery] frozen monitor loaded: theta_b5={theta_b5}")

    dataset = corpus_dir.name
    rows: list[dict] = []
    for i, entry in enumerate(selected):
        task = TASK_BY_NAME[entry["task_name"]]
        registry = build_registry(task.tools)
        steps = _load_steps(corpus_dir / entry["file"])
        print(f"[rollback-recovery] ({i+1}/{len(selected)}) "
             f"{entry['episode_id']} ({entry['failure_class']}, "
             f"tau={entry['tau']})...")

        monitor.start_episode()
        primary_ckpt = compute_checkpoint_primary(steps, monitor, theta_b5,
                                                   episode_from_trace)
        primary_row = _score_retry(entry, task, primary_ckpt, PRIMARY, registry,
                                   corpus_dir, dataset)
        rows.append(primary_row)
        print(f"    primary: {primary_row['outcome']} (k={primary_row['checkpoint_k']})")

        oracle_ckpt = compute_checkpoint_oracle(steps, tau=entry["tau"])
        oracle_row = _score_retry(entry, task, oracle_ckpt, ORACLE_UPPER_BOUND,
                                  registry, corpus_dir, dataset)
        rows.append(oracle_row)
        print(f"    oracle:  {oracle_row['outcome']} (k={oracle_row['checkpoint_k']})")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset", "episode_id", "task_name", "failure_class",
                     "tau", "arm", "checkpoint_k", "checkpoint_outcome",
                     "alarm_step", "checkpoint_at_start", "outcome",
                     "success", "error"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[rollback-recovery] wrote {OUT_CSV}")

    report = build_report(
        rows, corpus_label=str(corpus_dir.relative_to(REPO_ROOT)),
        n_selected=len(selected),
        classes=sorted(set(e["failure_class"] for e in selected)),
        theta_b5=theta_b5)
    OUT_REPORT.write_text(report, "utf-8")
    print(f"[rollback-recovery] wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
