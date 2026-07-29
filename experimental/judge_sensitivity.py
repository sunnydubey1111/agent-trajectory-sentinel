"""L8 - does H3b survive a REAL judge? A paired stipulated-vs-measured run.

`derail.experiments.run_judge_calibration` measures a real Gemini-Flash judge
on a labelled subset and finds it materially worse than the parameters H3b
assumes (`JudgeConfig`: p_detect 0.90, p_false 0.02). This script asks the
consequence question: re-run the escalation study with the MEASURED rates and
compare, at the SAME seed, against the stipulated ones.

Both arms run the full study at the SAME seed, so datasets, monitors, thresholds
and operating-point selection are identical and the only thing that moves is the
judge. (An earlier attempt compared two different seeds; that confounds judge
quality with dataset draw and is not a sensitivity analysis.) The sweep covers
all five published master seeds, so the answer is a distribution, not one draw.

Nothing here can touch the publication path: both arms are redirected to
results/_sensitivity/ via AGENTWATCH_RESULTS_ROOT, and run_experiment refuses
outright to write a judge-override run to results/.

Run:  py -m experimental.judge_sensitivity            (~30 min, no API calls)
      py -m experimental.judge_sensitivity --seeds 7  (one seed, ~6 min)
      py -m experimental.judge_sensitivity --keep-runs
Writes results/tables/judge_sensitivity.csv (one row per arm per seed)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from derail.common import MASTER_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]
TABLES = REPO_ROOT / "results" / "tables"
#: The five published master seeds. Running them here does NOT overwrite
#: results/ or results/seed<N>/: both arms are redirected to a disposable tree
#: via AGENTWATCH_RESULTS_ROOT, and run_experiment refuses a judge-override run
#: that has not been redirected.
STUDY_SEEDS = (MASTER_SEED, 7, 101, 202, 303)
SENSITIVITY_ROOT = "results/_sensitivity"
#: Measured on traces/ollama7b by run_judge_calibration (prompt v1, n=172
#: distinct prompts); see results/tables/judge_calibration_summary.json.
MEASURED_P_DETECT = 0.548
MEASURED_P_FALSE = 0.057


def _run_arm(seed: int, overrides: dict[str, str] | None) -> dict:
    env = dict(os.environ)
    env.pop("AGENTWATCH_JUDGE_P_DETECT", None)
    env.pop("AGENTWATCH_JUDGE_P_FALSE", None)
    # BOTH arms are redirected, including the stipulated one: the comparison
    # must be between two runs made the same way, and neither may touch the
    # published tree.
    env["AGENTWATCH_RESULTS_ROOT"] = SENSITIVITY_ROOT
    env.update(overrides or {})
    proc = subprocess.run(
        [sys.executable, "-m", "derail.experiments.run_experiment",
         "--seed", str(seed)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:] + proc.stderr[-4000:])
        raise SystemExit(f"[judge-sens] arm failed ({proc.returncode})")
    root = REPO_ROOT / SENSITIVITY_ROOT
    out = root / "results.json" if seed == MASTER_SEED else (
        root / f"seed{seed}" / "results.json")
    return json.loads(out.read_text("utf-8"))


def _rows(result: dict) -> dict:
    """The two policies the H3b claim is made of, plus the judge it used."""
    policies = result["h3_escalation"]["policies"]
    every = next(p for p in policies if p["policy"] == "judge_every_step")
    selected = next((p for p in policies if p.get("selected_on_cal")), None)
    if selected is None:      # no confidence-gated point selected on cal
        selected = next(p for p in policies
                        if p["policy"] == "escalate_on_alarm")
    return {
        "judge_p_detect": result["config"]["judge"]["p_detect"],
        "judge_p_false": result["config"]["judge"]["p_false"],
        "judge_every_step_detection": round(every["detection_rate"], 4),
        "judge_every_step_calls": round(every["mean_judge_calls"], 3),
        "escalate_detection": round(selected["detection_rate"], 4),
        "escalate_calls": round(selected["mean_judge_calls"], 3),
        "escalate_lead": (None if selected["mean_lead"] is None
                          else round(selected["mean_lead"], 3)),
        "cost_ratio_vs_judge": round(selected["cost_ratio_vs_judge"], 4),
        "detection_recovered": round(
            selected["detection_rate"] / every["detection_rate"], 4),
        "call_fraction": round(
            selected["mean_judge_calls"] / every["mean_judge_calls"], 4),
        "verdict": result["verdicts"]["H3b"].split(":")[0],
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(STUDY_SEEDS),
                    help="master seeds to sweep (default: the five study seeds)")
    ap.add_argument("--keep-runs", action="store_true",
                    help="keep the disposable results/_sensitivity/ tree")
    args = ap.parse_args(argv)

    rows: list[tuple[str, int, dict]] = []
    for i, seed in enumerate(args.seeds, 1):
        print(f"[judge-sens] seed {seed} ({i}/{len(args.seeds)}): "
              f"stipulated arm ...", flush=True)
        rows.append(("stipulated", seed, _rows(_run_arm(seed, None))))
        print(f"[judge-sens] seed {seed} ({i}/{len(args.seeds)}): "
              f"measured arm ({MEASURED_P_DETECT}/{MEASURED_P_FALSE}) ...",
              flush=True)
        rows.append(("measured", seed, _rows(_run_arm(seed, {
            "AGENTWATCH_JUDGE_P_DETECT": str(MEASURED_P_DETECT),
            "AGENTWATCH_JUDGE_P_FALSE": str(MEASURED_P_FALSE)}))))

    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "judge_sensitivity.csv"
    cols = list(rows[0][2])
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write("arm,seed," + ",".join(cols) + "\n")
        for name, seed, row in rows:
            fh.write(f"{name},{seed}," +
                     ",".join("" if row[c] is None else str(row[c])
                              for c in cols) + "\n")

    if not args.keep_runs:
        shutil.rmtree(REPO_ROOT / SENSITIVITY_ROOT, ignore_errors=True)

    print(f"\n[judge-sens] H3b across {len(args.seeds)} seeds "
          f"(paired: same seed, same data, only the judge differs)")
    for arm in ("stipulated", "measured"):
        got = [r for n, _s, r in rows if n == arm]
        rec = [r["detection_recovered"] for r in got]
        calls = [r["call_fraction"] for r in got]
        supported = sum(1 for r in got if r["verdict"] == "SUPPORTED")
        print(f"  {arm:11s} recovered {_mean(rec):.0%} "
              f"(range {min(rec):.0%}-{max(rec):.0%}) at {_mean(calls):.0%} "
              f"of judge calls; SUPPORTED at {supported}/{len(got)} seeds")
    print(f"[judge-sens] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
