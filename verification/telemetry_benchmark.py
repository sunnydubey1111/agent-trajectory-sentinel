"""Reproducible per-step telemetry-extraction cost (v4 grounding path).

Backs the paper's telemetry-cost figure (§10). Times step_signal_grd over
real recorded steps from traces/demo7b (warm cache), and writes the
median/p95/mean microseconds-per-step to results/tables/telemetry_runtime.csv.

Wall-clock latency is machine-sensitive (see the P2 findings on machine
drift); this script is the reproducible source of record, not a fixed
constant. Run: py -m verification.telemetry_benchmark
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from derail.telemetry.adapter import (ExtFeatureState, GrdFeatureState,
                                      step_signal_grd)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = sorted((ROOT / "traces" / "demo7b").glob("demo-healthy-0*.jsonl"))
    episodes = [[json.loads(x) for x in f.read_text("utf-8").splitlines() if x]
                for f in files[:40]]
    if not episodes:
        raise SystemExit("no demo7b traces found to benchmark")

    for steps in episodes[:2]:              # warm caches (hash projection)
        xs, gs = ExtFeatureState(), GrdFeatureState()
        for s in steps:
            step_signal_grd(s, xs, gs, use_sentence_transformers=False)

    per_step_us: list[float] = []
    for steps in episodes:
        xs, gs = ExtFeatureState(), GrdFeatureState()
        for s in steps:
            t0 = time.perf_counter()
            step_signal_grd(s, xs, gs, use_sentence_transformers=False)
            per_step_us.append((time.perf_counter() - t0) * 1e6)

    a = np.array(per_step_us)
    row = {"path": "telemetry_v4_grounding", "n_steps": int(a.size),
           "median_us": round(float(np.median(a)), 1),
           "p95_us": round(float(np.quantile(a, 0.95)), 1),
           "mean_us": round(float(a.mean()), 1)}
    out = ROOT / "results" / "tables" / "telemetry_runtime.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"[telemetry-bench] {row}")
    print(f"[telemetry-bench] wrote {out}")


if __name__ == "__main__":
    main()
