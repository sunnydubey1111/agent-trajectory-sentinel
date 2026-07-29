"""Runtime/footprint benchmark of the monitors (reviewer request).

Reports, per monitor: one-off fit time on the healthy train split, per-step
scoring latency (median and p95 over ~10k streamed steps, measured around
score_step exactly as deployed), and the parameter/state footprint. The
point of comparison: one LLM agent step is hundreds of milliseconds to
seconds, so a monitor must sit orders of magnitude below that.

Writes results/tables/runtime.csv.
Run:  py -m derail.experiments.run_benchmark   (~3 min)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import DatasetConfig, SimConfig, Standardizer
from derail.experiments.run_experiment import build_monitors
from derail.telemetry.generator import make_dataset

BASE = Path(__file__).resolve().parents[2] / "results"


def _footprint_mb(mon) -> float:
    """Rough parameter/state footprint from the monitor's numpy/torch arrays."""
    seen: set[int] = set()
    total = 0

    def visit(obj, depth: int = 0) -> None:
        nonlocal total
        if depth > 3 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            total += obj.nbytes
            return
        if hasattr(obj, "parameters"):          # torch module
            for p in obj.parameters():
                total += p.numel() * p.element_size()
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                visit(v, depth + 1)
            return
        if hasattr(obj, "__dict__"):
            for v in vars(obj).values():
                visit(v, depth + 1)

    visit(mon)
    return total / 1e6


def main() -> None:
    data = make_dataset(DatasetConfig(), SimConfig())
    std = Standardizer().fit(data["train"])
    stream_eps = data["val"][:100]
    n_steps = sum(ep.T for ep in stream_eps)

    rows: list[dict] = []
    for mon in build_monitors(std, quick=False):
        t0 = time.perf_counter()
        mon.fit(data["train"])
        fit_s = time.perf_counter() - t0

        lat: list[float] = []
        for ep in stream_eps:
            mon.start_episode()
            for x in ep.X:
                t1 = time.perf_counter()
                mon.score_step(x)
                lat.append(time.perf_counter() - t1)
        lat_us = 1e6 * np.asarray(lat)
        rows.append({
            "monitor": mon.name,
            "fit_seconds": round(fit_s, 3),
            "step_latency_us_median": round(float(np.median(lat_us)), 1),
            "step_latency_us_p95": round(float(np.percentile(lat_us, 95)), 1),
            "footprint_mb": round(_footprint_mb(mon), 2),
            "n_steps_timed": n_steps,
        })
        r = rows[-1]
        print(f"  {r['monitor']:>18s}: fit {r['fit_seconds']:7.2f}s  "
              f"step {r['step_latency_us_median']:8.1f}us "
              f"(p95 {r['step_latency_us_p95']:8.1f})  "
              f"{r['footprint_mb']:6.2f} MB")

    table = pd.DataFrame(rows)
    (BASE / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(BASE / "tables" / "runtime.csv", index=False)
    print("wrote", BASE / "tables" / "runtime.csv")


if __name__ == "__main__":
    main()
