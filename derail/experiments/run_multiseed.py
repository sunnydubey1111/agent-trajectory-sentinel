"""Multi-seed stability study (review weakness C, stability part).

Runs the full experiment at several master seeds (fresh datasets each time;
monitor weights are seeded separately and stay fixed) and reports
mean +/- std of the headline metrics per monitor, plus per-seed hypothesis
verdicts. Writes results/tables/multiseed_summary.csv and
results/multiseed.json.

Run:  py -m derail.experiments.run_multiseed  (~10-15 min for 5 seeds)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import MASTER_SEED
from derail.experiments import run_experiment

SEEDS = (MASTER_SEED, 7, 101, 202, 303)
KEY_MONITORS = ("esn_cusum_max", "esn_cusum", "esn_full", "esn_single",
                "gru", "lstm", "tcn", "linear_ar", "delta_mahalanobis",
                "mahalanobis", "self_drift")
KEY_METRICS = ("detection_rate", "mean_lead_all", "episode_auc",
               "healthy_fa_rate")
BASE = Path(__file__).resolve().parents[2] / "results"


def main() -> None:
    per_seed: dict[int, dict] = {}
    for seed in SEEDS:
        print(f"\n===== seed {seed} =====")
        run_experiment.main(["--seed", str(seed)])
        rdir = BASE if seed == MASTER_SEED else BASE / f"seed{seed}"
        per_seed[seed] = json.loads(
            (rdir / "results.json").read_text(encoding="utf-8"))

    rows: list[dict] = []
    for mon in KEY_MONITORS:
        row: dict = {"monitor": mon}
        ok = True
        for metric in KEY_METRICS:
            vals = []
            for seed in SEEDS:
                per_mon = per_seed[seed]["h1"]["per_monitor"]
                if mon not in per_mon:
                    ok = False
                    break
                vals.append(float(per_mon[mon][metric]))
            if not ok:
                break
            row[f"{metric}_mean"] = float(np.mean(vals))
            row[f"{metric}_std"] = float(np.std(vals, ddof=1))
        if ok:
            rows.append(row)
    table = pd.DataFrame(rows)
    (BASE / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(BASE / "tables" / "multiseed_summary.csv", index=False)

    verdicts = {str(seed): {k: v.split(":", 1)[0]
                            for k, v in per_seed[seed]["verdicts"].items()}
                for seed in SEEDS}
    summary = {
        "seeds": list(SEEDS),
        "verdicts_by_seed": verdicts,
        "all_supported_every_seed": all(
            v == "SUPPORTED"
            for by_seed in verdicts.values() for v in by_seed.values()),
        # Provenance for the atomic five-seed set: each seed's
        # config fingerprint, so a stale seed cannot be silently mixed in.
        "config_sha256_by_seed": {
            str(seed): per_seed[seed].get("config_sha256") for seed in SEEDS},
        "quick_by_seed": {
            str(seed): per_seed[seed]["config"].get("quick") for seed in SEEDS},
        "provenance": run_experiment._provenance(quick=False, seed=MASTER_SEED),
    }
    (BASE / "multiseed.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n===== multi-seed summary (mean +/- std over "
          f"{len(SEEDS)} seeds) =====")
    for _, r in table.iterrows():
        print(f"  {r['monitor']:>18s}: det {r['detection_rate_mean']:.3f}"
              f"+/-{r['detection_rate_std']:.3f}  "
              f"lead_all {r['mean_lead_all_mean']:.2f}"
              f"+/-{r['mean_lead_all_std']:.2f}  "
              f"auc {r['episode_auc_mean']:.3f}"
              f"+/-{r['episode_auc_std']:.3f}")
    print("verdicts by seed:", json.dumps(verdicts))
    print("wrote", BASE / "tables" / "multiseed_summary.csv")


if __name__ == "__main__":
    main()
