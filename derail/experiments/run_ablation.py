"""ESN hyperparameter ablation (review weakness E).

One-at-a-time sensitivity sweep around the primary monitor's defaults
(reservoir size, spectral radius, leak rate, ensemble size K, CUSUM drift
allowance). Each config is fit on healthy train, thresholded on healthy val
at the 5% FA budget, and evaluated on test — a SENSITIVITY report, not model
selection: the deployed defaults were fixed before this sweep and are not
re-picked from it.

Writes results/tables/esn_ablation.csv.
Run:  py -m derail.experiments.run_ablation   (~4 min)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from derail.common import DatasetConfig, SimConfig, Standardizer
from derail.evaluation.metrics import (
    episode_auc,
    evaluate_alarms,
    pick_threshold,
    summarize,
)
from derail.monitor.esn import ChannelMaxESNMonitor
from derail.telemetry.generator import make_dataset

BASE = Path(__file__).resolve().parents[2] / "results"
FA_BUDGET = 0.05

DEFAULTS = dict(K=8, reservoir_size=128, spectral_radius=0.9,
                leak_rate=0.3, cusum_k=0.5, beta_disagreement=0.5)
SWEEPS: dict[str, list] = {
    "reservoir_size": [32, 64, 128, 256],
    "spectral_radius": [0.6, 0.8, 0.9, 0.95, 1.05],
    "leak_rate": [0.15, 0.3, 0.5, 0.7],
    "K": [1, 2, 4, 8, 16],
    "cusum_k": [0.25, 0.5, 0.75, 1.0],
    # beta=0 turns the disagreement (ensemble prediction-spread) term OFF, so
    # this sweep is also the ablation the review asked for: it shows whether
    # the fixed disagreement weight of 0.5 is doing any work.
    "beta_disagreement": [0.0, 0.25, 0.5, 1.0, 2.0],
}


def main() -> None:
    data = make_dataset(DatasetConfig(), SimConfig())
    std = Standardizer().fit(data["train"])
    val, test = data["val"], data["test"]

    rows: list[dict] = []
    for param, values in SWEEPS.items():
        for value in values:
            cfg = dict(DEFAULTS)
            cfg[param] = value
            mon = ChannelMaxESNMonitor(std, cusum=True, seed=12,
                                       name="ablate", **cfg)
            mon.fit(data["train"])
            val_scores = [mon.score_episode(ep) for ep in val]
            theta = float(pick_threshold(val_scores, fa_budget=FA_BUDGET))
            test_scores = {ep.episode_id: mon.score_episode(ep)
                           for ep in test}
            summ = summarize(evaluate_alarms(test, test_scores, theta))
            row = {
                "param": param, "value": value,
                "is_default": value == DEFAULTS[param],
                "detection_rate": summ["detection_rate"],
                "mean_lead_all": summ["mean_lead_all"],
                "healthy_fa_rate": summ["healthy_fa_rate"],
                "episode_auc": float(episode_auc(test, test_scores)),
            }
            rows.append(row)
            print(f"  {param}={value!s:>6} det={row['detection_rate']:.3f} "
                  f"lead_all={row['mean_lead_all']:.2f} "
                  f"fa={row['healthy_fa_rate']:.3f} "
                  f"auc={row['episode_auc']:.3f}"
                  f"{'  (default)' if row['is_default'] else ''}")
    table = pd.DataFrame(rows)
    (BASE / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(BASE / "tables" / "esn_ablation.csv", index=False)
    print("wrote", BASE / "tables" / "esn_ablation.csv")


if __name__ == "__main__":
    main()
