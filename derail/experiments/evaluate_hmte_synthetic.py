"""Evaluate HMTE-ESN-M vs ChannelMax on a large synthetic dataset to ensure stable covariance estimation.

Run: py -m derail.experiments.evaluate_hmte_synthetic
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from derail.common import DatasetConfig, SimConfig, Standardizer
from derail.telemetry.generator import make_dataset
from derail.monitor.esn import ChannelMaxESNMonitor, HMTE_ESN_M_Monitor
from derail.evaluation.metrics import episode_auc

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"


def run_synthetic_evaluation():
    seeds = [101, 102, 103]

    results = []

    print("Evaluating HMTE-ESN-M vs ChannelMax on synthetic data (stable covariance)...")
    print("| Seed | ChannelMax AUC | HMTE-ESN-M AUC |")
    print("|------|----------------|----------------|")

    for seed in seeds:
        ds_cfg = DatasetConfig(
            n_train_healthy=100,
            n_val_healthy=50,
            n_cal_healthy=20,
            n_cal_injected_per_class=10,
            n_test_healthy=100,
            n_test_injected_per_class=20,
            master_seed=seed
        )
        sim_cfg = SimConfig()
        ds = make_dataset(ds_cfg, sim_cfg)

        # Fit standardizer
        std = Standardizer()
        std.fit(ds["train"])

        # Instantiate monitors
        channel_max = ChannelMaxESNMonitor(std, K=15, reservoir_size=100, seed=seed)
        hmte_esn_m = HMTE_ESN_M_Monitor(std, K=15, reservoir_size=100, seed=seed)

        # Fit monitors
        channel_max.fit(ds["train"])
        hmte_esn_m.fit(ds["train"])

        # Evaluate test AUC
        test_eps = ds["test"]

        def get_auc(monitor):
            scores_dict = {}
            for ep in test_eps:
                monitor.start_episode()
                step_scores = [monitor.score_step(x) for x in ep.X]
                scores_dict[ep.episode_id] = np.array(step_scores)
            return episode_auc(test_eps, scores_dict)

        max_auc = get_auc(channel_max)
        hmte_auc = get_auc(hmte_esn_m)

        print(f"| {seed:4d} | {max_auc:.3f}          | {hmte_auc:.3f}          |")
        results.append({
            "Seed": seed,
            "ChannelMax": max_auc,
            "HMTE-ESN-M": hmte_auc
        })

    df = pd.DataFrame(results)
    mean_max = df["ChannelMax"].mean()
    mean_hmte = df["HMTE-ESN-M"].mean()

    print("\n=== Average Results ===")
    print(f"ChannelMax Mean AUC : {mean_max:.3f}")
    print(f"HMTE-ESN-M Mean AUC : {mean_hmte:.3f}")


if __name__ == "__main__":
    run_synthetic_evaluation()
