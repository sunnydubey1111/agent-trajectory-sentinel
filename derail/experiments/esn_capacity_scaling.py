"""Experiment: Evaluate ESN reservoir capacity vs training dataset size.

Runs a grid search over:
  - N_train in [20, 40, 80]
  - reservoir_size in [100, 300, 500]
Evaluates mean test AUC across seeds to see if the optimal reservoir capacity
grows with available training data.

Run: py -m derail.experiments.esn_capacity_scaling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from derail.common import DatasetConfig, SimConfig, Standardizer
from derail.telemetry.generator import make_dataset
from derail.monitor.esn import ESNEnsembleMonitor
from derail.evaluation.metrics import episode_auc

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"


def run_capacity_scaling():
    N_sizes = [20, 40, 80]
    reservoir_sizes = [100, 300, 500]
    seeds = [101, 102]
    K = 30  # 30 ensemble members is optimal for low variance and execution speed

    results = []

    print("Starting ESN Capacity Scaling experiment...")
    print("| N_train | Reservoir | Seed | AUC |")
    print("|---------|-----------|------|-----|")

    for N in N_sizes:
        for res_size in reservoir_sizes:
            for seed in seeds:
                # Generate dataset
                ds_cfg = DatasetConfig(
                    n_train_healthy=N,
                    n_val_healthy=20,
                    n_cal_healthy=20,
                    n_cal_injected_per_class=10,
                    n_test_healthy=50,
                    n_test_injected_per_class=10,
                    master_seed=seed
                )
                sim_cfg = SimConfig()
                ds = make_dataset(ds_cfg, sim_cfg)

                # Fit standardizer
                std = Standardizer()
                std.fit(ds["train"])

                # Fit ESN
                esn = ESNEnsembleMonitor(std, K=K, reservoir_size=res_size, seed=seed)
                esn.fit(ds["train"])

                # Score test episodes
                test_eps = ds["test"]
                scores_dict = {}
                for ep in test_eps:
                    esn.start_episode()
                    step_scores = [esn.score_step(x) for x in ep.X]
                    scores_dict[ep.episode_id] = np.array(step_scores)

                auc = episode_auc(test_eps, scores_dict)
                print(f"| {N:7d} | {res_size:9d} | {seed:4d} | {auc:.3f} |")

                results.append({
                    "N_train": N,
                    "Reservoir": res_size,
                    "Seed": seed,
                    "AUC": auc
                })

    df = pd.DataFrame(results)
    summary = df.groupby(["N_train", "Reservoir"])["AUC"].mean().reset_index()

    print("\n=== Average Capacity Scaling Results ===")
    print("N_train | Reservoir | Mean AUC")
    print("--------|-----------|---------")
    for _, row in summary.iterrows():
        print(f"{int(row['N_train']):7d} | {int(row['Reservoir']):9d} | {row['AUC']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "esn_capacity_scaling.csv", index=False)
    print(f"\nSaved capacity scaling results to {RESULTS_DIR / 'esn_capacity_scaling.csv'}")


if __name__ == "__main__":
    run_capacity_scaling()
