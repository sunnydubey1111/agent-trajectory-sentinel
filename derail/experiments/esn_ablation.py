"""Ablation study: Separating ESN reservoir capacity and ensemble size effects at N=20.

Runs a 2x2 grid search:
  - reservoir_size in [100, 500]
  - K in [8, 50]
Fits on N=20 healthy train episodes, evaluates test AUC across 3 seeds.

Run: py -m derail.experiments.esn_ablation
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


def run_ablation():
    reservoir_sizes = [500, 100]
    ensemble_sizes = [8, 50]
    seeds = [101, 102, 103]

    results = []

    print("Starting ESN ablation study at N=20...")
    print("| Reservoir | Ensemble | Seed | AUC |")
    print("|-----------|----------|------|-----|")

    for res_size in reservoir_sizes:
        for K in ensemble_sizes:
            for seed in seeds:
                # Generate dataset with N=20 healthy train episodes
                ds_cfg = DatasetConfig(
                    n_train_healthy=20,
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

                # Instantiate ESN
                esn = ESNEnsembleMonitor(std, K=K, reservoir_size=res_size, seed=seed)
                esn.fit(ds["train"])

                # Evaluate AUC on test split
                test_eps = ds["test"]
                scores_dict = {}
                for ep in test_eps:
                    esn.start_episode()
                    step_scores = [esn.score_step(x) for x in ep.X]
                    scores_dict[ep.episode_id] = np.array(step_scores)

                auc = episode_auc(test_eps, scores_dict)
                print(f"| {res_size:9d} | {K:8d} | {seed:4d} | {auc:.3f} |")

                results.append({
                    "Reservoir": res_size,
                    "Ensemble": K,
                    "Seed": seed,
                    "AUC": auc
                })

    df = pd.DataFrame(results)
    summary = df.groupby(["Reservoir", "Ensemble"])["AUC"].mean().reset_index()

    print("\n=== Average Ablation Results ===")
    print("Reservoir | Ensemble | Mean AUC")
    print("----------|----------|---------")
    for _, row in summary.iterrows():
        print(f"{int(row['Reservoir']):9d} | {int(row['Ensemble']):8d} | {row['AUC']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "esn_ablation_grid.csv", index=False)
    print(f"\nSaved ablation results to {RESULTS_DIR / 'esn_ablation_grid.csv'}")


if __name__ == "__main__":
    run_ablation()
