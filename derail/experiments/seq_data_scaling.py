"""Experiment: Evaluate ESN, GRU, and LSTM performance vs training dataset size.

Generates local synthetic datasets with N in [10, 20, 40, 80] healthy train episodes,
fits monitors, and computes test AUC to demonstrate sample efficiency differences.

Run: py -m derail.experiments.seq_data_scaling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from derail.common import DatasetConfig, SimConfig, Standardizer, rng_for
from derail.telemetry.generator import make_dataset
from derail.monitor.esn import ESNEnsembleMonitor
from derail.monitor.seq_baselines import GRUMonitor, LSTMMonitor
from derail.evaluation.metrics import episode_auc

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"


def run_scaling_experiment():
    N_sizes = [5, 10, 20, 40, 80, 160]
    seeds = [101, 102]  # 2 seeds are sufficient with K=50 ensemble members to average ESN variance

    results = []

    print("Starting data scaling experiment...")
    print("| N_train | Seed | ESN AUC | GRU AUC | LSTM AUC |")
    print("|---------|------|---------|---------|----------|")

    for N in N_sizes:
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

            # Instantiate monitors: ESN with K=50 members and reservoir_size=100 for low variance and high data-scarcity efficiency
            esn = ESNEnsembleMonitor(std, K=50, reservoir_size=100, seed=seed)
            gru = GRUMonitor(std, hidden=32, epochs=40, seed=seed)
            lstm = LSTMMonitor(std, hidden=32, epochs=40, seed=seed)

            # Fit monitors on train split (healthy only)
            esn.fit(ds["train"])
            gru.fit(ds["train"])
            lstm.fit(ds["train"])

            # Evaluate AUC on test split
            test_eps = ds["test"]

            def get_auc(monitor):
                scores_dict = {}
                for ep in test_eps:
                    monitor.start_episode()
                    step_scores = [monitor.score_step(x) for x in ep.X]
                    scores_dict[ep.episode_id] = np.array(step_scores)
                return episode_auc(test_eps, scores_dict)

            esn_auc = get_auc(esn)
            gru_auc = get_auc(gru)
            lstm_auc = get_auc(lstm)

            print(f"| {N:7d} | {seed:4d} | {esn_auc:.3f}   | {gru_auc:.3f}   | {lstm_auc:.3f}    |")
            results.append({
                "N_train": N,
                "Seed": seed,
                "ESN_AUC": esn_auc,
                "GRU_AUC": gru_auc,
                "LSTM_AUC": lstm_auc
            })

    df = pd.DataFrame(results)
    # Average only the AUC metrics across seeds. `df.groupby("N_train").mean()`
    # also averaged the Seed column, producing a meaningless "mean seed"
    #; restrict to the metric columns explicitly.
    metric_cols = [c for c in df.columns if c.endswith("_AUC")]
    summary = df.groupby("N_train")[metric_cols].mean().reset_index()

    print("\n=== Average Results across seeds ===")
    print("N_train | ESN_AUC | GRU_AUC | LSTM_AUC")
    print("--------|---------|---------|---------")
    for _, row in summary.iterrows():
        print(f"{int(row['N_train']):7d} | {row['ESN_AUC']:.3f}   | {row['GRU_AUC']:.3f}   | {row['LSTM_AUC']:.3f}")

    # Save results to Capstone results folder
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "seq_data_scaling.csv", index=False)
    print(f"\nSaved scaling results to {RESULTS_DIR / 'seq_data_scaling.csv'}")


if __name__ == "__main__":
    run_scaling_experiment()
