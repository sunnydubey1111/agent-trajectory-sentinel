"""Ablation: Evaluate ESN reservoir capacity vs task type at N=20.

Defines 5 task domains by customizing SimConfig telemetry signatures:
  - RAG (long documents, high semantic variance)
  - Coding (frequent syntax errors, fast execution)
  - Planning (long episodes, many waypoints)
  - Math (focused reasoning, low semantic variance, low entropy)
  - Search (rapid web tool calls, service delays)

Runs ESN (K=30) across sizes [50, 100, 150, 200, 300, 500, 1000] to identify the optimal
capacity per task domain at N=20.

Run: py -m derail.experiments.task_capacity_ablation
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


def run_task_capacity_ablation():
    # Define custom SimConfigs simulating different task domains
    task_configs = {
        "RAG": SimConfig(
            waypoint_sigma=0.55,  # high semantic variation between docs
            outlen_lognorm={
                "plan": (4.5, 0.5), "tool_call": (3.0, 0.4),
                "tool_result": (5.8, 0.8),  # extremely long retrieved context
                "synthesis": (5.0, 0.5)
            }
        ),
        "Coding": SimConfig(
            healthy_error_rate=0.12,  # high rate of compiler/syntax errors in normal runs
            latency_lognorm={
                "plan": (-1.2, 0.4), "tool_call": (-1.0, 0.3),  # fast interpreter calls
                "tool_result": (-0.5, 0.4), "synthesis": (-0.9, 0.4)
            }
        ),
        "Planning": SimConfig(
            T_min=40, T_max=80,  # long planning trajectories
            n_waypoints_min=6, n_waypoints_max=12,  # many sequential checkpoints
            latency_lognorm={
                "plan": (1.8, 0.5), "tool_call": (0.8, 0.6),  # heavy planning delays
                "tool_result": (-0.5, 0.4), "synthesis": (-0.9, 0.4)
            }
        ),
        "Math": SimConfig(
            waypoint_sigma=0.10,  # highly focused, low-noise reasoning steps
            ar_rho=0.90,  # highly coherent trajectories
            entropy_base={
                "plan": 1.0, "tool_call": 0.8, "tool_result": 0.5, "synthesis": 0.4  # low entropy (high confidence)
            }
        ),
        "Search": SimConfig(
            n_waypoints_min=4, n_waypoints_max=8,
            latency_lognorm={
                "plan": (-1.2, 0.4), "tool_call": (1.2, 0.4),  # web search request delays
                "tool_result": (4.0, 0.7), "synthesis": (-0.9, 0.4)
            }
        )
    }

    reservoir_sizes = [50, 100, 150, 200, 300, 500, 1000]
    seeds = [101, 102]
    K = 15  # ensemble size to control ESN variance and optimize CPU execution speed

    results = []

    print("Starting ESN Task-Capacity Ablation Study...")
    print("| Task | Reservoir | Mean AUC |")
    print("|------|-----------|----------|")

    for task_name, sim_cfg in task_configs.items():
        for res_size in reservoir_sizes:
            aucs = []
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
                aucs.append(auc)

            mean_auc = float(np.mean(aucs))
            print(f"| {task_name:8s} | {res_size:9d} | {mean_auc:.3f} |")
            results.append({
                "Task": task_name,
                "Reservoir": res_size,
                "AUC": mean_auc
            })

    df = pd.DataFrame(results)

    # Pivot results for standard presentation
    pivot_df = df.pivot(index="Reservoir", columns="Task", values="AUC").reset_index()

    print("\n=== Pivot Table: Reservoir Size vs. Task Type Mean AUC ===")
    print("Reservoir | RAG   | Coding | Planning | Math  | Search")
    print("----------|-------|--------|----------|-------|-------")
    for _, row in pivot_df.iterrows():
        print(f"{int(row['Reservoir']):9d} | {row['RAG']:.3f} | {row['Coding']:.3f}  | {row['Planning']:.3f}    | {row['Math']:.3f} | {row['Search']:.3f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pivot_df.to_csv(RESULTS_DIR / "esn_task_capacity_grid.csv", index=False)
    print(f"\nSaved grid results to {RESULTS_DIR / 'esn_task_capacity_grid.csv'}")


if __name__ == "__main__":
    run_task_capacity_ablation()
