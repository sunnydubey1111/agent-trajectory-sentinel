"""Rigorous statistical analysis of ESN task capacity vs measured dataset characteristics.

Specifically measures:
  1. Average steps (T) vs. Best Reservoir Size
  2. Average Embedding Drift (step-to-step cosine distance)
  3. Average Entropy (token entropy)
  4. Calculates Pearson Correlation coefficients for Reservoir Size vs. these metrics.

Run: py -m derail.experiments.task_rigorous_analysis
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

from derail.common import DatasetConfig, SimConfig
from derail.telemetry.generator import make_dataset

# Optimal reservoir sizes observed in task_capacity_ablation
OPTIMAL_SIZES = {
    "RAG": 100,
    "Coding": 50,
    "Planning": 200,
    "Math": 150,
    "Search": 150
}

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"


def compute_embedding_drift(X: np.ndarray) -> float:
    """Compute mean step-to-step embedding drift (1 - cosine similarity)."""
    drifts = []
    for t in range(1, X.shape[0]):
        e_t0 = X[t-1, :32]
        e_t1 = X[t, :32]
        norm0 = np.linalg.norm(e_t0)
        norm1 = np.linalg.norm(e_t1)
        if norm0 > 1e-6 and norm1 > 1e-6:
            cos_sim = np.dot(e_t0, e_t1) / (norm0 * norm1)
            # clamp to [0, 2]
            drifts.append(np.clip(1.0 - cos_sim, 0.0, 2.0))
    return float(np.mean(drifts)) if drifts else 0.0


def run_analysis():
    # Define custom SimConfigs simulating different task domains
    task_configs = {
        "RAG": SimConfig(
            waypoint_sigma=0.55,
            outlen_lognorm={
                "plan": (4.5, 0.5), "tool_call": (3.0, 0.4),
                "tool_result": (5.8, 0.8),
                "synthesis": (5.0, 0.5)
            }
        ),
        "Coding": SimConfig(
            healthy_error_rate=0.12,
            latency_lognorm={
                "plan": (-1.2, 0.4), "tool_call": (-1.0, 0.3),
                "tool_result": (-0.5, 0.4), "synthesis": (-0.9, 0.4)
            }
        ),
        "Planning": SimConfig(
            T_min=40, T_max=80,
            n_waypoints_min=6, n_waypoints_max=12,
            latency_lognorm={
                "plan": (1.8, 0.5), "tool_call": (0.8, 0.6),
                "tool_result": (-0.5, 0.4), "synthesis": (-0.9, 0.4)
            }
        ),
        "Math": SimConfig(
            waypoint_sigma=0.10,
            ar_rho=0.90,
            entropy_base={
                "plan": 1.0, "tool_call": 0.8, "tool_result": 0.5, "synthesis": 0.4
            }
        ),
        "Search": SimConfig(
            n_waypoints_min=4, n_waypoints_max=8,
            latency_lognorm={
                "plan": (-1.2, 0.4), "tool_call": (1.2, 0.4),
                "tool_result": (4.0, 0.7), "synthesis": (-0.9, 0.4)
            }
        )
    }

    seeds = [101, 102, 103]

    measurements = []

    print("Measuring telemetry characteristics for ESN correlation analysis...")

    for task_name, sim_cfg in task_configs.items():
        lengths = []
        drifts = []
        entropies = []

        for seed in seeds:
            ds_cfg = DatasetConfig(
                n_train_healthy=30,
                n_val_healthy=10,
                n_cal_healthy=10,
                n_cal_injected_per_class=5,
                n_test_healthy=10,
                n_test_injected_per_class=5,
                master_seed=seed
            )
            ds = make_dataset(ds_cfg, sim_cfg)

            # Measure characteristics on healthy train episodes
            for ep in ds["train"]:
                lengths.append(len(ep.X))
                drifts.append(compute_embedding_drift(ep.X))
                entropies.append(ep.X[:, 32].mean()) # index 32 is mean token entropy

        measurements.append({
            "Task": task_name,
            "Optimal_Size": OPTIMAL_SIZES[task_name],
            "Avg_Steps": np.mean(lengths),
            "Avg_Drift": np.mean(drifts),
            "Avg_Entropy": np.mean(entropies)
        })

    df = pd.DataFrame(measurements)
    print("\n=== Measured Task Characteristics ===")
    print(df.to_string(index=False))

    # EXPLORATORY: the "optimal" reservoir size is selected on the
    # same small grid these correlations then use, and there are only a handful
    # of hand-designed task settings, so this is under-powered and
    # post-selected. The three correlation p-values are Holm-corrected and the
    # whole analysis is reported as exploratory, not confirmatory.
    from derail.evaluation.protocol import holm_bonferroni
    print("\n=== Pearson Correlation vs. Optimal Reservoir Size "
          "(EXPLORATORY, Holm-corrected) ===")
    raw = {col: pearsonr(df[col], df["Optimal_Size"])
           for col in ("Avg_Steps", "Avg_Drift", "Avg_Entropy")}
    holm = holm_bonferroni({c: rp[1] for c, rp in raw.items()})
    for col, (r, _) in raw.items():
        adj = holm[col]
        print(f"Optimal Size vs. {col:18s} | Pearson r = {r:+.3f} | "
              f"p = {adj['p_raw']:.3f} | Holm p = {adj['p_holm']:.3f}"
              f"{' *' if adj['reject'] else ''}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "esn_task_rigorous_metrics.csv", index=False)
    print(f"\nSaved metrics to {RESULTS_DIR / 'esn_task_rigorous_metrics.csv'}")


if __name__ == "__main__":
    run_analysis()
