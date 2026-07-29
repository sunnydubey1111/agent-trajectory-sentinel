"""Experiment: Evaluate HMTE-ESN-M vs ChannelMax on real Gemini agent traces.

Loads traces from traces/real/, splits them under the standard validation protocol,
fits ChannelMax and HMTE-ESN-M monitors, and compares their ROC-AUC.

Run: py -m derail.experiments.evaluate_hmte
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from derail.common import Episode, Standardizer, rng_for
from derail.telemetry.adapter import load_trace_jsonl
from derail.monitor.esn import ChannelMaxESNMonitor, HMTE_ESN_M_Monitor
from derail.evaluation.metrics import episode_auc

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces" / "real"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"
MIN_T = 4


def run_hmte_evaluation():
    manifest_path = TRACES_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest file not found at {manifest_path}. Please run collect_real_traces.py first.")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    episodes: list[Episode] = []
    for entry in manifest:
        if entry["T"] < MIN_T:
            continue
        ep = load_trace_jsonl(TRACES_DIR / entry["file"],
                              episode_id=entry["episode_id"],
                              tau=entry["tau"], failure_class=entry["failure_class"],
                              severity=None if entry["tau"] is None else 0.5,
                              use_sentence_transformers=False,
                              extended=False)
        episodes.append(ep)

    healthy = [ep for ep in episodes if ep.is_healthy]
    injected = [ep for ep in episodes if not ep.is_healthy]

    if len(healthy) < 4:
        print(f"Only {len(healthy)} usable healthy traces — need >= 4 to evaluate.")
        return

    # Split healthy into train/val/test
    perm = rng_for(0, "real-split").permutation(len(healthy))
    n_train = int(round(0.6 * len(healthy)))
    n_val = int(round(0.2 * len(healthy)))
    train = [healthy[i] for i in perm[:n_train]]
    val = [healthy[i] for i in perm[n_train:n_train + n_val]]
    test_h = [healthy[i] for i in perm[n_train + n_val:]]
    test = test_h + injected

    # Fit standardizer
    std = Standardizer().fit(train)

    # Instantiate monitors
    # Proposed baseline: ChannelMaxESNMonitor
    channel_max = ChannelMaxESNMonitor(std, K=15, reservoir_size=100, seed=0)
    # Proposed new architecture: HMTE-ESN-M
    hmte_esn_m = HMTE_ESN_M_Monitor(std, K=15, reservoir_size=100, seed=0)

    # Fit monitors
    channel_max.fit(train)
    hmte_esn_m.fit(train)

    # Evaluate test AUC
    def get_auc(monitor):
        scores_dict = {}
        for ep in test:
            monitor.start_episode()
            step_scores = [monitor.score_step(x) for x in ep.X]
            scores_dict[ep.episode_id] = np.array(step_scores)
        return episode_auc(test, scores_dict)

    max_auc = get_auc(channel_max)
    hmte_auc = get_auc(hmte_esn_m)

    print("\n=== HMTE-ESN-M vs ChannelMax ESN-CUSUM on Real Traces ===")
    print(f"ChannelMax (AUC) : {max_auc:.3f}")
    print(f"HMTE-ESN-M (AUC) : {hmte_auc:.3f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "Monitor": "ChannelMax",
        "Test_AUC": max_auc
    }, {
        "Monitor": "HMTE-ESN-M",
        "Test_AUC": hmte_auc
    }])
    df.to_csv(RESULTS_DIR / "hmte_vs_baseline.csv", index=False)
    print(f"\nSaved results to {RESULTS_DIR / 'hmte_vs_baseline.csv'}")


if __name__ == "__main__":
    run_hmte_evaluation()
