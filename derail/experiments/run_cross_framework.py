"""Experiment: Cross-Framework Generalization Matrix on real agent traces.

Loads real traces from:
  - LangGraph (Ollama-Qwen)
  - AutoGen (Ollama-Qwen)
  - Gemini (Gemini API traces in traces/real/)

Trains the one-class ESN monitor on healthy traces from each source framework,
and tests its detection ROC-AUC on all target frameworks without retraining.

Run: py -m derail.experiments.run_cross_framework
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from derail.common import Episode, Standardizer, rng_for
from derail.telemetry.adapter import load_trace_jsonl
from derail.monitor.esn import ChannelMaxESNMonitor
from derail.evaluation.metrics import episode_auc

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"
MIN_T = 3


def load_dataset(dir_path: Path, manifest_name: str = "manifest.json") -> tuple[list[Episode], list[Episode]]:
    manifest_path = dir_path / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    episodes: list[Episode] = []
    for entry in manifest:
        if entry["T"] < MIN_T:
            continue
        ep = load_trace_jsonl(
            dir_path / entry["file"],
            episode_id=entry["episode_id"],
            tau=entry["tau"],
            failure_class=entry["failure_class"],
            severity=None if entry["tau"] is None else 0.5,
            use_sentence_transformers=False,
            extended=False
        )
        episodes.append(ep)

    healthy = [ep for ep in episodes if ep.is_healthy]
    injected = [ep for ep in episodes if not ep.is_healthy]
    return healthy, injected


def run_cross_framework():
    # Framework trace paths
    frameworks = {
        "LangGraph": TRACES_DIR / "langgraph",
        "AutoGen": TRACES_DIR / "autogen",
        "GeminiAPI": TRACES_DIR / "real"
    }

    # Load all datasets
    data = {}
    for name, path in frameworks.items():
        try:
            healthy, injected = load_dataset(path)
            data[name] = {"healthy": healthy, "injected": injected}
            print(f"Loaded {name}: {len(healthy)} healthy, {len(injected)} injected traces.")
        except Exception as e:
            print(f"Error loading {name}: {e}")

    framework_names = list(data.keys())
    matrix_auc = pd.DataFrame(index=framework_names, columns=framework_names)

    print("\nStarting Cross-Framework Generalization Matrix...")
    print("| Train Domain | Test Domain  | Test AUC |")
    print("|--------------|--------------|----------|")

    # Precompute the on-diagonal healthy split for EVERY framework ONCE, so the
    # diagonal cell trains only on the train portion and tests only on the
    # held-out portion. The old code fit the monitor on ALL of a framework's
    # healthy episodes and then "held out" 40% of that same set, so on the
    # diagonal every held-out healthy test episode had been in training
    #. Split first, then fit.
    diag_split: dict[str, tuple[list, list]] = {}
    for fw in framework_names:
        healthy = data[fw]["healthy"]
        perm = rng_for(42, "cross-split", fw).permutation(len(healthy))
        n_train = int(round(0.6 * len(healthy)))
        train_ids = {healthy[i].episode_id for i in perm[:n_train]}
        diag_split[fw] = (
            [ep for ep in healthy if ep.episode_id in train_ids],
            [ep for ep in healthy if ep.episode_id not in train_ids])

    for train_name in framework_names:
        for test_name in framework_names:
            # On the diagonal, fit on the train portion and test on the DISJOINT
            # held-out portion; off the diagonal, the whole source is training
            # and the whole target is test (already disjoint corpora).
            if train_name == test_name:
                fit_healthy, test_h_split = diag_split[train_name]
            else:
                fit_healthy = data[train_name]["healthy"]
                test_h_split = data[test_name]["healthy"]
            test_injected = data[test_name]["injected"]

            std = Standardizer().fit(fit_healthy)
            monitor = ChannelMaxESNMonitor(std, K=15, reservoir_size=100, seed=0)
            monitor.fit(fit_healthy)

            # No test-healthy episode may share an id with a training episode.
            fit_ids = {ep.episode_id for ep in fit_healthy}
            assert not (fit_ids & {ep.episode_id for ep in test_h_split}), \
                f"healthy leakage on {train_name}->{test_name}"

            test_eps = test_h_split + test_injected
            scores = {}
            for ep in test_eps:
                monitor.start_episode()
                scores[ep.episode_id] = np.array([monitor.score_step(x) for x in ep.X])

            auc = episode_auc(test_eps, scores)
            matrix_auc.at[train_name, test_name] = f"{auc:.3f}"
            print(f"| {train_name:12s} | {test_name:12s} | {auc:.3f}    |")

    print("\n=== Transfer Matrix: Train Domain vs. Test Domain (ROC-AUC) ===")
    print(matrix_auc)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix_auc.to_csv(RESULTS_DIR / "real_cross_framework_matrix.csv")
    print(f"\nSaved cross-framework matrix to {RESULTS_DIR / 'real_cross_framework_matrix.csv'}")


if __name__ == "__main__":
    run_cross_framework()
