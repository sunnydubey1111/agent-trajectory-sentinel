"""Training-fairness diagnostics for the GRU/LSTM-vs-ESN comparison.

Answers three fairness questions with data (simulator testbed, default seed):

  1. CONVERGENCE — are the backprop baselines trained to a plateau at the
     default 40 epochs? (Per-epoch training MSE is logged; we report the
     loss improvement over the last 25% of epochs.)
  2. TRAINING BUDGET / CAPACITY — does 3x the epochs or 2x the hidden width
     close the gap to the ESN primary?
  3. ARCHITECTURE PARITY — the ESN primary uses a per-channel + max-fusion
     wrapper. Give the GRU the SAME wrapper (one GRU per channel, max of
     the three CUSUM streams): does the ESN's edge come from the reservoir
     or from the wrapper?

(The ESN itself has no convergence question: its readout is closed-form
ridge regression, and results/tables/esn_ablation.csv already shows flat
hyperparameter sensitivity.)

Writes results/tables/fairness.csv.
Run:  py -m derail.experiments.run_fairness   (~25 min, CPU)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import DatasetConfig, Episode, OnlineMonitor, SimConfig, Standardizer
from derail.evaluation.metrics import (
    episode_auc,
    evaluate_alarms,
    pick_threshold,
    summarize,
)
from derail.monitor.esn import ChannelMaxESNMonitor
from derail.monitor.seq_baselines import GRUMonitor, LSTMMonitor
from derail.telemetry.generator import make_dataset

BASE = Path(__file__).resolve().parents[2] / "results"
FA_BUDGET = 0.05


class GRUChannelMax(OnlineMonitor):
    """Per-channel GRU-CUSUM detectors fused by max — the exact wrapper the
    ESN primary uses, so the reservoir-vs-wrapper question is isolated."""

    name = "gru_cusum_max"

    def __init__(self, standardizer: Standardizer, epochs: int = 40) -> None:
        self.subs = [
            GRUMonitor(standardizer, epochs=epochs, seed=2000 + i,
                       channels=(c,), name=f"gru[{c}]")
            for i, c in enumerate(("e", "u", "m"))
        ]

    def fit(self, healthy_episodes: list[Episode]) -> None:
        for sub in self.subs:
            sub.fit(healthy_episodes)

    def start_episode(self) -> None:
        for sub in self.subs:
            sub.start_episode()

    def score_step(self, x_t: np.ndarray) -> float:
        return max(sub.score_step(x_t) for sub in self.subs)

    def score_episode(self, ep: Episode) -> np.ndarray:
        return np.max([sub.score_episode(ep) for sub in self.subs], axis=0)


def _evaluate(mon, data) -> dict:
    val_scores = [mon.score_episode(ep) for ep in data["val"]]
    theta = float(pick_threshold(val_scores, fa_budget=FA_BUDGET))
    scores = {ep.episode_id: mon.score_episode(ep) for ep in data["test"]}
    summ = summarize(evaluate_alarms(data["test"], scores, theta))
    return {
        "detection_rate": summ["detection_rate"],
        "healthy_fa_rate": summ["healthy_fa_rate"],
        "mean_lead_all": summ["mean_lead_all"],
        "episode_auc": float(episode_auc(data["test"], scores)),
    }


def _loss_plateau(loss_log: list[float]) -> dict:
    """Relative loss improvement over the final quarter of training."""
    q = max(1, len(loss_log) // 4)
    tail_drop = (loss_log[-q] - loss_log[-1]) / max(loss_log[-q], 1e-12)
    return {"loss_first": loss_log[0], "loss_last": loss_log[-1],
            "loss_tail_rel_drop": tail_drop}


def main() -> None:
    data = make_dataset(DatasetConfig(), SimConfig())
    std = Standardizer().fit(data["train"])

    rows: list[dict] = []

    def report(name: str, mon, loss_log: list[float] | None) -> None:
        stats = _evaluate(mon, data)
        row = {"config": name, **stats}
        if loss_log:
            row.update(_loss_plateau(loss_log))
        rows.append(row)
        extra = (f"  tail-drop={row.get('loss_tail_rel_drop', float('nan')):.3%}"
                 if loss_log else "")
        print(f"  {name:>22s}: det={stats['detection_rate']:.3f} "
              f"fa={stats['healthy_fa_rate']:.3f} "
              f"lead_all={stats['mean_lead_all']:.2f} "
              f"auc={stats['episode_auc']:.3f}{extra}")

    print("[fairness] 1+2: convergence and budget/capacity sweeps")
    for name, kwargs in [
        ("gru_default(e40,h64)", dict(epochs=40, hidden=64)),
        ("gru_long(e120,h64)", dict(epochs=120, hidden=64)),
        ("gru_big(e40,h128)", dict(epochs=40, hidden=128)),
        ("lstm_default(e40,h64)", dict(epochs=40, hidden=64)),
    ]:
        cls = LSTMMonitor if name.startswith("lstm") else GRUMonitor
        mon = cls(std, seed=14 if "gru" in name else 16, **kwargs)
        mon.fit(data["train"])
        report(name, mon, mon.loss_log)

    print("[fairness] 3: architecture parity (channel-max wrapper for GRU)")
    gmax = GRUChannelMax(std, epochs=40)
    gmax.fit(data["train"])
    report("gru_cusum_max", gmax, None)

    esn = ChannelMaxESNMonitor(std, K=8, cusum=True, seed=12,
                               name="esn_cusum_max")
    esn.fit(data["train"])
    report("esn_cusum_max (ref)", esn, None)

    table = pd.DataFrame(rows)
    (BASE / "tables").mkdir(parents=True, exist_ok=True)
    table.to_csv(BASE / "tables" / "fairness.csv", index=False)
    print(f"[fairness] wrote {BASE / 'tables' / 'fairness.csv'}")


if __name__ == "__main__":
    main()
