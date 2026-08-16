"""Adversarial-evasion robustness limit of the primary monitor.

The monitors are NOT designed against an adversary who shapes the telemetry
while still failing the task. This script measures how far such a white-box
adversary gets, deterministically, on the synthetic study (where we control the
telemetry and can apply the evasion transforms exactly).

Threat model. The adversary keeps the failure happening but, for a targeted
telemetry channel, replaces that channel's per-step values with the HEALTHY
training grand-mean (a zero-surprise, in-distribution signal the ESN predicts
perfectly and the Mahalanobis distance scores as normal). This is an upper
bound on single-adversary evasion: it fully removes that channel's anomaly. We
report the primary monitor's detection at the deployed 5%-FA threshold under:

  - none            : no evasion (baseline detection),
  - evade e / u / m : one behavioural channel healthy-ised,
  - evade e+u+m     : all three behavioural channels healthy-ised (full white-box).

The honest finding is the robustness LIMIT: per-channel evasion is blunted by
max-fusion (the other channels still fire), but a full-telemetry adversary who
can shape every channel drives detection toward the false-alarm floor. This is
a limitation to disclose, not a defended capability.

Run:  py -m experimental.adversarial_evasion
Writes results/tables/adversarial_evasion.csv
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import (CHANNEL_SLICES, DatasetConfig, Episode, SimConfig,
                           Standardizer)
from derail.evaluation.metrics import pick_threshold
from derail.monitor.esn import ChannelMaxESNMonitor
from derail.telemetry.generator import make_dataset

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"
FA_BUDGET = 0.05
EVASIONS = [("none", ()), ("evade_e", ("e",)), ("evade_u", ("u",)),
            ("evade_m", ("m",)), ("evade_e+u+m", ("e", "u", "m"))]


def _healthy_ise(ep: Episode, channels: tuple[str, ...],
                 healthy_mean: np.ndarray) -> Episode:
    """Return a copy of ep with the given channels replaced by the healthy
    grand-mean (per dim, broadcast over steps) - full single-channel evasion."""
    X = ep.X.copy()
    for ch in channels:
        sl = CHANNEL_SLICES[ch]
        X[:, sl] = healthy_mean[sl]
    return Episode(X=X, episode_id=ep.episode_id, is_healthy=ep.is_healthy,
                   failure_class=ep.failure_class, tau=ep.tau,
                   t_fail=ep.t_fail, severity=ep.severity)


def _detection(monitor, episodes: list[Episode], theta: float) -> float:
    hits = 0
    for ep in episodes:
        s = monitor.score_episode(ep)
        if np.any(s > theta):
            hits += 1
    return hits / len(episodes) if episodes else 0.0


def main() -> None:
    data = make_dataset(DatasetConfig(), SimConfig())
    train, val, test = data["train"], data["val"], data["test"]
    healthy_train = [ep for ep in train if ep.is_healthy]
    injected = [ep for ep in test if not ep.is_healthy]

    std = Standardizer().fit(healthy_train)
    monitor = ChannelMaxESNMonitor(std, K=8, cusum=True, seed=12,
                                   name="esn_cusum_max")
    monitor.fit(healthy_train)
    theta = float(pick_threshold(
        [monitor.score_episode(ep) for ep in val if ep.is_healthy],
        fa_budget=FA_BUDGET))

    # Healthy grand-mean in RAW telemetry space (the adversary's target signal).
    healthy_mean = np.concatenate([ep.X for ep in healthy_train]).mean(axis=0)

    # Realized healthy FA at theta (the floor evasion drives toward).
    fa = _detection(monitor, [ep for ep in test if ep.is_healthy], theta)

    rows = []
    base = None
    for label, channels in EVASIONS:
        evaded = [_healthy_ise(ep, channels, healthy_mean) for ep in injected] \
            if channels else injected
        det = _detection(monitor, evaded, theta)
        if base is None:
            base = det
        rows.append({"evasion": label, "channels": "+".join(channels) or "-",
                     "n_injected": len(injected), "detection": round(det, 3),
                     "detection_drop_vs_none": round(base - det, 3),
                     "healthy_fa": round(fa, 3), "theta": round(theta, 3)})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES / "adversarial_evasion.csv", index=False)
    print("[adversarial] primary=esn_cusum_max, synthetic test, "
          f"n_injected={len(injected)}, healthy FA={fa:.3f} at theta={theta:.2f}")
    print(table.to_string(index=False))
    full = table[table.evasion == "evade_e+u+m"]["detection"].iloc[0]
    print(f"\n[adversarial] ROBUSTNESS LIMIT: a full white-box adversary shaping "
          f"all three channels drops detection {base:.2f} -> {full:.2f} "
          f"(toward the {fa:.2f} FA floor); single-channel evasion is blunted by "
          "max-fusion. Reported as a limitation, not a defended capability.")
    print(f"[adversarial] wrote {TABLES / 'adversarial_evasion.csv'}")


if __name__ == "__main__":
    main()
