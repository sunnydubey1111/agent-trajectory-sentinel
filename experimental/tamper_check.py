"""A cross-channel tamper check, and its honest limit.

`adversarial_evasion.py` measured the robustness limit: an adversary who
replaces a telemetry channel with the healthy grand-mean blunts single-channel
detection, and one who shapes EVERY behavioural channel drives detection to the
false-alarm floor. That is a real limitation and it stays disclosed.

This adds the cheap defence that limitation invites. The evasion has a
signature the failure itself does not: a channel pinned to a constant has **no
temporal variability**, while every genuine episode - healthy or failing -
varies from step to step. So instead of asking "is this episode anomalous?",
the tamper check asks "is this telemetry physically plausible?".

    statistic   for channel c, the mean over c's dims of that dim's temporal
                standard deviation within the episode (on the healthy-fitted
                standardizer's scale);
    rule        flag the episode if ANY channel's statistic falls below the
                1st percentile of the healthy training distribution for that
                channel - a one-sided test, since only implausibly FLAT is
                suspicious;
    fusion      the deployed decision becomes "monitor alarms OR telemetry is
                implausible", so evasion has to beat both.

WHAT THIS IS NOT. It detects the *modelled* attack - constant-pinning - not
evasion in general. An adversary who replays realistic healthy variation
(sampling a real healthy episode's channel trace rather than its mean) keeps
the variability and defeats this check entirely; that variant is measured here
too, and it is why adversarial robustness stays FUTURE WORK rather than a
claimed capability.

Run:  py -m experimental.tamper_check
Writes results/tables/tamper_check.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from derail.common import (CHANNEL_SLICES, DatasetConfig, Episode, SimConfig,
                           Standardizer, rng_for)
from derail.evaluation.metrics import pick_threshold
from derail.monitor.esn import ChannelMaxESNMonitor
from derail.telemetry.generator import make_dataset

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"
FA_BUDGET = 0.05
CHANNELS = ("e", "u", "m")
#: One-sided: only implausibly LOW variability is evidence of pinning.
FLATNESS_PCT = 1.0


def flatness(ep: Episode, std: Standardizer) -> dict[str, float]:
    """Per-channel temporal variability of an episode, on the healthy scale."""
    Z = std.transform(ep.X)
    out = {}
    for ch in CHANNELS:
        block = Z[:, CHANNEL_SLICES[ch]]
        # std over steps for each dim, then averaged over the channel's dims.
        out[ch] = float(np.mean(np.std(block, axis=0)))
    return out


def fit_thresholds(healthy: list[Episode], std: Standardizer) -> dict[str, float]:
    """Lower bound per channel from HEALTHY episodes only (one-class)."""
    stats = {ch: [] for ch in CHANNELS}
    for ep in healthy:
        for ch, v in flatness(ep, std).items():
            stats[ch].append(v)
    return {ch: float(np.percentile(vals, FLATNESS_PCT))
            for ch, vals in stats.items()}


def is_tampered(ep: Episode, std: Standardizer,
                bounds: dict[str, float]) -> bool:
    return any(v < bounds[ch] for ch, v in flatness(ep, std).items())


def pin_channels(ep: Episode, channels: tuple[str, ...],
                 healthy_mean: np.ndarray) -> Episode:
    """The attack: replace channels with the healthy grand-mean (constant)."""
    X = ep.X.copy()
    for ch in channels:
        X[:, CHANNEL_SLICES[ch]] = healthy_mean[CHANNEL_SLICES[ch]]
    return Episode(X=X, episode_id=ep.episode_id, is_healthy=ep.is_healthy,
                   failure_class=ep.failure_class, tau=ep.tau,
                   t_fail=ep.t_fail, severity=ep.severity)


def replay_channels(ep: Episode, channels: tuple[str, ...],
                    donors: list[Episode], seed: int) -> Episode:
    """The ADAPTIVE attack: replay a real healthy episode's channel trace.

    Keeps genuine step-to-step variability, so the flatness test cannot see it.
    Donor steps are tiled/truncated to this episode's length.
    """
    rng = rng_for(seed, "replay", ep.episode_id)
    X = ep.X.copy()
    for ch in channels:
        donor = donors[int(rng.integers(len(donors)))]
        sl = CHANNEL_SLICES[ch]
        block = donor.X[:, sl]
        reps = int(np.ceil(len(X) / len(block)))
        X[:, sl] = np.tile(block, (reps, 1))[:len(X)]
    return Episode(X=X, episode_id=ep.episode_id, is_healthy=ep.is_healthy,
                   failure_class=ep.failure_class, tau=ep.tau,
                   t_fail=ep.t_fail, severity=ep.severity)


def main() -> int:
    data = make_dataset(DatasetConfig(), SimConfig())
    train, val, test = data["train"], data["val"], data["test"]
    std = Standardizer().fit(train)
    healthy_mean = np.mean(np.concatenate([ep.X for ep in train]), axis=0)

    monitor = ChannelMaxESNMonitor(std, seed=1300)
    monitor.fit(train)
    theta = float(pick_threshold([monitor.score_episode(ep) for ep in val],
                                 fa_budget=FA_BUDGET))
    bounds = fit_thresholds(train, std)
    healthy_test = [ep for ep in test if ep.is_healthy]
    injected = [ep for ep in test if not ep.is_healthy]

    def alarmed(ep):
        return bool(np.max(monitor.score_episode(ep)) >= theta)

    rows = []

    def record(arm: str, eps: list[Episode], note: str) -> None:
        mon = np.mean([alarmed(ep) for ep in eps])
        tam = np.mean([is_tampered(ep, std, bounds) for ep in eps])
        both = np.mean([alarmed(ep) or is_tampered(ep, std, bounds)
                        for ep in eps])
        rows.append({"arm": arm, "n": len(eps),
                     "monitor_alarm_rate": round(float(mon), 4),
                     "tamper_flag_rate": round(float(tam), 4),
                     "combined_rate": round(float(both), 4), "note": note})
        print(f"  {arm:28s} n={len(eps):4d}  monitor={mon:.2f}  "
              f"tamper={tam:.2f}  combined={both:.2f}   {note}")

    print("[tamper] healthy episodes (these rates are FALSE alarms)")
    record("healthy", healthy_test, "false-alarm budget")

    print("\n[tamper] injected episodes (these rates are DETECTION)")
    record("injected_no_evasion", injected, "no-evasion baseline")
    for chans in (("e",), ("u",), ("m",), ("e", "u", "m")):
        name = "+".join(chans)
        record(f"pinned_{name}",
               [pin_channels(ep, chans, healthy_mean) for ep in injected],
               "constant-pinning attack")
    record("replayed_e+u+m",
           [replay_channels(ep, CHANNELS, train, seed=17) for ep in injected],
           "ADAPTIVE attack: real healthy variation")

    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / "tamper_check.csv"
    df.to_csv(path, index=False)
    print(f"\n[tamper] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
