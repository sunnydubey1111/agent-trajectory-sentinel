"""Per-deployment healthy-only calibration for `ChannelMaxESNMonitor`.

A monitor frozen once on this project's native harness and scored
zero-shot on a different agent orchestrator (LangGraph, AutoGen, or any
future runtime) does not need to stay zero-shot: `ChannelMaxESNMonitor`
is already a one-class detector -- `fit()` takes only healthy episodes,
no failure labels -- so the SAME class, SAME procedure, SAME
hyperparameters can be refit on a target deployment's own verified
healthy runs instead of reused across deployments unchanged. Nothing
here is specific to any framework name; it operates on whatever
`Episode` list it is given.

Mirrors `framework_monitor_freeze.py`'s own train/val/test split and
threshold procedure exactly, generalized to any deployment's healthy
population rather than one corpus name list.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from derail.common import Standardizer, rng_for
from derail.evaluation.metrics import pick_threshold
from derail.monitor.esn import ChannelMaxESNMonitor

#: Floors read off a native-harness learning curve: at 12 healthy episodes the
#: picked threshold moves 100x across seeds and FA runs at 0.30; 30 reaches
#: 0.06, 80 holds at or below 0.03. Both are arguments, not fixed policy.
_MIN_HEALTHY = 30
_RECOMMENDED_HEALTHY = 80


@dataclass
class DeploymentCalibration:
    monitor: ChannelMaxESNMonitor
    standardizer: Standardizer
    theta: float
    train_ids: list[str] = field(default_factory=list)
    val_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)


def _split_60_20_20(episodes: list, seed: int):
    perm = rng_for(seed, "deployment-calibration", "split").permutation(len(episodes))
    n_train = max(int(round(0.6 * len(episodes))), 1)
    n_val = max(int(round(0.2 * len(episodes))), 1)
    idx_train = perm[:n_train]
    idx_val = perm[n_train:n_train + n_val]
    idx_test = perm[n_train + n_val:]
    pick = lambda idxs: [episodes[i] for i in idxs]
    return pick(idx_train), pick(idx_val), pick(idx_test)


def calibrate(healthy_episodes: list, *, channels: tuple[str, ...] = ("e", "m"),
             fa_budget: float = 0.05, K: int = 8, seed: int = 0,
             min_healthy: int = _MIN_HEALTHY,
             warn_below: int = _RECOMMENDED_HEALTHY) -> DeploymentCalibration:
    """Fit a fresh `ChannelMaxESNMonitor` on ONE deployment's own healthy
    episodes: no failure labels, no other deployment's data. 60/20/20
    train/val/test split of `healthy_episodes` (train fits the
    standardizer+monitor, val picks `theta` at `fa_budget` via
    `pick_threshold`, test is left for the caller to score FA on).

    Raises ValueError below `min_healthy`, and on a non-positive threshold.
    Both mean too few healthy episodes: the monitor's own 85/15 held fold
    sets its normalizers from a median and an IQR, and at 12 episodes that
    fold is one episode of a few post-washout steps.
    """
    if len(healthy_episodes) < min_healthy:
        raise ValueError(
            f"calibrate needs >= {min_healthy} healthy episodes, got "
            f"{len(healthy_episodes)}: the threshold is not identifiable below "
            f"that")
    if len(healthy_episodes) < warn_below:
        warnings.warn(
            f"calibrating on {len(healthy_episodes)} healthy episodes; "
            f"{warn_below}+ is where FA holds at or below 0.03", stacklevel=2)
    train, val, test = _split_60_20_20(healthy_episodes, seed)

    standardizer = Standardizer().fit(train)
    monitor = ChannelMaxESNMonitor(standardizer, K=K, cusum=True, seed=seed,
                                   name="esn_cusum_max_deployment",
                                   channels=channels)
    monitor.fit(train)

    val_streams = []
    for ep in val:
        monitor.start_episode()
        val_streams.append(np.array([monitor.score_step(ep.X[t])
                                     for t in range(ep.X.shape[0])]))
    theta = float(pick_threshold(val_streams, fa_budget=fa_budget,
                                 warn_infeasible=False))
    if not np.isfinite(theta) or theta <= 0.0:
        raise ValueError(
            f"degenerate threshold ({theta!r}) from {len(val)} validation "
            f"episodes: it would alarm on everything")

    return DeploymentCalibration(
        monitor=monitor, standardizer=standardizer, theta=theta,
        train_ids=[ep.episode_id for ep in train],
        val_ids=[ep.episode_id for ep in val],
        test_ids=[ep.episode_id for ep in test])


if __name__ == "__main__":
    from derail.telemetry.adapter import episode_from_trace

    def _steps(seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        return [{"text": f"step {t} {rng.integers(0, 1000)}",
                "token_logprobs": [-0.1] * 10, "action": "tool_call",
                "latency_s": float(rng.uniform(0.3, 0.8)), "output_tokens": 10,
                "error": False} for t in range(10)]

    healthy = [episode_from_trace(_steps(i), f"h{i}",
                                  use_sentence_transformers=False, extended=True)
              for i in range(_MIN_HEALTHY)]
    cal = calibrate(healthy, warn_below=0)
    assert cal.theta > 0
    assert set(cal.train_ids) | set(cal.val_ids) | set(cal.test_ids) == \
          {ep.episode_id for ep in healthy}
    assert not (set(cal.train_ids) & set(cal.val_ids))
    assert not (set(cal.train_ids) & set(cal.test_ids))
    assert not (set(cal.val_ids) & set(cal.test_ids))
    print("PASS deployment_calibration.py smoke test")
