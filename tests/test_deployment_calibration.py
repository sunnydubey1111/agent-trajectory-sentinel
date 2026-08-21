"""Per-deployment healthy-only monitor calibration. Offline, no live calls.
"""
from __future__ import annotations

import numpy as np
import pytest

from derail.monitor.deployment_calibration import (
    _MIN_HEALTHY, _RECOMMENDED_HEALTHY, calibrate)
from derail.telemetry.adapter import episode_from_trace


def _steps(seed: int, n: int = 10) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"text": f"step {t} {rng.integers(0, 1000)}",
             "token_logprobs": [-0.1] * 10, "action": "tool_call",
             "latency_s": float(rng.uniform(0.3, 0.8)), "output_tokens": 10,
             "error": False} for t in range(n)]


def _healthy_episodes(n: int = _MIN_HEALTHY):
    return [episode_from_trace(_steps(i), f"h{i}",
                               use_sentence_transformers=False, extended=True)
           for i in range(n)]


def _calibrate(episodes, **kw):
    kw.setdefault("warn_below", 0)
    return calibrate(episodes, **kw)


def test_calibrate_partitions_episodes_without_overlap():
    healthy = _healthy_episodes()
    cal = _calibrate(healthy, seed=3)
    train, val, test = set(cal.train_ids), set(cal.val_ids), set(cal.test_ids)
    assert train | val | test == {ep.episode_id for ep in healthy}
    assert not (train & val) and not (train & test) and not (val & test)
    assert train and val


def test_calibrate_is_deterministic():
    healthy = _healthy_episodes()
    a = _calibrate(healthy, seed=7)
    b = _calibrate(healthy, seed=7)
    assert a.train_ids == b.train_ids and a.val_ids == b.val_ids
    assert a.theta == b.theta


def test_calibrate_refuses_too_few_healthy_episodes():
    healthy = _healthy_episodes(_MIN_HEALTHY - 1)
    with pytest.raises(ValueError, match="healthy episodes"):
        _calibrate(healthy)


def test_calibrate_warns_below_the_recommended_healthy_count():
    healthy = _healthy_episodes(_MIN_HEALTHY)
    assert _MIN_HEALTHY < _RECOMMENDED_HEALTHY
    with pytest.warns(UserWarning, match="healthy episodes"):
        calibrate(healthy, seed=0)


def test_calibrated_monitor_scores_a_held_out_episode():
    healthy = _healthy_episodes()
    cal = _calibrate(healthy, seed=1)
    ep = healthy[0]
    cal.monitor.start_episode()
    scores = [cal.monitor.score_step(ep.X[t]) for t in range(ep.X.shape[0])]
    assert all(isinstance(s, float) for s in scores)
    assert cal.theta > 0


def test_calibrate_rejects_a_degenerate_threshold():
    healthy = _healthy_episodes()
    import derail.monitor.deployment_calibration as dc

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dc, "pick_threshold", lambda *a, **k: 0.0)
        with pytest.raises(ValueError, match="degenerate threshold"):
            _calibrate(healthy)
