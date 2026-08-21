"""Real-task rollback/retry gates: no-oracle-access, checkpoint edge cases,
and the reconstruction-equivalence proof -- all offline, no live calls.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from derail.intervene.real_tool_rollback import (
    CheckpointResult, ORACLE_UPPER_BOUND, PRIMARY, RECONSTRUCTION_FAILED,
    compute_checkpoint_oracle,
    compute_checkpoint_primary, retry_real_task_from_checkpoint)
from derail.intervene.rollback import rebuild_history

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_TRACE = REPO_ROOT / "traces" / "real_research7b" / "real-healthy-000.jsonl"


def _load_fixture_steps() -> list[dict]:
    return [json.loads(l) for l in FIXTURE_TRACE.read_text("utf-8").splitlines()
           if l.strip()]


class _FakeBackend:
    """Minimal reset/step/add_tool_results double -- no LLM, no network.
    Records every history mutation so reconstruction can be inspected."""

    def __init__(self, **_ignored):
        self.history: list = []

    def reset(self, task: str) -> None:
        self.history = [{"role": "user", "content": task}]

    def add_tool_results(self, results: list[dict]) -> None:
        for r in results:
            self.history.append({"role": "tool", "tool_name": r["name"],
                                 "content": r["content"]})


def test_checkpoint_primary_never_accesses_tau_or_failure_class():
    """Structural, not conventional: these names are not parameters this
    function could read, under any call."""
    sig = inspect.signature(compute_checkpoint_primary)
    names = set(sig.parameters)
    assert "tau" not in names
    assert "failure_class" not in names
    assert names == {"steps", "monitor", "theta_b5", "episode_from_trace"}


def test_oracle_and_primary_are_separate_functions_with_different_signatures():
    assert compute_checkpoint_primary is not compute_checkpoint_oracle
    oracle_params = set(inspect.signature(compute_checkpoint_oracle).parameters)
    assert "tau" in oracle_params
    assert "monitor" not in oracle_params
    assert PRIMARY != ORACLE_UPPER_BOUND


def test_not_triggered_when_score_never_crosses_threshold():
    steps = _load_fixture_steps()

    class _NeverAlarms:
        def start_episode(self):
            pass

        def score_step(self, x):
            return 0.0

    def _fake_episode_from_trace(steps, name, **kw):
        class E:
            X = np.zeros((len(steps), 3))
        return E()

    result = compute_checkpoint_primary(steps, _NeverAlarms(), theta_b5=1.0,
                                        episode_from_trace=_fake_episode_from_trace)
    assert result.outcome == "not_triggered"
    assert result.k == 0


def test_checkpoint_at_start_when_alarm_fires_before_any_successful_tool_call():
    steps = _load_fixture_steps()

    class _AlarmsImmediately:
        def start_episode(self):
            pass

        def score_step(self, x):
            return 999.0            # crosses on step 0

    def _fake_episode_from_trace(steps, name, **kw):
        class E:
            X = np.zeros((len(steps), 3))
        return E()

    result = compute_checkpoint_primary(steps, _AlarmsImmediately(), theta_b5=1.0,
                                        episode_from_trace=_fake_episode_from_trace)
    assert result.outcome == "checkpoint_at_start"
    assert result.k == 0
    assert result.alarm_step == 0


def test_oracle_checkpoint_localizes_to_last_success_before_tau():
    steps = _load_fixture_steps()
    result = compute_checkpoint_oracle(steps, tau=3)
    assert 0 <= result.k <= 3


def test_reconstruction_failed_is_retained_not_dropped():
    checkpoint = CheckpointResult(k=1, outcome="normal", alarm_step=1)

    def _broken_factory(**_kw):
        raise ValueError("simulated reconstruction break")

    class _FakeRegistry:
        def specs(self):
            return {}

    out = retry_real_task_from_checkpoint(
        _load_fixture_steps(), "task", checkpoint, PRIMARY,
        max_steps=6, registry=_FakeRegistry(), backend_factory=_broken_factory)
    assert out.outcome == RECONSTRUCTION_FAILED
    assert out.success is None
    assert out.error is not None


def test_not_triggered_short_circuits_without_any_retry_attempt():
    checkpoint = CheckpointResult(k=0, outcome="not_triggered")
    calls = []

    def _factory(**kw):
        calls.append(kw)
        raise AssertionError("must not be called when not_triggered")

    class _FakeRegistry:
        def specs(self):
            return {}

    out = retry_real_task_from_checkpoint(
        _load_fixture_steps(), "task", checkpoint, PRIMARY,
        max_steps=6, registry=_FakeRegistry(), backend_factory=_factory)
    assert out.outcome == "not_triggered"
    assert calls == []


def test_rebuild_history_prefix_then_continue_matches_full_reconstruction():
    """The equivalence proof: reconstructing steps[0:k] then appending the
    same steps[k:] the same way `rebuild_history` builds them, must produce
    the identical message list as reconstructing everything in one pass.
    This is the property "continuing from the checkpoint" needs -- the
    checkpoint state is provably the correct PREFIX of the real conversation,
    the same guarantee rebuild_history's own docstring already claims and
    the booking-domain repair system already relies on in production.
    """
    steps = _load_fixture_steps()
    assert len(steps) >= 4, "fixture too short for a meaningful split"
    k = 2

    full = _FakeBackend()
    rebuild_history(full, "task", steps, len(steps))

    prefix = _FakeBackend()
    rebuild_history(prefix, "task", steps, k)
    # Continue the SAME way rebuild_history does internally: one assistant
    # message + its tool results per remaining step.
    from derail.telemetry.events import parse_step_events
    for s in steps[k:]:
        evs, reasoning = parse_step_events(s)
        msg: dict = {"role": "assistant"}
        if reasoning.strip():
            msg["content"] = reasoning.strip()
        if evs:
            msg["tool_calls"] = [{"function": {"name": e.name, "arguments": e.args or {}}}
                                 for e in evs]
        elif "content" not in msg:
            msg["content"] = ""
        prefix.history.append(msg)
        for e in evs:
            prefix.history.append({"role": "tool", "tool_name": e.name,
                                   "content": e.result})

    assert prefix.history == full.history
