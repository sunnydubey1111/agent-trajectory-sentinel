"""The AFTraj-2K importer's mapping, which no downstream check would catch.

A wrong turn-to-step mapping does not crash and does not look wrong: it shifts
tau by a step or two and every lead-time and detection number moves with it.
These tests pin the mapping on hand-built trajectories whose correct answer is
obvious by inspection, so the corpus never has to be present.
"""
from __future__ import annotations

import json

import pytest

from derail.experiments import import_aftraj
from derail.telemetry.adapter import episode_from_trace


def _turn(role, content="", action="", thought=""):
    return {"role": role, "content": content, "action": action,
            "thought": thought}


def _call(name, args, call_id):
    return json.dumps([{"name": name, "arguments": json.dumps(args),
                        "id": call_id}])


def _result(name, result, call_id):
    return json.dumps([{"name": name, "result": result, "call_id": call_id}])


def _row(turns, mistake_step=None, source="injected"):
    row = {"conv_id": "probe", "domain": "math", "turns": turns,
           "num_turns": len(turns)}
    if mistake_step is not None:
        row["mistake_step"] = mistake_step
        row["unsafe_source"] = source
    return row


def test_environment_and_user_turns_are_not_steps() -> None:
    turns = [_turn("user", "the task"),
             _turn("Solver", action=_call("compute", {"x": 1}, "c0")),
             _turn("environment", _result("compute", "3", "c0")),
             _turn("Solver", "3")]
    steps, tau = import_aftraj._convert(_row(turns))
    assert len(steps) == 2, "only the two agent turns are steps"
    assert tau is None


def test_tau_is_the_agent_step_position_not_the_turn_index() -> None:
    """mistake_step indexes turns; tau indexes steps. They differ."""
    turns = [_turn("user", "task"),
             _turn("Solver", action=_call("compute", {}, "c0")),   # step 0
             _turn("environment", _result("compute", "ok", "c0")),
             _turn("Solver", "interim"),                            # step 1
             _turn("Verifier", "wrong answer")]                     # step 2
    steps, tau = import_aftraj._convert(_row(turns, mistake_step=4))
    assert len(steps) == 3
    assert tau == 2, "turn 4 is the third agent step"


def test_mistake_on_an_environment_turn_lands_on_the_calling_step() -> None:
    """A bad tool result belongs to the step whose record carries it."""
    turns = [_turn("user", "task"),
             _turn("Solver", action=_call("search", {}, "c0")),     # step 0
             _turn("environment", _result("search", "poisoned", "c0")),
             _turn("Solver", "used the bad result")]                # step 1
    steps, tau = import_aftraj._convert(_row(turns, mistake_step=2))
    assert tau == 0, "the environment turn belongs to the step that called"
    assert len(steps) == 2


def test_tool_results_are_attached_and_errors_flagged() -> None:
    turns = [_turn("Solver", action=_call("sql", {"q": "SELECT 1"}, "c0")),
             _turn("environment", _result("sql", "Error: timeout", "c0"))]
    steps, _ = import_aftraj._convert(_row(turns))
    events = steps[0]["tool_events"]
    assert len(events) == 1
    assert events[0]["name"] == "sql"
    assert events[0]["args"] == {"q": "SELECT 1"}
    assert events[0]["result"] == "Error: timeout"
    assert events[0]["is_error"] is True
    assert steps[0]["error"] is True


def test_every_step_declares_logprobs_unavailable() -> None:
    """AFTraj has no logprobs; the u channel must read as missing, not zero."""
    turns = [_turn("Solver", "a"), _turn("Solver", "b")]
    steps, _ = import_aftraj._convert(_row(turns))
    assert all(s["logprobs_available"] is False for s in steps)


def test_converted_steps_build_a_valid_episode() -> None:
    turns = [_turn("user", "task"),
             _turn("Solver", action=_call("compute", {}, "c0")),
             _turn("environment", _result("compute", "3", "c0")),
             _turn("Solver", "3"),
             _turn("Solver", action=_call("verify", {}, "c1")),
             _turn("environment", _result("verify", "no", "c1")),
             _turn("Verifier", "final")]
    steps, tau = import_aftraj._convert(_row(turns, mistake_step=6))
    episode = episode_from_trace(steps, "probe", tau=tau,
                                 failure_class="external", extended=True)
    assert episode.T == len(steps)
    assert episode.tau == tau
    assert not episode.is_healthy


def test_a_mistake_before_any_agent_turn_is_refused() -> None:
    """Silently defaulting tau to 0 would fabricate a detectable onset."""
    turns = [_turn("user", "task"), _turn("Solver", "answer")]
    with pytest.raises(ValueError, match="precedes every agent step"):
        import_aftraj._convert(_row(turns, mistake_step=0))


def test_external_is_an_accepted_failure_class() -> None:
    from derail.common import REAL_FAILURE_CLASSES
    assert "external" in REAL_FAILURE_CLASSES
