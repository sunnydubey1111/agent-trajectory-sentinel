"""The ATBench conversion, and the one thing it must never do.

ATBench has no onset label. If an episode were ever built with a tau, every
detection number would silently become a claim about earliness that the data
cannot support, so these tests pin the absence as hard as the presence.
"""
from __future__ import annotations

import json

from derail.experiments import run_atbench_study as atb
from derail.telemetry.adapter import episode_from_trace


def _turn(role, content="", action="", thought=""):
    return {"role": role, "content": content, "action": action,
            "thought": thought}


def _action(name, args):
    return json.dumps({"name": name, "arguments": args})


def test_user_and_environment_turns_are_not_steps() -> None:
    turns = [_turn("user", "do the thing"),
             _turn("agent", action=_action("search", {"q": "x"})),
             _turn("environment", '{"status": "success"}'),
             _turn("agent", thought="done")]
    assert len(atb.to_steps(turns)) == 2


def test_tool_result_is_folded_into_the_calling_step() -> None:
    turns = [_turn("agent", action=_action("fetch", {"url": "u"})),
             _turn("environment", '{"status": "success", "result": 1}')]
    steps = atb.to_steps(turns)
    assert len(steps) == 1
    event = steps[0]["tool_events"][0]
    assert event["name"] == "fetch"
    assert event["args"] == {"url": "u"}
    assert "success" in event["result"]


def test_error_status_in_a_tool_result_is_flagged() -> None:
    turns = [_turn("agent", action=_action("sql", {})),
             _turn("environment", '{"status": "error", "msg": "denied"}')]
    steps = atb.to_steps(turns)
    assert steps[0]["tool_events"][0]["is_error"] is True
    assert steps[0]["error"] is True


def test_the_terminal_complete_turn_is_an_answer_not_a_tool_call() -> None:
    """`Complete{...}` is the agent answering; counting it as a call would
    inflate the tool-call metadata on the last step of every trajectory."""
    turns = [_turn("agent", action='Complete{"response": "here it is"}')]
    steps = atb.to_steps(turns)
    assert steps[0]["tool_events"] == []
    assert steps[0]["action"] == "synthesis"


def test_every_step_declares_logprobs_unavailable() -> None:
    turns = [_turn("agent", thought="a"), _turn("agent", thought="b")]
    assert all(s["logprobs_available"] is False for s in atb.to_steps(turns))


def test_episodes_carry_no_tau_because_atbench_has_no_onset() -> None:
    """The label lives outside the Episode; inside it there is no onset."""
    turns = [_turn("user", "task")] + [_turn("agent", thought=f"s{i}")
                                       for i in range(4)]
    steps = atb.to_steps(turns)
    episode = episode_from_trace(steps, "atb-1", extended=True)
    assert episode.tau is None
    assert episode.is_healthy is True, (
        "is_healthy records the absence of an onset label, not a safety claim")


def test_short_trajectories_are_dropped(tmp_path) -> None:
    rows = [{"id": 1, "label": 0,
             "contents": [[_turn("agent", thought="only one step")]],
             "failure_mode": "benign"},
            {"id": 2, "label": 1, "failure_mode": "flawed_planning",
             "contents": [[_turn("agent", thought=f"s{i}") for i in range(5)]]}]
    path = tmp_path / "test.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    episodes, labels, kept = atb.load(path)
    assert [e.episode_id for e in episodes] == ["atb-2"]
    assert labels == [1]
    assert kept[0]["id"] == 2


def test_the_study_is_not_registered_as_a_hybrid_study_dataset() -> None:
    """It cannot be: run_hybrid_study needs a tau ATBench does not have."""
    from derail.experiments.run_hybrid_study import REAL_DATASETS
    assert "atbench" not in REAL_DATASETS
