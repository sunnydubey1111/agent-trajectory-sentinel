"""Collector safeguard: never overwrite an accepted recording; retries get a
distinct staging location. Offline -- no live calls.
"""

from derail.experiments.collect_framework_real_traces import (
    _fresh_staging_dir, _seed_for, _tau_for, check_admission,
    collect_one_episode)


def test_fresh_staging_dir_never_reuses_a_previous_attempt(tmp_path):
    first = _fresh_staging_dir(tmp_path, "ep-000")
    second = _fresh_staging_dir(tmp_path, "ep-000")
    assert first != second
    assert first.exists() and second.exists()
    assert first.name.endswith("__attempt0")
    assert second.name.endswith("__attempt1")


def test_collect_one_episode_refuses_when_final_dir_already_exists(tmp_path):
    final_dir = tmp_path / "_cassettes" / "langgraph-real-healthy-000"
    final_dir.mkdir(parents=True)
    (final_dir / "sentinel.json").write_text("do-not-touch", "utf-8")

    try:
        collect_one_episode("langgraph", "healthy", None, 0, out_dir=tmp_path)
        assert False, "must have raised FileExistsError before any live call"
    except FileExistsError:
        pass

    assert (final_dir / "sentinel.json").read_text("utf-8") == "do-not-touch"


def _step(tool_events):
    return {"text": "", "action": "tool_call", "tool_events": tool_events}


def test_seed_and_tau_are_deterministic_functions_of_the_attempt():
    assert _seed_for(0, 0) != _seed_for(0, 1) != _seed_for(1, 0)
    assert _seed_for(0, 0) == _seed_for(0, 0)
    t0 = _tau_for("autogen", "autogen-real-goal_drift-000", 0)
    t1 = _tau_for("autogen", "autogen-real-goal_drift-000", 1)
    assert _tau_for("autogen", "autogen-real-goal_drift-000", 0) == t0
    assert (t0, t1) != (t1, t0) or t0 != t1  # attempts are independently drawn


def test_check_admission_rejects_a_non_landed_injection():
    steps = [_step([{"name": "web_search", "result": "clean", "is_error": False}]),
            _step([])]
    ok, reason, facts = check_admission(steps, "goal_drift", tau=2, seed=1)
    assert not ok and "no-op" in reason
    assert facts["applied_count"] == 0


def test_check_admission_accepts_a_landed_injection_with_room():
    steps = [_step([{"name": "web_search", "result": "clean", "is_error": False}]),
            _step([{"name": "web_search", "result": "clean", "is_error": False}]),
            _step([])]
    ok, reason, facts = check_admission(steps, "goal_drift", tau=1, seed=1)
    assert ok and facts["applied_count"] > 0


def test_check_admission_never_filters_healthy_by_length():
    ok, reason, facts = check_admission([_step([])], None, tau=None, seed=1)
    assert ok and "outcome-independent" in reason
