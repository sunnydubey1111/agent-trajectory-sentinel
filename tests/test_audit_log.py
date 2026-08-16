"""The audit trail: its contract, its privacy rules, and its irrelevance.

"Irrelevance" is the load-bearing property. An audit log that can change a
verdict, slow a decision into a different one, or take the process down is
worse than no audit log, because it turns an observability feature into a
failure mode of the thing being observed.
"""
from __future__ import annotations

import json
import sys

import pytest

from derail.audit import (AUDIT_SCHEMA_VERSION, DECISIONS, EVENTS, AuditLog,
                          NullAuditLog, digest)


@pytest.fixture()
def log(tmp_path):
    return AuditLog("t", root=tmp_path)


# ------------------------------------------------------------- the contract
def test_a_run_records_every_stage_the_architecture_claims(tmp_path):
    """DESIGN.md promises an audit trail "every stage writes to"; these are
    the stages."""
    log = AuditLog("run", root=tmp_path)
    log.run_start(episode_id="e", model="m", task="t")
    log.step(0, episode_id="e", action="tool_call", latency_s=0.4,
             features={"fused": 0.2})
    log.tool(0, episode_id="e", name="lookup_flight", result="$100",
             is_error=False)
    log.score(0, episode_id="e", monitor="esn", score=0.2, threshold=1.0)
    log.check(0, episode_id="e", check="tool_contract", passed=True)
    log.decision(0, episode_id="e", kind="rollback", reason="alarm")
    log.outcome(status="repaired", episode_id="e", steps=3)
    log.run_end()

    got = [r["event"] for r in log.records()]
    assert got == ["run_start", "step", "tool", "score", "check", "decision",
                   "outcome", "run_end"]
    assert set(got) <= set(EVENTS)


def test_every_record_is_self_locating(log):
    """A line has to say which run, which step and when, or it cannot be
    joined to anything during an incident review."""
    log.run_start(episode_id="e")
    log.step(2, episode_id="e", action="plan")
    log.score(2, episode_id="e", monitor="esn", score=1.0)
    for r in log.records():
        assert r["schema"] == AUDIT_SCHEMA_VERSION
        assert r["run_id"] == "t"
        assert r["ts"].endswith("+00:00"), "timestamps must be UTC-explicit"
    stepped = [r for r in log.records() if r["event"] in ("step", "score")]
    assert all(r["t"] == 2 and r["episode_id"] == "e" for r in stepped)


def test_the_detector_record_carries_score_threshold_and_verdict(log):
    log.score(1, monitor="fused", score=12.5, threshold=9.0, alarmed=True)
    r = log.records()[0]
    assert (r["score"], r["threshold"], r["alarmed"]) == (12.5, 9.0, True)


def test_a_tool_record_carries_its_error_status(log):
    log.tool(0, name="calculator", result="Error: boom", is_error=True)
    r = log.records()[0]
    assert r["name"] == "calculator" and r["is_error"] is True


def test_decision_kinds_are_a_closed_vocabulary(log):
    """An ad-hoc decision name creates a category nobody knows to look in."""
    log.decision(0, kind="rollback")
    log.decision(1, kind="teleport")
    kinds = [r["kind"] for r in log.records()]
    assert kinds[0] in DECISIONS
    assert kinds[1] == "unknown:teleport"


# ------------------------------------------------------------------ privacy
def test_raw_content_is_not_persisted_by_default(tmp_path):
    secret = "the customer's card is 4111111111111111"
    log = AuditLog("p", root=tmp_path)
    log.run_start(task=secret)
    log.step(0, text=secret)
    log.tool(0, name="t", result=secret)
    body = log.path.read_text("utf-8")
    assert "4111111111111111" not in body
    assert "customer" not in body


def test_a_digest_still_identifies_the_same_bytes():
    """Metadata has to be enough to prove two runs saw the same payload."""
    a, b = digest("same"), digest("same")
    assert a == b and a["chars"] == 4
    assert digest("other") != a
    assert digest(None)["sha256"] is None


def test_opting_in_persists_content_but_still_redacts_credentials(tmp_path):
    log = AuditLog("c", root=tmp_path, include_content=True)
    log.tool(0, name="t",
             result="key ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 and text")
    body = log.path.read_text("utf-8")
    assert "ghp_ABCDEF" not in body, "a credential reached the audit log"
    assert "and text" in body, "opt-in content was not recorded at all"


def test_the_run_records_whether_content_logging_was_on(tmp_path):
    """A reader must be able to tell a quiet run from a redacted one."""
    off = AuditLog("off", root=tmp_path)
    off.run_start()
    assert off.records()[0]["content_logged"] is False


# -------------------------------------------------------- failure isolation
def test_a_broken_sink_drops_records_and_never_raises(tmp_path):
    log = AuditLog("b", root=tmp_path)
    log._path = tmp_path / "missing" / "deep" / "x.jsonl"
    log.run_start()
    log.step(0, action="plan")
    log.score(0, monitor="esn", score=1.0)
    assert log.dropped == 3
    assert log.records() == []


def test_an_unwritable_root_disables_the_sink_without_raising(tmp_path):
    target = tmp_path / "file-not-a-dir"
    target.write_text("x", "utf-8")
    log = AuditLog("u", root=target / "audit")
    log.step(0, action="plan")          # must not raise
    assert log.path is None or log.dropped >= 0


def test_a_disabled_sink_writes_nothing_and_still_answers(tmp_path):
    log = NullAuditLog()
    log.run_start()
    log.step(0, action="plan")
    log.outcome(status="done")
    assert log.records() == [] and log.path is None


def test_values_that_cannot_serialize_degrade_instead_of_raising(tmp_path):
    log = AuditLog("s", root=tmp_path)
    log.step(0, action=object(), features={"nan": float("nan"),
                                           "inf": float("inf"),
                                           "ok": 1.5})
    assert log.dropped == 0
    f = log.records()[0]["features"]
    assert f["ok"] == 1.5
    # NaN/Infinity are not JSON; recording them as null keeps a reader from
    # mistaking a parser's `NaN` for a real measurement.
    assert f["nan"] is None and f["inf"] is None


# ------------------------------------------------------------ serialization
def test_the_file_is_append_only_json_lines(tmp_path):
    log = AuditLog("j", root=tmp_path)
    for i in range(5):
        log.step(i, action="plan")
    lines = log.path.read_text("utf-8").splitlines()
    assert len(lines) == 5
    for line in lines:
        parsed = json.loads(line)          # each line stands alone
        assert parsed["event"] == "step"
    assert "\n" not in json.dumps(json.loads(lines[0]))


def test_records_survive_a_truncated_final_line(tmp_path):
    """A run killed mid-write must not make the whole trail unreadable."""
    log = AuditLog("k", root=tmp_path)
    log.step(0, action="plan")
    log.step(1, action="plan")
    with log.path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "step", "t":')      # torn write
    assert len(log.records()) == 2


def test_each_run_gets_its_own_file(tmp_path):
    a = AuditLog("run-a", root=tmp_path)
    b = AuditLog("run-b", root=tmp_path)
    a.step(0, action="plan")
    b.step(0, action="plan")
    assert a.path != b.path
    assert len(a.records()) == 1 and len(b.records()) == 1


# ------------------------------------- independence of the monitored path
def test_the_monitor_does_not_import_the_audit_log():
    """The dependency has to point one way. If a monitor module imported this,
    a logging change could alter a detection result."""
    import derail.audit as audit_mod

    for name in ("derail.monitor.esn", "derail.monitor.baseline",
                 "derail.monitor.hybrid", "derail.monitor.grounding",
                 "derail.evaluation.metrics", "derail.verify.checks"):
        __import__(name)
        src = sys.modules[name].__file__
        assert "audit" not in open(src, encoding="utf-8").read(), (
            f"{name} references the audit log; the monitored path must not "
            f"depend on the thing observing it")
    assert audit_mod is not None


def test_the_serving_loop_treats_the_sink_as_optional(tmp_path):
    """`run_real_episode` must produce identical telemetry with and without a
    log, or the audit trail is changing the thing it audits."""
    from derail.harness.agent_loop import run_real_episode
    from derail.harness.tools import SimpleTool, ToolRegistry

    def _backend():
        class B:
            model = "scripted"

            def reset(self, task): pass

            def add_tool_results(self, results): pass

            def step(self, t):
                if t < 2:
                    return {"stop_reason": "tool_use", "text": f"s{t}",
                            "tool_uses": [{"id": f"c{t}", "name": "search",
                                           "input": {"q": str(t)}}],
                            "output_tokens": 3, "token_logprobs": [-0.1]}
                return {"stop_reason": "end_turn", "text": "done",
                        "tool_uses": [], "output_tokens": 2,
                        "token_logprobs": [-0.2]}
        return B()

    reg = ToolRegistry([SimpleTool("search", "s", {"q": "q"},
                                   lambda q: f"r{q}")])
    plain = run_real_episode(_backend(), reg, "task", max_steps=5)
    log = AuditLog("wired", root=tmp_path)
    logged = run_real_episode(_backend(), reg, "task", max_steps=5,
                              audit=log, episode_id="e")

    assert plain == logged, "the audit sink changed the telemetry"
    events = [r["event"] for r in log.records()]
    assert events[0] == "run_start" and events[-1] == "run_end"
    assert "tool" in events and "step" in events


def test_a_failing_sink_does_not_change_the_serving_loop(tmp_path):
    from derail.harness.agent_loop import run_real_episode
    from derail.harness.tools import SimpleTool, ToolRegistry

    class B:
        model = "scripted"

        def reset(self, task): pass

        def add_tool_results(self, results): pass

        def step(self, t):
            return {"stop_reason": "end_turn", "text": "done",
                    "tool_uses": [], "output_tokens": 1,
                    "token_logprobs": []}

    reg = ToolRegistry([SimpleTool("search", "s", {"q": "q"},
                                   lambda q: "r")])
    good = run_real_episode(B(), reg, "task", max_steps=3)

    broken = AuditLog("x", root=tmp_path)
    broken._path = tmp_path / "gone" / "x.jsonl"
    same = run_real_episode(B(), reg, "task", max_steps=3,
                            audit=broken, episode_id="e")
    assert good == same
    assert broken.dropped > 0, "the test did not actually exercise a failure"
