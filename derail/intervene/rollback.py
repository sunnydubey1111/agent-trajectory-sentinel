"""Rollback to a checkpoint and retry, driven by what the checks localized.

The rollback is REAL, not simulated: a committed trace plus its task seed is
enough to rebuild the agent's conversation exactly as it stood at step k
(system + task, then each assistant turn and the tool result it received), and
the agent is then re-run live from there. Nothing about the outcome is replayed
— every step after the checkpoint is a fresh model call.

Repair ladder (each rung adds exactly one mechanism, so the study can say which
one does the work):

  resample  rollback, identical context, new sample. The control: a stochastic
            agent improves on retry alone, so nothing above this rung may be
            credited unless it beats this.
  generic   rollback + a task-independent reminder to re-check the work.
  located   rollback + WHICH check failed, with no computed values. Isolates
            localization from handing the agent a recomputed answer.
  specific  rollback + the check's own finding, in words.
  adaptive  specific when the ANSWER is wrong, generic when only completeness
            is. Added after `specific` was measured to damage correct work.

No rung can reach the task's ground truth: `repair_message` takes the findings
and nothing else, and a Finding is computed only from tool results the run
itself received. That is a structural guarantee, not a string test.

It is not, however, the whole story, and `located` exists because of it. The
check recomputes the total from the agent's own figures, so for a run that
looked everything up correctly and merely mis-added, that recomputed value IS
the correct answer — measured, 26 of 55 `specific` hints contain it. Such a
hint is legitimate in production, since nothing external is used, but it does
more than locate the fault: it supplies the answer. `located` states which
check failed and no computed value at all (0 of 55 contain the total), so the
study can report what localization alone is worth.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from derail.experiments.collect_traces import OllamaBackend, _make_world, _run_tool
from derail.experiments.demo import (DEMO_MAX_STEPS, DEMO_TOOL_SPECS,
                                     _make_demo_task, _step_record, _tool_bit)
from derail.preconditions import error_shaped
from derail.telemetry.events import make_tool_event, parse_step_events
from derail.verify.checks import BOOKING_SPEC, Finding, TaskSpec, verify

#: Rungs the offline repair comparison scores. `unstick` is deliberately
#: absent -- iterating it here would produce empty groups that read as a
#: measured zero rather than "not applicable" (see `repair_message` for why
#: it doesn't apply to this corpus).
SCORED_RUNGS = ("none", "resample", "generic", "located", "specific",
                "adaptive", "recompute")
#: Every rung `repair_message` can build, including the live-only one.
RUNGS = SCORED_RUNGS + ("unstick",)

_GENERIC_HINT = (
    "Before you give the final answer, re-check your work: make sure you have "
    "looked up every figure the task asks for, and that your total is the sum "
    "of exactly those figures."
)

_RECOMPUTE_HINT = (
    "Use the calculator tool to add up the figures you looked up, one "
    "expression containing every one of them, and state the total the "
    "calculator returns."
)

_UNSTICK_HINT = (
    "A tool is returning errors or asking you to repeat the same call. Do not "
    "call it again with the same arguments. Work with the figures you have "
    "already obtained and give the final answer, noting anything you could "
    "not look up."
)


@dataclass
class Attempt:
    """One post-rollback retry."""

    rung: str
    steps: list[dict]
    resumed_from: int
    model_calls: int
    findings_before: list[Finding]
    findings_after: list[Finding]


def rollback_step(steps: list[dict], spec: TaskSpec) -> int:
    """Index to resume from: just after the last priced lookup.

    That is the last point at which the run was still gathering facts, before
    it combined them. Both failure families localize there — a missing lookup
    can still be made, and a bad combination is redone — so the study does not
    need a different checkpoint rule per family.
    """
    priced = {li.tool for li in spec.line_items} | set(spec.required_tools)
    last = -1
    for i, s in enumerate(steps):
        # `parse_step_events`, not the text parser: where the collector
        # recorded what it executed, the checkpoint must be placed from that
        # record rather than from the rendered display form of it.
        evs, _ = parse_step_events(s)
        if any(e.name in priced and not e.is_error for e in evs):
            last = i
    return last + 1


def repair_message(rung: str, findings: list[Finding]) -> str | None:
    """The user-turn nudge appended at the checkpoint, or None.

    The leak guarantee is structural (see module docstring): there is no
    parameter here through which the task's ground truth could reach the
    prompt. A string test is not usable instead, because the check's
    recomputed total legitimately equals the true total whenever the agent
    looked everything up correctly and merely mis-added it, the dominant
    failure.
    """
    if rung in ("none", "resample"):
        return None
    if rung == "generic":
        return _GENERIC_HINT
    if rung == "located":
        # Names the fault and nothing else. Measured: 26 of 55 `specific`
        # hints contain the run's true total, because the check recomputes it
        # from the agent's own figures; this rung removes that so the two
        # effects can be told apart.
        if not findings:
            return _GENERIC_HINT
        detail = "; ".join(f.terse for f in findings)
        return (f"A consistency check of your run so far found: {detail}. "
                f"Correct that before giving the final answer.")
    if rung == "specific":
        if not findings:
            return _GENERIC_HINT
        return _specific_hint(findings)
    if rung == "adaptive":
        # Point the agent at a named defect ONLY when the ANSWER is wrong.
        # A coverage-only finding is real, but directing the agent at it
        # disturbs a total that was already right more often than it helps
        # (DESIGN.md Module 9).
        if any(f.check == "total_consistency" for f in findings):
            return _specific_hint(findings)
        return _GENERIC_HINT
    if rung == "recompute":
        # The dominant failure on this task is arithmetic over figures the
        # agent already looked up correctly, and the agent is holding a
        # calculator it did not use. Every other rung asks it to think again
        # with the same faculty that just failed; this one routes the step to
        # the tool that cannot make that error. It names no value, so it
        # carries no more information than `located`.
        if not findings:
            return _RECOMPUTE_HINT
        detail = "; ".join(f.terse for f in findings)
        return (f"A consistency check of your run so far found: {detail}. "
                f"{_RECOMPUTE_HINT}")
    if rung == "unstick":
        # NOT part of the scored offline comparison, and it cannot be: that
        # corpus holds answer-level failures on a working tool layer, so it
        # contains no run this rung applies to. Its evidence is the live
        # alarm-repair study instead (`results/tables/alarm_repair.csv`).
        # For a run stuck against a failing tool, "re-check your work" is the
        # wrong instruction: it invites another call to the tool that is
        # feeding the loop. This rung tells the agent to stop calling it and
        # finish with what it has, which is the only move that can end the
        # episode without the tool recovering.
        return _UNSTICK_HINT
    raise ValueError(f"unknown rung {rung!r}")


def _specific_hint(findings: list[Finding]) -> str:
    detail = "; ".join(f.detail for f in findings)
    return (f"A consistency check of your run so far found: {detail}. "
            f"Correct that before giving the final answer.")


def rebuild_history(backend: OllamaBackend, task: str, steps: list[dict],
                    k: int) -> None:
    """Restore the conversation exactly as it stood after step k-1.

    Read from `parse_step_events`, so a step that carries structured
    `tool_events` is replayed from what the collector executed: the real
    argument objects and the untruncated result. Rebuilding from the rendered
    text instead hands the retried agent the display form of its own history —
    arguments re-parsed out of a string, results cut to the display cap — and
    the retry then answers a slightly different question from the original run.
    """
    backend.reset(task)
    for s in steps[:k]:
        evs, reasoning = parse_step_events(s)
        msg: dict = {"role": "assistant"}
        if reasoning.strip():
            msg["content"] = reasoning.strip()
        if evs:
            msg["tool_calls"] = [
                {"function": {"name": e.name, "arguments": e.args or {}}}
                for e in evs]
        elif "content" not in msg:
            msg["content"] = ""
        backend.history.append(msg)
        for e in evs:
            backend.history.append({"role": "tool", "tool_name": e.name,
                                    "content": e.result})


def retry_from_checkpoint(steps: list[dict], seed: int, rung: str,
                          model: str = "qwen2.5:7b",
                          temperature: float = 0.2,
                          spec: TaskSpec = BOOKING_SPEC) -> Attempt:
    """Roll back, apply the rung's repair, and re-run the agent live.

    Booking-bound by construction, not just by the default `spec`: it rebuilds
    the world and the task with `_make_world`/`_make_demo_task`. Another domain
    needs its own runner, not another argument here. `rollback_step` above is
    the generic half and takes no default.
    """
    before = verify(steps, spec).findings
    k = rollback_step(steps, spec)
    world = _make_world(seed)
    task, _ = _make_demo_task(seed, world)

    if rung == "none":
        return Attempt("none", list(steps), len(steps), 0, before, before)

    backend = OllamaBackend(model, tool_specs=DEMO_TOOL_SPECS,
                            temperature=temperature)
    rebuild_history(backend, task, steps, k)
    hint = repair_message(rung, before)
    if hint:
        backend.history.append({"role": "user", "content": hint})

    out_steps = list(steps[:k])
    calls = 0
    for t in range(k, DEMO_MAX_STEPS):
        t0 = time.perf_counter()
        try:
            out = backend.step(t)
        except Exception:                       # noqa: BLE001
            break
        calls += 1
        lat = time.perf_counter() - t0
        bits = []
        events = []
        step_error = False
        if out["tool_uses"]:
            results = []
            for u in out["tool_uses"]:
                r = _run_tool(u["name"], u["input"], world)
                # `_run_tool` reports failure as an "Error: ..." string, so a
                # hardcoded False here records a failed call as a success. The
                # retried traces are re-graded and re-scored, and the error
                # flag feeds two live monitor features, so the retry arm would
                # be measured on telemetry that says nothing ever failed.
                is_err = error_shaped(r)
                step_error = step_error or is_err
                results.append({"id": u["id"], "name": u["name"],
                                "content": r, "is_error": is_err})
                bits.append(_tool_bit(u["name"], u["input"], r))
                events.append(make_tool_event(u["name"], u["input"], r,
                                              is_error=is_err,
                                              call_id=u["id"]))
            backend.add_tool_results(results)
        record = _step_record(out, bits, lat, error=step_error)
        # Carry the structured form as well, so a retried trace is read the
        # same way a collected one is rather than re-parsed from its text.
        record["tool_events"] = events
        out_steps.append(record)
        if out["stop_reason"] == "end_turn":
            break
    return Attempt(rung, out_steps, k, calls, before,
                   verify(out_steps, spec).findings)


if __name__ == "__main__":       # self-test: no model, no network
    trace = [
        {"text": 'I will price the legs. [lookup_flight({"i": 1}) -> $100]'},
        {"text": '[lookup_flight({"i": 2}) -> $200]'},
        {"text": '[lookup_hotel({"c": "x"}) -> $10/night]'},
        {"text": 'The grand total is $999 USD.'},
    ]
    k = rollback_step(trace, BOOKING_SPEC)
    assert k == 3, k          # resume just after the last priced lookup

    f = verify(trace, BOOKING_SPEC).findings
    assert f
    msg = repair_message("specific", f)
    assert "consistency check" in msg
    assert repair_message("resample", f) is None
    assert repair_message("none", f) is None
    assert repair_message("generic", f) == _GENERIC_HINT
    # Structural leak guarantee: the hint depends on findings and nothing else.
    import inspect as _i
    assert list(_i.signature(repair_message).parameters) == ["rung", "findings"]

    class _B:
        def __init__(self):
            self.history = []

        def reset(self, task):
            self.history = [{"role": "system", "content": "s"},
                            {"role": "user", "content": task}]

    b = _B()
    rebuild_history(b, "task text", trace, 3)
    roles = [m["role"] for m in b.history]
    assert roles == ["system", "user", "assistant", "tool", "assistant",
                     "tool", "assistant", "tool"], roles
    assert b.history[3]["content"] == "$100"
    assert b.history[2]["tool_calls"][0]["function"]["name"] == "lookup_flight"
    print("PASS: intervene.rollback self-test")
