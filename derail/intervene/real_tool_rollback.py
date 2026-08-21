"""Domain-neutral rollback/retry continuation for real-tool episodes.

Extends rollback.py's precedent to REAL_TASKS instead of the booking world.
`rebuild_history` (rollback.py) is reused unmodified -- reading it shows it
was already domain-neutral (task text + parse_step_events only, no booking
reference); `retry_from_checkpoint` was the booking-bound half, and this
module is its real-tool counterpart.

Two checkpoint rules, kept structurally separate so the oracle arm can never
be mistaken for the deployable primary result:

  compute_checkpoint_primary -- online-observable only: the frozen monitor's
                                 own causal alarm. Takes no tau, no
                                 failure_class -- enforced by signature, not
                                 by convention.
  compute_checkpoint_oracle  -- ground-truth tau. Explicitly named, reported
                                 only as arm="oracle_upper_bound".

No injected-ground-truth repair hint exists for RealTask (no Finding/
checks.verify() layer applies to it) -- the retry is rollback-and-retry with
no hint, honestly labeled "live rollback/retry recovery", never "repair".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from derail.harness.record_replay import Cassette
from derail.harness.tools import ToolRegistry, format_tool_bit
from derail.intervene.rollback import rebuild_history
from derail.telemetry.events import SCHEMA_VERSION, make_tool_event, parse_step_events
from derail.preconditions import error_shaped

PRIMARY = "primary"
ORACLE_UPPER_BOUND = "oracle_upper_bound"

#: Outcome taxonomy. `not_triggered` and `reconstruction_failed` are
#: terminal (no retry ran); the rest are set after a retry actually executes.
NOT_TRIGGERED = "not_triggered"
RECONSTRUCTION_FAILED = "reconstruction_failed"
RECOVERED = "recovered"
STILL_WRONG = "still_wrong"
HALTED = "halted"


def _last_successful_tool_step_before(steps: list[dict], limit: int) -> int:
    """Index just after the last step with >=1 non-error tool call, strictly
    before `limit`. 0 if none exists (a valid `checkpoint_at_start`)."""
    last = 0
    for t, s in enumerate(steps[:limit]):
        evs, _ = parse_step_events(s)
        if evs and any(not e.is_error for e in evs):
            last = t + 1
    return last


@dataclass
class CheckpointResult:
    k: int
    outcome: str          # "normal" | "checkpoint_at_start" | "not_triggered"
    alarm_step: int | None = None


def compute_checkpoint_primary(steps: list[dict], monitor, theta_b5: float,
                               episode_from_trace: Callable) -> CheckpointResult:
    """Online-observable checkpoint. Signature is the enforcement: this
    function receives only `steps` and the frozen monitor -- there is no
    parameter through which tau or a failure_class could reach it."""
    ep = episode_from_trace(steps, "checkpoint-scan",
                            use_sentence_transformers=False, extended=True)
    monitor.start_episode()
    alarm_step = None
    for t in range(ep.X.shape[0]):
        if monitor.score_step(ep.X[t]) > theta_b5:
            alarm_step = t
            break
    if alarm_step is None:
        return CheckpointResult(k=0, outcome=NOT_TRIGGERED)
    k = _last_successful_tool_step_before(steps, alarm_step)
    return CheckpointResult(
        k=k, outcome=("checkpoint_at_start" if k == 0 else "normal"),
        alarm_step=alarm_step)


def compute_checkpoint_oracle(steps: list[dict], tau: int) -> CheckpointResult:
    """Ground-truth-tau checkpoint -- SECONDARY, oracle-localized upper
    bound only. Never the primary recovery result."""
    k = _last_successful_tool_step_before(steps, tau)
    return CheckpointResult(k=k, outcome="checkpoint_at_start" if k == 0 else "normal")


@dataclass
class RetryOutcome:
    arm: str                       # PRIMARY | ORACLE_UPPER_BOUND
    checkpoint: CheckpointResult
    steps: list[dict] = field(default_factory=list)
    outcome: str = ""              # RECOVERED | STILL_WRONG | HALTED |
                                    # NOT_TRIGGERED | RECONSTRUCTION_FAILED
    success: bool | None = None
    error: str | None = None


def retry_real_task_from_checkpoint(
        steps: list[dict], task_prompt: str, checkpoint: CheckpointResult,
        arm: str, *, model: str = "qwen2.5:7b", temperature: float = 0.2,
        max_steps: int, registry: ToolRegistry,
        backend_factory: Callable[..., Any]) -> RetryOutcome:
    """Roll back to `checkpoint.k` and re-run live to completion or halt.

    Continuation tool calls pass `cassette=None` explicitly (not
    `mode="live"`) -- the ToolRegistry.call/Cassette.call dispatch calls the
    live function unconditionally whenever no cassette is given, so every
    call issued here is structurally live, never a replay, verified by the
    `source` field this same call path records.

    `backend_factory(model, tool_specs, temperature) -> OllamaBackend`-shaped
    object exposing reset/step/add_tool_results, matching rollback.py's
    existing contract -- kept as a parameter rather than importing
    OllamaBackend directly, so this stays a pure continuation runner testable
    against a fixture backend (see the equivalence proof).
    """
    if checkpoint.outcome == NOT_TRIGGERED:
        return RetryOutcome(arm=arm, checkpoint=checkpoint,
                            outcome=NOT_TRIGGERED, success=None)

    try:
        backend = backend_factory(model=model, tool_specs=registry.specs(),
                                  temperature=temperature)
        rebuild_history(backend, task_prompt, steps, checkpoint.k)
    except Exception as exc:                              # noqa: BLE001
        return RetryOutcome(arm=arm, checkpoint=checkpoint,
                            outcome=RECONSTRUCTION_FAILED, success=None,
                            error=f"{type(exc).__name__}: {exc}")

    out_steps = list(steps[:checkpoint.k])
    for t in range(checkpoint.k, max_steps):
        t0 = time.perf_counter()
        try:
            out = backend.step(t)
        except Exception as exc:                          # noqa: BLE001
            return RetryOutcome(arm=arm, checkpoint=checkpoint,
                                steps=out_steps, outcome=HALTED, success=False,
                                error=f"{type(exc).__name__}: {exc}")
        lat = time.perf_counter() - t0
        bits, events, step_error, results = [], [], False, []
        for use in out.get("tool_uses", []):
            res = registry.call(use["name"], use["input"], cassette=None,
                                record_provenance=True)
            step_error = step_error or res.is_error
            bits.append(res.step_bit())
            results.append({"id": use["id"], "name": use["name"],
                            "content": res.content, "is_error": res.is_error})
            events.append(make_tool_event(use["name"], use["input"],
                                          res.content, is_error=res.is_error,
                                          latency_s=res.latency_s,
                                          call_id=use["id"], source=res.source))
            assert res.source in ("live_external", "live_local"), (
                f"continuation call for {use['name']!r} returned "
                f"source={res.source!r} -- must be live, never a replay")
        if results:
            backend.add_tool_results(results)
        record = {
            "text": out.get("text", ""), "action": out.get("action", ""),
            "latency_s": round(lat, 4), "error": step_error,
            "tool_events": events, "schema": SCHEMA_VERSION,
            "resumed_from": {"checkpoint_step": checkpoint.k},
        }
        out_steps.append(record)
        if out.get("stop_reason") == "end_turn":
            break
    else:
        return RetryOutcome(arm=arm, checkpoint=checkpoint, steps=out_steps,
                            outcome=HALTED, success=False)

    return RetryOutcome(arm=arm, checkpoint=checkpoint, steps=out_steps,
                        outcome="_scored_by_caller", success=None)
