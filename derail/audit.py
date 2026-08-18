"""The audit trail the architecture has always claimed and never had.

`DESIGN.md` describes "the audit trail every stage writes to" and the
architecture diagram carries an Audit / Log Store box holding events,
decisions, alerts, explanations and lineage. Nothing implemented it: the only
durable runtime output was provider cassettes and the SQL fixture, both of
which exist to make runs *repeatable*, not to make a past run *answerable*.

What this is
------------
An append-only JSONL sink, one file per run, under
``runtime_root()/audit/``. One JSON object per line, flushed as it is written,
so a run that is killed mid-episode still leaves everything up to that point.
No database, no service, no daemon: the question "what did the monitor see and
what did it decide?" is answered by reading a file, and a file is the only
dependency a post-incident review can be certain still exists.

What it deliberately is NOT
---------------------------
It is not a second telemetry path. Nothing here is read back by the monitor,
and no monitor decision depends on a write succeeding — see "Failure
isolation". A log that can change a verdict is not an audit log.

Privacy
-------
Raw content is NOT persisted by default. Prompts, tool arguments, tool results
and model output are recorded as a length plus a truncated SHA-256, which is
enough to prove two runs saw the same bytes and to line a record up against a
trace, and not enough to leak a customer record or a credential. Enable
`include_content` (or ``AGENTWATCH_AUDIT_CONTENT=1``) only where the payloads
are known to be safe; even then every string goes through the same secret
redaction the tool layer uses before it reaches a model or a cassette.

Failure isolation
-----------------
Every public method is wrapped: a full disk, a revoked permission, a value
that will not serialize, a vanished directory — none of them can raise into
the caller. Drops are counted and reported once, so a silently dead log is
visible without ever being fatal. The monitoring and repair paths therefore
run identically whether the sink works, is disabled, or is on fire.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Mapping

from derail.harness.record_replay import runtime_root
from derail.harness.sandbox import redact_secrets

#: Bumped when the meaning of a recorded field changes. A reader that does not
#: know the version must refuse the file rather than guess at it.
AUDIT_SCHEMA_VERSION = 1

#: Opt-in switch for persisting redacted raw content. Off unless set to one of
#: these, because the safe default for a log nobody has reviewed is metadata.
_CONTENT_ENV = "AGENTWATCH_AUDIT_CONTENT"
_TRUTHY = {"1", "true", "yes", "on"}

#: Cap on any single recorded string when content logging IS enabled. A log
#: line is evidence, not a copy of the payload.
MAX_CONTENT_CHARS = 2000

#: The event vocabulary. Closed on purpose: an audit trail whose record types
#: are ad-hoc cannot be queried, and a typo would create a category nobody
#: knows to look in.
EVENTS = ("run_start", "step", "tool", "score", "check", "decision",
          "outcome", "run_end")

#: Decisions the intervention path can take. Also closed, same reason.
DECISIONS = ("checkpoint", "rollback", "repair", "escalate", "halt", "resume")


def content_logging_enabled() -> bool:
    return os.environ.get(_CONTENT_ENV, "").strip().lower() in _TRUTHY


def digest(text: str | None) -> dict:
    """Identity of a string without the string (see module docstring, "Privacy")."""
    if text is None:
        return {"chars": 0, "sha256": None}
    s = str(text)
    return {"chars": len(s),
            "sha256": hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]}


def _jsonable(value: Any) -> Any:
    """Coerce to something json can write, without raising on the odd type."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # NaN/Infinity are not JSON; a reader that gets `NaN` back from a
        # non-strict parser cannot tell it from a real measurement.
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


class AuditLog:
    """Append-only JSONL audit sink for one run."""

    def __init__(self, run_id: str | None = None, *,
                 root: Path | None = None,
                 enabled: bool = True,
                 include_content: bool | None = None) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:16]
        self.enabled = bool(enabled)
        self.include_content = (content_logging_enabled()
                                if include_content is None
                                else bool(include_content))
        self.dropped = 0
        self.written = 0
        self._warned = False
        self._lock = threading.Lock()
        self._path: Path | None = None
        if self.enabled:
            try:
                base = Path(root) if root is not None else runtime_root() / "audit"
                base.mkdir(parents=True, exist_ok=True)
                self._path = base / f"{self.run_id}.jsonl"
            except Exception:                                   # noqa: BLE001
                self._note_drop()
                self.enabled = False

    # ------------------------------------------------------------- plumbing
    @property
    def path(self) -> Path | None:
        """Where records land, or None when the sink is disabled/unusable."""
        return self._path

    def _note_drop(self) -> None:
        self.dropped += 1
        if not self._warned:
            self._warned = True
            print("[audit] log sink unavailable; the run continues and "
                  "monitoring is unaffected. Records are being dropped.")

    def _emit(self, event: str, **fields: Any) -> None:
        """Write one record. Cannot raise: an audit sink must never be able
        to take down the path it is auditing."""
        if not self.enabled or self._path is None:
            return
        try:
            record = {"schema": AUDIT_SCHEMA_VERSION,
                      "run_id": self.run_id,
                      "ts": datetime.datetime.now(
                          datetime.timezone.utc).isoformat(),
                      "event": event}
            record.update({k: _jsonable(v) for k, v in fields.items()
                           if v is not None})
            line = json.dumps(record, sort_keys=True, separators=(",", ":"))
            with self._lock:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
                    fh.flush()
                self.written += 1
        except Exception:                                       # noqa: BLE001
            self._note_drop()

    def _text(self, text: str | None) -> dict:
        """Digest, plus the redacted content itself only when opted in."""
        out = digest(text)
        if self.include_content and text is not None:
            out["content"] = redact_secrets(str(text))[:MAX_CONTENT_CHARS]
        return out

    # --------------------------------------------------------------- events
    def run_start(self, *, episode_id: str | None = None,
                  model: str | None = None, task: str | None = None,
                  config: Mapping[str, Any] | None = None) -> None:
        self._emit("run_start", episode_id=episode_id, model=model,
                   task=self._text(task), config=config,
                   content_logged=self.include_content)

    def step(self, t: int, *, episode_id: str | None = None,
             action: str | None = None, latency_s: float | None = None,
             output_tokens: int | None = None, text: str | None = None,
             features: Mapping[str, float] | None = None) -> None:
        """One agent step. `features` carries the numeric telemetry a
        diagnosis needs (channel values, top contributing dims) — never the
        text those features were derived from."""
        self._emit("step", t=t, episode_id=episode_id, action=action,
                   latency_s=latency_s, output_tokens=output_tokens,
                   text=self._text(text), features=features)

    def tool(self, t: int, *, name: str, args_key: str | None = None,
             result: str | None = None, is_error: bool = False,
             truncated: bool = False, latency_s: float | None = None,
             episode_id: str | None = None) -> None:
        """One executed tool call.

        `args_key` is DIGESTED, not stored. Callers naturally pass
        `canonical_args(...)`, which is the arguments themselves — a live demo
        run put the task's city names straight into the trail that way. The
        digest keeps what the key is for (same arguments produce the same
        value, so a retry is still identifiable) and drops what it should
        never have carried. Enforcing it here rather than at each call site
        makes the guarantee structural: no caller can leak by being careless.
        """
        self._emit("tool", t=t, episode_id=episode_id, name=name,
                   args=self._text(args_key), result=self._text(result),
                   is_error=bool(is_error), truncated=bool(truncated),
                   latency_s=latency_s)

    def score(self, t: int, *, monitor: str, score: float,
              threshold: float | None = None, alarmed: bool = False,
              episode_id: str | None = None) -> None:
        self._emit("score", t=t, episode_id=episode_id, monitor=monitor,
                   score=score, threshold=threshold, alarmed=bool(alarmed))

    def check(self, t: int | None, *, check: str, passed: bool,
              detail: str | None = None,
              episode_id: str | None = None) -> None:
        """A deterministic verdict. `detail` is the check's own terse text,
        which is computed from the run's own tool results — not model output —
        so it is recorded verbatim."""
        self._emit("check", t=t, episode_id=episode_id, check=check,
                   passed=bool(passed), detail=detail)

    def decision(self, t: int | None, *, kind: str, reason: str | None = None,
                 detail: Mapping[str, Any] | None = None,
                 episode_id: str | None = None) -> None:
        if kind not in DECISIONS:
            kind = f"unknown:{kind}"
        self._emit("decision", t=t, episode_id=episode_id, kind=kind,
                   reason=reason, detail=detail)

    def outcome(self, *, status: str, episode_id: str | None = None,
                steps: int | None = None,
                detail: Mapping[str, Any] | None = None) -> None:
        self._emit("outcome", episode_id=episode_id, status=status,
                   steps=steps, detail=detail)

    def run_end(self) -> None:
        self._emit("run_end", written=self.written, dropped=self.dropped)

    # ---------------------------------------------------------- reading back
    def records(self) -> list[dict]:
        """Every record written by this run, for a reader or a test.

        Returns [] rather than raising when the sink never opened: a caller
        inspecting an audit trail should not have to handle the case where
        auditing was switched off.
        """
        if self._path is None or not self._path.exists():
            return []
        out = []
        for line in self._path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue          # a torn final line is not a reason to fail
        return out


class NullAuditLog(AuditLog):
    """A sink that records nothing, for callers that want no file at all."""

    def __init__(self) -> None:
        super().__init__(enabled=False)


if __name__ == "__main__":       # module self-test
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        log = AuditLog("selftest", root=Path(d))
        log.run_start(episode_id="ep-1", model="m", task="a secret task")
        log.step(0, action="tool_call", latency_s=1.25,
                 features={"e": 0.5, "u": 1.5})
        log.tool(0, name="lookup_flight", args_key='{"city":"Osaka"}',
                 result="Error: boom", is_error=True)
        log.score(0, monitor="esn_cusum_max", score=12.5, threshold=9.0,
                  alarmed=True)
        log.check(0, check="tool_contract", passed=False, detail="malformed")
        log.decision(0, kind="rollback", reason="alarm")
        log.outcome(status="repaired", steps=4)
        log.run_end()

        recs = log.records()
        assert [r["event"] for r in recs] == [
            "run_start", "step", "tool", "score", "check", "decision",
            "outcome", "run_end"], [r["event"] for r in recs]
        # Content is NOT persisted by default.
        assert recs[0]["task"]["sha256"] and "content" not in recs[0]["task"]
        body = log.path.read_text("utf-8")
        assert "secret task" not in body
        assert "Osaka" not in body, "tool arguments reached the audit log"
        assert recs[3]["alarmed"] is True and recs[3]["threshold"] == 9.0
        assert log.dropped == 0 and log.written == 8

        # Opt-in content, still redacted.
        log2 = AuditLog("selftest2", root=Path(d), include_content=True)
        log2.tool(0, name="t", result="token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345")
        body = log2.path.read_text("utf-8")
        assert "ghp_ABCDEF" not in body, "credential reached the audit log"

        # A broken sink drops records and never raises.
        broken = AuditLog("selftest3", root=Path(d))
        broken._path = Path(d) / "no-such-dir" / "x.jsonl"
        broken.step(0, action="plan")
        assert broken.dropped == 1 and broken.records() == []

        # Non-serializable values degrade to a string rather than exploding.
        log4 = AuditLog("selftest4", root=Path(d))
        log4.step(0, features={"weird": float("nan")}, action=object())
        assert log4.dropped == 0
        assert log4.records()[0]["features"]["weird"] is None

    print("PASS: audit self-test")
