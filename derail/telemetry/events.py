"""Structured tool-call telemetry (step schema v5).

Real structured tool calls were flattened into
model-controlled text and recovered with a regex.  That made the telemetry
spoofable (a model can type tool syntax without calling a tool), lossy (results
truncated to 100 characters), and ambiguous (an inner ``]`` cut arrays and
Markdown short; non-word tool names were not parsed at all; arguments were not
canonicalised, so a whitespace difference hid a retry).

A step written by this project now carries its tool calls as data:

    {"text": "...", "action": "tool_call", "latency_s": 1.4,
     "token_logprobs": [...], "logprobs_available": true,
     "output_tokens": 213, "error": false,
     "task": "<the original user task>",
     "schema": 5,
     "tool_events": [
        {"id": "call-3", "name": "arxiv_search",
         "args": {"query": "echo state networks"},
         "result": "<full result>", "result_chars": 812,
         "result_truncated": false, "is_error": false, "latency_s": 0.42}
     ]}

`parse_step_events` prefers that structured form and falls back to parsing the
text for the older corpora, so no already-collected trace is lost:

  * ``[name({args}) -> result]``   - the v2/v3 in-bracket form;
  * ``[name({args})] -> result``   - the arrow-outside-bracket form; 41
    committed LangGraph-7B traces carry recoverable results in it;
  * ``[name({args})]``             - the v1 form with no result recorded.

The fallback is bracket-depth aware, so a result containing ``]`` survives, and
it accepts dotted/hyphenated tool names.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from derail.preconditions import error_shaped

SCHEMA_VERSION = 5

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-]*")
_ARROW_RE = re.compile(r"\s*->\s*")


@dataclass(frozen=True)
class ToolCallEvent:
    """One tool call as the feature extractors see it."""

    name: str
    args: dict | None          # None when the text form held unparseable args
    args_key: str              # canonical form, used for retry equality
    result: str                # "" when the trace recorded no result
    has_result: bool
    is_error: bool
    latency_s: float | None    # measured tool time; None in legacy traces
    truncated: bool
    source: str                # "structured" | "text"


def canonical_args(args: object) -> str:
    """Whitespace- and order-independent rendering of call arguments."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return " ".join(args.split())
    try:
        return json.dumps(args, sort_keys=True, separators=(",", ":"),
                          default=str)
    except (TypeError, ValueError):
        return str(args)


def make_tool_event(name: str, args: dict, result: str, *, is_error: bool,
                    latency_s: float | None = None, call_id: str | None = None,
                    max_result_chars: int | None = None,
                    source: str | None = None) -> dict:
    """Build one structured tool event for a step record.

    `source` ("live_external" | "live_local" | "cassette_replay" | "mock"),
    opt-in: only added as a key when a caller passes one, so every existing
    call site's event dict is byte-identical to before -- an event with no
    `source` key means "not recorded", never inferred after the fact.
    """
    text = str(result)
    truncated = (max_result_chars is not None and len(text) > max_result_chars)
    event = {"id": call_id or "", "name": str(name), "args": dict(args or {}),
            "result": text[:max_result_chars] if truncated else text,
            "result_chars": len(text), "result_truncated": truncated,
            "is_error": bool(is_error),
            "latency_s": None if latency_s is None else float(latency_s)}
    if source is not None:
        event["source"] = source
    return event


# ------------------------------------------------------------ text fallback
def _match_closing(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index of the delimiter closing the one at `start`, or -1.

    String-aware, so a bracket inside a JSON string does not shift the depth.
    """
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _next_bit_start(text: str, pos: int) -> int:
    """Index of the next `[name(` tool bit at or after `pos` (else len(text))."""
    i = pos
    while True:
        j = text.find("[", i)
        if j < 0:
            return len(text)
        m = _NAME_RE.match(text, j + 1)
        if m and m.end() < len(text) and text[m.end()] == "(":
            return j
        i = j + 1


def parse_tool_bits(text: str) -> tuple[list[ToolCallEvent], str]:
    """(tool calls, reasoning text) parsed out of one step's text."""
    events: list[ToolCallEvent] = []
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(text):
        j = text.find("[", i)
        if j < 0:
            break
        m = _NAME_RE.match(text, j + 1)
        if not m or m.end() >= len(text) or text[m.end()] != "(":
            i = j + 1
            continue
        name = m.group(0)
        p_end = _match_closing(text, m.end(), "(", ")")
        if p_end < 0:
            i = j + 1
            continue
        args_raw = text[m.end() + 1:p_end]
        seg_end = _next_bit_start(text, p_end + 1)
        rest = text[p_end + 1:]

        if rest.startswith("]"):
            close = p_end + 1
            arrow = _ARROW_RE.match(text, close + 1)
            if arrow and arrow.end() <= seg_end:
                # "[name(args)] -> result" (framework collectors)
                result, end = text[arrow.end():seg_end].rstrip(), seg_end
            else:
                result, end = "", close + 1          # v1: no result recorded
            has_result = bool(result)
        else:
            arrow = _ARROW_RE.match(text, p_end + 1)
            if not arrow:
                i = j + 1
                continue
            # "[name(args) -> result]": prefer true bracket matching, fall back
            # to the last ']' in this bit's segment so a truncated or
            # bracket-bearing result is still recovered whole.
            b_end = _match_closing(text, j, "[", "]")
            if not (0 <= b_end < seg_end):
                b_end = text.rfind("]", arrow.end(), seg_end)
            if b_end < 0:
                result, end = text[arrow.end():seg_end].rstrip(), seg_end
            else:
                result, end = text[arrow.end():b_end], b_end + 1
            has_result = True

        try:
            args_obj = json.loads(args_raw) if args_raw.strip() else {}
            if not isinstance(args_obj, dict):
                args_obj = None
        except (json.JSONDecodeError, ValueError):
            args_obj = None
        events.append(ToolCallEvent(
            name=name,
            args=args_obj,
            args_key=canonical_args(args_obj if args_obj is not None else args_raw),
            result=result,
            has_result=has_result,
            is_error=error_shaped(result),
            latency_s=None,
            truncated=False,          # unknowable from text alone
            source="text"))
        spans.append((j, end))
        i = end

    if not spans:
        return events, text
    out, prev = [], 0
    for start, end in spans:
        out.append(text[prev:start])
        prev = end
    out.append(text[prev:])
    return events, " ".join(" ".join(out).split())


def parse_step_events(step: dict) -> tuple[list[ToolCallEvent], str]:
    """(tool calls, reasoning text) for one step record.

    Structured `tool_events` win outright: when a collector recorded what it
    actually executed, model-written text can no longer fabricate a tool call
    or hide one.

    "Outright" includes the EMPTY list. A collector that writes the key is
    reporting what it executed, and an empty list is the report "nothing"; the
    step's text can then only be the model's prose, which may well contain
    tool-like syntax it merely described. Falling back to the parser there is
    what lets written syntax read as an executed call. Only an ABSENT key means
    the corpus has no structured record and the text is all there is.
    """
    raw = step.get("tool_events")
    text = str(step.get("text", ""))
    if isinstance(raw, list):
        events = []
        for ev in raw:
            if not isinstance(ev, dict):
                continue
            args = ev.get("args") if isinstance(ev.get("args"), dict) else {}
            result = "" if ev.get("result") is None else str(ev.get("result"))
            latency = ev.get("latency_s")
            events.append(ToolCallEvent(
                name=str(ev.get("name", "")),
                args=args,
                args_key=canonical_args(args),
                result=result,
                has_result=bool(result) or bool(ev.get("result_chars")),
                is_error=bool(ev.get("is_error")),
                latency_s=None if latency is None else float(latency),
                truncated=bool(ev.get("result_truncated")),
                source="structured"))
        # The rendered bits are display only; strip them from the reasoning
        # text so the two forms produce the same reasoning string.
        _, reasoning = parse_tool_bits(text)
        return events, reasoning
    return parse_tool_bits(text)
