"""WS0.5 — the real-tool telemetry contract.

Every real tool (WS1: Tavily, weather, GitHub, SQL, Python REPL, ...) plugs
in here so its calls come out in exactly the shape the rest of the pipeline
already understands:

  * the step-text bit  "[name({json args}) -> result]"  that
    derail.telemetry.events parses for the derived x channel
    (reasoning depth, tool-success rate, retries, per-tool latency);
  * a per-call latency and error flag for the m channel;
  * transparent cassette record/replay + cost-free repeat runs (WS0.3).

A Tool is just: a name, a description, a {param: description} schema, and a
run(**args) -> str. Raising from run() (or returning a string that starts
with "Error:") marks the call as an error, matching the existing collectors.
This module owns NO live integrations — those are WS1; it owns the contract
they satisfy so the adapter and monitors never change.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from derail.harness.record_replay import Cassette, request_key
from derail.harness.sandbox import redact_secrets

_RESULT_TRUNCATE = 100   # chars kept in the step-text bit (matches collectors)


@runtime_checkable
class Tool(Protocol):
    """Structural type for a real tool. Implement these four members."""

    name: str
    description: str
    parameters: dict[str, str]        # param name -> human description

    def run(self, **kwargs: Any) -> str: ...


@dataclass
class SimpleTool:
    """Concrete Tool wrapping a plain function (for tools without state)."""

    name: str
    description: str
    parameters: dict[str, str]
    fn: Callable[..., str]

    def run(self, **kwargs: Any) -> str:
        return self.fn(**kwargs)


@dataclass
class ToolResult:
    """Outcome of one tool call, in telemetry-ready form."""

    name: str
    args: dict[str, Any]
    content: str          # the result string (already error-normalized)
    is_error: bool
    latency_s: float

    def step_bit(self) -> str:
        """The "[name({args}) -> result]" fragment for the step text."""
        return format_tool_bit(self.name, self.args, self.content)


def format_tool_bit(name: str, args: dict[str, Any], result: str) -> str:
    """Canonical step-text fragment — byte-compatible with the collectors and
    with derail.telemetry.events. Args are sorted so the same call
    always renders (and hashes) identically."""
    return (f"[{name}({json.dumps(args, sort_keys=True)}) -> "
            f"{str(result)[:_RESULT_TRUNCATE]}]")


#: Bumped when the recorded shape of a tool result changes.
CASSETTE_SCHEMA_VERSION = "tool/v2"

#: Constructor-time settings that change what a tool returns. They belong in
#: the cassette key, otherwise a reconfigured tool replays a recording made by
#: a different implementation.
_FINGERPRINT_ATTRS = ("root", "db_path", "max_results", "max_rows",
                      "max_file_chars", "max_output_bytes", "timeout_s",
                      "allow_network", "allow_hosts", "servers")


def _source_digest(cls: type) -> str:
    """Hash of the tool class's own source, or "" when it cannot be read.

    The class NAME does not identify an implementation: edit what a tool
    returns, keep its name and settings, and every recorded cassette replays
    the old answers under the new code with nothing to notice. Reading the
    source ties a recording to the implementation that produced it.

    Falls back to "" for a class defined interactively or in a zipimport,
    where no source exists; the fingerprint is then no weaker than before.
    """
    try:
        return hashlib.sha256(
            inspect.getsource(cls).encode("utf-8")).hexdigest()[:16]
    except (OSError, TypeError):
        return ""


def tool_fingerprint(tool: Tool) -> str:
    """Stable identity of a tool *implementation plus configuration*."""
    cls = type(tool)
    cfg = {a: str(getattr(tool, a)) for a in _FINGERPRINT_ATTRS if hasattr(tool, a)}
    return json.dumps(
        {"class": f"{cls.__module__}.{cls.__qualname__}",
         "source": _source_digest(cls),
         "params": sorted(getattr(tool, "parameters", {})),
         "config": cfg},
        sort_keys=True, separators=(",", ":"), default=str)


_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               dict: "object", list: "array"}


def tool_json_schema(tool: Tool) -> dict:
    """JSON Schema for one tool's arguments, derived from ``run``'s signature.

    A parameter with a default is optional, and its annotation gives the type.
    Declaring *every* parameter required and typing everything as a string
    rejects valid calls like ``list_dir()`` and advertises action-dependent
    parameters as mandatory.
    """
    sig = inspect.signature(tool.run)
    properties: dict[str, dict] = {}
    required: list[str] = []
    descriptions = getattr(tool, "parameters", {}) or {}
    var_keyword = False
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL,
                                            param.VAR_KEYWORD):
            var_keyword = var_keyword or param.kind is param.VAR_KEYWORD
            continue
        annotation = param.annotation
        if isinstance(annotation, str):        # from __future__ annotations
            annotation = {"str": str, "int": int, "float": float,
                          "bool": bool, "dict": dict, "list": list}.get(
                              annotation.split("|")[0].strip(), str)
        spec = {"type": _JSON_TYPES.get(annotation, "string"),
                "description": str(descriptions.get(name, ""))}
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            spec["default"] = param.default
        properties[name] = spec
    if var_keyword:
        # `run(**kwargs)` (SimpleTool and friends) exposes no signature to read,
        # so fall back to the declared parameter dict. Optionality is unknowable
        # there, so those parameters stay required - the previous behaviour.
        for name, description in descriptions.items():
            if name not in properties:
                properties[name] = {"type": "string",
                                    "description": str(description)}
                required.append(name)
    return {"type": "object", "properties": properties, "required": required}


class ToolRegistry:
    """Holds the tools available to one agent cell and runs their calls."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> dict[str, tuple[str, dict[str, str]]]:
        """{name: (description, params)} — the shape the existing Gemini
        declaration builder (collect_traces.TOOL_SPECS) already consumes."""
        return {n: (t.description, dict(t.parameters))
                for n, t in self._tools.items()}

    def schemas(self) -> dict[str, dict]:
        """{name: JSON Schema} for every tool (types, defaults, required)."""
        return {n: tool_json_schema(t) for n, t in self._tools.items()}

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def call(self, name: str, args: dict[str, Any],
             cassette: Cassette | None = None,
             injector: Any | None = None) -> ToolResult:
        """Run one tool call: time it, normalize errors, record/replay.

        With a cassette the (name, args) result is replayed if seen before,
        so live tools (paid search, flaky network) are hit at most once.
        Latency is re-measured on replay (near-zero), which correctly shows
        up as a metadata/x-channel signal rather than a fabricated delay.

        An `injector` (WS4 harness.inject.ToolInjector) is applied AFTER the
        cassette, so the same clean recording serves a healthy run and any
        injected variant; the transformed result is what the agent sees and
        what telemetry records.
        """
        if name not in self._tools:
            res = ToolResult(name, args, f"Error: unknown tool {name}",
                             True, 0.0)
            return injector.apply(res) if injector is not None else res
        # Key covers the implementation and its configuration; the pre-review
        # key is kept as a legacy fallback so recordings that cost real money
        # still replay and are migrated on first hit.
        key = request_key("tool", name, args,
                          tool_fingerprint(self._tools[name]),
                          namespace=CASSETTE_SCHEMA_VERSION)
        legacy_keys = (request_key("tool", name, args),)

        def _live() -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                out = str(self._tools[name].run(**args))
                is_err = out.startswith("Error:")
            except Exception as exc:  # noqa: BLE001 — surfaced to the agent
                out, is_err = f"Error: {type(exc).__name__}: {exc}", True
            # Central redaction: applied before the result can reach a model,
            # a trace file or a cassette, so no individual tool can leak a
            # credential by forgetting to scrub.
            return {"content": redact_secrets(out), "is_error": is_err,
                    "latency_s": round(time.perf_counter() - t0, 4)}

        t0 = time.perf_counter()
        rec = (cassette.call(key, _live, legacy_keys=legacy_keys,
                             is_error=lambda r: bool(r.get("is_error")))
               if cassette is not None else _live())
        # On a replay the recorded latency is stale; report the real elapsed
        # (tiny) time so downstream latency features reflect what happened.
        latency = (rec["latency_s"] if cassette is None
                   else round(time.perf_counter() - t0, 4))
        res = ToolResult(name, args, rec["content"], rec["is_error"], latency)
        return injector.apply(res) if injector is not None else res


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import tempfile

    from derail.common import D_TOTAL_EXT, IDX_REASON_DEPTH, IDX_TOOL_SUCCESS
    from derail.telemetry.adapter import ExtFeatureState, step_signal_ext

    calls = {"n": 0}

    def flaky_search(query: str) -> str:
        calls["n"] += 1
        return f"results for {query}: 3 hits"

    def boom(x: str) -> str:
        raise ValueError("service down")

    reg = ToolRegistry([
        SimpleTool("web_search", "Search the web.", {"query": "Search query"},
                   flaky_search),
        SimpleTool("break_it", "Always fails.", {"x": "anything"}, boom),
    ])
    assert "web_search" in reg and set(reg.specs()) == {"web_search", "break_it"}

    ok = reg.call("web_search", {"query": "gemini pricing"})
    assert not ok.is_error and ok.latency_s >= 0.0
    bad = reg.call("break_it", {"x": "y"})
    assert bad.is_error and bad.content.startswith("Error:")
    unknown = reg.call("nope", {})
    assert unknown.is_error

    # The step bit must parse back through the adapter's x-channel extractor:
    # one successful call -> reasoning depth 1, tool-success 1.0.
    step = {"text": "Now searching. " + ok.step_bit(),
            "token_logprobs": [-0.1] * 12, "action": "tool_call",
            "latency_s": ok.latency_s, "output_tokens": 12, "error": ok.is_error}
    x = step_signal_ext(step, ExtFeatureState(), use_sentence_transformers=False)
    assert x.shape[0] == D_TOTAL_EXT
    assert x[IDX_REASON_DEPTH] == 1.0, x[IDX_REASON_DEPTH]
    assert x[IDX_TOOL_SUCCESS] == 1.0, x[IDX_TOOL_SUCCESS]

    # An error call is visible to the contract (success rate drops to 0).
    estep = {"text": bad.step_bit(), "token_logprobs": [-0.1] * 5,
             "action": "tool_call", "latency_s": bad.latency_s,
             "output_tokens": 5, "error": bad.is_error}
    ex = step_signal_ext(estep, ExtFeatureState(), use_sentence_transformers=False)
    assert ex[IDX_TOOL_SUCCESS] == 0.0, ex[IDX_TOOL_SUCCESS]

    # Cassette: a second identical call replays (no live hit).
    with tempfile.TemporaryDirectory() as d:
        cas = Cassette(d, mode="auto")
        before = calls["n"]                          # 1 (the `ok` call above)
        r1 = reg.call("web_search", {"query": "same"}, cassette=cas)  # records
        r2 = reg.call("web_search", {"query": "same"}, cassette=cas)  # replays
        assert r1.content == r2.content
        assert calls["n"] == before + 1              # exactly one live hit
        assert cas.n_recorded == 1 and cas.n_replayed == 1

    print("PASS tools.py smoke test | step bit:", ok.step_bit())
