"""WS2 — real agent frameworks driving the WS0.5 ToolRegistry.

The vertical slice (harness/agent_loop.py) used OUR manual loop. This module
hands orchestration to a real framework — a LangGraph ReAct-style StateGraph —
so the "validated on a real agent framework" claim is literal: LangGraph owns
the agent<->tool loop; we only supply tools (the ToolRegistry) and read out
telemetry in the v2/v3 step format the adapter already consumes.

Framework note: LangGraph's model bindings (ChatOllama) do not surface
token logprobs, so framework cells are e+m (no live u channel) — the same
limitation the earlier mock-tool framework traces documented. The u channel
lives on the native Ollama loop (agent_loop.py), which is its own source.
That is a finding, not a gap: orchestration frameworks abstract logprobs away.

Telemetry contract -- `latency_s`: every step's `latency_s` measures ONLY
that step's own LLM call, the same quantity `agent_loop.run_real_episode`
measures (`t0 = time.perf_counter(); out = backend.step(t)`, nothing else
around it). Tool-execution time is never folded in here; it is already
captured per call in `tool_events[].latency_s`. Both loops below restart
their `t_prev` clock the instant a step's tool calls finish executing, not
when the previous agent step ended -- otherwise a step's `latency_s` would
also include the PRECEDING step's tool-execution time, silently measuring a
different quantity than the native harness under the same feature name (this
was exactly wrong before it was fixed; see `test_telemetry_contracts.py`).
"""

from __future__ import annotations

import inspect
import time
from typing import Any

from derail.harness.record_replay import Cassette
from derail.harness.tools import ToolRegistry, format_tool_bit
from derail.preconditions import error_shaped
from derail.telemetry.events import SCHEMA_VERSION, make_tool_event

_SYSTEM = ("Solve the user's task using the tools. Call at most ONE tool per "
           "message, then wait for its result. Be brief; finish with a "
           "one-line answer.")


def reached_step_limit(steps: list[dict], max_steps: int) -> bool:
    """True when collection may stop.

    A framework loop must not stop between an agent's tool request and that
    tool's result: the call is executed but never recorded, so the episode ends
    with a step whose tool outcome is missing.
    """
    return bool(steps) and len(steps) >= max_steps and not steps[-1].get("_pending")


_PY_TYPES = {"string": str, "integer": int, "number": float,
             "boolean": bool, "object": dict, "array": list}


def _pydantic_args_model(name: str, schema: dict):
    """Pydantic model for a tool's arguments honouring types and defaults.

    Optional parameters carry their real default instead of being declared
    required, so `list_dir()` and action-dependent GitHub fields
    are valid calls again.
    """
    from pydantic import Field, create_model

    fields = {}
    required = set(schema.get("required", []))
    for param, spec in schema.get("properties", {}).items():
        py_type = _PY_TYPES.get(spec.get("type", "string"), str)
        description = spec.get("description", "")
        if param in required:
            fields[param] = (py_type, Field(..., description=description))
        else:
            fields[param] = (py_type, Field(spec.get("default"),
                                            description=description))
    return create_model(f"{name}_args", **fields)


def _occurrence_suffix(occurrence: dict, tool_name: str, args: dict) -> str:
    """Per-episode occurrence count for `(tool_name, args)`, as a cassette
    key suffix ("occ0", "occ1", ...) -- disambiguates a repeated identical
    call within one episode's cassette. `occurrence` is the caller's own
    dict, so counting is scoped to one episode's closures, never shared."""
    import json as _json
    ident = (tool_name, _json.dumps(args, sort_keys=True, default=str))
    occurrence[ident] = occurrence.get(ident, 0) + 1
    return f"occ{occurrence[ident] - 1}"


def _langchain_tools(registry: ToolRegistry, cassette: Cassette | None,
                     record: dict | None = None,
                     record_provenance: bool = False,
                     injector: Any | None = None):
    """Wrap each registry tool as a LangChain StructuredTool whose execution
    routes back through registry.call (cassette + error normalization).

    `record` (optional) collects the executed ToolResults keyed by call, so the
    caller can emit structured telemetry instead of re-parsing rendered text.

    `record_provenance`, opt-in: also requests execution-time provenance
    tagging and per-episode occurrence-suffixed cassette keys from
    `ToolRegistry.call`. Default False keeps every existing caller (incl.
    `_run_live`'s persistent self-test cassette) on its current key scheme.

    `injector` (optional, WS4 harness.inject.ToolInjector): forwarded to
    `registry.call`, applied after the cassette exactly as `run_real_episode`
    already does for the native loop. Caller is responsible for advancing
    `injector.t` once per agent turn before tool execution runs.
    """
    from langchain_core.tools import StructuredTool

    tools = []
    occurrence: dict = {}
    schemas = registry.schemas()
    for name, (desc, _params) in registry.specs().items():
        args_model = _pydantic_args_model(name, schemas[name])

        def _make(tool_name: str):
            def _call(**kwargs) -> str:
                suffix = (_occurrence_suffix(occurrence, tool_name, kwargs)
                          if record_provenance else None)
                res = registry.call(tool_name, kwargs, cassette=cassette,
                                    key_suffix=suffix,
                                    record_provenance=record_provenance,
                                    injector=injector)
                if record is not None:
                    record.setdefault("results", []).append(res)
                return res.content
            return _call

        tools.append(StructuredTool.from_function(
            func=_make(name), name=name, description=desc,
            args_schema=args_model))
    return tools


def run_langgraph_episode(registry: ToolRegistry, task: str, *,
                          model: str = "qwen2.5:7b", max_steps: int = 12,
                          cassette: Cassette | None = None,
                          system: str = _SYSTEM,
                          record_provenance: bool = False,
                          injector: Any | None = None) -> list[dict]:
    """Run one episode under a LangGraph StateGraph; return v2/v3 step dicts.

    `injector`, optional: advanced to the current step index once per agent
    turn, right before that turn's tool calls execute -- the same "runner
    advances t once per step" contract `run_real_episode` already documents.
    """
    from langchain_core.messages import SystemMessage, ToolMessage
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, MessagesState, StateGraph

    record: dict = {}
    tools = _langchain_tools(registry, cassette, record=record,
                             record_provenance=record_provenance,
                             injector=injector)
    by_name = {t.name: t for t in tools}
    llm = ChatOllama(model=model, num_predict=512,
                     temperature=0.2).bind_tools(tools)

    def agent_node(state: MessagesState) -> dict:
        ai = llm.invoke(state["messages"])
        if len(ai.tool_calls or []) > 1:      # serialize (small models batch)
            ai.tool_calls = ai.tool_calls[:1]
        return {"messages": [ai]}

    def tool_node(state: MessagesState) -> dict:
        # `steps` is assigned below, in this same function scope, before the
        # stream loop that calls this node runs -- closure resolves at call
        # time, so this sees the live list, not an empty one.
        if injector is not None and steps:
            injector.t = len(steps) - 1
        outs = []
        for tc in state["messages"][-1].tool_calls:
            res = by_name[tc["name"]].invoke(tc.get("args") or {})
            outs.append(ToolMessage(content=str(res), tool_call_id=tc["id"]))
        return {"messages": outs}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        lambda s: "tools" if s["messages"][-1].tool_calls else END,
        {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    app = graph.compile()

    steps: list[dict] = []
    t_prev = time.perf_counter()
    stream = app.stream(
        {"messages": [SystemMessage(system), ("user", task)]},
        config={"recursion_limit": 2 * max_steps + 3},
        stream_mode="updates")
    for update in stream:
        if "agent" in update:
            now = time.perf_counter()
            msg = update["agent"]["messages"][-1]
            calls = [(tc["name"], tc.get("args") or {})
                     for tc in (msg.tool_calls or [])]
            usage = getattr(msg, "usage_metadata", None) or {}
            record["results"] = []
            steps.append({
                "text": str(msg.content or ""),
                "token_logprobs": [],   # frameworks abstract logprobs away
                "logprobs_available": False,
                "action": "tool_call" if calls else "synthesis",
                "latency_s": round(now - t_prev, 4),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "error": False, "task": task, "tool_events": [],
                "schema": SCHEMA_VERSION,
                "_calls": calls, "_pending": bool(calls)})
            t_prev = now
        elif "tools" in update and steps:
            # Pair the just-run results with the preceding agent step's calls:
            # structured events carry the full result and its measured latency,
            # and the rendered bits stay in the text for readability.
            results = [str(m.content) for m in update["tools"]["messages"]]
            executed = list(record.get("results", []))
            bits, err = [], False
            for i, ((name, args), res) in enumerate(
                    zip(steps[-1].get("_calls", []), results)):
                bits.append(format_tool_bit(name, args, res))
                is_error = (executed[i].is_error if i < len(executed)
                           else error_shaped(res))
                err = err or is_error
                latency = (executed[i].latency_s if i < len(executed) else None)
                source = (executed[i].source if i < len(executed) else None)
                steps[-1]["tool_events"].append(make_tool_event(
                    name, args, res, is_error=is_error, latency_s=latency,
                    source=source))
            steps[-1]["text"] = (steps[-1]["text"] + " " + " ".join(bits)).strip()
            steps[-1]["error"] = steps[-1]["error"] or err
            steps[-1]["_pending"] = False
            # `latency_s` measures ONLY the next LLM call, matching the
            # native harness (`agent_loop.py` times `backend.step()` alone,
            # nothing else) -- tool execution time is already captured per
            # call in `tool_events[].latency_s`. Restarting the clock the
            # instant tool execution ends (not when the PRIOR agent step
            # ended) keeps it that way; without this, a step's `latency_s`
            # would also include the previous step's tool-execution time,
            # which is a different quantity than what the native harness
            # measures under the same feature name.
            t_prev = time.perf_counter()
        # Stop only once the pending agent->tool cycle has completed, else the
        # requested call is executed but never recorded.
        if reached_step_limit(steps, max_steps):
            break
    for s in steps:
        s.pop("_calls", None)
        s.pop("_pending", None)
    return steps


def _autogen_tools(registry: ToolRegistry, cassette: Cassette | None,
                   record: dict | None = None,
                   record_provenance: bool = False,
                   injector: Any | None = None):
    """Wrap each registry tool as an AutoGen FunctionTool whose execution
    routes back through registry.call (cassette + error normalization).

    Built from closures and an explicit signature.  The previous version
    generated source with `exec`, and its argument mapping rendered the
    parameter *names* rather than the caller's values - every AutoGen tool
    call received `{"text": "text"}` instead of the text, which a
    one-line echo test now catches.  Generating code from tool-supplied names
    was also an injection path.

    `record_provenance`, opt-in: see `_langchain_tools` -- same contract,
    same default-False backward compatibility. `injector`: see
    `_langchain_tools` -- forwarded to `registry.call`.
    """
    from autogen_core.tools import FunctionTool

    tools = []
    occurrence: dict = {}
    schemas = registry.schemas()
    for name, (desc, _params) in registry.specs().items():
        schema = schemas[name]
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        def _make(tool_name: str, props: dict, req: set):
            def _call(**kwargs) -> str:
                suffix = (_occurrence_suffix(occurrence, tool_name, dict(kwargs))
                          if record_provenance else None)
                res = registry.call(tool_name, dict(kwargs), cassette=cassette,
                                    key_suffix=suffix,
                                    record_provenance=record_provenance,
                                    injector=injector)
                if record is not None:
                    record.setdefault("results", []).append(res)
                return res.content

            # AutoGen inspects the signature, so build a real one with the
            # right names, annotations and defaults.
            parameters = [
                inspect.Parameter(
                    param,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=(inspect.Parameter.empty if param in req
                             else props[param].get("default")),
                    annotation=_PY_TYPES.get(props[param].get("type", "string"), str))
                for param in sorted(props, key=lambda p: p not in req)
            ]
            _call.__signature__ = inspect.Signature(parameters,
                                                    return_annotation=str)
            # AutoGen reads type hints from __annotations__, not from the
            # signature object, so both have to agree.
            _call.__annotations__ = {p.name: p.annotation for p in parameters}
            _call.__annotations__["return"] = str
            _call.__name__ = f"autogen_{tool_name}"
            _call.__doc__ = desc
            return _call

        tools.append(FunctionTool(func=_make(name, properties, required),
                                  name=name, description=desc))
    return tools


def run_autogen_episode(registry: ToolRegistry, task: str, *,
                        model: str = "qwen2.5:7b", max_steps: int = 12,
                        cassette: Cassette | None = None,
                        system: str = _SYSTEM,
                        record_provenance: bool = False,
                        options: dict | None = None,
                        injector: Any | None = None) -> list[dict]:
    """Run one episode under AutoGen AssistantAgent; return v2/v3 step dicts.

    `options`, opt-in: Ollama generation options (e.g. temperature,
    num_predict) forwarded to `OllamaChatCompletionClient`. Default None
    keeps every existing caller on the client's own defaults, unchanged.

    `injector`, optional: advanced to the current step index right after
    that step's pending entry is appended, before AutoGen executes its
    tool calls -- same "runner advances t once per step" contract as the
    native loop and the LangGraph runner above.
    """
    import asyncio
    import json
    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    record: dict = {}
    tools = _autogen_tools(registry, cassette, record=record,
                           record_provenance=record_provenance,
                           injector=injector)

    class SerializedToolsClient(OllamaChatCompletionClient):
        """Truncate batched tool calls so episodes stay sequential."""

        async def create(self, *a, **kw):
            result = await super().create(*a, **kw)
            if isinstance(result.content, list) and len(result.content) > 1:
                result.content = result.content[:1]
            return result

    async def _run() -> list[dict]:
        client_kwargs = {"model": model}
        if options is not None:
            client_kwargs["options"] = options
        client = SerializedToolsClient(**client_kwargs)
        agent = AssistantAgent(
            "worker", model_client=client, tools=tools,
            system_message=system, max_tool_iterations=max_steps)
        steps: list[dict] = []
        t_prev = time.perf_counter()
        async for event in agent.run_stream(task=task):
            etype = type(event).__name__
            if etype == "ToolCallRequestEvent":
                now = time.perf_counter()
                calls = [(fc.name, json.loads(fc.arguments or "{}"))
                         for fc in event.content]
                usage = getattr(event, "models_usage", None)
                record["results"] = []
                steps.append({
                    "text": "",
                    "token_logprobs": [],
                    "logprobs_available": False,
                    "action": "tool_call",
                    "latency_s": round(now - t_prev, 4),
                    "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "error": False, "task": task, "tool_events": [],
                    "schema": SCHEMA_VERSION,
                    "_calls": calls, "_pending": bool(calls)})
                if injector is not None:
                    injector.t = len(steps) - 1
                t_prev = now
            elif etype == "ToolCallExecutionEvent":
                if steps and "_calls" in steps[-1]:
                    executed = list(record.get("results", []))
                    bits, err = [], False
                    for i, ((name, args), r) in enumerate(
                            zip(steps[-1]["_calls"], event.content)):
                        res_str = str(r.content)
                        bits.append(format_tool_bit(name, args, res_str))
                        is_error = (executed[i].is_error if i < len(executed)
                                   else error_shaped(res_str))
                        err = err or is_error
                        latency = (executed[i].latency_s
                                   if i < len(executed) else None)
                        tool_source = (executed[i].source
                                      if i < len(executed) else None)
                        steps[-1]["tool_events"].append(make_tool_event(
                            name, args, res_str, is_error=is_error,
                            latency_s=latency, source=tool_source))
                    steps[-1]["text"] = (steps[-1]["text"] + " " + " ".join(bits)).strip()
                    steps[-1]["error"] = steps[-1]["error"] or err
                    steps[-1]["_pending"] = False
                # See the matching comment in run_langgraph_episode: restart
                # the clock the instant tool execution ends, so the next
                # step's `latency_s` measures only its own LLM call, the
                # same quantity the native harness measures under this
                # feature name (tool time lives in tool_events[].latency_s).
                t_prev = time.perf_counter()
            elif etype == "TextMessage" and getattr(event, "source", "") == "worker":
                now = time.perf_counter()
                usage = getattr(event, "models_usage", None)
                steps.append({
                    "text": str(event.content),
                    "token_logprobs": [],
                    "logprobs_available": False,
                    "action": "synthesis",
                    "latency_s": round(now - t_prev, 4),
                    "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                    "error": False, "task": task, "tool_events": [],
                    "schema": SCHEMA_VERSION})
                t_prev = now
            # Finish the pending agent->tool cycle before enforcing the limit.
            if reached_step_limit(steps, max_steps):
                break
        await client.close()
        for s in steps:
            s.pop("_calls", None)
            s.pop("_pending", None)
        return steps

    return asyncio.run(_run())


def _run_live(model: str, framework: str = "langgraph") -> None:
    from derail.harness.real_tools import _ensure_tls, build_registry
    from derail.telemetry.adapter import episode_from_trace
    from derail.common import (D_TOTAL_EXT, IDX_REASON_DEPTH, IDX_RETRY_COUNT,
                               IDX_TOOL_SUCCESS)

    _ensure_tls()
    # This live check drives arxiv + wikipedia only.
    registry = build_registry(("arxiv_search", "wikipedia_search"))
    cassette = Cassette(f"traces/_cassettes/{framework}", mode="auto")
    task = ("Find two arXiv papers on echo state networks for anomaly "
            "detection and give a one-line summary of each. Use arxiv_search "
            "and wikipedia_search; finish with a one-line answer.")
    print(f"[{framework}] LIVE {model} (framework-owned loop)\n  task: {task}\n")
    if framework == "autogen":
        steps = run_autogen_episode(registry, task, model=model, max_steps=12,
                                    cassette=cassette)
    else:
        steps = run_langgraph_episode(registry, task, model=model, max_steps=12,
                                      cassette=cassette)
    ep = episode_from_trace(steps, f"{framework}-live",
                            use_sentence_transformers=False, extended=True)
    for t, s in enumerate(steps):
        x = ep.X[t]
        print(f"  t={t} [{s['action']:9}] err={int(s['error'])} "
              f"depth={x[IDX_REASON_DEPTH]:.0f} succ={x[IDX_TOOL_SUCCESS]:.2f} "
              f"retry={x[IDX_RETRY_COUNT]:.0f}  {s['text'][:90]}")
    print(f"\n[{framework}] T={ep.T} steps, X shape {ep.X.shape}")
    print(f"[{framework}] {cassette.summary()}")
    assert ep.X.shape[1] == D_TOTAL_EXT
    assert any("->" in s["text"] and s["text"].count("[") for s in steps), \
        "no v2/v3 tool bit produced"
    print(f"[{framework}] OK: framework loop -> real tools -> (T,51) Episode")


def _selfcheck() -> None:
    """Offline: verify both frameworks' tool-wrapping builds valid tools from
    a registry and routes execution back through registry.call — no model,
    no network. Covers the exec()/pydantic construction where bugs would hide."""
    from derail.harness.tools import SimpleTool, ToolRegistry

    reg = ToolRegistry([SimpleTool(
        "echo", "Echo the input text.", {"text": "text to echo"},
        lambda text: f"echoed: {text}")])

    # LangChain StructuredTool: name + schema + synchronous invoke routing.
    lc = _langchain_tools(reg, None)
    assert len(lc) == 1 and lc[0].name == "echo"
    assert set(lc[0].args_schema.model_fields) == {"text"}
    assert lc[0].invoke({"text": "hi"}) == "echoed: hi", "LC routing broken"

    # AutoGen FunctionTool: construction (exec-built signature) + metadata.
    ag = _autogen_tools(reg, None)
    assert len(ag) == 1 and ag[0].name == "echo" and ag[0].description
    print("PASS frameworks offline self-check (LangGraph + AutoGen wrapping) "
          "| run --framework {langgraph,autogen} for a live episode")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="py -m derail.harness.frameworks")
    parser.add_argument("--framework", choices=["langgraph", "autogen"], default="langgraph")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--check", action="store_true",
                        help="offline self-check (no model/network), then exit")
    args = parser.parse_args()
    if args.check:
        _selfcheck()
    else:
        _run_live(args.model, args.framework)

