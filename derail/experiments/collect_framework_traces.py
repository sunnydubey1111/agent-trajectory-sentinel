"""Collect traces from REAL agent frameworks (LangGraph, AutoGen) on Ollama.

External validation across orchestrators: the same mock-tool task suite and
the same telemetry schema as collect_traces.py, but the agent loop is owned
by a real framework — LangGraph (create_react_agent) or AutoGen
(AssistantAgent) — driving a local Ollama model. No API costs.

Injection is TOOL-LAYER only, which is framework-agnostic (the tools are
ours, whoever orchestrates):

  tool_cascade        tools return errors with ramping probability from tau
  looping             the tool returns non-progress "retry" text from tau
  context_corruption  tool results are garbled at the source from tau on
                      (the agent's context accumulates corrupt data)

goal_drift needs history rewriting inside the framework's state and is
deliberately out of scope here — it is covered by the gemini/ollama
collectors in collect_traces.py, which own their message history.

CrewAI is not included: it does not install on Python 3.14 (dependency
build failure); it needs Python <= 3.12. Documented as environment-blocked.

Usage (Ollama server running, model pulled, e.g. qwen2.5:7b):

  py -m derail.experiments.collect_framework_traces --framework langgraph
  py -m derail.experiments.collect_framework_traces --framework autogen
  ...then evaluate:
  py -m derail.experiments.run_real_traces --dir traces/langgraph
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from derail.common import rng_for, stable_hash
from derail.harness.collection import (ModelUnavailable, Provenance,
                                       _sha256_text, accept_episode,
                                       require_ollama_model, reusable,
                                       write_episode, write_manifest)
from derail.harness.frameworks import reached_step_limit
from derail.harness.tools import format_tool_bit
from derail.telemetry.events import SCHEMA_VERSION, make_tool_event
from derail.experiments.collect_traces import (
    MAX_STEPS,
    TOOL_SPECS,
    _make_task,
    _make_world,
    _run_tool,
)

TRACES_ROOT = Path(__file__).resolve().parents[2] / "traces"
# qwen2.5:3b was the previous default and is no longer pulled on the collection
# machine. The 3b corpora it produced stay frozen and still reproduce; new
# collection defaults to the surviving family model.
MODEL_DEFAULT = "qwen2.5:7b"
INJECT_CLASSES = ("tool_cascade", "looping", "context_corruption")
SYSTEM_TEXT = ("Solve the user's task using the tools. Call at most ONE "
               "tool per message, then wait for its result. Be brief; give "
               "a one-line final answer.")


# --------------------------------------------------------- tool-layer layer
class ToolLayerInjection:
    """Framework-agnostic injection: manipulates tool RESULTS from tau on.

    The adapter advances `self.t` before each model call; the tool closures
    consult it, so tau is ground truth no matter who runs the loop.
    """

    def __init__(self, failure_class: str | None, tau: int | None, rng) -> None:
        if failure_class is not None and failure_class not in INJECT_CLASSES:
            raise ValueError(f"{failure_class!r} is not injectable here; "
                             f"known: {INJECT_CLASSES}")
        self.failure_class = failure_class
        self.tau = tau
        self.rng = rng
        self.t = 0
        self.error_this_step = False
        # What actually happened, so an episode is never labelled on intent
        # alone.
        self.applied_count = 0
        self.first_applied_t: int | None = None
        self.applied_tools: list[str] = []

    def _record(self, name: str) -> None:
        self.applied_count += 1
        if self.first_applied_t is None:
            self.first_applied_t = self.t
        self.applied_tools.append(name)

    def transform(self, name: str, result: str) -> str:
        is_error = result.startswith("Error:")
        if (self.failure_class is not None and self.tau is not None
                and self.t >= self.tau):
            if self.failure_class == "tool_cascade":
                p = min(0.9, 0.4 + 0.15 * (self.t - self.tau))
                if self.rng.random() < p:
                    result = f"Error: {name} service unavailable (HTTP 503)."
                    is_error = True
                    self._record(name)
            elif self.failure_class == "looping":
                result = ("Temporary data inconsistency detected — please "
                          "retry the exact same query to confirm the value.")
                self._record(name)
            elif self.failure_class == "context_corruption":
                words = result.split()
                self.rng.shuffle(words)
                result = " ".join(words) + f" ${int(self.rng.integers(1, 999))}"
                self._record(name)
        self.error_this_step = self.error_this_step or is_error
        return result


def _make_tool_fns(world: dict, inj: ToolLayerInjection) -> dict:
    """Five plainly-typed closures (frameworks infer schemas from these)."""

    def lookup_flight(origin: str, destination: str) -> str:
        """Get the one-way flight price in USD between two cities."""
        return inj.transform("lookup_flight", _run_tool(
            "lookup_flight", {"origin": origin, "destination": destination},
            world))

    def lookup_hotel(city: str) -> str:
        """Get the nightly hotel price in USD for a city."""
        return inj.transform("lookup_hotel",
                             _run_tool("lookup_hotel", {"city": city}, world))

    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return inj.transform("get_weather",
                             _run_tool("get_weather", {"city": city}, world))

    def search_catalog(item: str) -> str:
        """Get the price in USD of a catalog item (item-1..item-8)."""
        return inj.transform("search_catalog",
                             _run_tool("search_catalog", {"item": item}, world))

    def calculator(expression: str) -> str:
        """Evaluate a basic arithmetic expression (+ - * / and parentheses)."""
        return inj.transform("calculator", _run_tool(
            "calculator", {"expression": expression}, world))

    return {"lookup_flight": lookup_flight, "lookup_hotel": lookup_hotel,
            "get_weather": get_weather, "search_catalog": search_catalog,
            "calculator": calculator}


def _step_record(text: str, tool_calls: list[tuple[str, dict]], action: str,
                 latency: float, out_tokens: int, error: bool,
                 task: str = "") -> dict:
    return {"text": text.strip(),
            "token_logprobs": [],   # not exposed via the framework bindings
            "logprobs_available": False,
            "action": action, "latency_s": round(latency, 4),
            "output_tokens": out_tokens, "error": error,
            "task": task, "tool_events": [], "schema": SCHEMA_VERSION,
            "_calls": list(tool_calls), "_pending": bool(tool_calls)}


def _attach_results(step: dict, results: list[str],
                    errors: list[bool] | None = None) -> None:
    """Fold executed tool results into the step as structured events.

    The old collector appended "-> result" *outside* the bracket, a form the
    adapter never parsed, so grounding features were absent from every
    framework corpus.  Results are now data first; the rendered bit
    uses the canonical in-bracket form for readability.
    """
    bits = []
    for i, ((name, args), result) in enumerate(zip(step.get("_calls", []),
                                                   results)):
        is_error = (bool(errors[i]) if errors and i < len(errors)
                    else str(result).lstrip().lower().startswith("error"))
        step["tool_events"].append(
            make_tool_event(name, args, result, is_error=is_error))
        bits.append(format_tool_bit(name, args, result))
        step["error"] = step["error"] or is_error
    step["text"] = (step["text"] + " " + " ".join(bits)).strip()
    step["_pending"] = False


# ------------------------------------------------------- LangGraph adapter
def run_langgraph_episode(model: str, seed: int,
                          inj: ToolLayerInjection) -> list[dict]:
    """Explicit LangGraph StateGraph: agent node <-> tool node.

    Owning the graph (instead of the prebuilt ReAct agent) lets us serialize
    tool calls — small local models batch calls regardless of instructions,
    which collapses episodes below the monitor's washout.
    """
    from langchain_core.messages import SystemMessage, ToolMessage
    from langchain_core.tools import StructuredTool
    from langchain_ollama import ChatOllama
    from langgraph.graph import END, START, MessagesState, StateGraph

    world = _make_world(seed)
    task, _ = _make_task(seed, world)
    fns = _make_tool_fns(world, inj)
    tools = [StructuredTool.from_function(func=f, name=n,
                                          description=TOOL_SPECS[n][0])
             for n, f in fns.items()]
    llm = ChatOllama(model=model, num_predict=512).bind_tools(tools)

    def agent_node(state: MessagesState) -> dict:
        ai = llm.invoke(state["messages"])
        if len(ai.tool_calls or []) > 1:
            ai.tool_calls = ai.tool_calls[:1]   # serialize tools
        return {"messages": [ai]}

    def tool_node(state: MessagesState) -> dict:
        outs = []
        for tc in state["messages"][-1].tool_calls:
            try:
                res = fns[tc["name"]](**(tc.get("args") or {}))
            except Exception as exc:  # noqa: BLE001 — surfaced to the agent
                res = f"Error: {exc}"
            outs.append(ToolMessage(content=str(res),
                                    tool_call_id=tc["id"]))
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
        {"messages": [SystemMessage(SYSTEM_TEXT), ("user", task)]},
        config={"recursion_limit": 2 * MAX_STEPS + 3},
        stream_mode="updates")
    for update in stream:
        if "agent" in update:
            now = time.perf_counter()
            msg = update["agent"]["messages"][-1]
            calls = [(tc["name"], tc.get("args") or {})
                     for tc in (msg.tool_calls or [])]
            usage = getattr(msg, "usage_metadata", None) or {}
            steps.append(_step_record(
                text=str(msg.content or ""),
                tool_calls=calls,
                action="tool_call" if calls else "synthesis",
                latency=now - t_prev,
                out_tokens=int(usage.get("output_tokens") or 0),
                error=False, task=task))
            t_prev = now
            inj.error_this_step = False
            # The injector's clock must equal the index of the step being
            # built, not one past it: the tool result is folded back into THIS
            # step, so an incremented clock recorded an onset one step later
            # than the mutation actually landed.
            inj.t = len(steps) - 1
        elif "tools" in update:
            if steps:
                results = [str(m.content) for m in update["tools"]["messages"]]
                errors = [inj.error_this_step] * len(results)
                _attach_results(steps[-1], results, errors)
        # Finish the pending agent->tool cycle before stopping.
        if reached_step_limit(steps, MAX_STEPS):
            break
    for s in steps:
        s.pop("_calls", None)
        s.pop("_pending", None)
    return steps


# --------------------------------------------------------- AutoGen adapter
def run_autogen_episode(model: str, seed: int,
                        inj: ToolLayerInjection) -> list[dict]:
    import asyncio

    from autogen_agentchat.agents import AssistantAgent
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    world = _make_world(seed)
    task, _ = _make_task(seed, world)
    fns = _make_tool_fns(world, inj)

    class SerializedToolsClient(OllamaChatCompletionClient):
        """Truncate batched tool calls so episodes stay sequential."""

        async def create(self, *a, **kw):
            result = await super().create(*a, **kw)
            if isinstance(result.content, list) and len(result.content) > 1:
                result.content = result.content[:1]
            return result

    async def _run() -> list[dict]:
        client = SerializedToolsClient(model=model)
        agent = AssistantAgent(
            "worker", model_client=client, tools=list(fns.values()),
            system_message=SYSTEM_TEXT, max_tool_iterations=MAX_STEPS)
        steps: list[dict] = []
        t_prev = time.perf_counter()
        async for event in agent.run_stream(task=task):
            etype = type(event).__name__
            if etype == "ToolCallRequestEvent":
                now = time.perf_counter()
                calls = [(fc.name, json.loads(fc.arguments or "{}"))
                         for fc in event.content]
                usage = getattr(event, "models_usage", None)
                steps.append(_step_record(
                    "", calls, "tool_call", now - t_prev,
                    int(getattr(usage, "completion_tokens", 0) or 0), False,
                    task=task))
                t_prev = now
                inj.error_this_step = False
                inj.t = len(steps) - 1      # see the LangGraph note above
            elif etype == "ToolCallExecutionEvent":
                if steps:
                    results = [str(r.content) for r in event.content]
                    errors = [bool(getattr(r, "is_error", False))
                              or inj.error_this_step for r in event.content]
                    _attach_results(steps[-1], results, errors)
            elif etype == "TextMessage" and getattr(event, "source", "") == "worker":
                now = time.perf_counter()
                usage = getattr(event, "models_usage", None)
                steps.append(_step_record(
                    str(event.content), [], "synthesis", now - t_prev,
                    int(getattr(usage, "completion_tokens", 0) or 0), False,
                    task=task))
                t_prev = now
                inj.t = len(steps) - 1
            # Finish the pending agent->tool cycle before stopping.
            if reached_step_limit(steps, MAX_STEPS):
                break
        await client.close()
        for s in steps:
            s.pop("_calls", None)
            s.pop("_pending", None)
        return steps

    return asyncio.run(_run())


ADAPTERS = {"langgraph": run_langgraph_episode, "autogen": run_autogen_episode}


# ------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.collect_framework_traces")
    parser.add_argument("--framework", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--healthy", type=int, default=24)
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", default=None,
                        help="trace output dir (default: traces/<framework>). "
                             "Use a fresh dir per agent model — healthy "
                             "distributions must not be mixed.")
    args = parser.parse_args(argv)

    run_episode = ADAPTERS[args.framework]
    # Preflight before touching the output directory: a missing model must not
    # cost an existing corpus or an hour of failing episodes.
    try:
        require_ollama_model(args.model)
    except ModelUnavailable as exc:
        raise SystemExit(f"[collect:{args.framework}] {exc}")
    out_dir = Path(args.out_dir) if args.out_dir else TRACES_ROOT / args.framework
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[collect:{args.framework}] local model {args.model} via Ollama — "
          f"no API cost; {args.healthy} healthy + "
          f"{args.per_class}x{len(INJECT_CLASSES)} injected")

    manifest: list[dict] = []
    rejected: list[dict] = []
    previous = {}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        previous = {e["episode_id"]: e
                    for e in json.loads(manifest_path.read_text("utf-8"))}

    # Healthy episode i and injected episode i share the task index, so the
    # classes are counterfactual pairs on the same task.
    plan = ([("healthy", None, i) for i in range(args.healthy)]
            + [(fc, fc, i) for fc in INJECT_CLASSES
               for i in range(args.per_class)])
    for kind, fc, i in plan:
        seed = args.seed * 1000 + stable_hash(args.framework, "task", i) % 100000
        episode_id = f"{args.framework}-{kind}-{i:03d}"
        rng = rng_for(args.seed, "inject", episode_id)
        tau = None if fc is None else int(rng.integers(2, 4))
        task_text, _ = _make_task(seed, _make_world(seed))
        provenance = Provenance(
            collector="collect_framework_traces", backend=args.framework,
            model=args.model, temperature=None, episode_seed=seed,
            task_name=f"task-{i}", task_sha256=_sha256_text(task_text),
            tools=tuple(sorted(TOOL_SPECS)),
            tool_roster_sha256=_sha256_text(json.dumps(sorted(TOOL_SPECS))),
            requested_class=fc, requested_tau=tau,
            injector_seed=args.seed)

        if args.resume:
            ok, why = reusable(out_dir, previous.get(episode_id), provenance)
            if ok:
                manifest.append(previous[episode_id])
                print(f"  [resume] {episode_id}: unchanged")
                continue
            if previous.get(episode_id):
                print(f"  [resume] {episode_id}: re-collecting ({why})")

        injector = ToolLayerInjection(fc, tau, rng)
        try:
            steps = run_episode(args.model, seed, injector)
        except Exception as exc:  # noqa: BLE001 — skip episode, keep batch
            print(f"  [error] {episode_id}: {type(exc).__name__}: {exc}")
            continue
        verdict = accept_episode(steps, injector=injector, min_steps=4)
        if not verdict.accepted:
            rejected.append({"episode_id": episode_id, "requested_class": fc,
                             "reason": verdict.reason, "facts": verdict.facts})
            print(f"  [reject] {episode_id}: {verdict.reason}")
            continue
        entry = write_episode(out_dir, episode_id, steps, provenance, verdict,
                              extra={"framework": args.framework})
        manifest.append(entry)
        write_manifest(out_dir, manifest)
        print(f"  [ok] {episode_id}: T={len(steps)}"
              + (f" onset={verdict.tau} ({fc})" if fc else ""))

    write_manifest(out_dir, manifest)
    if rejected:
        (out_dir / "rejected.json").write_text(json.dumps(rejected, indent=2),
                                               "utf-8")
    print(f"[collect:{args.framework}] wrote {len(manifest)} traces "
          f"({len(rejected)} rejected) to {out_dir}")


if __name__ == "__main__":
    main()
