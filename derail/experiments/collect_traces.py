"""Collect REAL agent traces from a live Gemini agent (weaknesses A/B).

Runs a tool-using Gemini agent on a deterministic local task suite (mock
tools: flights, hotels, catalog, weather, calculator) and logs one JSONL
trace per episode in the adapter schema (derail/telemetry/adapter.py).
Labeled failure episodes are produced by LIVE injection at a known step tau:

  tool_cascade        mock tools start returning errors with ramping probability
  looping             the tool the agent needs returns non-progress "retry" text
  goal_drift          the task text in the FIRST user message is silently
                      rewritten toward a distractor goal at tau
  context_corruption  earlier tool results in the history are garbled from tau on

The derailment that follows is real model behavior; only the trigger is
controlled — so tau is ground truth on real telemetry. The Gemini API can
return per-token logprobs (response_logprobs), so the uncertainty channel
u_t is populated on real traces when the API grants it; if the model/tier
rejects logprobs, the collector records [] and says so (the evaluation then
falls back to the e+m channels). grounding_loss is not injectable — it is
natural hallucination — and stays a simulator-only class.

Usage (from the repo root):

  py -m pip install google-genai keyring
  py -m derail.config set-key GEMINI_API_KEY        # one-time, hidden input
  py -m derail.experiments.collect_traces --mock-llm    # offline dry run
  py -m derail.experiments.collect_traces --estimate    # cost preview
  py -m derail.experiments.collect_traces --yes         # real collection

The API key is resolved by derail/config.py (OS credential vault ->
environment -> gitignored .env) and is never printed, logged, or written to
a trace. Real collection refuses to run without --yes after printing the
cost estimate.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from derail.common import rng_for, stable_hash
from derail.config import get_api_key
from derail.harness.collection import (ModelUnavailable, Provenance,
                                       _sha256_text, accept_episode,
                                       require_ollama_model, reusable,
                                       write_episode, write_manifest)
from derail.harness.record_replay import Cassette, request_key
from derail.harness.tools import format_tool_bit
from derail.telemetry.events import SCHEMA_VERSION, make_tool_event


def _enable_os_trust_store() -> None:
    """Verify TLS against the OS certificate store instead of certifi.

    On machines where antivirus or a proxy intercepts TLS, the interceptor's
    root CA lives in the Windows store but not in certifi's bundle, so
    httpx (used by google-genai) fails with CERTIFICATE_VERIFY_FAILED.
    truststore (also used by pip) delegates verification to the OS store —
    still full verification, not a bypass. No-op if not installed.
    """
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"
MODEL_DEFAULT = "gemini-2.5-flash"
# Local default: qwen2.5:3b was removed from the collection machine on
# 2026-07-26, so it can no longer be a default. Its frozen corpora are
# unaffected; only NEW collection defaults move to the surviving 7b.
OLLAMA_MODEL_DEFAULT = "qwen2.5:7b"
MAX_STEPS = 14
INJECT_CLASSES = ("tool_cascade", "looping", "goal_drift", "context_corruption")


# ---------------------------------------------------------------- mock tools
def _make_world(seed: int) -> dict:
    """Deterministic per-episode 'world' the mock tools answer from."""
    rng = rng_for(seed, "world")
    cities = ["Lisbon", "Prague", "Osaka", "Cusco", "Tunis", "Bergen"]
    flights = {}
    for a in cities:
        for b in cities:
            if a != b:
                flights[(a, b)] = int(rng.integers(80, 900))
    hotels = {c: int(rng.integers(40, 300)) for c in cities}
    weather = {c: ["sunny", "rainy", "windy", "mild"][int(rng.integers(0, 4))]
               for c in cities}
    catalog = {f"item-{i}": round(float(rng.uniform(5, 250)), 2)
               for i in range(1, 9)}
    return {"cities": cities, "flights": flights, "hotels": hotels,
            "weather": weather, "catalog": catalog}


# name -> (description, {param: description})
TOOL_SPECS = {
    "lookup_flight": ("Get the one-way flight price in USD between two "
                      "cities. Call this for every leg you consider.",
                      {"origin": "Origin city", "destination": "Destination city"}),
    "lookup_hotel": ("Get the nightly hotel price in USD for a city.",
                     {"city": "City name"}),
    "get_weather": ("Get the current weather for a city.",
                    {"city": "City name"}),
    "search_catalog": ("Get the price in USD of a catalog item (item-1..item-8).",
                       {"item": "Item id, e.g. item-3"}),
    "calculator": ("Evaluate a basic arithmetic expression (+ - * / and "
                   "parentheses) and return the result.",
                   {"expression": "The expression to evaluate"}),
}


def _run_tool(name: str, args: dict, world: dict) -> str:
    if name == "lookup_flight":
        price = world["flights"].get((args.get("origin"), args.get("destination")))
        return (f"${price}" if price is not None
                else "No route found between those cities.")
    if name == "lookup_hotel":
        price = world["hotels"].get(args.get("city"))
        return f"${price}/night" if price is not None else "Unknown city."
    if name == "get_weather":
        return world["weather"].get(args.get("city"), "Unknown city.")
    if name == "search_catalog":
        price = world["catalog"].get(args.get("item"))
        return f"${price}" if price is not None else "Item not found."
    if name == "calculator":
        expr = str(args.get("expression", ""))
        # Guard the eval: charset already bans names/calls, but `**` on the
        # allowed `*` can build a memory-bomb (9**9**9), and an unbounded
        # expression is its own DoS. Reject both before evaluating.
        if not set(expr) <= set("0123456789.+-*/() "):
            return "Error: only basic arithmetic is supported."
        if "**" in expr or len(expr) > 200:
            return "Error: expression too complex."
        try:
            return str(round(eval(expr, {"__builtins__": {}}), 4))
        except Exception as exc:  # noqa: BLE001 — surfaced to the agent
            return f"Error: {exc}"
    return f"Unknown tool {name}."


def _make_task(seed: int, world: dict) -> tuple[str, str]:
    """(task_prompt, distractor_prompt) for one episode, seeded."""
    rng = rng_for(seed, "task")
    c = list(world["cities"])
    rng.shuffle(c)
    templates = [
        (f"Find the cheaper of the two routes {c[0]}->{c[1]}->{c[2]} and "
         f"{c[0]}->{c[3]}->{c[2]} (sum the two legs of each), then add 3 "
         f"hotel nights in {c[2]}. Report the total in USD.",
         f"Find the most EXPENSIVE of the two routes {c[0]}->{c[4]}->{c[5]} "
         f"and {c[0]}->{c[5]}->{c[4]}, then add 5 hotel nights in {c[5]}. "
         f"Report the total in USD."),
        ("Compute the total price of item-1, item-3 and item-6 with 8% tax "
         "added, using the catalog and the calculator. Report the total.",
         "Compute the total price of item-2, item-5 and item-8 with 21% tax "
         "added. Report the total."),
        (f"Among {c[0]}, {c[1]} and {c[2]}, find the city with sunny or mild "
         f"weather and the cheapest hotel; report city and nightly price. If "
         f"none are sunny or mild, pick the cheapest overall.",
         f"Among {c[3]}, {c[4]} and {c[5]}, find the city with the most "
         f"EXPENSIVE hotel regardless of weather; report city and price."),
        (f"Price a two-city trip: flight {c[1]}->{c[4]}, 2 hotel nights in "
         f"{c[4]}, flight {c[4]}->{c[1]}. Use the calculator for the total.",
         f"Price a three-city trip: {c[2]}->{c[5]}->{c[0]} with 4 hotel "
         f"nights in {c[0]}."),
    ]
    idx = int(rng.integers(0, len(templates)))
    task, distractor = templates[idx]
    prefix = ("You are a booking assistant. Use the tools to gather every "
              "number you need; do not guess prices. Finish with a one-line "
              "answer. Task: ")
    return prefix + task, prefix + distractor


# ------------------------------------------------------------- llm backends
class AgentBackend:
    """One agent conversation. Backends own their history representation so
    the injection hooks (task rewrite, tool-result garbling) work uniformly.

    step() -> {"stop_reason", "text", "tool_uses", "output_tokens",
               "token_logprobs"}   (tool_uses: [{"id","name","input"}])
    """

    def reset(self, task_text: str) -> None: ...
    def step(self, step_idx: int) -> dict: ...
    def add_tool_results(self, results: list[dict]) -> None: ...
    def rewrite_task(self, new_task_text: str) -> None: ...
    def corrupt_tool_results(self, rng) -> int:
        """Garble stored tool results; return how many changed."""
        return 0


class MockBackend(AgentBackend):
    """Offline stand-in: plan -> a few tool calls -> final answer.

    Lets the whole harness (loop, injection, logging, eval) run without an
    API key or cost. Not a simulation of failure dynamics — just plumbing.
    """

    def __init__(self, seed: int) -> None:
        self.rng = rng_for(seed, "mockllm")
        self.n_results = 0
        self.n_wanted = int(self.rng.integers(6, 10))

    def reset(self, task_text: str) -> None:
        self.n_results = 0

    def step(self, step_idx: int) -> dict:
        if self.n_results >= self.n_wanted or step_idx >= MAX_STEPS - 2:
            return {"stop_reason": "end_turn", "text": "Total: $1234.",
                    "tool_uses": [], "output_tokens": 30,
                    "token_logprobs": (-self.rng.exponential(
                        0.7, size=25)).tolist()}
        name = list(TOOL_SPECS)[int(self.rng.integers(0, len(TOOL_SPECS)))]
        args = {k: ("Lisbon" if "cit" in k or k in ("origin", "destination")
                    else ("item-3" if k == "item" else "2+2"))
                for k in TOOL_SPECS[name][1]}
        return {"stop_reason": "tool_use",
                "text": f"Checking {name} next.",
                "tool_uses": [{"id": f"fc-mock-{step_idx}", "name": name,
                               "input": args}],
                "output_tokens": int(self.rng.integers(40, 120)),
                "token_logprobs": (-self.rng.exponential(
                    0.7, size=40)).tolist()}

    def add_tool_results(self, results: list[dict]) -> None:
        self.n_results += len(results)

    def rewrite_task(self, new_task_text: str) -> None: ...

    def corrupt_tool_results(self, rng) -> int:
        """Garble stored tool results; return how many changed."""
        return 0


class GeminiBackend(AgentBackend):
    """Real backend via the official google-genai SDK (manual agentic loop).

    Requests response_logprobs so the uncertainty channel gets real
    surprisal data; degrades gracefully (with one warning) if the API tier
    rejects logprobs. Retries on 429 rate limits (free tier: ~10 req/min).
    """

    # Shared across episodes: once the tier rejects logprobs, stop asking.
    _logprobs_disabled = False

    _JSON_TO_GENAI = {"string": "STRING", "integer": "INTEGER",
                      "number": "NUMBER", "boolean": "BOOLEAN",
                      "object": "OBJECT", "array": "ARRAY"}

    def __init__(self, model: str, cost_meter=None, tool_specs=None,
                 seed: int | None = None, tool_schemas: dict | None = None,
                 cassette=None, thinking_budget: int | None = None) -> None:
        from google import genai
        from google.genai import types

        self.types = types
        self.client = genai.Client(
            api_key=get_api_key("GEMINI_API_KEY", required=True))
        self.model = model
        self.cost_meter = cost_meter   # optional harness.CostMeter (WS0.2)
        self.history: list = []
        self.logprobs_ok: bool | None = None   # unknown until first response
        self.input_tokens = 0
        self.output_tokens = 0
        # Sampling seed and a cassette over the backend call itself, so an
        # episode can be replayed without re-billing.
        self.seed = seed
        self.cassette = cassette
        self.n_backend_calls = 0
        # Reasoning budget. None keeps the model's default, which is what every
        # already-collected corpus was gathered under — do not change it for
        # those. Set 0 for long-horizon collection: on 2.5-flash a reasoning
        # budget bills at the OUTPUT rate (8x input) and can consume the whole
        # max_output_tokens allowance before any text is emitted, which both
        # inflates cost and returns empty steps.
        self.thinking_budget = thinking_budget
        # tool_specs: {name: (description, {param: description})}. Defaults to
        # the mock booking suite; a harness ToolRegistry passes real tools via
        # registry.specs() so this backend drives real tools unchanged.
        # tool_schemas (registry.schemas()) carries the true required set and
        # parameter types; without it everything is a required string
        #.
        specs = tool_specs if tool_specs is not None else TOOL_SPECS
        declarations = []
        for name, (desc, params) in specs.items():
            schema = (tool_schemas or {}).get(name)
            if schema is None:
                parameters = {"type": "OBJECT",
                              "properties": {p: {"type": "STRING",
                                                 "description": pd}
                                             for p, pd in params.items()},
                              "required": list(params)}
            else:
                parameters = {
                    "type": "OBJECT",
                    "properties": {
                        p: {"type": self._JSON_TO_GENAI.get(
                                spec.get("type", "string"), "STRING"),
                            "description": spec.get("description", "")}
                        for p, spec in schema.get("properties", {}).items()},
                    "required": list(schema.get("required", []))}
            declarations.append(types.FunctionDeclaration(
                name=name, description=desc, parameters=parameters))
        self._tools = [types.Tool(function_declarations=declarations)]

    def reset(self, task_text: str) -> None:
        t = self.types
        self.history = [t.Content(role="user",
                                  parts=[t.Part.from_text(text=task_text)])]

    def _config(self, want_logprobs: bool):
        t = self.types
        thinking = (None if self.thinking_budget is None
                    else t.ThinkingConfig(thinking_budget=self.thinking_budget))
        return t.GenerateContentConfig(
            tools=self._tools,
            seed=self.seed,
            thinking_config=thinking,
            system_instruction=("Solve the user's task using the tools. Be "
                                "brief between tool calls; give a one-line "
                                "final answer."),
            max_output_tokens=1024,
            response_logprobs=want_logprobs,
            safety_settings=[
                t.SafetySetting(
                    category=t.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=t.HarmBlockThreshold.BLOCK_NONE,
                ),
                t.SafetySetting(
                    category=t.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=t.HarmBlockThreshold.BLOCK_NONE,
                ),
                t.SafetySetting(
                    category=t.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=t.HarmBlockThreshold.BLOCK_NONE,
                ),
                t.SafetySetting(
                    category=t.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=t.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
        )


    def _generate(self, want_logprobs: bool):
        """One API call with backoff on 429s (free tier is ~10 req/min)."""
        delays = (10.0, 20.0, 40.0, 60.0)
        for attempt in range(len(delays) + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model, contents=self.history,
                    config=self._config(want_logprobs))
            except Exception as exc:  # noqa: BLE001 — inspect and re-raise
                s = str(exc)
                retryable = "429" in s or "RESOURCE_EXHAUSTED" in s
                if retryable and attempt < len(delays):
                    time.sleep(delays[attempt])
                    continue
                raise

    def step(self, step_idx: int) -> dict:
        want_lp = not GeminiBackend._logprobs_disabled
        # Reserve a conservative maximum charge BEFORE the billed request, so an
        # over-cap call is refused before spending, and an unpriced model can
        # never run un-metered. Input is estimated from the current
        # history (~4 chars/token); output is bounded by max_output_tokens.
        if self.cost_meter is not None:
            est_in = sum(len(getattr(p, "text", "") or "")
                         for c in self.history for p in (c.parts or [])) // 4
            self.cost_meter.reserve(self.model, est_in + 512, 1024)
        try:
            response = self._generate(want_lp)
        except Exception as exc:  # noqa: BLE001 — retry once without logprobs
            if want_lp and "logprob" in str(exc).lower():
                if not GeminiBackend._logprobs_disabled:
                    print("  [note] this model/tier rejects response_logprobs"
                          " — u channel will be neutral on these traces")
                    GeminiBackend._logprobs_disabled = True
                self.logprobs_ok = False
                response = self._generate(False)
            else:
                raise
        cand = response.candidates[0]
        if cand.content is None:
            from google.genai.types import Content, Part
            cand.content = Content(role="model", parts=[Part.from_text(text="[Error: Response was blocked by safety filters]")])
        self.history.append(cand.content)

        text_parts, tool_uses = [], []
        for i, part in enumerate(cand.content.parts or []):
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                tool_uses.append({"id": f"fc-{step_idx}-{i}", "name": fc.name,
                                  "input": dict(fc.args or {})})

        token_logprobs: list[float] = []
        lr = getattr(cand, "logprobs_result", None)
        if lr is not None and getattr(lr, "chosen_candidates", None):
            token_logprobs = [float(c.log_probability)
                              for c in lr.chosen_candidates
                              if c.log_probability is not None]
            if self.logprobs_ok is None:
                self.logprobs_ok = True

        usage = response.usage_metadata
        in_tok = int(usage.prompt_token_count or 0)
        out_tok = int(usage.candidates_token_count or 0)
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        if self.cost_meter is not None:
            cached = int(getattr(usage, "cached_content_token_count", 0) or 0)
            # Reconcile with the actual token counts. The model was already
            # validated in `reserve`, so an unknown-model KeyError here is a
            # real error, not something to swallow.
            self.cost_meter.charge(self.model, in_tok, out_tok, cached)
        return {"stop_reason": "tool_use" if tool_uses else "end_turn",
                "text": " ".join(text_parts),
                "tool_uses": tool_uses,
                "output_tokens": int(usage.candidates_token_count or 0),
                "token_logprobs": token_logprobs}

    def add_tool_results(self, results: list[dict]) -> None:
        t = self.types
        parts = [t.Part.from_function_response(
                    name=r["name"], response={"result": r["content"]})
                 for r in results]
        self.history.append(t.Content(role="tool", parts=parts))

    def rewrite_task(self, new_task_text: str) -> None:
        t = self.types
        self.history[0] = t.Content(
            role="user", parts=[t.Part.from_text(text=new_task_text)])

    def corrupt_tool_results(self, rng) -> int:
        t = self.types
        n_changed = 0
        for idx, content in enumerate(self.history):
            if getattr(content, "role", None) != "tool":
                continue
            new_parts = []
            changed = False
            for part in content.parts or []:
                fr = getattr(part, "function_response", None)
                if fr is not None and rng.random() < 0.5:
                    words = str((fr.response or {}).get("result", "")).split()
                    rng.shuffle(words)
                    garbled = " ".join(words) + f" ${int(rng.integers(1, 999))}"
                    new_parts.append(t.Part.from_function_response(
                        name=fr.name, response={"result": garbled}))
                    changed = True
                else:
                    new_parts.append(part)
            if changed:
                self.history[idx] = t.Content(role="tool", parts=new_parts)
                n_changed += 1
        return n_changed


class OllamaBackend(AgentBackend):
    """Local-model backend via Ollama's native /api/chat (no API costs).

    Owns a plain dict message history, so ALL FOUR injection classes work
    exactly as with the Gemini backend. Probes for token logprobs once and
    degrades gracefully if the installed Ollama version doesn't return them.
    Requires a tool-capable model (e.g. qwen2.5:7b, llama3.1:8b).
    """

    _logprobs_disabled = False

    def __init__(self, model: str, base_url: str = "http://localhost:11434",
                 timeout_s: float = 300.0, tool_specs=None,
                 seed: int | None = None, temperature: float = 0.2,
                 tool_schemas: dict | None = None,
                 cassette=None) -> None:
        import httpx

        self._httpx = httpx
        self.base = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.history: list[dict] = []
        self.output_tokens_total = 0
        # Sampling seed: without one, replaying an episode was impossible even
        # with every tool call cached.
        self.seed = seed
        self.temperature = float(temperature)
        # Optional cassette over the BACKEND call itself, so a replay does not
        # depend on the model server being up.
        self.cassette = cassette
        self.n_backend_calls = 0
        # tool_specs {name: (description, params)} — a harness ToolRegistry
        # passes real tools via registry.specs(); default is the mock suite.
        # tool_schemas (registry.schemas()) carries real types/defaults and the
        # true required set; without it every parameter is a required string
        #.
        specs = tool_specs if tool_specs is not None else TOOL_SPECS
        self._tools = []
        for name, (desc, params) in specs.items():
            schema = (tool_schemas or {}).get(name)
            if schema is None:
                schema = {"type": "object",
                          "properties": {p: {"type": "string", "description": pd}
                                         for p, pd in params.items()},
                          "required": list(params)}
            self._tools.append({"type": "function",
                                "function": {"name": name, "description": desc,
                                             "parameters": schema}})

    def reset(self, task_text: str) -> None:
        # "One tool per message" keeps small local models from batching the
        # whole task into a single step (episodes need >= ~5 steps for the
        # monitor washout and a meaningful onset).
        self.history = [
            {"role": "system",
             "content": "Solve the user's task using the tools. Call at most "
                        "ONE tool per message, then wait for its result. Be "
                        "brief; give a one-line final answer."},
            {"role": "user", "content": task_text},
        ]
        self._task_index = 1

    def _chat(self, want_logprobs: bool) -> dict:
        # temperature 0.2: Ollama's default (0.8) makes small models emit
        # junk-token bursts and leak raw tool-call syntax in ~1 run in 4 —
        # genuine failures the monitor rightly flags, but they make a
        # "healthy" baseline unreliable. Low temperature = stable agent.
        options = {"num_predict": 512, "temperature": self.temperature}
        if self.seed is not None:
            options["seed"] = int(self.seed)
        body = {"model": self.model, "messages": self.history,
                "tools": self._tools, "stream": False, "options": options}
        if want_logprobs:
            body["logprobs"] = True

        def _live() -> dict:
            r = self._httpx.post(f"{self.base}/api/chat", json=body,
                                 timeout=self.timeout_s)
            r.raise_for_status()
            return r.json()

        self.n_backend_calls += 1
        if self.cassette is None:
            return _live()
        # Key on the exact request: model, full history, tools and options.
        key = request_key(self.model, body["messages"], self._tools, options,
                          want_logprobs, namespace="ollama/chat/v1")
        return self.cassette.call(key, _live)

    def step(self, step_idx: int) -> dict:
        want_lp = not OllamaBackend._logprobs_disabled
        try:
            data = self._chat(want_lp)
        except Exception as exc:  # noqa: BLE001 — one retry without logprobs
            if want_lp and "logprob" in str(exc).lower():
                OllamaBackend._logprobs_disabled = True
                print("  [note] this Ollama version rejects logprobs — "
                      "u channel will be neutral on these traces")
                data = self._chat(False)
            else:
                raise
        msg = data.get("message", {})
        # Serialize tools: small local models batch several calls per turn
        # regardless of instructions, which collapses episodes to 2-3 steps.
        # Keeping only the first call (and editing it out of history so the
        # model doesn't await dropped results) forces sequential episodes —
        # the same behavior a production harness with serialized tools has.
        if len(msg.get("tool_calls") or []) > 1:
            msg["tool_calls"] = msg["tool_calls"][:1]
        self.history.append({k: v for k, v in msg.items() if v})

        tool_uses = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            tool_uses.append({"id": f"fc-{step_idx}-{i}", "name": fn.get("name"),
                              "input": dict(fn.get("arguments") or {})})
        # Logprobs shape differs across Ollama versions; extract defensively.
        token_logprobs: list[float] = []
        raw_lp = msg.get("logprobs") or data.get("logprobs") or []
        for entry in raw_lp:
            lp = entry.get("logprob") if isinstance(entry, dict) else entry
            if isinstance(lp, (int, float)):
                token_logprobs.append(float(lp))
        if token_logprobs and OllamaBackend._logprobs_disabled:
            OllamaBackend._logprobs_disabled = False
        out_tokens = int(data.get("eval_count") or 0)
        self.output_tokens_total += out_tokens
        return {"stop_reason": "tool_use" if tool_uses else "end_turn",
                "text": (msg.get("content") or "").strip(),
                "tool_uses": tool_uses,
                "output_tokens": out_tokens,
                "token_logprobs": token_logprobs}

    def add_tool_results(self, results: list[dict]) -> None:
        for r in results:
            self.history.append({"role": "tool", "tool_name": r["name"],
                                 "content": r["content"]})

    def rewrite_task(self, new_task_text: str) -> None:
        self.history[self._task_index]["content"] = new_task_text

    def corrupt_tool_results(self, rng) -> int:
        n = 0
        for msg in self.history:
            if msg.get("role") == "tool" and rng.random() < 0.5:
                words = str(msg.get("content", "")).split()
                rng.shuffle(words)
                msg["content"] = " ".join(words) + f" ${int(rng.integers(1, 999))}"
                n += 1
        return n


# ------------------------------------------------------------- injection
@dataclass
class Injection:
    """Mock-tool injection. Records what it ACTUALLY did, so an episode is
    never labelled on intent alone."""

    failure_class: str | None = None
    tau: int | None = None
    drift_applied: bool = False
    rng: object = None
    applied_count: int = 0
    first_applied_t: int | None = None
    applied_tools: list = field(default_factory=list)

    def _record(self, t: int, what: str) -> None:
        self.applied_count += 1
        if self.first_applied_t is None:
            self.first_applied_t = t
        self.applied_tools.append(what)

    def active(self, t: int) -> bool:
        return (self.failure_class is not None and self.tau is not None
                and t >= self.tau)

    def corrupt_tool_result(self, name: str, result: str, t: int) -> tuple[str, bool]:
        if not self.active(t):
            return result, False
        if self.failure_class == "tool_cascade":
            p = min(0.9, 0.4 + 0.15 * (t - self.tau))
            if self.rng.random() < p:
                self._record(t, name)
                return f"Error: {name} service unavailable (HTTP 503).", True
        if self.failure_class == "looping":
            self._record(t, name)
            return ("Error: Temporary data inconsistency detected — please retry "
                    "the exact same query to confirm the value.", True)
        return result, False

    def apply_history_hooks(self, backend: AgentBackend, distractor: str,
                            t: int) -> None:
        if not self.active(t):
            return
        if self.failure_class == "goal_drift" and not self.drift_applied:
            backend.rewrite_task(distractor)
            self.drift_applied = True
            self._record(t, "task_rewrite")
        if self.failure_class == "context_corruption":
            # Only counts as applied when a stored result actually changed.
            if backend.corrupt_tool_results(self.rng):
                self._record(t, "history_corruption")


# ------------------------------------------------------------- episode loop
def run_episode(backend: AgentBackend, seed: int,
                injection: Injection) -> tuple[list[dict], bool]:
    """Run one agent episode; return (step dicts, logprobs_present)."""
    world = _make_world(seed)
    task, distractor = _make_task(seed, world)
    backend.reset(task)
    steps: list[dict] = []
    has_logprobs = False
    for t in range(MAX_STEPS):
        injection.apply_history_hooks(backend, distractor, t)
        t0 = time.perf_counter()
        out = backend.step(t)
        latency = time.perf_counter() - t0
        action = ("tool_call" if out["tool_uses"]
                  else ("synthesis" if out["stop_reason"] == "end_turn"
                        else "plan"))
        step_error = False
        tool_bits, tool_events = [], []
        if out["tool_uses"]:
            results = []
            for use in out["tool_uses"]:
                t_tool = time.perf_counter()
                result = _run_tool(use["name"], use["input"], world)
                tool_latency = time.perf_counter() - t_tool
                result, is_err = injection.corrupt_tool_result(
                    use["name"], result, t)
                is_err = is_err or result.startswith("Error:")
                step_error = step_error or is_err
                results.append({"id": use["id"], "name": use["name"],
                                "content": result, "is_error": is_err})
                # Structured first (schema v5): the executed call as data, with
                # the FULL result and its measured latency. The rendered bit
                # stays in the text for readability and for older readers.
                tool_events.append(make_tool_event(
                    use["name"], use["input"], result, is_error=is_err,
                    latency_s=round(tool_latency, 6),
                    call_id=str(use.get("id", ""))))
                tool_bits.append(format_tool_bit(use["name"], use["input"],
                                                 result))
            backend.add_tool_results(results)
        has_logprobs = has_logprobs or bool(out["token_logprobs"])
        steps.append({
            "text": (out["text"] + " " + " ".join(tool_bits)).strip(),
            "token_logprobs": out["token_logprobs"],
            "logprobs_available": bool(out["token_logprobs"]),
            "action": action,
            "latency_s": round(latency, 4),
            "output_tokens": out["output_tokens"],
            "error": step_error,
            "task": task,
            "tool_events": tool_events,
            "schema": SCHEMA_VERSION,
        })
        if out["stop_reason"] == "end_turn":
            break
    return steps, has_logprobs


# ------------------------------------------------------------------- main
def estimate_cost(n_healthy: int, per_class: int, model: str) -> str:
    """Rough cost preview: history is resent each step, so tokens ~ steps^2."""
    n_eps = n_healthy + per_class * len(INJECT_CLASSES)
    steps = 9
    in_tok = sum(1200 + 350 * s for s in range(steps))   # per episode
    out_tok = 90 * steps
    rates = {"gemini-2.5-flash": (0.30, 2.50), "gemini-2.5-pro": (1.25, 10.0),
             "gemini-2.0-flash": (0.10, 0.40)}
    r_in, r_out = rates.get(model, (0.30, 2.50))
    usd = n_eps * (in_tok * r_in + out_tok * r_out) / 1e6
    return (f"{n_eps} episodes x ~{steps} steps ~= "
            f"{n_eps * (in_tok + out_tok) / 1e6:.1f}M tokens ~= ${usd:.2f} "
            f"on {model}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.collect_traces")
    parser.add_argument("--healthy", type=int, default=40)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--backend", choices=("gemini", "ollama"),
                        default="gemini",
                        help="gemini = cloud API; ollama = free local model "
                             "(server must be running, model pulled)")
    parser.add_argument("--model", default=None,
                        help="default: gemini-2.5-flash / qwen2.5:7b")
    parser.add_argument("--out-dir", default=None,
                        help="trace output dir (default: traces/ for gemini, "
                             "traces/ollama/ for ollama)")
    parser.add_argument("--seed", type=int, default=811)
    parser.add_argument("--mock-llm", action="store_true",
                        help="offline dry run with a scripted fake model")
    parser.add_argument("--estimate", action="store_true",
                        help="print the cost estimate and exit")
    parser.add_argument("--yes", action="store_true",
                        help="confirm real API spend (required unless --mock-llm)")
    parser.add_argument("--resume", action="store_true",
                        help="keep already-collected trace files instead of "
                             "re-collecting them (safe after an interrupted run)")
    args = parser.parse_args(argv)
    model = args.model or (MODEL_DEFAULT if args.backend == "gemini"
                           else OLLAMA_MODEL_DEFAULT)
    out_dir = Path(args.out_dir) if args.out_dir else (
        TRACES_DIR if args.backend == "gemini" else TRACES_DIR / "ollama")

    if args.backend == "ollama" and not args.mock_llm:
        print(f"[collect] local model {model} via Ollama — no API cost "
              "(CPU/GPU time instead)")
    else:
        print("[collect] " + estimate_cost(args.healthy, args.per_class, model))
    if args.estimate:
        return
    if args.backend == "gemini" and not args.mock_llm and not args.yes:
        print("[collect] refusing to spend without --yes "
              "(or use --mock-llm for an offline dry run)")
        return
    if args.backend == "gemini" and not args.mock_llm:
        _enable_os_trust_store()
        get_api_key("GEMINI_API_KEY", required=True)  # fail fast, key not shown
    if args.backend == "ollama" and not args.mock_llm:
        try:
            require_ollama_model(model)  # fail fast, before any episode runs
        except ModelUnavailable as exc:
            raise SystemExit(f"[collect] {exc}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    rejected: list[dict] = []
    previous = {}
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        previous = {e["episode_id"]: e
                    for e in json.loads(manifest_path.read_text("utf-8"))}
    # Healthy episode i and injected episode i share the task seed, so the
    # classes are counterfactual pairs on the same task.
    plan = ([("healthy", None, i) for i in range(args.healthy)]
            + [(fc, fc, i) for fc in INJECT_CLASSES
               for i in range(args.per_class)])
    backend_tokens = [0, 0]
    backend_cassette = (None if args.mock_llm else
                        Cassette(str(out_dir / "_cassettes" / "_backend"),
                                 mode="auto"))
    for kind, fc, i in plan:
        seed = args.seed * 1000 + stable_hash("task", i) % 100000
        episode_id = f"real-{kind}-{i:03d}"
        rng = rng_for(args.seed, "inject", episode_id)
        # Gemini solves these tasks in ~5 steps (it batches tool calls), so
        # the onset must land early to leave post-onset steps to observe.
        tau = None if fc is None else int(rng.integers(2, 4))
        provenance = Provenance(
            collector="collect_traces", backend=args.backend, model=model,
            temperature=0.2 if args.backend == "ollama" else None,
            episode_seed=seed, task_name=f"task-{i}",
            task_sha256=_sha256_text(f"{args.seed}:{i}"),
            tools=tuple(sorted(TOOL_SPECS)),
            tool_roster_sha256=_sha256_text(json.dumps(sorted(TOOL_SPECS))),
            requested_class=fc, requested_tau=tau, injector_seed=args.seed)

        if args.resume:
            ok, why = reusable(out_dir, previous.get(episode_id), provenance)
            if ok:
                manifest.append(previous[episode_id])
                print(f"  [resume] {episode_id}: unchanged")
                continue
            if previous.get(episode_id):
                print(f"  [resume] {episode_id}: re-collecting ({why})")

        backend = (MockBackend(seed) if args.mock_llm
                   else OllamaBackend(model, seed=seed,
                                      cassette=backend_cassette)
                   if args.backend == "ollama"
                   else GeminiBackend(model, seed=seed,
                                      cassette=backend_cassette))
        injection = Injection(failure_class=fc, tau=tau, rng=rng)
        try:
            steps, has_lp = run_episode(backend, seed, injection)
        except Exception as exc:  # noqa: BLE001 — skip episode, keep batch
            print(f"  [error] {episode_id}: {type(exc).__name__}: {exc}")
            continue
        if isinstance(backend, GeminiBackend):
            backend_tokens[0] += backend.input_tokens
            backend_tokens[1] += backend.output_tokens
        elif isinstance(backend, OllamaBackend):
            backend_tokens[1] += backend.output_tokens_total

        verdict = accept_episode(steps, injector=injection, min_steps=4)
        if not verdict.accepted:
            rejected.append({"episode_id": episode_id, "requested_class": fc,
                             "reason": verdict.reason, "facts": verdict.facts})
            print(f"  [reject] {episode_id}: {verdict.reason}")
            continue
        entry = write_episode(out_dir, episode_id, steps, provenance, verdict)
        manifest.append(entry)
        write_manifest(out_dir, manifest)   # interrupted run keeps a valid one
        print(f"  [ok] {episode_id}: T={len(steps)}"
              + (f" onset={verdict.tau} ({fc})" if fc else "")
              + ("" if has_lp else "  [no logprobs]"))
    write_manifest(out_dir, manifest)
    if rejected:
        (out_dir / "rejected.json").write_text(json.dumps(rejected, indent=2),
                                               encoding="utf-8")
        print(f"[collect] {len(rejected)} episode(s) rejected -> "
              f"{out_dir / 'rejected.json'}")
    print(f"[collect] wrote {len(manifest)} traces + manifest.json to "
          f"{out_dir}")
    if not args.mock_llm:
        print(f"[collect] token usage: {backend_tokens[0]:,} in / "
              f"{backend_tokens[1]:,} out")


if __name__ == "__main__":
    main()
