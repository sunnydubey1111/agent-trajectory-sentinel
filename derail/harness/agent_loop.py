"""WS1/WS2 vertical slice — drive a real agent on real tools, end to end.

`run_real_episode` runs one manual agentic loop: the backend proposes tool
calls, the ToolRegistry executes them (cassette + cost-metered), and each
step is turned into the v2/v3 step dict the adapter already consumes. The
output is a plain step list -> Episode via telemetry.adapter, with the
derived x channel populated from the real tool calls. Provider-agnostic: any
backend exposing reset/step/add_tool_results works (GeminiBackend here).

`--live` runs one real Gemini episode on an arXiv/Wikipedia task and prints
the trace, the per-step x-channel signals, and the dollar cost. The offline
smoke test drives the same loop with a scripted backend (no spend, no keys).
"""

from __future__ import annotations

import time
from typing import Any

from derail.harness.record_replay import Cassette, CostMeter
from derail.harness.tools import ToolRegistry
from derail.telemetry.events import (SCHEMA_VERSION, canonical_args,
                                     make_tool_event)


def run_real_episode(backend: Any, registry: ToolRegistry, task: str, *,
                     max_steps: int = 12, cassette: Cassette | None = None,
                     injector: Any | None = None,
                     audit: Any | None = None,
                     episode_id: str | None = None) -> list[dict]:
    """Run one episode; return the list of v2/v3 step dicts.

    An optional WS4 injector transforms tool results from its onset tau; the
    runner advances injector.t once per step so tau is exact ground truth.

    `audit` is an optional `derail.audit.AuditLog`. It is write-only and
    side-effect-free: nothing read back from it reaches the returned steps, and
    every call into it is failure-isolated, so a broken sink changes neither
    the telemetry this produces nor any decision made from it. Collectors leave
    it None — a collection run's evidence IS the trace it writes.
    """
    from derail.audit import NullAuditLog

    log = audit if audit is not None else NullAuditLog()
    log.run_start(episode_id=episode_id, task=task,
                  model=getattr(backend, "model", None),
                  config={"max_steps": max_steps,
                          "injected": getattr(injector, "failure_class", None),
                          "tau": getattr(injector, "tau", None)})
    backend.reset(task)
    steps: list[dict] = []
    for t in range(max_steps):
        if injector is not None:
            injector.t = t
        t0 = time.perf_counter()
        out = backend.step(t)
        latency = time.perf_counter() - t0

        bits, step_error, results, events = [], False, [], []
        for use in out.get("tool_uses", []):
            res = registry.call(use["name"], use["input"], cassette=cassette,
                                injector=injector)
            step_error = step_error or res.is_error
            bits.append(res.step_bit())
            results.append({"id": use["id"], "name": use["name"],
                            "content": res.content, "is_error": res.is_error})
            # Structured event: the executed call as data, with the FULL result
            # and its measured latency, so telemetry no longer depends on
            # re-parsing model-controlled text.
            events.append(make_tool_event(
                res.name, res.args, res.content, is_error=res.is_error,
                latency_s=res.latency_s, call_id=str(use.get("id", ""))))
            log.tool(t, episode_id=episode_id, name=res.name,
                     args_key=canonical_args(res.args), result=res.content,
                     is_error=res.is_error, latency_s=res.latency_s)
        if results:
            backend.add_tool_results(results)

        action = ("tool_call" if out.get("tool_uses")
                  else ("synthesis" if out["stop_reason"] == "end_turn"
                        else "plan"))
        text = (out.get("text", "") + " " + " ".join(bits)).strip()
        logprobs = out.get("token_logprobs") or []
        steps.append({"text": text,
                      "token_logprobs": logprobs,
                      "logprobs_available": bool(logprobs),
                      "action": action,
                      "latency_s": round(latency, 4),
                      "output_tokens": int(out.get("output_tokens", 0)),
                      "error": step_error,
                      "task": task,
                      "tool_events": events,
                      "schema": SCHEMA_VERSION})
        log.step(t, episode_id=episode_id, action=action, latency_s=latency,
                 output_tokens=int(out.get("output_tokens", 0)), text=text,
                 features={"n_tool_calls": len(events),
                           "n_tool_errors": sum(1 for e in events
                                                if e.get("is_error")),
                           "step_error": float(step_error)})
        if out["stop_reason"] == "end_turn":
            break
    log.outcome(status="completed", episode_id=episode_id, steps=len(steps))
    log.run_end()
    return steps


def _run_live(model: str, budget_usd: float, backend_kind: str = "gemini",
              inject_class: str | None = None, tau: int = 4) -> None:
    """One real episode on a real research task (Gemini or local Ollama)."""
    from derail.harness.real_tools import _ensure_tls, default_registry
    from derail.telemetry.adapter import episode_from_trace

    from derail.audit import AuditLog

    _ensure_tls()   # this machine's AV intercepts TLS; fix before any HTTPS
    registry = default_registry()
    meter = CostMeter(budget_usd=budget_usd)
    if backend_kind == "ollama":
        from derail.experiments.collect_traces import OllamaBackend
        backend = OllamaBackend(model, tool_specs=registry.specs())
    else:
        from derail.experiments.collect_traces import GeminiBackend
        backend = GeminiBackend(model, cost_meter=meter,
                                tool_specs=registry.specs())
    injector = None
    if inject_class:
        from derail.harness.inject import ToolInjector
        injector = ToolInjector(inject_class, tau=tau, seed=7)
    # Serving, not collection: this runs one ad-hoc live slice, so it replays
    # the committed recordings but records new ones under `runs/`.
    cassette = Cassette("traces/_cassettes/slice", mode="auto", serving=True)
    task = ("Find the two most recent arXiv papers about echo state networks "
            "for anomaly detection, and give a one-line summary of each. "
            "Use the arxiv_search and wikipedia_search tools; use the python "
            "tool if you need to compute anything. Finish with a one-line "
            "answer.")

    inj_note = f"; INJECT {inject_class}@tau={tau}" if inject_class else ""
    print(f"[slice] LIVE {backend_kind}:{model}; budget ${budget_usd:.2f}"
          f"{inj_note}\n  task: {task}\n")
    # A serving run leaves an audit trail. It is write-only: nothing below
    # reads it back, so the slice behaves identically if the sink fails.
    audit = AuditLog()
    steps = run_real_episode(backend, registry, task, max_steps=12,
                             cassette=cassette, injector=injector,
                             audit=audit, episode_id="slice-live")
    ep = episode_from_trace(
        steps, "slice-live",
        tau=(tau if inject_class else None),
        failure_class=(inject_class or None),
        severity=(0.5 if inject_class else None),
        use_sentence_transformers=False, extended=True)

    from derail.common import (D_TOTAL_EXT, IDX_MEAN_ENTROPY,
                               IDX_REASON_DEPTH, IDX_RETRY_COUNT,
                               IDX_TOOL_SUCCESS)
    n_lp = sum(len(s["token_logprobs"]) for s in steps)
    for t, s in enumerate(steps):
        x = ep.X[t]
        print(f"  t={t} [{s['action']:9}] err={int(s['error'])} "
              f"depth={x[IDX_REASON_DEPTH]:.0f} succ={x[IDX_TOOL_SUCCESS]:.2f} "
              f"retry={x[IDX_RETRY_COUNT]:.0f} u={x[IDX_MEAN_ENTROPY]:.2f}  "
              f"{s['text'][:90]}")
    u_live = "LIVE" if n_lp > 0 else "neutral (no logprobs)"
    print(f"\n[slice] T={ep.T} steps, X shape {ep.X.shape} "
          f"(expect width {D_TOTAL_EXT})")
    print(f"[slice] u-channel: {u_live} ({n_lp} token logprobs total)")
    print(f"[slice] {meter.summary()}")
    print(f"[slice] {cassette.summary()}")
    assert ep.X.shape[1] == D_TOTAL_EXT


# --------------------------------------------------------------- smoke test
class _ScriptedBackend:
    """Offline stand-in: `n_calls` arxiv_search calls, then a final answer.
    Each call uses a distinct query so nothing looks like a retry."""

    def __init__(self, n_calls: int = 1) -> None:
        self.history: list = []
        self.n_calls = n_calls

    def reset(self, task: str) -> None:
        self.history = [("user", task)]

    def add_tool_results(self, results: list[dict]) -> None:
        self.history.append(("tool", results))

    def step(self, t: int) -> dict:
        if t < self.n_calls:
            return {"stop_reason": "tool_use",
                    "text": "Let me search arXiv.",
                    "tool_uses": [{"id": f"c{t}", "name": "arxiv_search",
                                   "input": {"query": f"echo state network {t}"}}],
                    "output_tokens": 8, "token_logprobs": [-0.2] * 8}
        return {"stop_reason": "end_turn",
                "text": "Based on the results, here is the summary.",
                "tool_uses": [], "output_tokens": 10,
                "token_logprobs": [-0.3] * 10}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="py -m derail.harness.agent_loop")
    parser.add_argument("--live", action="store_true",
                        help="run one real episode (Gemini spends money; "
                             "Ollama is free and gives a live u-channel)")
    parser.add_argument("--backend", choices=["gemini", "ollama"],
                        default="gemini")
    parser.add_argument("--model", default=None,
                        help="default: gemini-2.5-flash or qwen2.5:7b")
    parser.add_argument("--budget", type=float, default=0.50)
    parser.add_argument("--inject", default=None,
                        help="WS4 failure class to inject (e.g. looping, "
                             "rate_limit, wrong_document, malformed_json)")
    parser.add_argument("--tau", type=int, default=4, help="injection onset")
    args = parser.parse_args()

    if args.live:
        model = args.model or ("qwen2.5:7b" if args.backend == "ollama"
                               else "gemini-2.5-flash")
        _run_live(model, args.budget, args.backend, args.inject, args.tau)
    else:
        # Offline: scripted backend + REAL tools (arXiv runs live but free).
        from derail.harness.real_tools import ArxivSearch
        from derail.harness.tools import ToolRegistry
        from derail.telemetry.adapter import episode_from_trace
        from derail.common import D_TOTAL_EXT, IDX_REASON_DEPTH

        # Fake arXiv payload so the test never needs the network.
        xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
               '<title>ESN anomaly detection</title>'
               '<id>http://arxiv.org/abs/2601.00001v1</id>'
               '<author><name>X</name></author></entry></feed>').encode()
        reg = ToolRegistry([ArxivSearch(get=lambda url: xml)])
        steps = run_real_episode(_ScriptedBackend(), reg,
                                 "find ESN anomaly papers", max_steps=6)
        assert len(steps) == 2, steps
        assert steps[0]["action"] == "tool_call" and not steps[0]["error"]
        assert "arxiv_search" in steps[0]["text"]
        assert steps[1]["action"] == "synthesis"
        ep = episode_from_trace(steps, "slice-offline",
                                use_sentence_transformers=False, extended=True)
        assert ep.X.shape == (2, D_TOTAL_EXT)
        assert ep.X[0, IDX_REASON_DEPTH] == 1.0

        # --- WS4: healthy vs injected on the SAME scripted run ------------
        from derail.harness.inject import ToolInjector
        from derail.common import IDX_ERROR_FLAG

        reg2 = ToolRegistry([ArxivSearch(get=lambda url: xml)])
        healthy = run_real_episode(_ScriptedBackend(n_calls=3), reg2,
                                   "find papers", max_steps=8)
        tau = 1
        inj = ToolInjector("looping", tau=tau, seed=7)   # deterministic error
        injected = run_real_episode(_ScriptedBackend(n_calls=3), reg2,
                                    "find papers", max_steps=8, injector=inj)
        # Healthy: every tool call succeeded; injected: errors from tau on.
        assert not any(s["error"] for s in healthy), "healthy run should be clean"
        assert all(s["error"] for t, s in enumerate(injected)
                   if t >= tau and s["action"] == "tool_call"), \
            "injected tool steps at/after tau should error"
        assert not injected[0]["error"], "pre-tau step must stay clean"
        # Label it and confirm the x channel reflects the failure post-tau.
        ep_inj = episode_from_trace(injected, "slice-injected", tau=tau,
                                    failure_class="looping", severity=0.5,
                                    use_sentence_transformers=False, extended=True)
        assert not ep_inj.is_healthy and ep_inj.tau == tau
        post_fail = ep_inj.X[tau:, IDX_ERROR_FLAG].max()
        assert post_fail == 1.0, "error flag not set post-tau in telemetry"
        print("PASS agent_loop.py offline slice + WS4 injection "
              "| run --live for a real episode")
