"""Real-tools demo engine — the mock booking demo, rebuilt on real tools.

Same idea as experiments/demo.py (a live agent scored by the ESN channel-max
monitor, with failure-injection buttons), but the agent drives REAL tools
(arXiv / Wikipedia / web / python) on a fixed research task, injection uses
the WS4 ToolInjector, and everything runs through the cassette so a stage run
is REAL yet deterministic and free (Ollama, live u channel).

This module is the engine: collect a healthy calibration set for the fixed
demo task, cross-fit the monitor + threshold on it, and run/score one healthy
or injected episode. The web UI is wired on top of this (next increment); the
headless `--demo` mode proves detection works on real-tool injections first.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from derail.common import Standardizer, rng_for
from derail.evaluation.metrics import pick_threshold
from derail.experiments.demo import StreamingChannelMax
from derail.harness.agent_loop import run_real_episode
from derail.harness.inject import ToolInjector
from derail.harness.record_replay import Cassette
from derail.telemetry.adapter import episode_from_trace, load_trace_jsonl
from derail.verify.checks import RESEARCH_SPEC, required_coverage

TRACES = Path(__file__).resolve().parents[2] / "traces" / "demo_real"
MODEL = "qwen2.5:7b"
FA_BUDGET = 0.10

# Multi-part task designed for LENGTH (~7 steps) so the monitor has runway.
# The TOPIC varies per episode: a fixed task starves the monitor of healthy
# diversity (residual stds collapse -> every deviation explodes), exactly like
# the original demo needed seeded per-run variation. Same STRUCTURE every run
# (so calibration matches the runtime distribution), different CONTENT.
_DEMO_TOPICS = (
    ("echo state networks", "transformers", "time-series anomaly detection"),
    ("isolation forests", "autoencoders", "network intrusion detection"),
    ("graph neural networks", "LSTMs", "financial fraud detection"),
    ("one-class SVM", "variational autoencoders", "sensor fault detection"),
    ("Gaussian processes", "convolutional networks", "ECG anomaly detection"),
    ("hidden Markov models", "diffusion models", "industrial process monitoring"),
)


# Exactly the tools demo_task names below.
DEMO_TASK_TOOLS = ("arxiv_search", "wikipedia_search", "web_search", "python")


def demo_task(seed: int) -> str:
    a, b, app = _DEMO_TOPICS[seed % len(_DEMO_TOPICS)]
    return (f"You are a research assistant comparing {a} and {b} for {app}. "
            f"Do all of the following, one tool call per step: (1) arxiv_search "
            f"for recent {a} papers, (2) arxiv_search for {b} papers, "
            f"(3) wikipedia_search for '{a}', (4) wikipedia_search for '{b}', "
            f"(5) web_search for a recent comparison, (6) use the python tool "
            f"to print how many of the two approaches you found papers for. "
            f"Finish with a one-line comparison.")


def _registry():
    from derail.harness.real_tools import _ensure_tls, build_registry
    _ensure_tls()
    # The demo task names its tools explicitly.
    return build_registry(DEMO_TASK_TOOLS)


def _backend():
    from derail.experiments.collect_traces import OllamaBackend
    return OllamaBackend(MODEL, tool_specs=_registry().specs())


def collect_healthy(n: int) -> None:
    """Collect n healthy episodes of the FIXED demo task for calibration."""
    from derail.harness.collect_real import collect_dataset

    registry = _registry()
    cassette = Cassette("traces/_cassettes/demo_real", mode="auto")
    collect_dataset(TRACES, lambda s: _backend(), registry,
                    n_healthy=n, n_inject_per_class=0, classes=(),
                    task_fn=demo_task, model=MODEL, cassette=cassette)


def fit_monitor(src: Path = TRACES) -> tuple[StreamingChannelMax, float, float]:
    """Cross-fit the channel-max monitor + threshold on healthy demo traces."""
    manifest = json.loads((src / "manifest.json").read_text("utf-8"))
    entries = [e for e in manifest
               if e["failure_class"] is None and e["T"] >= 4]

    # A healthy null must hold runs that DID the task: a run omitting required
    # calls is strongly anomalous to the monitor, and leaving it in inflates
    # the null and lifts the threshold above where real failures live
    # (DESIGN.md Amendment 7). Unlike the booking demo this task has no
    # computable ground-truth answer, so completeness is checkable here and
    # correctness is not -- that half of the policy is a stated limitation.
    def _complete(path: Path) -> bool:
        steps = [json.loads(x) for x in
                 path.read_text("utf-8").splitlines() if x.strip()]
        return not required_coverage(steps, RESEARCH_SPEC)

    done = [e for e in entries if _complete(src / e["file"])]
    if len(done) >= 10 and len(done) < len(entries):
        print(f"[demo_real] task-completeness policy: "
              f"{len(entries) - len(done)}/{len(entries)} runs excluded from "
              f"the healthy null (required calls missing).")
        entries = done

    def _is_clean(path: Path) -> bool:
        # Exclude REAL glitch episodes (qwen unicode/CJK bursts, empty steps)
        # that inflate theta — a raw-telemetry filter, not score-peeking.
        for line in path.read_text("utf-8").splitlines():
            txt = json.loads(line).get("text", "")
            if not txt.strip():
                return False
            non_ascii = sum(1 for ch in txt if ord(ch) > 127)
            if non_ascii > max(8, 0.2 * len(txt)):
                return False
        return True

    clean = [e for e in entries if _is_clean(src / e["file"])]
    if len(clean) >= 10:
        entries = clean
    healthy = [load_trace_jsonl(src / e["file"], episode_id=e["episode_id"],
                                use_sentence_transformers=False, extended=True)
               for e in entries]
    if len(healthy) < 5:
        raise SystemExit(f"only {len(healthy)} healthy demo episodes — "
                         f"collect more: py -m derail.harness.demo_real "
                         f"--collect-healthy 30")

    perm = rng_for(0, "demo-real-split").permutation(len(healthy))
    n_folds = 5 if len(healthy) >= 15 else 3
    folds = [perm[k::n_folds] for k in range(n_folds)]
    val_scores = []
    for k in range(n_folds):
        rest = [healthy[i] for j in range(n_folds) if j != k for i in folds[j]]
        std_k = Standardizer().fit(rest)
        mon_k = StreamingChannelMax(std_k)
        mon_k.fit(rest)
        val_scores += [mon_k.score_episode(healthy[i]) for i in folds[k]]
    std = Standardizer().fit(healthy)
    mon = StreamingChannelMax(std)
    mon.fit(healthy)
    theta = float(pick_threshold(val_scores, fa_budget=FA_BUDGET))
    theta5 = float(pick_threshold(val_scores, fa_budget=0.05))
    print(f"[demo_real] fitted on {len(healthy)} healthy demo episodes; "
          f"theta(10% FA)={theta:.2f}, theta(5% FA)={theta5:.2f}")
    return mon, theta, theta5


def run_and_score(monitor: StreamingChannelMax, theta: float, registry,
                  backend, *, inject_class: str | None = None, tau: int = 2,
                  seed: int = 0, cassette: Cassette | None = None) -> dict:
    """Run one healthy/injected demo episode; return steps + scores + alarm."""
    injector = (ToolInjector(inject_class, tau=tau, seed=7)
                if inject_class else None)
    steps = run_real_episode(backend, registry, demo_task(seed), max_steps=12,
                             cassette=cassette, injector=injector)
    ep = episode_from_trace(
        steps, "demo", tau=(tau if inject_class else None),
        failure_class=(inject_class or None),
        severity=(0.5 if inject_class else None),
        use_sentence_transformers=False, extended=True)
    scores = monitor.score_episode(ep)
    alarm = next((t for t, s in enumerate(scores) if s > theta), None)
    return {"steps": steps, "scores": [float(s) for s in scores],
            "alarm_step": alarm, "inject_class": inject_class,
            "inject_tau": tau if inject_class else None, "T": ep.T}


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="py -m derail.harness.demo_real")
    parser.add_argument("--collect-healthy", type=int, default=0)
    parser.add_argument("--demo", action="store_true",
                        help="fit on collected healthy + run healthy and each "
                             "injected class, print detection")
    args = parser.parse_args()

    if args.collect_healthy:
        collect_healthy(args.collect_healthy)
    elif args.demo:
        registry = _registry()
        mon, theta, theta5 = fit_monitor()
        cas = Cassette("traces/_cassettes/demo_real", mode="auto")
        print(f"\n{'scenario':<18} {'alarm':>6} {'max_score':>10}  T")
        for i, fc in enumerate((None, "looping", "rate_limit", "wrong_document",
                                "malformed_json")):
            r = run_and_score(mon, theta, registry, _backend(),
                              inject_class=fc, tau=2, seed=100 + i, cassette=cas)
            mx = max(r["scores"]) if r["scores"] else 0.0
            print(f"{fc or 'healthy':<18} {str(r['alarm_step']):>6} "
                  f"{mx:>10.1f}  {r['T']}")
        print(f"\ntheta(10% FA)={theta:.1f}")
    else:
        # Offline: fit + score on SCRIPTED healthy episodes (no Ollama/network).
        import tempfile
        from derail.harness.real_tools import ArxivSearch
        from derail.harness.tools import ToolRegistry

        class _Scripted:
            def __init__(self, n=5):
                self.n = n
            def reset(self, task): pass
            def add_tool_results(self, results): pass
            def step(self, t):
                if t < self.n:
                    return {"stop_reason": "tool_use", "text": "searching",
                            "tool_uses": [{"id": f"c{t}", "name": "arxiv_search",
                                           "input": {"query": f"q{t}"}}],
                            "output_tokens": 6, "token_logprobs": [-0.2] * 6}
                return {"stop_reason": "end_turn", "text": "done",
                        "tool_uses": [], "output_tokens": 5,
                        "token_logprobs": [-0.3] * 5}

        xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
               '<title>Paper</title><id>http://arxiv.org/abs/1v1</id>'
               '<author><name>A</name></author></entry></feed>').encode()
        reg = ToolRegistry([ArxivSearch(get=lambda url: xml)])

        # Build a handful of healthy episodes, fit, then score healthy vs injected.
        with tempfile.TemporaryDirectory() as d:
            src = Path(d)
            from derail.harness.collect_real import collect_dataset
            collect_dataset(src, lambda s: _Scripted(5), reg, n_healthy=8,
                            n_inject_per_class=0, classes=(),
                            task_fn=lambda s: "demo task", model="scripted",
                            verbose=False)
            mon, theta, theta5 = fit_monitor(src)
            healthy = run_and_score(mon, theta, reg, _Scripted(5))
            injected = run_and_score(mon, theta, reg, _Scripted(5),
                                     inject_class="looping", tau=2)
            assert healthy["inject_class"] is None
            assert injected["inject_class"] == "looping"
            # Injected episode should score no lower than healthy at its peak.
            assert max(injected["scores"]) >= max(healthy["scores"]) - 1e-9
            assert all(np.isfinite(injected["scores"]))
        print("PASS demo_real.py offline self-test | "
              "--collect-healthy N then --demo for the real thing")
