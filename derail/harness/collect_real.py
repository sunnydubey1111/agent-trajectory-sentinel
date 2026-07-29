"""WS5 — collect labeled real-tool datasets the study evaluator can score.

Runs the harness (real tools + WS4 injection) to produce one JSONL trace per
episode plus a manifest, in exactly the schema run_real_traces.py consumes.
So the "matrix" is: collect_real (real tools/agent/injection) ->
run_real_traces --dir traces/<source> -> detection/AUC per source, alongside
the existing simulator and mock-tool tables (added, not overwritten).

Healthy and injected episodes share the cassette, so injected variants replay
the same clean tool responses at a controlled tau — cheap, deterministic
ground truth. Ollama runs are free with a live u channel; Gemini is metered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from derail.harness.agent_loop import run_real_episode
from derail.harness.collection import (accept_episode, make_provenance,
                                       reusable, write_episode,
                                       write_manifest)
from derail.harness.inject import ToolInjector
from derail.harness.record_replay import Cassette, CostMeter

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"

# Research topics -> one varied task per episode (real arXiv/Wikipedia/web).
# 40 DISTINCT research topics — one per episode gives the one-class monitor
# genuine healthy diversity (too-few-distinct-trajectories was the overfitting
# cause). No repetition until seed wraps past 40.
_TOPICS = (
    "echo state networks for anomaly detection",
    "reservoir computing hardware accelerators",
    "CUSUM change-point detection",
    "one-class support vector machines",
    "graph neural networks for fraud detection",
    "conformal prediction for time series",
    "self-supervised representation learning",
    "Kalman filtering for sensor fusion",
    "isolation forests for outlier detection",
    "variational autoencoders for novelty detection",
    "transformer models for log analysis",
    "LSTM networks for predictive maintenance",
    "Gaussian process regression for forecasting",
    "hidden Markov models for sequence labeling",
    "diffusion models for image generation",
    "contrastive learning for embeddings",
    "federated learning for privacy",
    "reinforcement learning for control",
    "spectral clustering algorithms",
    "Bayesian optimization for hyperparameters",
    "attention mechanisms in vision",
    "knowledge graph embeddings",
    "causal inference from observational data",
    "survival analysis with neural networks",
    "optical flow estimation",
    "point cloud segmentation",
    "speech recognition acoustic models",
    "neural machine translation",
    "recommender systems with matrix factorization",
    "topic modeling with latent Dirichlet allocation",
    "adversarial robustness in deep learning",
    "quantization for model compression",
    "meta-learning for few-shot classification",
    "graph attention networks",
    "time-series motif discovery",
    "anomaly detection in network traffic",
    "protein structure prediction",
    "molecular property prediction",
    "electricity load forecasting",
    "seismic event detection",
)


# Capability allowlist for the research task below (and for the long-horizon
# and organic variants that reuse it) - exactly the tools the prompt names,
# nothing else.
RESEARCH_TASK_TOOLS = ("arxiv_search", "wikipedia_search", "web_search", "python")


def _default_task(seed: int) -> str:
    topic = _TOPICS[seed % len(_TOPICS)]
    return (f"You are a research assistant investigating {topic}. Do all of the "
            f"following, one tool call per step: (1) arxiv_search for recent "
            f"papers on {topic}, (2) wikipedia_search for the core concept, "
            f"(3) web_search for a recent development, (4) arxiv_search for a "
            f"specific application, (5) use the python tool to print how many "
            f"papers you found. Finish with a one-line summary.")


def collect_dataset(source_dir: Path, make_backend: Callable[[int], object],
                    registry, *, n_healthy: int, n_inject_per_class: int,
                    classes: tuple[str, ...], tau: int = 3, max_steps: int = 12,
                    model: str = "?", task_fn: Callable[[int], str] = _default_task,
                    cassette: Cassette | None = None, verbose: bool = True,
                    collector: str = "collect_real", backend: str = "?",
                    temperature: float | None = None,
                    verify: Callable[[list[dict]], bool] | None = None
                    ) -> list[dict]:
    """Collect healthy + injected episodes into source_dir; write manifest.

    Design points forced by

    * healthy episode k and every injected episode k run the SAME task with the
      SAME model seed, so the classes are counterfactual pairs rather than
      "healthy gets the early topics, failures get the later ones";
    * an episode is written only if `accept_episode` says the evidence supports
      its label - a no-op injection or an unsuccessful healthy run is rejected
      and logged in `rejected.json`, never quietly kept;
    * every entry carries a provenance fingerprint and a trace checksum, and a
      resume re-collects rather than relabels when either differs.
    """
    source_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    rejected: list[dict] = []
    previous = {}
    manifest_path = source_dir / "manifest.json"
    if manifest_path.exists():
        previous = {e["episode_id"]: e
                    for e in json.loads(manifest_path.read_text("utf-8"))}

    def _run_one(episode_id: str, task_index: int, failure_class: str | None):
        task_text = task_fn(task_index)
        injector = (None if failure_class is None
                    else ToolInjector(failure_class, tau=tau,
                                      seed=1000 + task_index))
        provenance = make_provenance(
            collector=collector, backend=backend, model=model,
            temperature=temperature, episode_seed=task_index,
            task_text=task_text, registry=registry,
            task_name=f"task-{task_index}", injector=injector)

        ok, why = reusable(source_dir, previous.get(episode_id), provenance)
        if ok:
            manifest.append(previous[episode_id])
            if verbose:
                print(f"  [{episode_id}] resumed ({why})")
            return
        if previous.get(episode_id) and verbose:
            print(f"  [{episode_id}] re-collecting: {why}")

        steps = run_real_episode(make_backend(task_index), registry, task_text,
                                 max_steps=max_steps, cassette=cassette,
                                 injector=injector)
        success = None if verify is None else bool(verify(steps))
        verdict = accept_episode(steps, injector=injector, success=success)
        if not verdict.accepted:
            rejected.append({"episode_id": episode_id,
                             "requested_class": failure_class,
                             "reason": verdict.reason, "facts": verdict.facts})
            if verbose:
                print(f"  [{episode_id}] REJECTED: {verdict.reason}")
            return
        manifest.append(write_episode(source_dir, episode_id, steps,
                                      provenance, verdict))
        if verbose:
            onset = "" if verdict.tau is None else f" onset={verdict.tau}"
            print(f"  [{episode_id}] T={len(steps)}{onset}")

    for k in range(n_healthy):
        _run_one(f"real-healthy-{k:03d}", k, None)
    for fc in classes:
        for k in range(n_inject_per_class):
            # SAME task index as healthy k: the only difference is the fault.
            _run_one(f"real-{fc}-{k:03d}", k, fc)

    write_manifest(source_dir, manifest)
    if rejected:
        (source_dir / "rejected.json").write_text(
            json.dumps(rejected, indent=2), "utf-8")
    if verbose:
        n_lab = sum(1 for e in manifest if e["failure_class"])
        print(f"[collect] {len(manifest)} episodes ({n_lab} labeled), "
              f"{len(rejected)} rejected -> {source_dir}")
    return manifest


def _make_backend_factory(backend_kind: str, model: str, registry,
                          meter: CostMeter | None,
                          temperature: float = 0.2,
                          backend_cassette: Cassette | None = None,
                          thinking_budget: int | None = None):
    """Backend factory whose sampling is SEEDED per episode.

    The seed is the episode index, so a healthy episode and its injected
    counterfactual sample the model identically and differ only by the fault.

    `thinking_budget` is Gemini-only and defaults to None (the model's own
    default), which is what every already-collected corpus used; pass 0 for
    long-horizon collection, where reasoning tokens would otherwise dominate
    both the bill and the output-token allowance.
    """

    def _factory(seed: int):
        if backend_kind == "ollama":
            from derail.experiments.collect_traces import OllamaBackend
            return OllamaBackend(model, tool_specs=registry.specs(),
                                 tool_schemas=registry.schemas(),
                                 seed=seed, temperature=temperature,
                                 cassette=backend_cassette)
        from derail.experiments.collect_traces import GeminiBackend
        return GeminiBackend(model, cost_meter=meter,
                             tool_specs=registry.specs(), seed=seed,
                             tool_schemas=registry.schemas(),
                             cassette=backend_cassette,
                             thinking_budget=thinking_budget)
    return _factory


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="py -m derail.harness.collect_real")
    parser.add_argument("--collect", action="store_true",
                        help="live collection (else: offline self-test)")
    parser.add_argument("--source", default="real_ollama7b")
    parser.add_argument("--backend", choices=["ollama", "gemini"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--healthy", type=int, default=20)
    parser.add_argument("--inject-per-class", type=int, default=6)
    parser.add_argument("--tau", type=int, default=2,
                        help="injection onset; keep low — real research "
                             "episodes are short (T~3-6), so tau=2 lands "
                             "while tau=3+ often misses")
    parser.add_argument("--budget", type=float, default=5.0)
    args = parser.parse_args()

    if args.collect:
        from derail.harness.real_tools import _ensure_tls, build_registry
        _ensure_tls()
        registry = build_registry(RESEARCH_TASK_TOOLS)
        meter = CostMeter(budget_usd=args.budget)
        cassette = Cassette(f"traces/_cassettes/{args.source}", mode="auto")
        # All tool-layer scenarios: error-based + retrieval + content-corruption.
        classes = ("looping", "tool_cascade", "rate_limit", "timeout",
                   "wrong_document", "malformed_json", "context_corruption",
                   "goal_drift")   # prompt-layer hijack
        collect_dataset(
            TRACES_DIR / args.source,
            _make_backend_factory(args.backend, args.model, registry, meter,
                                  backend_cassette=Cassette(
                                      f"traces/_cassettes/{args.source}_backend",
                                      mode="auto")),
            registry, n_healthy=args.healthy,
            n_inject_per_class=args.inject_per_class, classes=classes,
            tau=args.tau, model=args.model, cassette=cassette,
            collector="collect_real", backend=args.backend, temperature=0.2)
        if args.backend == "gemini":
            print(f"[collect] {meter.summary()}")
        print(f"[collect] evaluate: py -m derail.experiments.run_real_traces "
              f"--dir traces/{args.source} --extended")
    else:
        # Offline: scripted backend + real (canned) tool -> a valid dataset.
        import tempfile
        from derail.common import D_TOTAL_EXT
        from derail.harness.tools import ToolRegistry
        from derail.harness.real_tools import ArxivSearch
        from derail.telemetry.adapter import load_trace_jsonl

        class _Scripted:
            def __init__(self, n=3):
                self.n = n
            def reset(self, task): pass
            def add_tool_results(self, results): pass
            def step(self, t):
                if t < self.n:
                    return {"stop_reason": "tool_use", "text": "search",
                            "tool_uses": [{"id": f"c{t}", "name": "arxiv_search",
                                           "input": {"query": f"q{t}"}}],
                            "output_tokens": 5, "token_logprobs": [-0.2] * 5}
                return {"stop_reason": "end_turn", "text": "done",
                        "tool_uses": [], "output_tokens": 4,
                        "token_logprobs": [-0.3] * 4}

        xml = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
               '<title>Paper</title><id>http://arxiv.org/abs/1v1</id>'
               '<author><name>A</name></author></entry></feed>').encode()
        reg = ToolRegistry([ArxivSearch(get=lambda url: xml)])
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src"
            man = collect_dataset(
                src, lambda s: _Scripted(4), reg, n_healthy=6,
                n_inject_per_class=2, classes=("looping", "wrong_document"),
                tau=2, max_steps=8, model="scripted", verbose=False)
            assert len(man) == 6 + 2 * 2, len(man)
            assert (src / "manifest.json").exists()
            labeled = [e for e in man if e["failure_class"]]
            assert len(labeled) == 4 and all(e["tau"] == 2 for e in labeled)
            # Every trace round-trips back into an Episode.
            for e in man:
                ep = load_trace_jsonl(src / e["file"], episode_id=e["episode_id"],
                                      tau=e["tau"], failure_class=e["failure_class"],
                                      use_sentence_transformers=False, extended=True)
                assert ep.X.shape[1] == D_TOTAL_EXT
                assert ep.is_healthy == (e["failure_class"] is None)
        print("PASS collect_real.py offline self-test | "
              "run --collect to build a live labeled dataset")
