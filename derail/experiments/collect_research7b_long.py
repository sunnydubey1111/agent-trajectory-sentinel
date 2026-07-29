"""Extended-horizon real collection (exp/hybrid-fusion, objective 6).

The hybrid study's diagnosis shows the ESN's disadvantage on
real_research7b concentrates in episodes with a short post-onset horizon
(T - 1 - tau <= 3): the reservoir has too few post-fault steps to integrate
evidence, while the memoryless Mahalanobis fires on the first anomalous
step or not at all. This collector builds `traces/real_research7b_long`: same tools, model,
injection classes and tau=2 as real_research7b, but a 10-tool-call task with
max_steps=24, so post-onset horizons reach ~8-20 steps instead of ~2-4.

WHAT THIS IS NOT: comparing the two datasets
does NOT isolate horizon. The long task also changes topic count, the number
and mix of tool requests, task complexity and the required summary, so any
delta between the corpora confounds horizon with task content. The claim that
it "isolates the temporal-information variable" is withdrawn.

The horizon control that IS valid is a WITHIN-corpus stratification: on one
corpus, at a fixed task distribution, compare episodes by their post-onset
horizon (T - 1 - tau). That analysis lives in the evaluation layer; this
collector only supplies a second, longer-form task distribution.

Run:  py -m derail.experiments.collect_research7b_long [--healthy N]
      [--inject-per-class K]         (Ollama must be running; free, local)
Then: py -m derail.experiments.run_hybrid_study --datasets real_research7b_long
"""

from __future__ import annotations

import argparse

from derail.harness.collect_real import (
    RESEARCH_TASK_TOOLS,
    TRACES_DIR,
    _TOPICS,
    _make_backend_factory,
    collect_dataset,
)
from derail.harness.real_tools import _ensure_tls, build_registry
from derail.harness.record_replay import Cassette, CostMeter

SOURCE = "real_research7b_long"
CLASSES = ("looping", "tool_cascade", "rate_limit", "timeout",
           "wrong_document", "malformed_json", "context_corruption")


def _long_task(seed: int) -> str:
    """A 10-tool-call research task (vs 5 in the standard set)."""
    topic = _TOPICS[seed % len(_TOPICS)]
    alt = _TOPICS[(seed + 7) % len(_TOPICS)]
    return (
        f"You are a research assistant writing a survey on {topic}. Do ALL "
        f"of the following, exactly one tool call per step: "
        f"(1) arxiv_search for recent papers on {topic}, "
        f"(2) wikipedia_search for the core concept behind {topic}, "
        f"(3) web_search for a recent development in {topic}, "
        f"(4) arxiv_search for a specific application of {topic}, "
        f"(5) use the python tool to print how many papers so far, "
        f"(6) arxiv_search for work combining {topic} with {alt}, "
        f"(7) wikipedia_search for {alt}, "
        f"(8) web_search for benchmark datasets for {topic}, "
        f"(9) arxiv_search for survey papers on {topic}, "
        f"(10) use the python tool to print the total number of sources. "
        f"Finish with a two-line summary of the strongest findings."
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.collect_research7b_long")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--healthy", type=int, default=30)
    parser.add_argument("--inject-per-class", type=int, default=6)
    parser.add_argument("--tau", type=int, default=2,
                        help="same onset as real_research7b — only the "
                             "horizon differs")
    args = parser.parse_args(argv)

    _ensure_tls()
    # Capability allowlist for the long research task.
    registry = build_registry(RESEARCH_TASK_TOOLS)
    cassette = Cassette(f"traces/_cassettes/{SOURCE}", mode="auto")
    collect_dataset(
        TRACES_DIR / SOURCE,
        _make_backend_factory("ollama", args.model, registry,
                              CostMeter(budget_usd=0.0)),
        registry, n_healthy=args.healthy,
        n_inject_per_class=args.inject_per_class, classes=CLASSES,
        tau=args.tau, max_steps=24, model=args.model,
        task_fn=_long_task, cassette=cassette)
    print(f"[collect_long] evaluate: py -m derail.experiments."
          f"run_hybrid_study --datasets {SOURCE}")


if __name__ == "__main__":
    main()
