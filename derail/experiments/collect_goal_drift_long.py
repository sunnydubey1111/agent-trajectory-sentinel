"""Long-runway real goal_drift collection — the data the conceptor arm needs.

Slow goal drift is the one failure the behavioural monitor has never detected
(det 0.0125 for every monitor), and conceptors are the proposed mechanism for
it. That proposal could not be evaluated on real traces, for a reason that is
about the CORPUS and not the method: across every committed corpus, only
**three** real `goal_drift` episodes have the >= 9 post-onset steps that the
horizon law identifies as where temporal detection pays, and all three sit in
`real_research3b` (5 goal_drift episodes total). The long-form corpora
(`real_research7b_long`, `..._long_ext`) reach further than the short ones but
their class list omits `goal_drift` entirely.

This collector closes exactly that gap: the SAME long-form 10-tool-call task,
model, tool suite, tau and max_steps as `collect_research7b_long`, with
`classes=("goal_drift",)`.

What it actually yielded, measured on the collected manifest with the
definition the studies use (`H = T - 1 - tau`): median 8, range 1-10, with
**19 of 120** injected episodes reaching `H >= 9`. That is one step short of
the band this collection was aimed at, and it is a property of the task -
`real_research7b_long` and `..._long_ext` also sit at median 8. Read a `>= 9`
figure from any of the three as resting on tens of episodes, not hundreds.

It writes a SEPARATE corpus, `traces/real_research7b_long_drift`, rather than
extending `real_research7b_long`. That corpus feeds published hybrid-study
numbers; adding episodes to it would silently move them. A new directory keeps
the provenance unambiguous and leaves every existing result untouched.

WHAT THIS IS NOT: this is not a horizon-controlled comparison against
`real_research7b`'s short goal_drift episodes. The long task differs in topic
count, tool mix and required summary, so a delta between the two corpora
confounds horizon with task content — the same caveat
`collect_research7b_long` states about itself. The valid use is a WITHIN-corpus
comparison of monitors on this corpus, at a fixed task distribution.

Run:  py -m derail.experiments.collect_goal_drift_long [--healthy N]
      [--inject N]                  (Ollama must be running; free, local)
"""

from __future__ import annotations

import argparse

from derail.experiments.collect_research7b_long import _long_task
from derail.harness.collect_real import (
    RESEARCH_TASK_TOOLS,
    TRACES_DIR,
    _make_backend_factory,
    collect_dataset,
)
from derail.harness.real_tools import _ensure_tls, build_registry
from derail.harness.record_replay import Cassette, CostMeter

SOURCE = "real_research7b_long_drift"
CLASSES = ("goal_drift",)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.collect_goal_drift_long")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--healthy", type=int, default=24)
    parser.add_argument("--inject", type=int, default=24,
                        help="goal_drift episodes to collect")
    parser.add_argument("--tau", type=int, default=2,
                        help="same onset as the other real corpora — only the "
                             "horizon differs")
    args = parser.parse_args(argv)

    _ensure_tls()
    registry = build_registry(RESEARCH_TASK_TOOLS)
    cassette = Cassette(f"traces/_cassettes/{SOURCE}", mode="auto")
    collect_dataset(
        TRACES_DIR / SOURCE,
        _make_backend_factory("ollama", args.model, registry,
                              CostMeter(budget_usd=0.0)),
        registry, n_healthy=args.healthy,
        n_inject_per_class=args.inject, classes=CLASSES,
        tau=args.tau, max_steps=24, model=args.model,
        task_fn=_long_task, cassette=cassette,
        # Long-form research task: no computable ground truth, so nothing can
        # verify a healthy run at collection time.
        allow_unverified_healthy=True)
    print(f"[collect_drift_long] wrote {TRACES_DIR / SOURCE}")


if __name__ == "__main__":
    main()
