"""L5 - a Gemini corpus long enough to be scoreable.

The committed Gemini corpus (`traces/real`) is 18 episodes with ONE positive
and a mean length of 5 steps. With an ESN washout of 3, most of those episodes
contribute one or two scored steps, so no labelled detection claim can rest on
them. That leaves two options: lengthen the tasks, or scope Gemini out of
labelled detection entirely.

This lengthens them. It reuses the EXACT long research task, injection classes,
onset and step budget as `collect_research7b_long` (the qwen long corpus), so
`real_gemini_long` and `real_research7b_long` differ only in the provider. That
buys two things at once: a Gemini corpus with real post-onset horizon, and a
clean cross-PROVIDER pair (local open-weights vs commercial API) on an
identical task distribution.

Spend discipline (this is the only paid collector in the project besides the
original `real` set):
  * a hard CostMeter cap, enforced by reserve() BEFORE each billed request;
  * --estimate runs the arithmetic and exits without calling;
  * every backend call is recorded to a cassette, so the corpus can be rebuilt
    offline for free and the money is spent exactly once;
  * thinking_budget=0 - on 2.5-flash reasoning tokens bill at the output rate
    (8x input) and can eat the whole max_output_tokens allowance before any
    text is produced, which would both inflate the bill and yield empty steps.

    py -m derail.experiments.collect_gemini_long --estimate
    py -m derail.experiments.collect_gemini_long --pilot 2 --yes
    py -m derail.experiments.collect_gemini_long --yes
Then: py -m derail.experiments.run_hybrid_study --datasets real_gemini_long
"""
from __future__ import annotations

import argparse

from derail.experiments.collect_research7b_long import CLASSES, _long_task
from derail.harness.collect_real import (
    RESEARCH_TASK_TOOLS,
    TRACES_DIR,
    _make_backend_factory,
    collect_dataset,
)
from derail.harness.real_tools import _ensure_tls, build_registry
from derail.harness.record_replay import Cassette, CostMeter

SOURCE = "real_gemini_long"
MODEL_DEFAULT = "gemini-2.5-flash"
#: Operator-approved hard cap for L5.
BUDGET_DEFAULT = 5.0
MAX_STEPS = 24
TAU = 2


def estimate(n_healthy: int, per_class: int, model: str) -> str:
    """Conservative arithmetic: every episode runs to the step budget.

    Each step resends the whole history, so an episode of T steps costs
    ~T*(T+1)/2 * (tokens per turn) input. Assuming ~700 input tokens of growth
    per turn and 120 output tokens per step at the 24-step ceiling is a
    deliberate over-estimate; real episodes stop earlier.
    """
    from derail.harness.record_replay import price_call

    episodes = n_healthy + per_class * len(CLASSES)
    turns = MAX_STEPS
    in_tok = int(turns * (turns + 1) / 2 * 700)
    out_tok = turns * 120
    per_ep = price_call(model, in_tok, out_tok)
    return (f"{episodes} episodes x <= ${per_ep:.3f} worst case "
            f"= <= ${episodes * per_ep:.2f} (real cost is well below this: "
            f"episodes that finish early stop billing)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="py -m derail.experiments.collect_gemini_long",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--healthy", type=int, default=30)
    ap.add_argument("--inject-per-class", type=int, default=6)
    ap.add_argument("--tau", type=int, default=TAU)
    ap.add_argument("--budget", type=float, default=BUDGET_DEFAULT,
                    help="hard USD cap, refused before any over-cap call")
    ap.add_argument("--pilot", type=int, default=0,
                    help="collect only N healthy episodes to price the run")
    ap.add_argument("--estimate", action="store_true",
                    help="print the cost estimate and exit without calling")
    ap.add_argument("--yes", action="store_true",
                    help="confirm real API spend")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    n_healthy = args.pilot or args.healthy
    per_class = 0 if args.pilot else args.inject_per_class
    print(f"[collect_gemini_long] {estimate(n_healthy, per_class, args.model)}")
    print(f"[collect_gemini_long] hard cap ${args.budget:.2f}")
    if args.estimate:
        return 0
    if not args.yes:
        print("[collect_gemini_long] refusing to spend without --yes")
        return 2

    _ensure_tls()
    registry = build_registry(RESEARCH_TASK_TOOLS)   # capability allowlist
    meter = CostMeter(budget_usd=args.budget)
    out_dir = TRACES_DIR / (args.out_dir or SOURCE)
    backend_cassette = Cassette(f"traces/_cassettes/{SOURCE}_backend",
                                mode="auto")
    collect_dataset(
        out_dir,
        _make_backend_factory("gemini", args.model, registry, meter,
                              backend_cassette=backend_cassette,
                              thinking_budget=0),
        registry, n_healthy=n_healthy, n_inject_per_class=per_class,
        classes=() if args.pilot else CLASSES,
        tau=args.tau, max_steps=MAX_STEPS, model=args.model,
        task_fn=_long_task,
        cassette=Cassette(f"traces/_cassettes/{SOURCE}", mode="auto"),
        collector="collect_gemini_long", backend="gemini", temperature=0.2,
        # Long-form research task: no computable ground truth, so nothing can
        # verify a healthy run at collection time.
        allow_unverified_healthy=True)
    print(f"[collect_gemini_long] {meter.summary()}")
    print(f"[collect_gemini_long] evaluate: py -m derail.experiments."
          f"run_hybrid_study --datasets {SOURCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
