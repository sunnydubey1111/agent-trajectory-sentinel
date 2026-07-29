"""Re-collect the framework corpora under the v5 collection contract.

The original four corpora recorded no tool results, which left their injected
labels unverifiable from file - 62 positives could not be checked. Re-collection
under v5 records the results, so every label can be confirmed against the
evidence the run actually produced. It is affordable because the models are
local.

Episode counts match what the original collection requested, so the corpora
stay comparable in size; ACCEPTED counts are lower wherever the acceptance gate
rejects an episode the old run kept. Each corpus directory is cleared and
rewritten in place, and every rejection is recorded in its rejected.json.

    py -m devtools.recollect_frameworks              # all four
    py -m devtools.recollect_frameworks --only langgraph7b
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from derail.harness.collection import ModelUnavailable, require_ollama_model

REPO_ROOT = Path(__file__).resolve().parents[1]

# (out dir, framework, model, healthy, per-class) - matching the original plan.
# Attempt counts, not accepted counts. The first re-collection (2026-07-24)
# measured the acceptance rate of each cell under the new evidence gate:
# ~95% of healthy episodes at 7B but only ~65% at 3B (short episodes), and
# only ~1/3 of tool_cascade attempts (the ramp needs a post-onset tool call to
# bite). Attempts are sized from those rates to land near the original corpus
# sizes; every rejection is recorded in the corpus's rejected.json, so the
# acceptance rate stays visible rather than being hidden by a bigger N.
#
# The model column is part of each corpus's IDENTITY, not a convenience
# default: `langgraph`/`autogen` ARE the 3b corpora the cross-model results
# rest on. qwen2.5:3b was removed from this machine on 2026-07-26, so those
# two cells can no longer be re-collected as themselves - the preflight below
# refuses them rather than quietly producing a 7b corpus under a 3b name.
# Re-pull qwen2.5:3b to re-collect them; changing the model here would need a
# new corpus name and a ledger entry.
PLAN = [
    ("langgraph7b", "langgraph", "qwen2.5:7b", 110, 36),
    ("autogen7b", "autogen", "qwen2.5:7b", 62, 36),
    ("langgraph", "langgraph", "qwen2.5:3b", 60, 36),
    ("autogen", "autogen", "qwen2.5:3b", 32, 36),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", help="restrict to a corpus")
    ap.add_argument("--seed", type=int, default=811)
    args = ap.parse_args(argv)

    selected = [cell for cell in PLAN
                if not args.only or cell[0] in args.only]
    if not selected:
        print(f"[recollect] no corpus matches --only {args.only}")
        return 2
    # Preflight EVERY selected model before the first rmtree below: this script
    # clears a corpus directory before re-collecting it, so an unavailable
    # model would otherwise delete frozen traces and then fail. Two cells in
    # PLAN pin qwen2.5:3b, which is no longer installed here.
    missing = []
    for corpus, _framework, model, _healthy, _per_class in selected:
        try:
            require_ollama_model(model)
        except ModelUnavailable as exc:
            missing.append(f"  {corpus}: {exc}")
    if missing:
        print("[recollect] refusing to start - nothing was deleted:\n"
              + "\n".join(missing))
        return 1

    started = time.time()
    for corpus, framework, model, healthy, per_class in selected:
        out_dir = REPO_ROOT / "traces" / corpus
        print(f"\n=== {corpus}: {framework} on {model} "
              f"({healthy} healthy + {per_class}x3 injected) ===", flush=True)
        # Clear first: a rejected episode must not leave last run's file behind.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        proc = subprocess.run(
            [sys.executable, "-m", "derail.experiments.collect_framework_traces",
             "--framework", framework, "--model", model,
             "--healthy", str(healthy), "--per-class", str(per_class),
             "--seed", str(args.seed), "--out-dir", str(out_dir)],
            cwd=REPO_ROOT)
        if proc.returncode != 0:
            print(f"[recollect] {corpus} FAILED ({proc.returncode})")
            return proc.returncode
    print(f"\n[recollect] done in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
