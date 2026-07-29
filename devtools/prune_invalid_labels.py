"""Apply the collection acceptance gate retroactively to an existing corpus.

Some corpora contain a handful of episodes that the gate would refuse today -
chiefly positives whose stochastic injector never fired.  Where the
corpus is otherwise sound, re-collecting hundreds of good episodes to remove a
few bad labels would be wasteful, so this tool applies the same rule after the
fact: the offending episodes leave `manifest.json` and are recorded in
`rejected.json` with the auditor's reason.

The trace files are NOT deleted - they stay as provenance of what was
collected.  What changes is the claim the manifest makes about them.

    py -m devtools.prune_invalid_labels --corpus real_research7b --dry-run
    py -m devtools.prune_invalid_labels --corpus real_research7b --apply

Demo calibration corpora are deliberately out of scope: whether an
unsuccessful demo run counts as healthy is the threshold decision,
not a labelling error.
"""
from __future__ import annotations

import argparse
import json
import sys

from devtools.trace_audit import BLOCKING, TRACES, audit_corpus

EXCLUDED = {"demo7b", "demo7b_scoped", "organic7b", "organic_demo7b"}


def prune(corpus: str, apply: bool) -> dict:
    if corpus in EXCLUDED:
        raise SystemExit(f"{corpus} is out of scope for this tool "
                         f"(see the module docstring)")
    corpus_dir = TRACES / corpus
    manifest_path = corpus_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    audits = {a.episode_id: a for a in audit_corpus(corpus_dir)}

    keep, drop = [], []
    for entry in manifest:
        audit = audits.get(entry["episode_id"])
        defects = sorted(set(audit.defects) & BLOCKING) if audit else []
        if defects:
            drop.append({"episode_id": entry["episode_id"],
                         "requested_class": entry.get("failure_class"),
                         "reason": f"retroactive gate: {', '.join(defects)}",
                         "facts": dict(audit.detail) | {"T": audit.T,
                                                        "tau": audit.tau}})
        else:
            keep.append(entry)

    if apply and drop:
        rejected_path = corpus_dir / "rejected.json"
        existing = (json.loads(rejected_path.read_text("utf-8"))
                    if rejected_path.exists() else [])
        rejected_path.write_text(json.dumps(existing + drop, indent=2), "utf-8")
        manifest_path.write_text(json.dumps(keep, indent=2), "utf-8")
    return {"corpus": corpus, "kept": len(keep), "dropped": len(drop),
            "dropped_ids": [d["episode_id"] for d in drop]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    for corpus in args.corpus:
        report = prune(corpus, apply=args.apply)
        verb = "would drop" if args.dry_run else "dropped"
        print(f"{report['corpus']}: kept {report['kept']}, {verb} "
              f"{report['dropped']}")
        for episode_id in report["dropped_ids"]:
            print(f"    {episode_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
