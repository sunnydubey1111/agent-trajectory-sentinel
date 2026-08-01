"""End-to-end behavioural regression snapshot for the synthetic study.

The pipeline is bit-deterministic for a fixed master seed *within one
environment* (verified: two consecutive runs produced 1107/1107 identical leaf
values), so a stored ``results.json`` is a precise tripwire.  Every change is
run against it; an unexplained diff is a regression, an explained diff is
re-snapshotted with the reason recorded in the commit message.

Bit-exactness does not survive a change of platform, so the comparison is
tolerant in the two places where it cannot be exact - see ``_FLOAT_RTOL`` and
``_CROSS_PLATFORM_MONITORS``.

A disposable seed is used so the run never touches published artifacts, and the
temporary results directory is removed afterwards.

    py -m devtools.behavior_snapshot --write --reason "why the numbers moved"
    py -m devtools.behavior_snapshot --check
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys
import time

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SNAPSHOT_SEED = 424242
SNAPSHOT_DIR = REPO_ROOT / "tests" / "baseline"
SNAPSHOT_PATH = SNAPSHOT_DIR / f"quick_seed{SNAPSHOT_SEED}.results.json"
RUN_DIR = REPO_ROOT / "results" / f"seed{SNAPSHOT_SEED}"

# BLAS kernels differ between builds, so leaf floats land a few ulp apart on
# another platform (thresholds move at the 1e-13 level). This tolerance ignores
# that and stays far tighter than any deliberate change.
_FLOAT_RTOL = 1e-9

# torch conv and recurrent kernels are not bit-reproducible across builds, and
# the divergence selects a different threshold: tcn's theta moves ~4%, shifting
# its discrete delay and lead metrics by a whole step. No tolerance separates
# that from a regression, so these monitors' numbers are not compared. Their
# self-test covers them, and /config/monitors still catches them disappearing.
_CROSS_PLATFORM_MONITORS = ("gru", "lstm", "tcn")


def _run_experiment() -> dict:
    """Run the quick study at the disposable seed; return its results.json."""
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    proc = subprocess.run(
        [sys.executable, "-m", "derail.experiments.run_experiment",
         "--quick", "--seed", str(SNAPSHOT_SEED)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0:
        raise SystemExit(f"experiment failed ({proc.returncode})\n"
                         f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    payload = json.loads((RUN_DIR / "results.json").read_text(encoding="utf-8"))
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    return payload


def _flatten(obj: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten(value, f"{prefix}/{key}"))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            out.update(_flatten(value, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def _not_bit_reproducible(key: str) -> bool:
    return any(f"/per_monitor/{name}/" in key for name in _CROSS_PLATFORM_MONITORS)


def _same(current: object, stored: object) -> bool:
    """Equality, except that leaf floats are compared to _FLOAT_RTOL."""
    if isinstance(current, float) and isinstance(stored, float):
        return math.isclose(current, stored, rel_tol=_FLOAT_RTOL, abs_tol=0.0)
    return current == stored


def compare(current: dict, stored: dict) -> list[str]:
    """Human-readable list of differences (empty when identical)."""
    cur, old = _flatten(current), _flatten(stored)
    lines = []
    for key in sorted(set(old) - set(cur)):
        if _not_bit_reproducible(key):
            continue
        lines.append(f"  removed  {key} = {old[key]!r}")
    for key in sorted(set(cur) - set(old)):
        if _not_bit_reproducible(key):
            continue
        lines.append(f"  added    {key} = {cur[key]!r}")
    for key in sorted(set(cur) & set(old)):
        if _not_bit_reproducible(key):
            continue
        if not _same(cur[key], old[key]):
            lines.append(f"  changed  {key}: {old[key]!r} -> {cur[key]!r}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    ap.add_argument("--reason", default="", help="why the snapshot changed (--write)")
    args = ap.parse_args(argv)

    t0 = time.time()
    current = _run_experiment()
    print(f"quick study at seed {SNAPSHOT_SEED} finished in {time.time() - t0:.0f}s")

    if args.write:
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        current["_snapshot_reason"] = args.reason
        SNAPSHOT_PATH.write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {SNAPSHOT_PATH.relative_to(REPO_ROOT)} ({args.reason or 'no reason given'})")
        return 0

    if not SNAPSHOT_PATH.exists():
        raise SystemExit(f"no snapshot at {SNAPSHOT_PATH}; run --write first")
    stored = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    stored.pop("_snapshot_reason", None)
    diffs = compare(current, stored)
    if diffs:
        print(f"BEHAVIOUR CHANGED: {len(diffs)} difference(s)")
        print("\n".join(diffs[:80]))
        if len(diffs) > 80:
            print(f"  ... and {len(diffs) - 80} more")
        return 1
    print("behaviour identical to snapshot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
