"""End-to-end behavioural tripwire.

`tests/baseline/quick_seed424242.results.json` is the full result payload of a
quick study at a disposable seed.  The pipeline is bit-deterministic for a fixed
seed, so any difference here is a real behavioural change - intended or not.

When a change intentionally moves the numbers, re-snapshot with

    py -m devtools.behavior_snapshot --write --reason "<finding id>: <what changed>"

and record the same reason in the commit message.  Never re-snapshot to make
a red test go green without that explanation.
"""
from __future__ import annotations

import json

import pytest

from devtools import behavior_snapshot


@pytest.mark.slow
def test_quick_study_matches_snapshot() -> None:
    assert behavior_snapshot.SNAPSHOT_PATH.exists(), (
        "no behavioural snapshot; run py -m devtools.behavior_snapshot --write")
    stored = json.loads(behavior_snapshot.SNAPSHOT_PATH.read_text(encoding="utf-8"))
    stored.pop("_snapshot_reason", None)
    current = behavior_snapshot._run_experiment()
    diffs = behavior_snapshot.compare(current, stored)
    assert not diffs, "quick study drifted from the snapshot:\n" + "\n".join(diffs[:60])
