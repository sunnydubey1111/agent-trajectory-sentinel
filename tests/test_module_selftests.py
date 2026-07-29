"""Make the modules' embedded ``__main__`` self-tests discoverable.

Historically every library module carried its own assertion block that only ran
when someone typed ``py -m derail.monitor.esn`` by hand, so nothing checked them
in bulk and ``python -O`` silently removed them.  Each block is executed here as
a subprocess, which keeps the assertions live and gives the suite a real gate.

New behaviour is covered by focused unit tests in the other ``tests/`` modules;
this file only guarantees that the historical self-checks keep passing.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from conftest import REPO_ROOT

# (module, marks) - runtimes measured on the dev box, 2026-07-24.
FAST = [
    "derail.evaluation.stats",
    "derail.harness.agent_loop",
    "derail.harness.collect_real",
    "derail.harness.inject",
    "derail.harness.record_replay",
    "derail.harness.tools",
    "derail.monitor.baseline",
    "derail.monitor.baselines",
    "derail.monitor.calibration",
    "derail.monitor.escalation",
    "derail.monitor.esn",
    "derail.monitor.grounding_verify",
    "derail.telemetry.adapter",
    "derail.telemetry.generator",
    "derail.verify.checks",
    "verification.organic_hallucination",
]
SLOW = [
    "derail.evaluation.metrics",
    "derail.monitor.grounding",
    "derail.monitor.hmt_esn",
    "derail.monitor.hybrid",
    "derail.monitor.seq_baselines",
    "derail.harness.demo_real",
    "derail.harness.frameworks",
]


def _run_selftest(module: str, timeout: int = 900) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", module],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"`py -m {module}` exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )


@pytest.mark.parametrize("module", FAST)
def test_selftest_fast(module: str) -> None:
    _run_selftest(module)


@pytest.mark.slow
@pytest.mark.parametrize("module", SLOW)
def test_selftest_slow(module: str) -> None:
    _run_selftest(module)


@pytest.mark.slow
@pytest.mark.network
def test_selftest_real_tools() -> None:
    # fixed in Phase 6: vector search is now BM25 lexical retrieval (no
    # qdrant init-failure branch), so the self-test passes. Marked `network`
    # because its final probe hits live Wikipedia.
    _run_selftest("derail.harness.real_tools")
