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

# (module, marks) - runtimes measured on the dev box.
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
    "derail.harness.frameworks",
    "verification.organic_hallucination",
]
SLOW = [
    "derail.evaluation.metrics",
    "derail.monitor.grounding",
    "derail.monitor.hmt_esn",
    "derail.monitor.hybrid",
    "derail.monitor.seq_baselines",
    "derail.harness.demo_real",
]

# Modules whose bare `__main__` would reach a live model or the network, and
# the flag that runs their offline half instead. The live path is exercised by
# its own marked test below, so the default gate stays deterministic: a live
# agent picks its own search queries, which misses the committed cassettes and
# writes new ones into `traces/_cassettes/`.
OFFLINE_ARGS = {"derail.harness.frameworks": ["--check"]}


def _run_selftest(module: str, args: list[str] | None = None,
                  timeout: int = 900) -> None:
    argv = args if args is not None else OFFLINE_ARGS.get(module, [])
    proc = subprocess.run(
        [sys.executable, "-m", module, *argv],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"`py -m {module} {' '.join(argv)}`".rstrip()
            + f" exited {proc.returncode}\n"
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
    # Vector search is BM25 lexical retrieval, so there is no qdrant
    # init-failure branch to trip over. Marked `network` because the final
    # probe hits live Wikipedia.
    _run_selftest("derail.harness.real_tools")


@pytest.mark.slow
@pytest.mark.ollama
@pytest.mark.network
def test_selftest_frameworks_live() -> None:
    """The live half of the frameworks self-test: a real agent episode.

    Needs a served model and network access, and records any cassette it
    misses, so it is excluded from the default gate. The offline `--check`
    half runs there instead and covers the LangGraph/AutoGen wrapping.
    """
    _run_selftest("derail.harness.frameworks", args=[])
