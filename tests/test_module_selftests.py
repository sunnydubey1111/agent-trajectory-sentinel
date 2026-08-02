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


@pytest.fixture
def no_new_cassettes():
    """Leave `traces/_cassettes/` exactly as the test found it.

    A live agent picks its own search queries, so it misses the committed
    cassettes and records replacements. Those recordings are incidental --- an
    artefact of whatever the model happened to ask this time --- not corpus
    data, and leaving them behind dirties the working tree and shifts the
    artifact manifest's trace count. The marks on the live test do not prevent
    this: `slow`/`network`/`ollama` only exclude when a `-m` filter is passed,
    and a bare `py -m pytest` runs it.
    """
    root = REPO_ROOT / "traces" / "_cassettes"
    before = {p for p in root.rglob("*.json")} if root.exists() else set()
    yield
    if not root.exists():
        return
    for path in root.rglob("*.json"):
        if path not in before:
            path.unlink()
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()),
                            key=lambda p: -len(p.parts)):
        if not any(directory.iterdir()):
            directory.rmdir()


@pytest.mark.slow
@pytest.mark.ollama
@pytest.mark.network
def test_selftest_frameworks_live(no_new_cassettes) -> None:
    """The live half of the frameworks self-test: a real agent episode.

    Needs a served model and network access. It records any cassette it
    misses, so it runs under `no_new_cassettes`, which removes recordings this
    test created rather than leaving them in the tree. The offline `--check`
    half covers the LangGraph/AutoGen wrapping without a model.
    """
    _run_selftest("derail.harness.frameworks", args=[])
