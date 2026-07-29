"""Shared fixtures. The suite always runs against the repository checkout."""
from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_report_header(config) -> str:  # noqa: ARG001
    return f"repo root: {REPO_ROOT}"
