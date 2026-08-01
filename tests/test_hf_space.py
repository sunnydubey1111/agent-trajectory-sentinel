"""The Space bundle: a static page plus the scores it replays.

Two things can go wrong silently. The page and the data can disagree on field
names, which no Python test would notice and which shows up as a blank chart.
And the shipped runs can drift into a curated set the monitor always catches,
which would make the demo an advertisement. Both are checked here.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

from devtools import hf_space

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("space") / "bundle"
    hf_space.build(out, "someone/some-space")
    return out


@pytest.fixture(scope="module")
def data(bundle) -> dict:
    return json.loads((bundle / "data.json").read_text("utf-8"))


def test_the_bundle_is_three_static_files(bundle) -> None:
    """A static Space runs no code: anything else shipped is dead weight."""
    names = sorted(p.name for p in bundle.rglob("*") if p.is_file())
    assert names == ["README.md", "data.json", "index.html"]


def test_the_card_declares_a_static_space(bundle) -> None:
    card = (bundle / "README.md").read_text("utf-8")
    assert card.startswith("---\n")
    assert "sdk: static" in card, "gradio and docker Spaces are not free"
    assert "app_file: index.html" in card


def test_frontmatter_respects_the_hubs_field_limits(bundle) -> None:
    """The hub rejects the whole upload over a long short_description, and
    only after the payload has been built and the repo created."""
    card = (bundle / "README.md").read_text("utf-8")
    header = card.split("---\n")[1]
    fields = dict(line.split(":", 1) for line in header.splitlines()
                  if ":" in line and not line.startswith(" "))
    description = fields["short_description"].strip()
    assert len(description) <= 60, f"{len(description)} chars: {description!r}"
    assert description, "the hub shows this under the Space title"


#: The contract between the builder and the page. A rename on either side
#: leaves a blank chart and raises nothing anywhere, so it is pinned.
TOP_LEVEL_FIELDS = ("theta", "runs", "n_val", "fa_budget")
RUN_FIELDS = ("id", "cls", "tau", "scores", "alarm", "steps")


def test_every_field_the_page_reads_exists_in_the_data(bundle, data) -> None:
    page = (bundle / "index.html").read_text("utf-8")
    for field in TOP_LEVEL_FIELDS:
        assert field in data, f"data.json is missing {field}"
        assert f"data.{field}" in page, f"page never reads {field}"
    run = data["runs"][0]
    for field in RUN_FIELDS:
        assert field in run, f"a run record is missing {field}"
        assert re.search(rf"\b(r|run|S\.run)\.{field}\b", page), (
            f"page never reads run.{field}")


def test_run_records_are_complete_and_consistent(data) -> None:
    for run in data["runs"]:
        assert len(run["scores"]) == len(run["steps"]), run["id"]
        assert run["tau"] is None or 0 <= run["tau"] < len(run["scores"])
        if run["alarm"] is not None:
            assert 0 <= run["alarm"] < len(run["scores"])
            assert run["scores"][run["alarm"]] > data["theta"]


def test_the_alarm_index_is_the_first_crossing(data) -> None:
    """The page marks run.alarm as the first crossing; if it is not, the star
    lands on the wrong step and every lead time shown is wrong."""
    for run in data["runs"]:
        above = [i for i, s in enumerate(run["scores"]) if s > data["theta"]]
        assert run["alarm"] == (above[0] if above else None), run["id"]


def test_every_failure_class_is_represented(data) -> None:
    source = json.loads((REPO_ROOT / "traces" / hf_space.CORPUS /
                         "manifest.json").read_text("utf-8"))
    available = {e["failure_class"] for e in source if e["failure_class"]}
    shipped = {r["cls"] for r in data["runs"] if r["cls"]}
    assert shipped == available, f"missing: {available - shipped}"


def test_both_healthy_and_failing_runs_are_shipped(data) -> None:
    assert sum(1 for r in data["runs"] if r["tau"] is None) >= 3
    assert sum(1 for r in data["runs"] if r["tau"] is not None) >= 8


def test_the_demo_is_not_curated_to_only_show_catches(data) -> None:
    """The selection rule is 'two per class by horizon', not 'ones that work'.

    If a future change ever makes every shipped failure a catch, that is worth
    a deliberate decision rather than a silent slide into marketing.
    """
    failing = [r for r in data["runs"] if r["tau"] is not None]
    missed = [r for r in failing if r["alarm"] is None]
    assert missed, ("every shipped failure is caught; the card claims misses "
                    "are included on purpose, so either the claim or the "
                    "selection needs updating")


def test_the_card_states_that_scores_are_precomputed(bundle) -> None:
    """The page must not imply a model is running when none is."""
    card = " ".join((bundle / "README.md").read_text("utf-8").split())
    assert "not when you clicked" in card or "when this page was built" in card
    page = " ".join((bundle / "index.html").read_text("utf-8").split())
    assert "not in your browser" in page


def test_the_page_carries_no_external_dependency(bundle) -> None:
    """A Space that fetches a CDN breaks when the CDN does, and leaks a
    visitor's IP to a third party."""
    page = (bundle / "index.html").read_text("utf-8")
    for pattern in ("http://", "src=\"//", "cdn."):
        offenders = [ln for ln in page.splitlines()
                     if pattern in ln and "github.com" not in ln]
        assert not offenders, offenders[:2]
