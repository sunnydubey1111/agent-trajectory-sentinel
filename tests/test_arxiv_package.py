"""The arXiv upload must compile without this repository around it.

arXiv receives a flat directory, not a checkout. A figure path that resolves
here and nowhere else does not fail the build — it leaves a blank box in the
published PDF, which is discovered by a reader rather than by the author.
"""
from __future__ import annotations

import pathlib
import re

import pytest

from devtools import arxiv_package

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


#: The manuscripts are local-only (see .gitignore), so a checkout without them
#: has no package to assemble and every test here skips rather than fails.
pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "paper" / "main.tex").exists(),
    reason="manuscripts are local-only; no arXiv source to package")


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("arxiv") / "pkg"
    arxiv_package.build(out)
    return out


def test_nothing_the_source_needs_is_missing(tmp_path) -> None:
    summary = arxiv_package.build(tmp_path / "pkg")
    assert not summary["missing"], summary["missing"]
    assert summary["figures"] >= 5


def test_every_figure_sits_beside_the_source(package) -> None:
    tex = (package / f"{arxiv_package.STEM}.tex").read_text("utf-8")
    for name in arxiv_package._figures(tex):
        assert (package / name).exists(), f"{name} not in the upload"


def test_the_checkout_relative_graphics_path_is_gone(package) -> None:
    """It resolves here and nowhere else; on arXiv it yields empty boxes."""
    tex = (package / f"{arxiv_package.STEM}.tex").read_text("utf-8")
    assert "../results/figures" not in tex


def test_the_bibliography_source_always_ships(package) -> None:
    """Without references.bib arXiv cannot build a bibliography at all."""
    assert (package / "references.bib").exists()


def test_the_compiled_bibliography_is_renamed_with_the_source(package) -> None:
    """When a .bbl exists it must carry the job name, not `main`.

    latexmk resolves the bibliography by job name, so a main.bbl beside a
    renamed source is silently ignored and every citation degrades to a
    question mark --- visible only after submission. The .bbl is a build
    product and is gitignored, so a fresh checkout has none until the paper
    has been compiled; that is not a packaging error, and arXiv will run
    BibTeX from references.bib instead.
    """
    if not (REPO_ROOT / "paper" / "main.bbl").exists():
        pytest.skip("no main.bbl in this checkout; compile paper/main.tex")
    assert (package / f"{arxiv_package.STEM}.bbl").exists()
    assert not (package / "main.bbl").exists(), "the old job name shipped too"


def test_the_style_file_is_included(package) -> None:
    tex = (package / f"{arxiv_package.STEM}.tex").read_text("utf-8")
    for match in re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", tex):
        for name in match.split(","):
            local = package / f"{name.strip()}.sty"
            if (REPO_ROOT / "paper" / f"{name.strip()}.sty").exists():
                assert local.exists(), f"{name} is local but was not shipped"


def test_the_arxiv_version_is_not_anonymous(package) -> None:
    """A preprint is attributed; that is the point of a preprint server."""
    tex = (package / f"{arxiv_package.STEM}.tex").read_text("utf-8")
    assert "Sunny Dubey" in tex
    assert "0009-0002-8296-8631" in tex, "ORCID missing from the preprint"
    assert "Anonymous" not in tex


def test_the_preprint_points_at_the_public_artifacts(package) -> None:
    """A preprint whose artifact is public should say where it is."""
    tex = (package / f"{arxiv_package.STEM}.tex").read_text("utf-8")
    for target in ("github.com/sunnydubey1111/agent-trajectory-sentinel",
                   "huggingface.co/datasets/sunnydubey1111",
                   "huggingface.co/spaces/sunnydubey1111"):
        assert target in tex, f"{target} not cited in the preprint"


def test_an_author_blind_manuscript_carries_no_identifier() -> None:
    """An author-blind manuscript must not name its author or its artifacts.

    Manuscript sources are local to the author (see .gitignore), so a fresh
    checkout has nothing to check and skips. Where one exists, this is what
    stops an edit made for the attributed preprint from reaching it.
    """
    source = REPO_ROOT / "paper" / "workshop.tex"
    if not source.exists():
        pytest.skip("manuscript sources are local-only; nothing to check here")
    tex = source.read_text("utf-8")
    for probe in ("Sunny", "Dubey", "0009-0002", "github.com", "huggingface"):
        assert probe not in tex, f"{probe!r} would identify the author"
