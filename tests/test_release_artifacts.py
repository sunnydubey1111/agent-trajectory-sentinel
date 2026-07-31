"""Gate the release artifacts: the ledger, the data card, and the docs they name.

These are the files a reader trusts without running anything, which is exactly
why they need a test. A claim that no longer matches its artifact, or a data
card describing a corpus that has since changed size, is worse than no document
at all.
"""
from __future__ import annotations

import pathlib

import pytest

from conftest import REPO_ROOT

RELEASE_DOCS = ["LICENSE", "CITATION.cff", "README.md", "DESIGN.md",
                "CLAIMS.md", "DATA_CARD.md", "CHECKSUMS.md", "REPRODUCE.md",
                "requirements.txt", "requirements.lock.txt",
                "requirements-core.lock.txt"]


@pytest.mark.parametrize("name", RELEASE_DOCS)
def test_release_document_is_present_and_not_empty(name: str) -> None:
    path = REPO_ROOT / name
    assert path.exists(), f"{name} is missing from the release"
    assert path.stat().st_size > 0, f"{name} is empty"


def test_every_published_claim_matches_its_artifact() -> None:
    """The headline numbers are recomputed from `results/`, not trusted."""
    from devtools import claims_ledger

    mismatched = [c for c in claims_ledger.build() if not c.check()]
    assert not mismatched, "\n".join(
        f"{c.id}: claimed {c.expected!r}, artifact gives {c.actual!r} ({c.source})"
        for c in mismatched)


def test_claims_ledger_file_is_current() -> None:
    """`CLAIMS.md` is generated; a hand-edit or a stale copy fails here."""
    from devtools import claims_ledger

    claims = claims_ledger.build()
    ok = all(c.check() for c in claims)
    expected = claims_ledger._render(claims, ok)
    current = claims_ledger.LEDGER_PATH.read_text("utf-8").replace("\r\n", "\n")
    assert current == expected, "run `py -m devtools.claims_ledger --write`"


def test_data_card_matches_the_committed_corpora() -> None:
    from devtools import data_card

    current = data_card.CARD_PATH.read_text("utf-8").replace("\r\n", "\n")
    assert current == data_card.render(), "run `py -m devtools.data_card --write`"


def test_data_card_describes_every_corpus() -> None:
    """A corpus added without a purpose line would ship an unexplained row."""
    from devtools import data_card

    corpora = {name for name, _ in data_card._corpora()}
    undescribed = sorted(corpora - set(data_card.PURPOSE))
    assert not undescribed, f"corpora with no purpose line: {undescribed}"


def test_readme_links_resolve() -> None:
    """Every relative markdown link in the README points at something."""
    import re

    text = (REPO_ROOT / "README.md").read_text("utf-8")
    broken = []
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = match.group(1)
        if target.startswith(("http", "#", "mailto")):
            continue
        if not (REPO_ROOT / target).exists():
            broken.append(target)
    assert not broken, f"broken README links: {broken}"


def test_manuscript_pdfs_are_published() -> None:
    """Both papers ship as PDFs so a reader needs no LaTeX toolchain."""
    for name in ("main.pdf", "paper.pdf"):
        pdf = REPO_ROOT / "paper" / name
        assert pdf.exists(), f"paper/{name} is missing; see REPRODUCE.md"
        assert pdf.read_bytes()[:5] == b"%PDF-", f"paper/{name} is not a PDF"


def test_markdown_to_latex_converts_the_manuscript_without_loss() -> None:
    """The converter must refuse to drop content rather than silently skip it."""
    from devtools import md_to_latex

    source = md_to_latex.SOURCE.read_text("utf-8")
    tex = md_to_latex.convert(source)
    import re

    md_headings = len(re.findall(r"^#{2,6} ", source, re.M))
    tex_sections = len(re.findall(r"\\(?:sub)*section\{", tex))
    assert md_headings == tex_sections, (md_headings, tex_sections)
    md_tables = len(re.findall(r"^\|[-: |]+\|$", source, re.M))
    assert md_tables == tex.count(r"\begin{longtable}")


def test_markdown_to_latex_raises_on_an_unmapped_character() -> None:
    with pytest.raises(Exception):
        from devtools import md_to_latex

        md_to_latex._escape("\u2603")          # snowman: deliberately unmapped


def test_license_names_a_holder() -> None:
    text = (REPO_ROOT / "LICENSE").read_text("utf-8")
    assert "MIT License" in text
    assert "Copyright (c)" in text and text.count("<") == 0, (
        "the LICENSE still carries a placeholder")


def test_citation_file_parses_as_yaml_like_records() -> None:
    """No YAML dependency: check the fields a citation consumer needs."""
    text = (REPO_ROOT / "CITATION.cff").read_text("utf-8")
    for key in ("cff-version:", "title:", "authors:", "license:", "version:",
                "date-released:"):
        assert key in text, f"CITATION.cff is missing {key!r}"


def test_reproduce_record_names_every_verification_gate() -> None:
    """A gate that exists but is undocumented will not be run by a reader."""
    text = (REPO_ROOT / "REPRODUCE.md").read_text("utf-8")
    for command in ("devtools.behavior_snapshot --check",
                    "devtools.artifact_manifest --check",
                    "devtools.claims_ledger --check",
                    "devtools.data_card --check"):
        assert command in text, f"REPRODUCE.md does not mention `{command}`"


def test_limitations_survive_in_the_readme() -> None:
    """Negative results are load-bearing; silently dropping one is a defect."""
    text = (REPO_ROOT / "README.md").read_text("utf-8")
    for phrase in ("Repair coverage is partial",
                   "only half sensitive",
                   "did not work",
                   "do not transfer across deployments",
                   "halved by measurement"):
        assert phrase in text, f"README no longer states: {phrase!r}"
