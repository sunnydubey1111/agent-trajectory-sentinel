"""Gate the release artifacts: the ledger, the data card, and the docs they name.

These are the files a reader trusts without running anything, which is exactly
why they need a test. A claim that no longer matches its artifact, or a data
card describing a corpus that has since changed size, is worse than no document
at all.
"""
from __future__ import annotations


import pytest

from conftest import REPO_ROOT

RELEASE_DOCS = ["LICENSE", "CITATION.cff", "README.md", "DESIGN.md",
                "CLAIMS.md", "DATA_CARD.md", "CHECKSUMS.md", "REPRODUCE.md",
                "USER_GUIDE.md",
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


def _anchor(heading: str) -> str:
    """GitHub's slug for a markdown heading."""
    import re

    slug = heading.strip().lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    return re.sub(r"\s+", "-", slug)


def test_readme_links_resolve() -> None:
    """Every relative link in the README points at something.

    A `file.md#section` link is checked in both halves: the file must exist and
    the heading must too. Treating the fragment as part of the path fails every
    anchored link, correct ones included.

    The centred header is raw HTML, so `href=` and `src=` are scanned too --
    otherwise the badge row and the hero image would be the one part of the
    page no test covers.
    """
    import re

    text = (REPO_ROOT / "README.md").read_text("utf-8")
    targets = [m.group(1) for m in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text)]
    targets += [m.group(1) for m in
                re.finditer(r'(?:href|src)="([^"]+)"', text)]

    broken = []
    for target in targets:
        if target.startswith(("http", "#", "mailto")):
            continue
        path, _, fragment = target.partition("#")
        resolved = REPO_ROOT / path
        if not resolved.exists():
            broken.append(target)
            continue
        if fragment and resolved.suffix == ".md":
            headings = {_anchor(h) for h in re.findall(
                r"(?m)^#{1,6}\s+(.+?)\s*$", resolved.read_text("utf-8"))}
            if fragment.lower() not in headings:
                broken.append(f"{target} (no such heading)")
    assert not broken, f"broken README links: {broken}"


def test_manuscript_pdfs_rebuild_from_their_sources() -> None:
    """Where `paper/` exists, it must still hold everything a build needs.

    The manuscripts are local-only (see .gitignore), so this cannot assert
    they are present — on any checkout but the author's the directory is
    simply absent, and skipping is the correct outcome. What it still pins is
    that a tree which HAS them is not missing a piece, and that REPRODUCE.md
    keeps saying how to build them.
    """
    paper = REPO_ROOT / "paper"
    if not paper.is_dir():
        pytest.skip("manuscripts are local-only; nothing to check here")

    for name in ("main.tex", "paper.tex", "references.bib", "preprint.sty"):
        assert (paper / name).exists(), f"paper/{name} missing"

    reproduce = (REPO_ROOT / "REPRODUCE.md").read_text("utf-8")
    assert "latexmk -pdf main.tex" in reproduce, (
        "REPRODUCE.md no longer says how to build the manuscripts")

    for name in ("main.pdf", "paper.pdf"):
        pdf = paper / name
        if pdf.exists():
            assert pdf.read_bytes()[:5] == b"%PDF-", f"paper/{name} is not a PDF"


def test_markdown_to_latex_converts_the_manuscript_without_loss() -> None:
    """The converter must refuse to drop content rather than silently skip it."""
    from devtools import md_to_latex

    if not md_to_latex.SOURCE.exists():
        pytest.skip("manuscripts are local-only; nothing to convert here")
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


def test_license_is_apache_and_the_notice_names_a_holder() -> None:
    """Apache-2.0 is applied verbatim, so the holder lives in NOTICE, not LICENSE.

    Editing a name into the licence body is the usual mistake here: it makes the
    text non-standard and licence scanners stop recognising it. Section 4(d) is
    what makes NOTICE binding on anyone redistributing this, so the attribution
    and the citation have to survive in there.
    """
    text = (REPO_ROOT / "LICENSE").read_text("utf-8")
    assert "Apache License" in text and "Version 2.0, January 2004" in text
    assert "Copyright [yyyy] [name of copyright owner]" in text, (
        "the appendix boilerplate was edited; Apache-2.0 must stay verbatim")

    notice = (REPO_ROOT / "NOTICE").read_text("utf-8")
    assert "Copyright 2026 Sunny Dubey" in notice
    assert "arXiv:2608.02464" in notice, "the NOTICE must carry the citation"


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
    """Negative results are load-bearing; silently dropping one is a defect.

    They are framed as future work rather than as caveats, but the measured
    content must stay: reframing is not the same as deleting.
    """
    text = (REPO_ROOT / "README.md").read_text("utf-8")
    assert "## Future work" in text
    for phrase in ("Repair coverage is partial",
                   "only half sensitive",
                   "did not work",
                   "do not transfer across deployments",
                   "halved by measurement",
                   # The breadth corpora are the most heavily filtered, which
                   # weakens exactly the cross-framework and cross-model
                   # claims. It was in the data card and not the README, where
                   # a reader forms their impression.
                   "most heavily filtered"):
        assert phrase in text, f"README no longer states: {phrase!r}"


def test_the_readme_gives_the_real_discard_rates() -> None:
    """The range is the disclosure; a vague 'some episodes were dropped' is not.

    Each rate is read back from DATA_CARD.md, which is generated from the
    rejected.json files, so this fails if the README and the corpora disagree.
    """
    import re

    readme = (REPO_ROOT / "README.md").read_text("utf-8")
    card = (REPO_ROOT / "DATA_CARD.md").read_text("utf-8")

    for corpus in ("langgraph", "real_research7b"):
        row = re.search(rf"\|\s*`{corpus}`\s*\|[^|]*\|[^|]*\|\s*([\d.]+)%",
                        card)
        assert row, f"{corpus} has no discard row in DATA_CARD.md"
        rate = row.group(1)
        assert rate in readme, (
            f"README does not carry {corpus}'s discard rate of {rate}%")


DIAGRAMS = {
    "assets/Architecture_D2.png": "DESIGN.md",
    "assets/AgentTrajectorySentinel_GIF.gif": "README.md",
    "assets/Runtime_Flow.png": "DESIGN.md",
    "assets/Sequence_Diagram_1.png": "DESIGN.md",
    "assets/Class_Diagram.png": "DESIGN.md",
}


def test_social_card_still_states_the_published_numbers() -> None:
    """The link preview is a claim surface, and it is the least-read file here.

    It is uploaded to a GitHub setting rather than rendered from the repository,
    so nothing else would ever catch it disagreeing with the README.
    """
    from devtools import social_card

    readme = (REPO_ROOT / "README.md").read_text("utf-8")
    for figure in social_card.STAT_ASSERTIONS:
        assert figure in readme, (
            f"the social card shows {figure!r} and the README no longer does")


def test_social_card_is_committed_at_the_size_the_platforms_want() -> None:
    """1280x640. Off-size cards are cropped or refused by the link unfurlers.

    Read from the PNG header rather than by redrawing: text rasterisation
    depends on the installed font files and the freetype version, so a Linux
    CI run draws the same picture with different bytes. `--check` covers the
    byte-level question on the machine that produced the card.
    """
    from devtools import social_card

    card = REPO_ROOT / "assets" / "social_preview.png"
    assert card.exists(), "run `py -m devtools.social_card`"

    header = card.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    assert (width, height) == (social_card.WIDTH, social_card.HEIGHT)
    assert (width, height) == (1280, 640)


@pytest.mark.parametrize("path,doc", sorted(DIAGRAMS.items()))
def test_published_diagram_is_present_and_referenced(path: str, doc: str) -> None:
    """A diagram that moves or is dropped leaves a broken image in the README."""
    asset = REPO_ROOT / path
    assert asset.exists(), f"{path} is missing"
    assert asset.stat().st_size > 0
    assert path in (REPO_ROOT / doc).read_text("utf-8"), (
        f"{path} is committed but {doc} does not reference it")


def test_every_diagram_carries_alt_text() -> None:
    """Alt text is what a screen reader and a failed image load fall back to.

    Both spellings count. The README's hero image is an HTML `<img>` so that it
    can be centred, and an image that escapes the markdown regex would escape
    this guard with it.
    """
    import re

    for doc in ("README.md", "DESIGN.md"):
        text = (REPO_ROOT / doc).read_text("utf-8")
        for match in re.finditer(r"!\[([^\]]*)\]\(([^)]+)\)", text):
            alt, target = match.group(1).strip(), match.group(2)
            assert len(alt) > 20, f"{doc}: {target} has thin alt text {alt!r}"

        for tag in re.findall(r"<img\b[^>]*>", text, re.S):
            src = re.search(r'src="([^"]+)"', tag)
            assert src, f"{doc}: <img> with no src: {tag!r}"
            alt = re.search(r'alt="([^"]*)"', tag, re.S)
            assert alt, f"{doc}: {src.group(1)} has no alt text"
            words = " ".join(alt.group(1).split())

            # A badge is a label, not a diagram: "CI status" is the whole of
            # what it conveys, so it is held to being present, not to length.
            is_badge = (src.group(1).startswith("https://img.shields.io")
                        or src.group(1).endswith("badge.svg"))
            floor = 0 if is_badge else 20
            assert len(words) > floor, (
                f"{doc}: {src.group(1)} has thin alt text {words!r}")


def test_the_uncounted_root_corpus_stays_disclosed() -> None:
    """`traces/manifest.json` holds episodes no total on the data card counts.

    Every corpus count in the project globs `traces/*/manifest.json`, which
    matches subdirectories only, so the top-level manifest is invisible to the
    data card, the claims ledger and the Hugging Face export. Those episodes are
    committed and a published claim rests on them, so the card must say they
    exist. Without this test the disclosure can be regenerated away silently.
    """
    import json

    manifest = REPO_ROOT / "traces" / "manifest.json"
    if not manifest.exists():                     # nothing to disclose
        return
    n = len(json.loads(manifest.read_text("utf-8")))
    assert n, "traces/manifest.json is empty"

    card = (REPO_ROOT / "DATA_CARD.md").read_text("utf-8")
    assert "One corpus this card does not count" in card, (
        "the data card no longer discloses the uncounted root corpus")
    assert str(n) in card, f"the card does not state the {n} uncounted episodes"

    notice = (REPO_ROOT / "traces" / "NOTICE_gemini.md").read_text("utf-8")
    assert str(n) in notice, (
        f"the Gemini notice does not cover the {n} root-corpus episodes")
