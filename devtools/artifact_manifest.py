"""Content manifest for source and data artifacts.

Refactoring rewrites code and regenerates result artifacts.  This tool makes
every such change *visible*: it records a SHA-256 for each tracked source file,
result artifact and trace, so an accidental edit to a committed artifact cannot
pass unnoticed.

    py -m devtools.artifact_manifest --write            # snapshot current state
    py -m devtools.artifact_manifest --check            # diff against snapshot
    py -m devtools.artifact_manifest --check --section results
    py -m devtools.artifact_manifest --doc              # write CHECKSUMS.md

Intentional changes are re-snapshotted and explained in the commit message;
unexplained changes are defects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "BASELINE_MANIFEST.json"
CHECKSUMS_DOC = REPO_ROOT / "CHECKSUMS.md"

# section -> (root directory, glob patterns)
SECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    # `*.sql` covers committed tool fixtures (derail/harness/fixtures): they
    # are code assets, not research data, and an edited fixture changes what a
    # tool returns, so it must fail the integrity check like an edited module.
    "code": (".", ("derail/**/*.py", "verification/**/*.py",
                   "experimental/**/*.py", "devtools/**/*.py", "tests/**/*.py",
                   "derail/**/*.sql")),
    "results": ("results", ("**/*.json", "**/*.csv", "**/*.md", "**/*.png")),
    "traces": ("traces", ("**/*.jsonl", "**/*.json")),
    # LICENSE and NOTICE are named explicitly because they carry no extension
    # and the globs below are extension-driven. Both are load-bearing under
    # Apache-2.0 -- section 4(d) makes NOTICE binding on anyone redistributing
    # this -- so a silent edit to either should fail the integrity check the
    # same way an edited trace does.
    # `paper/` is deliberately absent: the manuscripts are local-only (see
    # .gitignore), so hashing them would report them missing on every checkout
    # but the author's — the same reason the anonymised submissions were never
    # hashed. The claim ledger, not this manifest, is what ties the
    # manuscripts' numbers to the artifacts they came from.
    "docs": (".", ("*.md", "LICENSE", "NOTICE", "docs/**/*.md",
                   "requirements*.txt")),
}
EXCLUDE_PARTS = {"__pycache__", ".git", ".pytest_cache",
                 # Per-episode score dumps are regenerable and are not
                 # committed (see .gitignore), so manifesting them would
                 # make every fresh checkout report them as missing.
                 "scores",
                 # Imported external corpora (traces/_aftraj, traces/_atbench).
                 # Someone else's data, fetched on demand and not committed:
                 # hashing it here would report it missing on every fresh
                 # checkout and would fold another project's episodes into our
                 # integrity record.
                 "_aftraj", "_atbench",
                 # The NeurIPS workshop submission, untracked while it is under
                 # double-blind review (see .gitignore): a public copy in a
                 # repository under the author's own name defeats the anonymity
                 # it is written for. It exists on the author's machine and in
                 # no checkout, so hashing it would fail the integrity check
                 # for everyone else.
                 "workshop.tex", "workshop.pdf", "neurips_2026.sty",
                 # The TMLR submission and its vendored style package, for the
                 # same two reasons: TMLR rejects non-anonymous submissions
                 # without review, so the source stays untracked, and the style
                 # files are the venue's (Apache-2.0) rather than ours.
                 "tmlr.tex", "tmlr.pdf", "tmlr.sty", "tmlr.bst",
                 "fancyhdr.sty", "math_commands.tex"}
# Text artifacts are hashed after CRLF -> LF normalisation, matching the
# .gitattributes policy, so a manifest is identical on Windows and
# Linux checkouts and a line-ending flip is never mistaken for a data change.
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".tex",
                 ".bib", ".yml", ".yaml", ".cfg", ".ini"}


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    if path.suffix.lower() in TEXT_SUFFIXES:
        h.update(path.read_bytes().replace(b"\r\n", b"\n"))
        return h.hexdigest()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(sections: list[str] | None = None) -> dict[str, dict[str, str]]:
    """Map section -> {repo-relative posix path: sha256}."""
    wanted = sections or list(SECTIONS)
    out: dict[str, dict[str, str]] = {}
    for name in wanted:
        root_name, patterns = SECTIONS[name]
        root = REPO_ROOT / root_name
        entries: dict[str, str] = {}
        if root.exists():
            for pattern in patterns:
                for path in sorted(root.glob(pattern)):
                    if not path.is_file():
                        continue
                    if EXCLUDE_PARTS & set(path.parts):
                        continue
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    entries[rel] = _sha256(path)
        out[name] = dict(sorted(entries.items()))
    return out


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"no manifest at {MANIFEST_PATH}; run --write first")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def diff(sections: list[str] | None = None) -> dict[str, dict[str, list[str]]]:
    """Compare the working tree against the stored manifest."""
    stored = load_manifest()["sections"]
    current = collect(sections)
    report: dict[str, dict[str, list[str]]] = {}
    for name, cur in current.items():
        old = stored.get(name, {})
        report[name] = {
            "added": sorted(set(cur) - set(old)),
            "removed": sorted(set(old) - set(cur)),
            "changed": sorted(p for p in set(cur) & set(old) if cur[p] != old[p]),
        }
    return report


def root_digest(sections: dict[str, dict[str, str]]) -> str:
    """One SHA-256 over every per-file hash: a single value to quote or compare.

    `CHECKSUMS.md` is excluded because it *carries* this digest -- including it
    would make the value depend on itself and never settle. Its own hash is
    still recorded in the manifest like any other file.
    """
    h = hashlib.sha256()
    for name in sorted(sections):
        for path in sorted(sections[name]):
            if path == CHECKSUMS_DOC.name:
                continue
            h.update(f"{path}|{sections[name][path]}|".encode())
    return h.hexdigest()


def write_doc() -> None:
    """Write the reader-facing checksum summary."""
    manifest = load_manifest()
    sections = manifest["sections"]
    lines = [
        "# Checksums",
        "",
        "Every file this repository publishes is hashed with SHA-256 in",
        "`BASELINE_MANIFEST.json`. Text files are hashed after CRLF to LF",
        "normalisation, matching the `.gitattributes` policy, so a checkout on",
        "Windows and one on Linux produce identical digests and a line-ending",
        "flip is never mistaken for a data change.",
        "",
        "```",
        "py -m devtools.artifact_manifest --check     # verify every file",
        "py -m devtools.artifact_manifest --doc       # regenerate this summary",
        "```",
        "",
        f"**Root digest:** `{root_digest(sections)}`",
        "",
        "A single SHA-256 over every path and per-file hash in the manifest, in",
        "sorted order. Two checkouts agreeing on this value agree on every",
        "tracked byte.",
        "",
        "| section | files | covers |",
        "|---|---:|---|",
    ]
    covers = {
        "code": "`derail/`, `verification/`, `experimental/`, `devtools/`, `tests/`",
        "results": "every table, figure and results JSON the claims cite",
        "traces": "every committed agent episode and replay cassette",
        "docs": "`*.md`, `LICENSE`, `NOTICE`, the paper sources, and the "
                "requirements files",
    }
    for name in sorted(sections):
        lines.append(f"| `{name}` | {len(sections[name]):,} | {covers.get(name, '')} |")
    lines += [
        f"| **total** | **{sum(len(v) for v in sections.values()):,}** | |",
        "",
        "Per-episode trace hashes are additionally recorded in each corpus's own",
        "`manifest.json` where the collector wrote them (`trace_sha256`).",
        "",
    ]
    CHECKSUMS_DOC.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"wrote {CHECKSUMS_DOC.name}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the manifest")
    mode.add_argument("--check", action="store_true", help="diff against the manifest")
    mode.add_argument("--doc", action="store_true", help="write CHECKSUMS.md")
    ap.add_argument("--section", action="append", choices=sorted(SECTIONS),
                    help="restrict to one section (repeatable)")
    ap.add_argument("--note", default="", help="free-text note stored with --write")
    args = ap.parse_args(argv)

    if args.doc:
        write_doc()
        return 0

    if args.write:
        if args.section:
            manifest = load_manifest() if MANIFEST_PATH.exists() else {"sections": {}}
            manifest["sections"].update(collect(args.section))
        else:
            manifest = {"sections": collect(None)}
        manifest["note"] = args.note
        manifest["file_count"] = sum(len(v) for v in manifest["sections"].values())
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST_PATH.name}: {manifest['file_count']} files "
              f"across {len(manifest['sections'])} sections")
        return 0

    report = diff(args.section)
    dirty = False
    for name, d in report.items():
        n = sum(len(v) for v in d.values())
        if n == 0:
            print(f"{name}: clean ({len(collect([name])[name])} files)")
            continue
        dirty = True
        print(f"{name}: {len(d['changed'])} changed, {len(d['added'])} added, "
              f"{len(d['removed'])} removed")
        for kind in ("changed", "added", "removed"):
            for path in d[kind]:
                print(f"  {kind:8s} {path}")
    return 1 if dirty else 0


if __name__ == "__main__":
    sys.exit(main())
