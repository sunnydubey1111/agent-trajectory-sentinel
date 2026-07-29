"""Content manifest for source and data artifacts.

The remediation of rewrites a large amount of code and
regenerates several result artifacts.  This tool makes every such change
*visible*: it records a SHA-256 for each tracked source file, result artifact
and trace, so an accidental edit to a committed artifact cannot pass unnoticed
between phases.

    py -m devtools.artifact_manifest --write            # snapshot current state
    py -m devtools.artifact_manifest --check            # diff against snapshot
    py -m devtools.artifact_manifest --check --section results

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

# section -> (root directory, glob patterns)
SECTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "code": (".", ("derail/**/*.py", "verification/**/*.py",
                   "experimental/**/*.py", "devtools/**/*.py", "tests/**/*.py")),
    "results": ("results", ("**/*.json", "**/*.csv", "**/*.md", "**/*.png")),
    "traces": ("traces", ("**/*.jsonl", "**/*.json")),
    "docs": (".", ("*.md", "docs/**/*.md", "paper/**/*.md", "paper/**/*.tex",
                   "paper/**/*.bib", "requirements*.txt")),
}
EXCLUDE_PARTS = {"__pycache__", ".git", ".pytest_cache",
                 # Per-episode score dumps are regenerable and are not
                 # committed (see .gitignore), so manifesting them would
                 # make every fresh checkout report them as missing.
                 "scores"}
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write the manifest")
    mode.add_argument("--check", action="store_true", help="diff against the manifest")
    ap.add_argument("--section", action="append", choices=sorted(SECTIONS),
                    help="restrict to one section (repeatable)")
    ap.add_argument("--note", default="", help="free-text note stored with --write")
    args = ap.parse_args(argv)

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
