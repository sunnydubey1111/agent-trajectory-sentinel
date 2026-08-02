"""Assemble the arXiv submission from `paper/main.tex`.

arXiv compiles a self-contained upload: it does not have this repository, so
`\\graphicspath{{../results/figures/}}` resolves to nothing there and every
figure silently becomes a missing-image box. This module builds a flat
directory that compiles on its own - sources, style, bibliography and the five
figures side by side - and rewrites the one path that assumed a checkout.

    py -m devtools.arxiv_package --build          # write build/arxiv/
    py -m devtools.arxiv_package --build --check  # ... and compile it there

The submission is NOT anonymous: arXiv is a preprint server, so the author
block, the ORCID and the artifact links all belong in it. That is the opposite
of `paper/workshop.tex`, which is anonymised for double-blind review. Both are
generated from their own source and neither is derived from the other.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "paper"
FIGURES = REPO_ROOT / "results" / "figures"
BUILD_DIR = REPO_ROOT / "build" / "arxiv"

#: Files the upload needs beside main.tex. The .bbl is included deliberately:
#: arXiv runs BibTeX only when it must, and shipping the compiled bibliography
#: removes a class of build failure that is invisible until after submission.
SUPPORT = ("neurips.sty", "references.bib", "main.bbl")


def _figures(tex: str) -> list[str]:
    return re.findall(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)


def build(out_dir: pathlib.Path) -> dict:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    tex = (PAPER / "main.tex").read_text("utf-8")

    # Everything sits in one directory in the upload, so the checkout-relative
    # graphics path is not just wrong there, it fails silently.
    tex = tex.replace("\\graphicspath{{../results/figures/}}",
                      "% graphicspath removed: the arXiv upload is flat")
    (out_dir / "main.tex").write_text(tex, encoding="utf-8", newline="\n")

    missing = []
    for name in SUPPORT:
        source = PAPER / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
        else:
            missing.append(name)

    figures = _figures(tex)
    for name in figures:
        source = FIGURES / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
        else:
            missing.append(name)

    return {"figures": len(figures), "missing": missing,
            "bytes": sum(p.stat().st_size for p in out_dir.iterdir())}


def check(out_dir: pathlib.Path) -> tuple[bool, str]:
    """Compile the package in its own directory, as arXiv will."""
    try:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                        "main.tex"], cwd=out_dir, capture_output=True,
                       text=True)
    except OSError as exc:
        return False, f"could not run latexmk ({exc}); compile it by hand"

    log_path = out_dir / "main.log"
    if not log_path.exists():
        # latexmk exited without producing a log at all, which on this
        # toolchain means it never really started rather than that the
        # document is broken. Say which, instead of blaming the source.
        return False, ("latexmk produced no log; it likely did not run. "
                       f"Compile by hand: cd {out_dir} && latexmk -pdf main.tex")
    if not (out_dir / "main.pdf").exists():
        return False, "no PDF produced; see main.log"

    log = log_path.read_text("utf-8", errors="ignore")
    problems = []
    if "Undefined control sequence" in log:
        problems.append("undefined control sequence")
    if re.search(r"LaTeX Warning: Citation .* undefined", log):
        problems.append("undefined citation")
    if re.search(r"LaTeX Warning: Reference .* undefined", log):
        problems.append("undefined reference")
    # A missing figure does not fail the build; it leaves a box.
    if "File `" in log and "' not found" in log:
        problems.append("missing include")
    return not problems, ", ".join(problems)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m devtools.arxiv_package")
    parser.add_argument("--build", action="store_true", required=True)
    parser.add_argument("--check", action="store_true",
                        help="compile the package in place, as arXiv will")
    parser.add_argument("--out", default=str(BUILD_DIR))
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out)
    summary = build(out_dir)
    print(f"[arxiv] {summary['figures']} figures, "
          f"{summary['bytes'] / 1e6:.1f} MB -> {out_dir}")
    if summary["missing"]:
        print(f"[arxiv] MISSING: {summary['missing']}", file=sys.stderr)
        return 1

    if args.check:
        ok, detail = check(out_dir)
        print(f"[arxiv] standalone compile: {'OK' if ok else 'FAILED'}"
              + (f" ({detail})" if detail else ""))
        return 0 if ok else 1
    print("[arxiv] not compiled (pass --check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
