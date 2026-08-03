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

#: What the submitted files are called. `main` says nothing once the upload
#: leaves this repository - on arXiv, in a referee's downloads folder and in
#: the source tarball anyone can fetch, the name is the only label the file
#: carries. The bibliography is renamed with it, because latexmk resolves the
#: .bbl by job name.
STEM = "agent_trajectory_sentinel"

#: Files the upload needs beside the source. The .bbl is included deliberately:
#: arXiv runs BibTeX only when it must, and shipping the compiled bibliography
#: removes a class of build failure that is invisible until after submission.
SUPPORT = ("preprint.sty", "references.bib")


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
    (out_dir / f"{STEM}.tex").write_text(tex, encoding="utf-8", newline="\n")

    missing = []
    for name in SUPPORT:
        source = PAPER / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
        else:
            missing.append(name)

    # The .bbl is a build product and is gitignored, so a fresh checkout will
    # not have one until paper/main.tex has been compiled. That is not a
    # packaging error: arXiv runs BibTeX from references.bib when no .bbl is
    # supplied. Ship it when it exists, because doing so removes a class of
    # remote build failure, and say so plainly when it does not.
    #
    # It is renamed with the source: latexmk resolves the bibliography by job
    # name, so a main.bbl beside a renamed .tex is silently ignored and every
    # citation degrades to a question mark.
    bbl = PAPER / "main.bbl"
    shipped_bbl = bbl.exists()
    if shipped_bbl:
        shutil.copy2(bbl, out_dir / f"{STEM}.bbl")

    figures = _figures(tex)
    for name in figures:
        source = FIGURES / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
        else:
            missing.append(name)

    return {"figures": len(figures), "missing": missing, "bbl": shipped_bbl,
            "bytes": sum(p.stat().st_size for p in out_dir.iterdir())}


def check(out_dir: pathlib.Path) -> tuple[bool, str]:
    """Compile the package in its own directory, as arXiv will."""
    try:
        run = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode",
                              f"{STEM}.tex"], cwd=out_dir,
                             capture_output=True, text=True)
    except OSError as exc:
        return False, f"could not run latexmk ({exc}); compile it by hand"

    log_path = out_dir / f"{STEM}.log"
    if not log_path.exists():
        # latexmk exited without producing a log at all, which on this
        # toolchain means it never really started rather than that the
        # document is broken. Say which, instead of blaming the source.
        #
        # Surface its exit code and stderr too. Without them "it likely did
        # not run" is untestable: a sandbox that blocks the subprocess and a
        # machine with no latexmk installed produce the identical message,
        # and the first is a false alarm about a package that is actually
        # fine while the second is a real missing dependency.
        tail = (run.stderr or run.stdout or "").strip().splitlines()
        hint = f" (exit {run.returncode}" + (
            f"; last output: {tail[-1][:120]!r})" if tail else ")")
        return False, ("latexmk produced no log; it likely did not run"
                       + hint + f". Compile by hand: cd {out_dir} && "
                       f"latexmk -pdf {STEM}.tex")
    if not (out_dir / f"{STEM}.pdf").exists():
        return False, f"no PDF produced; see {STEM}.log"

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
    if not summary["bbl"]:
        print("[arxiv] no main.bbl to ship — arXiv will run BibTeX from "
              "references.bib. To include it, compile paper/main.tex first "
              "(cd paper && latexmk -pdf main.tex).")

    if args.check:
        ok, detail = check(out_dir)
        print(f"[arxiv] standalone compile: {'OK' if ok else 'FAILED'}"
              + (f" ({detail})" if detail else ""))
        return 0 if ok else 1
    print("[arxiv] not compiled (pass --check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
