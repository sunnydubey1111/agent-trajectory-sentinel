"""Typeset the long-form manuscript (`paper/paper.md`) as a PDF.

`paper/main.tex` is the conference-format paper and is written directly in
LaTeX. `paper/paper.md` is the full-length version, written in Markdown so it
stays diffable and reviewable in the repository, which leaves it with no PDF of
its own. This converter closes that gap: it emits `paper/paper.tex` from the
Markdown and leaves compilation to the same `latexmk` that builds `main.tex`, so
both PDFs come off one toolchain.

It is deliberately narrow. It handles exactly the constructs `paper.md` uses --
ATX headings, ordered and unordered lists, pipe tables, fenced and inline code,
bold/italic, block quotes -- and raises on anything it does not recognise rather
than silently dropping content. A converter that guesses would let a paragraph
vanish from a published PDF without anyone noticing.

    py -m devtools.md_to_latex                 # paper/paper.md -> paper/paper.tex
    py -m devtools.md_to_latex --build         # ... and run latexmk to paper.pdf
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "paper" / "paper.md"
TEX_OUT = REPO_ROOT / "paper" / "paper.tex"

#: Every non-ASCII character `paper.md` uses, mapped to a math-safe macro. The
#: converter raises on any character missing from this table, so a new symbol
#: cannot reach the PDF as a silently dropped glyph.
UNICODE: dict[str, str] = {
    "§": r"\S{}", "±": r"$\pm$", "µ": r"$\mu$",
    "×": r"$\times$", "Δ": r"$\Delta$", "θ": r"$\theta$",
    "τ": r"$\tau$", "–": "--", "—": "---",
    "→": r"$\rightarrow$", "↔": r"$\leftrightarrow$",
    "⇒": r"$\Rightarrow$", "−": "$-$", "≈": r"$\approx$",
    "≠": r"$\neq$", "≤": r"$\leq$", "≥": r"$\geq$",
}

#: Characters LaTeX reads as markup in ordinary text.
ESCAPES = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=1in]{geometry}
\usepackage{array}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{amsmath,amssymb}
\usepackage[protrusion=true,expansion=false]{microtype}
\usepackage{parskip}
\usepackage[colorlinks=true,linkcolor=black,urlcolor=black]{hyperref}
\usepackage{fancyvrb}
\setcounter{secnumdepth}{0}
\setlength{\emergencystretch}{3em}
\title{%(title)s}
\author{Sunny Dubey \\ \texttt{sjkumardube@gmail.com}}
\date{}
\begin{document}
\maketitle
"""


class ConversionError(RuntimeError):
    """Raised when the source contains a construct this converter cannot map."""


def _escape(text: str) -> str:
    """Escape LaTeX specials in a plain-text run (no inline markup left)."""
    out = []
    for ch in text:
        if ch in ESCAPES:
            out.append(ESCAPES[ch])
        elif ch in UNICODE:
            out.append(UNICODE[ch])
        elif ord(ch) > 127:
            raise ConversionError(f"unmapped character {ch!r} (U+{ord(ch):04X})")
        else:
            out.append(ch)
    return "".join(out)


def _inline(text: str) -> str:
    """Convert inline markup, escaping the plain runs between the markers.

    Code spans are extracted first and escaped under verbatim rules, so a `$`
    or `_` inside `\\texttt{}` cannot be read as math or a subscript.
    """
    spans: list[str] = []

    def stash(m: re.Match[str]) -> str:
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"\x01{m.group(1)}\x02", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: f"\x03{m.group(1)}\x04", text)
    out = _escape(text)
    out = out.replace("\x01", r"\textbf{").replace("\x02", "}")
    out = out.replace("\x03", r"\emph{").replace("\x04", "}")

    def restore(m: re.Match[str]) -> str:
        # Long identifiers (`judge_calibration_summary.json`) are a single
        # unbreakable word inside \texttt, which overruns a narrow table
        # cell. Allow a break after each underscore and slash.
        code = _escape(spans[int(m.group(1))])
        code = code.replace(r"\_", r"\_\allowbreak{}")
        code = code.replace("/", r"/\allowbreak{}")
        return r"\texttt{" + code + "}"

    return re.sub("\x00(\\d+)\x00", restore, out)


#: A column wider than this many characters is set as a wrapping paragraph
#: column rather than a rigid one. Below it, `l`/`r` keep numbers and short
#: labels tight; above it a rigid column runs off the page instead of wrapping,
#: which is what made the manuscript's telemetry table overflow the text block.
_WRAP_AT = 24

_NUMERIC = re.compile(r"^[-+]?[\d.,]+\s*(%|x|s|ms|us|MB)?$")


def _column_kind(values: list[str]) -> str:
    """Pick `r`, `l` or a wrapping `p{}` from what a column actually holds."""
    stripped = [re.sub(r"[*`$\\]", "", v).strip() for v in values if v.strip()]
    if not stripped:
        return "l"
    if max(len(v) for v in stripped) > _WRAP_AT:
        return "p"
    if all(_NUMERIC.match(v) for v in stripped):
        return "r"
    return "l"


def _column_spec(header: list[str], body: list[list[str]]) -> str:
    """Build a column spec, sharing the free width between wrapping columns."""
    kinds = [_column_kind([header[i]] + [row[i] for row in body])
             for i in range(len(header))]
    n_wrap = kinds.count("p")
    if not n_wrap:
        return "".join(kinds)
    # Budget the text width. Every column carries inter-column padding, so
    # shares that sum to 1 overrun the margin; rigid columns are charged a
    # flat estimate and the remainder is split between the wrapping ones.
    ncol = len(kinds)
    rigid = ncol - n_wrap
    share = (1.0 - 0.04 * ncol - 0.11 * rigid) / n_wrap
    share = max(share, 0.15)
    wrap = rf">{{\raggedright\arraybackslash}}p{{{share:.3f}\linewidth}}"
    return "".join(wrap if k == "p" else k for k in kinds)


def _table(rows: list[str]) -> list[str]:
    """Render a pipe table as a booktabs longtable.

    A longtable is used because several tables in the manuscript are taller
    than a page; a plain tabular would silently overflow the text block. Column
    types are inferred from the cells (see `_column_kind`) so numbers stay
    right-aligned and prose wraps instead of running into the margin.
    """
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]          # cells[1] is the ---|--- rule
    ncol = len(header)
    for row in body:
        if len(row) != ncol:
            raise ConversionError(f"ragged table row: {row!r} (expected {ncol})")
    spec = _column_spec(header, body)
    out = [rf"\begin{{longtable}}{{{spec}}}", r"\toprule",
           " & ".join(rf"\textbf{{{_inline(c)}}}" for c in header) + r" \\",
           r"\midrule", r"\endhead"]
    out += [" & ".join(_inline(c) for c in row) + r" \\" for row in body]
    out += [r"\bottomrule", r"\end{longtable}", ""]
    return out


def convert(md: str) -> str:
    """Convert the manuscript body to a complete LaTeX document."""
    lines = md.replace("\r\n", "\n").split("\n")
    title, body_start = None, 0
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title, body_start = _inline(line[2:].strip()), i + 1
            break
    if title is None:
        raise ConversionError("no level-1 heading to use as the title")

    out: list[str] = []
    list_env: str | None = None
    i = body_start

    def close_list() -> None:
        nonlocal list_env
        if list_env:
            out.append(rf"\end{{{list_env}}}")
            list_env = None

    while i < len(lines):
        line = lines[i]

        if line.startswith("```"):                      # fenced code
            close_list()
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            if i >= len(lines):
                raise ConversionError("unterminated code fence")
            out += [r"\begin{Verbatim}[fontsize=\small,samepage=false]",
                    *block, r"\end{Verbatim}", ""]
            i += 1
            continue

        if line.startswith("|"):                        # pipe table
            close_list()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            if len(rows) < 2:
                raise ConversionError(f"table with no rule: {rows!r}")
            out += _table(rows)
            continue

        heading = re.match(r"^(#{2,6}) +(.*)$", line)
        if heading:
            close_list()
            depth = len(heading.group(1))
            cmd = {2: "section", 3: "subsection"}.get(depth, "subsubsection")
            out += [rf"\{cmd}{{{_inline(heading.group(2).strip())}}}", ""]
            i += 1
            continue

        bullet = re.match(r"^ *[-*] +(.*)$", line)
        numbered = re.match(r"^ *\d+\. +(.*)$", line)
        if bullet or numbered:
            want = "itemize" if bullet else "enumerate"
            if list_env != want:
                close_list()
                out.append(rf"\begin{{{want}}}")
                list_env = want
            text = (bullet or numbered).group(1)
            i += 1
            while (i < len(lines) and lines[i].strip()
                   and not re.match(r"^ *([-*]|\d+\.) +", lines[i])
                   and not lines[i].startswith(("|", "#", "```", ">"))
                   and lines[i].startswith(" ")):
                text += " " + lines[i].strip()
                i += 1
            out.append(r"\item " + _inline(text))
            continue

        if line.startswith(">"):                        # block quote
            close_list()
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out += [r"\begin{quote}", _inline(" ".join(quote)),
                    r"\end{quote}", ""]
            continue

        if not line.strip():                            # blank line
            close_list()
            out.append("")
            i += 1
            continue

        para = [line]                                   # paragraph
        i += 1
        while (i < len(lines) and lines[i].strip()
               and not lines[i].startswith(("|", "#", "```", ">"))
               and not re.match(r"^ *([-*]|\d+\.) +", lines[i])):
            para.append(lines[i])
            i += 1
        out += [_inline(" ".join(p.strip() for p in para)), ""]

    close_list()
    return (PREAMBLE % {"title": title}) + "\n".join(out) + "\n\\end{document}\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="py -m devtools.md_to_latex")
    parser.add_argument("--build", action="store_true",
                        help="run latexmk on the generated .tex")
    args = parser.parse_args(argv)

    tex = convert(SOURCE.read_text(encoding="utf-8"))
    TEX_OUT.write_text(tex, encoding="utf-8", newline="\n")
    print(f"wrote {TEX_OUT.relative_to(REPO_ROOT).as_posix()} "
          f"({len(tex.splitlines())} lines)")

    if args.build:
        proc = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
             TEX_OUT.name],
            cwd=TEX_OUT.parent, capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:])
            return proc.returncode
        print(f"wrote {TEX_OUT.with_suffix('.pdf').relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
