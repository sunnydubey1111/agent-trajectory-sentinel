"""Inline every image into the deck template, producing one self-contained HTML file.

    py -m deck.build            # -> deck/theory_2min.html

Edit deck/theory_2min.template.html, then re-run. The output is the file you
present from; it needs no network, no fonts and no sibling directory.
"""

from __future__ import annotations

import base64
import mimetypes
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DECK = ROOT / "deck"
TEMPLATE = DECK / "theory_2min.template.html"
OUTPUT = DECK / "theory_2min.html"

# token in the template -> file it stands for
IMAGES = {
    "{{IMG_TRACES}}": ROOT / "results" / "figures" / "fig1_score_traces_real.png",
    "{{IMG_HORIZON}}": ROOT / "results" / "figures" / "fig6_horizon_law.png",
    "{{IMG_ARCH}}": ROOT / "assets" / "Architecture_D2.png",
    "{{IMG_SEQUENCE}}": ROOT / "assets" / "Sequence_Diagram_1.png",
}


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in IMAGES.values() if not p.exists()]
    if missing:
        print("missing figures: " + ", ".join(missing), file=sys.stderr)
        print("run `py -m derail.experiments.plots` first", file=sys.stderr)
        return 1

    html = TEMPLATE.read_text(encoding="utf-8")
    for token, path in IMAGES.items():
        if token not in html:
            print(f"warning: {token} unused in template", file=sys.stderr)
        html = html.replace(token, data_uri(path))
        print(f"  inlined {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.0f} KB)")

    left = [t for t in IMAGES if t in html]
    if left:
        print("unsubstituted tokens: " + ", ".join(left), file=sys.stderr)
        return 1

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"\nwrote {OUTPUT.relative_to(ROOT)}  ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
