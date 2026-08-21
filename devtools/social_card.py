"""Draw the 1280x640 card GitHub, X, Slack and LinkedIn show for a repo link.

GitHub does not generate this from the README. Without one, every paste of the
repository URL renders as a grey placeholder with an owner avatar, which is the
first thing most people will ever see of this project.

Two constraints shaped it:

1. **Every number on the card is a claim.** The three figures here are the same
   ones the README leads with, and `tests/test_release_artifacts.py` asserts
   they still appear there -- so a result that moves and a card that does not
   is a test failure rather than a stale image nobody re-checks.
2. **It must be reproducible.** No randomness and no timestamps, so redrawing
   on the same machine yields the same bytes. `--check` compares them -- but
   only meaningfully where the card was drawn: text rasterisation differs by
   font file and freetype version, so a Linux run redraws the same picture
   with different bytes. The test suite therefore checks the dimensions and
   the numbers, not the pixels.

    py -m devtools.social_card              # write assets/social_preview.png
    py -m devtools.social_card --check      # same machine: bytes are current

Upload it at Settings -> General -> Social preview; GitHub does not read it
from the repository.
"""
from __future__ import annotations

import argparse
import io
import math
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "assets" / "social_preview.png"

WIDTH, HEIGHT = 1280, 640
SCALE = 2                      # draw at 2x, downsample: cheap antialiasing

BG = (11, 18, 32)
PANEL = (17, 26, 43)
INK = (242, 245, 250)
MUTED = (139, 151, 173)
ACCENT = (125, 211, 252)
ALARM = (242, 84, 91)

REPO_URL = "github.com/sunnydubey1111/agent-trajectory-sentinel"
TITLE = "AgentTrajectorySentinel"
TAGLINE = "Real-time detection and repair of LLM agent failures"
FOOTER = "arXiv:2608.02464     Apache-2.0 licence     Python 3.13+"

#: (figure, caption). Each figure is asserted to still appear in the README.
STATS = (
    ("52% → 73%", "task success, with rollback-and-retry"),
    ("0", "contract-check false positives, 2,080 healthy episodes"),
    ("3,294", "committed agent episodes, 31 corpora"),
)

#: Substrings of the figures above that must survive in the README verbatim.
STAT_ASSERTIONS = ("52%", "73%", "2,080", "3,294")

_FONTS = {
    "bold": ("arialbd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"),
    "regular": ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"),
    "mono": ("consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"),
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    """First installed face of a kind, at `size` scaled to the draw canvas.

    Windows carries Arial and Consolas; Linux CI carries DejaVu or Liberation.
    Falling back to PIL's bitmap default would silently ruin the card, so an
    exhausted list raises instead.
    """
    for name in _FONTS[kind]:
        try:
            return ImageFont.truetype(name, size * SCALE)
        except OSError:
            continue
    raise RuntimeError(f"no {kind} font found; tried {_FONTS[kind]}")


def _monitor_series(n: int = 44, onset: int = 18) -> list[float]:
    """A healthy stretch, then a derailment: the shape the monitor exists for.

    Deterministic by construction -- the wobble is a sum of sines, not noise --
    because a card that redraws differently on every run cannot be diffed.
    """
    out = []
    for i in range(n):
        wobble = 0.18 + 0.05 * math.sin(i * 1.7) + 0.03 * math.sin(i * 0.6)
        ramp = 0.0
        if i >= onset:
            ramp = 0.78 * ((i - onset) / (n - 1 - onset)) ** 1.15
        out.append(wobble + ramp)
    return out


def _draw_chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                threshold: float = 0.55) -> None:
    """The score crossing its alarm line, in the card's bottom-right panel."""
    s = SCALE
    x0, y0, x1, y1 = (v * s for v in box)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=14 * s, fill=PANEL)

    pad = 26 * s
    px0, py0, px1, py1 = x0 + pad, y0 + pad, x1 - pad, y1 - pad
    series = _monitor_series()

    def point(i: int, value: float) -> tuple[float, float]:
        x = px0 + (px1 - px0) * i / (len(series) - 1)
        y = py1 - (py1 - py0) * min(value, 1.0)
        return x, y

    # Alarm line, dashed so it reads as a threshold rather than another signal.
    ty = py1 - (py1 - py0) * threshold
    dash, gap = 10 * s, 8 * s
    x = px0
    while x < px1:
        draw.line((x, ty, min(x + dash, px1), ty), fill=ALARM, width=max(1, s))
        x += dash + gap

    crossing = next(i for i, v in enumerate(series) if v > threshold)
    for i in range(len(series) - 1):
        colour = ACCENT if i + 1 <= crossing else ALARM
        draw.line((*point(i, series[i]), *point(i + 1, series[i + 1])),
                  fill=colour, width=3 * s)

    cx, cy = point(crossing, series[crossing])
    r = 6 * s
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ALARM)
    # Left of the crossing and above the threshold: the only quiet corner of
    # the panel, since the score is below the line there and the post-alarm
    # curve climbs to the right of it.
    draw.text((cx - 14 * s, cy - 22 * s), "alarm", font=_font("mono", 15),
              fill=ALARM, anchor="rm")


def render() -> Image.Image:
    s = SCALE
    img = Image.new("RGB", (WIDTH * s, HEIGHT * s), BG)
    draw = ImageDraw.Draw(img)

    draw.text((80 * s, 74 * s), REPO_URL, font=_font("mono", 20), fill=MUTED)
    draw.text((80 * s, 128 * s), TITLE, font=_font("bold", 68), fill=INK)
    draw.text((80 * s, 226 * s), TAGLINE, font=_font("regular", 31), fill=ACCENT)
    draw.line((80 * s, 300 * s, 1200 * s, 300 * s), fill=(38, 51, 74), width=s)

    for row, (figure, caption) in enumerate(STATS):
        y = (338 + row * 78) * s
        draw.text((80 * s, y), figure, font=_font("bold", 38), fill=INK)
        draw.text((80 * s, y + 46 * s), caption, font=_font("regular", 20),
                  fill=MUTED)

    _draw_chart(draw, (712, 332, 1200, 552))
    draw.text((80 * s, 592 * s), FOOTER, font=_font("mono", 19), fill=MUTED)

    return img.resize((WIDTH, HEIGHT), Image.LANCZOS)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed PNG is not what this draws")
    args = parser.parse_args(argv)

    drawn = _png_bytes(render())
    if args.check:
        if not args.out.exists():
            print(f"{args.out} is missing; run `py -m devtools.social_card`")
            return 1
        if args.out.read_bytes() != drawn:
            print(f"{args.out} is stale; run `py -m devtools.social_card`")
            return 1
        print(f"{args.out.name} is current")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(drawn)
    print(f"wrote {args.out} ({len(drawn) / 1024:.0f} KB, {WIDTH}x{HEIGHT})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
