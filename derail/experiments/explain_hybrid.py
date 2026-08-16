"""Explain WHY the logistic hybrid works (exp/hybrid-explain).

Consumes the per-episode records written by run_hybrid_study
(`hybrid_explain.csv`, `hybrid_diagnosis.csv`) — the same scoring path as
the benchmark, so figure and tables cannot disagree — and produces:

  1. `hybrid_coefficients.csv` — the learned fusion weights per dataset:
     which detector the logistic leans on, and by how much.
  2. `hybrid_complementarity.csv` — the ESN x Mahalanobis detection
     quadrants (both / ESN-only / Maha-only / neither) with the logistic's
     detection rate inside each cell: the direct evidence of complementary
     coverage and of what the fusion recovers.
  3. `results/figures/hybrid_explain.png` — per-dataset scatter of every
     test episode in the hybrid's own 2-D feature space (calibrated,
     clipped robust-z of Mahalanobis on x, ESN on y, taken at the step
     that maximizes the fused score), with the logistic decision boundary
     at the deployed threshold and alarm outcomes encoded by color+shape.

Run:  py -m derail.experiments.explain_hybrid [--prefix hybrid]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES = Path(__file__).resolve().parents[2] / "results" / "tables"
FIGURES = Path(__file__).resolve().parents[2] / "results" / "figures"

# Validated palette (dataviz reference instance, light mode; all-pairs PASS
# for 4 slots, contrast WARN on slots 3-4 relieved by shapes + legend +
# the accompanying tables).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_DETECTED = "#2a78d6"   # slot 1 blue   — injected, alarm at/after tau
C_SILENT = "#008300"     # slot 2 green  — healthy, correctly silent
C_MISSED = "#e87ba4"     # slot 3 magenta— injected, no alarm
C_FALSE = "#eda100"      # slot 4 yellow — false or early alarm

OUTCOME_STYLE = {   # label -> (color, marker)
    "detected (true alarm)": (C_DETECTED, "o"),
    "missed failure": (C_MISSED, "X"),
    "healthy, silent": (C_SILENT, "s"),
    "false / early alarm": (C_FALSE, "^"),
}


def _bucket(row: pd.Series) -> str:
    o = row["outcome_logistic"]
    if o == "true_alarm":
        return "detected (true alarm)"
    if o == "miss":
        return "missed failure"
    if o == "correct_silence":
        return "healthy, silent"
    return "false / early alarm"     # false_alarm or early_alarm


def coefficients_table(ex: pd.DataFrame) -> pd.DataFrame:
    """Learned fusion weights per dataset.

    These are the MEAN-over-folds coefficients the cross-fit reports; the
    per-fold decision boundaries differ and are what actually scored each
    episode. The weight share is a share of MAGNITUDE
    (|coef|/sum|coef|), which is bounded in [0, 1]. A signed ratio
    coef_maha/(coef_esn+coef_maha) is not: it exceeds 100% whenever the
    coefficients have opposite signs.
    """
    rows = []
    for ds, g in ex.groupby("dataset", sort=False):
        r = g.iloc[0]
        denom = max(abs(r["coef_esn"]) + abs(r["coef_maha"]), 1e-12)
        share = abs(r["coef_maha"]) / denom
        rows.append({"dataset": ds, "coef_esn": round(r["coef_esn"], 4),
                     "coef_maha": round(r["coef_maha"], 4),
                     "intercept": round(r["intercept"], 4),
                     "maha_magnitude_share": round(share, 3)})
    return pd.DataFrame(rows)


def complementarity_table(diag: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [(ds, g) for ds, g in diag.groupby("dataset", sort=False)]
    groups.append(("ALL", diag))
    for ds, g in groups:
        cells = {
            "both": g[g.det_esn & g.det_maha],
            "esn_only": g[g.det_esn & ~g.det_maha],
            "maha_only": g[~g.det_esn & g.det_maha],
            "neither": g[~g.det_esn & ~g.det_maha],
        }
        row = {"dataset": ds, "n_injected": len(g)}
        for k, sub in cells.items():
            row[f"n_{k}"] = len(sub)
            row[f"logistic_det_in_{k}"] = (round(sub["det_logistic"].mean(), 3)
                                           if len(sub) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def _boundary_xy(coef_e: float, coef_m: float, b: float, theta: float,
                 xlim: tuple, ylim: tuple):
    """Points of the line coef_e*y + coef_m*x + b = theta inside the axes."""
    if abs(coef_e) < 1e-12 and abs(coef_m) < 1e-12:
        return None
    pts = []
    if abs(coef_e) > 1e-12:                     # y at the x borders
        for x in xlim:
            y = (theta - b - coef_m * x) / coef_e
            if ylim[0] <= y <= ylim[1]:
                pts.append((x, y))
    if abs(coef_m) > 1e-12:                     # x at the y borders
        for y in ylim:
            x = (theta - b - coef_e * y) / coef_m
            if xlim[0] <= x <= xlim[1]:
                pts.append((x, y))
    pts = sorted(set(pts))
    return pts if len(pts) >= 2 else None


def make_figure(ex: pd.DataFrame, out: Path) -> None:
    datasets = list(dict.fromkeys(ex["dataset"]))
    n = len(datasets)
    ncols = 4 if n > 4 else n
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.0 * ncols, 3.6 * nrows + 0.7),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()

    for ax, ds in zip(axes, datasets):
        g = ex[ex["dataset"] == ds]
        ax.set_facecolor(SURFACE)
        lo = min(-2.0, g["z_maha_at_alarm"].min(), g["z_esn_at_alarm"].min())
        hi = 52.0
        xlim = ylim = (lo - 1, hi)
        for label, (color, marker) in OUTCOME_STYLE.items():
            sub = g[g.apply(_bucket, axis=1) == label]
            if not len(sub):
                continue
            ax.scatter(sub["z_maha_at_alarm"], sub["z_esn_at_alarm"],
                       s=34, c=color, marker=marker, label=label,
                       edgecolors=SURFACE, linewidths=0.8, zorder=3)
        r = g.iloc[0]
        pts = _boundary_xy(r["coef_esn"], r["coef_maha"], r["intercept"],
                           r["theta_logistic"], xlim, ylim)
        if pts:
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            ax.plot([x0, x1], [y0, y1], ls="--", lw=1.4, c=MUTED, zorder=2)
            ax.annotate("alarm boundary", xy=((x0 + x1) / 2, (y0 + y1) / 2),
                        xytext=(4, 6), textcoords="offset points",
                        fontsize=7.5, color=MUTED)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(ds, fontsize=10, color=INK, pad=6)
        ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[n:]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_2,
               bbox_to_anchor=(0.5, 1.0))
    fig.supxlabel("Mahalanobis confidence (calibrated robust-z, clipped)",
                  fontsize=10, color=INK_2)
    fig.supylabel("ESN confidence (calibrated robust-z, clipped)",
                  fontsize=10, color=INK_2)
    fig.suptitle("Hybrid logistic fusion: episode positions at the deciding "
                 "step, with the deployed alarm boundary",
                 fontsize=12, color=INK, y=1.06)
    fig.tight_layout(rect=(0.015, 0.01, 1, 0.965))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.explain_hybrid")
    parser.add_argument("--prefix", default="hybrid")
    args = parser.parse_args(argv)

    ex = pd.read_csv(TABLES / f"{args.prefix}_explain.csv")
    diag = pd.read_csv(TABLES / f"{args.prefix}_diagnosis.csv")

    coefs = coefficients_table(ex)
    coefs.to_csv(TABLES / f"{args.prefix}_coefficients.csv", index=False)
    print("[explain] learned fusion weights per dataset:")
    print(coefs.to_string(index=False))

    comp = complementarity_table(diag)
    comp.to_csv(TABLES / f"{args.prefix}_complementarity.csv", index=False)
    print("\n[explain] ESN x Mahalanobis complementarity "
          "(logistic det rate inside each cell):")
    print(comp.to_string(index=False))

    fig_path = FIGURES / f"{args.prefix}_explain.png"
    make_figure(ex, fig_path)
    print(f"\n[explain] wrote {fig_path}")


if __name__ == "__main__":
    main()
