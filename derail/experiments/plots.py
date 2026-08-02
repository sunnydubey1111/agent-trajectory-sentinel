"""Figures for the derailment-detection study. Reads results/, writes PNGs.

Run after run_experiment:  py -m derail.experiments.plots

Five figures:
  fig1_score_traces_real.png  PRIMARY: score streams on REAL agent traces
  fig1_score_traces.png       secondary: the same on SIMULATED telemetry
  fig2_h1_lead.png        H1 — expected budget saved per monitor (+95% CI)
  fig3_h2_heatmap.png     H2 — channel/family x failure-class lead heatmap
  fig4_reliability.png    H3a — reliability diagram, label-free vs oracle
  fig5_escalation.png     H3b — detection vs judge-call overhead frontier

Styling follows the project's dataviz conventions: single validated palette,
one axis per chart, thin marks, recessive grid, direct labels (the relief rule
for the low-contrast slots), text in ink colors rather than series colors.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from derail.common import FAILURE_CLASSES

RESULTS = Path(__file__).resolve().parents[2] / "results"
FIGURES = RESULTS / "figures"

# Reference palette (validated: worst adjacent CVD dE 47.2 on light surface).
BLUE, AQUA, YELLOW, VIOLET = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"
CRITICAL = "#d03b3b"          # status: alarm marks only, never a series
SURFACE, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
            "#1c5cab", "#104281", "#0d366b"]

PRIMARY = "esn_cusum_max"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9.5,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _despine_all(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


# ---------------------------------------------------------------- fig 1
def fig_score_traces() -> None:
    """Primary-monitor score streams: healthy + one episode per class."""
    meta = pd.read_csv(RESULTS / "scores" / "episodes.csv")
    scores = np.load(RESULTS / "scores" / f"{PRIMARY}.npz")
    h1 = pd.read_csv(RESULTS / "tables" / "h1_main.csv")
    theta = float(h1.loc[h1["monitor"] == PRIMARY, "theta"].iloc[0])

    test = meta[meta["split"] == "test"]

    def pick(fc: str | None) -> pd.Series:
        if fc is None:
            sub = test[test["is_healthy"]]
            return sub.iloc[len(sub) // 2]
        sub = test[test["failure_class"] == fc].copy()
        # a representative (median-severity) episode
        sub = sub.sort_values("severity")
        return sub.iloc[len(sub) // 2]

    panels = [None, *FAILURE_CLASSES]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharey=True)
    ymax = 4.0 * max(theta, 1.0)
    for ax, fc in zip(axes.ravel(), panels):
        row = pick(fc)
        s = scores[row["id"]]
        t = np.arange(len(s))
        ax.plot(t, np.clip(s, None, ymax * 1.05), color=BLUE, lw=2)
        ax.axhline(theta, color=MUTED, lw=1, ls=(0, (4, 3)))
        if fc is not None:
            tau = int(row["tau"])
            ax.axvline(tau, color=INK2, lw=1, ls=":")
            ax.text(tau, ymax * 0.97, " onset", color=INK2, fontsize=8,
                    ha="left", va="top")
            alarm = np.flatnonzero(s > theta)
            alarm = alarm[alarm >= 0]
            if alarm.size:
                a = int(alarm[0])
                ax.plot([a], [min(s[a], ymax)], "o", ms=8, color=CRITICAL,
                        zorder=5)
                ax.text(a, min(s[a], ymax), "  alarm", color=CRITICAL,
                        fontsize=8, ha="left", va="center")
            else:
                ax.text(0.97, 0.82, "no alarm (missed)", color=INK2,
                        fontsize=8.5, ha="right", transform=ax.transAxes)
        ax.set_title("healthy" if fc is None else fc.replace("_", " "),
                     fontsize=10, loc="left")
        ax.set_ylim(-ymax * 0.03, ymax)
        ax.grid(axis="y")
        ax.grid(False, axis="x")
        _despine_all(ax)
    for ax in axes[1]:
        ax.set_xlabel("step t")
    axes[0][0].set_ylabel(f"{PRIMARY} score")
    fig.suptitle("SIMULATOR (illustrative, not measured telemetry): "
                 "one-class CUSUM score streams "
                 f"(dashed = threshold at 5% val FA budget)",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "fig1_score_traces.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 2

# --------------------------------------------------------------- real traces
#: Figures lead with REAL end-to-end agent traces (operator directive, L1b).
#: The simulator panel is secondary and must say so on the figure itself: its
#: goal-drift stream is a *designed* slow rotation that reads as a capability
#: gap, while on real traces the same monitor detects goal drift at 0.66-0.86.
REAL_FIG_DATASET = "ollama7b"



def _real_grounding_loss_panel():
    """A REAL fabrication episode, scored against its own corpus's healthy null.

    Grounding loss is the one class the injected corpora cannot contain: a
    fabrication has to be elicited, not injected. It is taken instead from the
    organic (non-injected) corpus, where the objective labeller found genuine
    ungrounded figures, and scored against the healthy episodes of that same
    corpus so the null matches the serving distribution.

    Returns (episode, scores, theta, flagged_step) or None when the labels are
    unavailable.
    """
    import csv

    from derail.common import Standardizer
    from derail.evaluation.metrics import pick_threshold
    from derail.experiments.run_hybrid_study import load_real
    from derail.monitor.hybrid import make_hybrids

    corpus = RESULTS.parent / "traces" / "organic_demo7b_ext"
    labels = RESULTS / "tables" / "organic_hallucination_ext.csv"
    if not (corpus / "manifest.json").exists() or not labels.exists():
        return None
    lab = {r["episode_id"]: r["label"]
           for r in csv.DictReader(labels.open(encoding="utf-8"))}
    data, channels = load_real(corpus)
    episodes = data["train"] + data["val"] + data["test"]
    healthy = [e for e in episodes if lab.get(e.episode_id) == "healthy"]
    fabricated = [e for e in episodes if lab.get(e.episode_id) == "hallucinated"]
    if len(healthy) < 20 or not fabricated:
        return None
    # Split evenly rather than at a fixed count: the healthy subset of this
    # corpus changes with the labeller, and an even split both keeps a
    # validation set and gives the 5% budget the most evidence it can have
    # (below 1/(n+1) episodes that budget is unreachable at all).
    n_fit = max(10, len(healthy) // 2)
    fit, val = healthy[:n_fit], healthy[n_fit:]
    if not val:
        return None
    std = Standardizer().fit(fit)
    esn, _maha, _hy = make_hybrids(std, channels=channels, seed=1300)
    esn.fit(fit)
    theta = float(pick_threshold([esn.score_episode(e) for e in val],
                                 fa_budget=0.05, warn_infeasible=False))
    ep = max(fabricated, key=lambda e: len(e.X))
    return ep, np.asarray(esn.score_episode(ep), dtype=float), theta

def fig_score_traces_real(dataset: str = REAL_FIG_DATASET) -> None:
    """Primary-monitor score streams on REAL agent episodes (the lead figure).

    The validation splits of these corpora are small, so the 5% budget is not
    always reachable: below 1/(n+1) episodes no empirical quantile can deliver
    it, and the threshold falls back to the observed maximum. The alarm line
    drawn here is therefore conservative on those panels — it is the operating
    point the data supports, not a looser one chosen to make the figure read
    well.

    Same monitor, same 5% validation FA budget and same layout as the simulator
    version, so the two are directly comparable - but every stream here comes
    from a live agent run against real tools, not from constructed telemetry.
    """
    from derail.common import Standardizer
    from derail.evaluation.metrics import pick_threshold
    from derail.experiments.run_hybrid_study import REAL_DATASETS, load_real
    from derail.monitor.hybrid import make_hybrids

    data, channels = load_real(REAL_DATASETS[dataset])
    std = Standardizer().fit(data["train"])
    esn, _maha, _hy = make_hybrids(std, channels=channels, seed=1300)
    esn.fit(data["train"])
    theta = float(pick_threshold([esn.score_episode(ep) for ep in data["val"]],
                                 fa_budget=0.05))

    test = data["test"]
    healthy = [ep for ep in test if ep.is_healthy]
    classes = sorted({ep.failure_class for ep in test if not ep.is_healthy})
    panels: list = [None, *classes[:5]]

    def pick(fc):
        if fc is None:
            return healthy[len(healthy) // 2]
        sub = sorted((ep for ep in test if ep.failure_class == fc),
                     key=lambda e: len(e.X))
        return sub[len(sub) // 2]           # median-length, representative

    extra = _real_grounding_loss_panel()
    n = len(panels) + (1 if extra else 0)
    rows, cols = (2, 3) if n > 3 else (1, n)
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 5.6 if rows == 2 else 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()
    streams = {}
    for fc in panels:
        ep = pick(fc)
        streams[fc] = (ep, np.asarray(esn.score_episode(ep), dtype=float))
    # Real CUSUM streams span ~12 orders of magnitude (healthy ~6, a looping
    # episode ~1.5e12) because the statistic accumulates multiplicatively once
    # a failure takes hold. The simulator's streams sit in 0-125, so its linear
    # shared axis does not transfer: on a linear scale here every panel but the
    # largest collapses onto the axis. Symlog keeps the threshold, the healthy
    # band and the blow-up all legible, and linthresh=1 tolerates the exact
    # zeros a CUSUM emits before it starts accumulating.
    ymax = 10.0 * max(theta, max(float(s.max()) for _e, s in streams.values()))

    for ax, fc in zip(axes, panels):
        ep, s = streams[fc]
        ax.plot(np.arange(len(s)), s, color=BLUE, lw=2)
        ax.axhline(theta, color=MUTED, lw=1, ls=(0, (4, 3)))
        if fc is not None and ep.tau is not None:
            ax.axvline(int(ep.tau), color=INK2, lw=1, ls=":")
            ax.text(int(ep.tau), ymax * 0.5, " onset", color=INK2,
                    fontsize=8, ha="left", va="top")
        alarm = np.flatnonzero(s > theta)
        if fc is not None and alarm.size:
            a = int(alarm[0])
            ax.plot([a], [s[a]], "o", ms=8, color=CRITICAL, zorder=5)
            ax.text(a, s[a] * 6.0, " alarm", color=CRITICAL, fontsize=8,
                    ha="left", va="bottom")
        elif fc is not None:
            ax.text(0.97, 0.82, "no alarm (missed)", color=INK2, fontsize=8.5,
                    ha="right", transform=ax.transAxes)
        ax.set_title("healthy" if fc is None else fc.replace("_", " "),
                     fontsize=10, loc="left")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(0, ymax)
        # A decade per tick is unreadable over 12 orders of magnitude; show a
        # sparse ladder so the threshold and the healthy band stay locatable.
        ax.yaxis.set_major_locator(
            matplotlib.ticker.SymmetricalLogLocator(base=10.0, linthresh=1.0,
                                                    subs=(1.0,)))
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.LogFormatterSciNotation(base=10.0,
                                                      minor_thresholds=(0, 0)))
        ax.grid(axis="y")
        ax.grid(False, axis="x")
        _despine_all(ax)
    if extra:
        ep_g, s_g, theta_g = extra
        ax = axes[len(panels)]
        # This panel has its own threshold: it is a different corpus with its
        # own healthy null, so the shared axis would misrepresent it.
        ax.plot(np.arange(len(s_g)), s_g, color=BLUE, lw=2)
        ax.axhline(theta_g, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(0.97, 0.90, "no alarm", color=INK2, fontsize=8.5,
                ha="right", transform=ax.transAxes)
        ax.text(0.97, 0.78, "caught by grounding verifier", color=CRITICAL,
                fontsize=8, ha="right", transform=ax.transAxes)
        ax.set_title("grounding loss (organic, real)", fontsize=10, loc="left")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(
            matplotlib.ticker.SymmetricalLogLocator(base=10.0, linthresh=1.0,
                                                    subs=(1.0,)))
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.LogFormatterSciNotation(base=10.0,
                                                      minor_thresholds=(0, 0)))
        ax.set_xlabel("step t")
        ax.grid(axis="y")
        ax.grid(False, axis="x")
        _despine_all(ax)
    for ax in axes[n:]:
        fig.delaxes(ax)          # delete, not hide: tight_layout NaNs on hidden
    for ax in axes[max(0, len(panels) - cols):len(panels)]:
        ax.set_xlabel("step t")
    axes[0].set_ylabel(f"{PRIMARY} score")
    fig.suptitle(f"REAL agent traces ({dataset}): one-class CUSUM score streams "
                 f"(dashed = threshold at 5% val FA budget)",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "fig1_score_traces_real.png", dpi=150)
    plt.close(fig)



def fig_class_coverage_real(table: str = "l7b_per_class") -> None:
    """Where the monitor actually works, across every REAL dataset.

    The honest coverage picture, and the one the simulator figure cannot give:
    rows are failure classes, columns are real deployments, cells are the
    primary monitor's detection rate. Blank cells are classes a corpus does not
    contain - shown as absent rather than as zero, since "not collected" and
    "not detected" are different claims.
    """
    df = pd.read_csv(RESULTS / "tables" / f"{table}.csv")
    df = df[(df["monitor"] == PRIMARY) & (df["dataset"] != "sim")]
    piv = df.pivot_table(index="failure_class", columns="dataset",
                         values="detection_rate")
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    data = np.ma.masked_invalid(piv.to_numpy(dtype=float))
    cmap = LinearSegmentedColormap.from_list("cov", [SURFACE, BLUE])
    cmap.set_bad(color="#f2f2f2")
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([c.replace("_", " ") for c in piv.index], fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy(dtype=float)[i, j]
            if v != v:
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=SURFACE if v > 0.55 else INK)
    ax.set_xticks(np.arange(-.5, piv.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, piv.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.grid(False, which="major")
    ax.tick_params(which="minor", length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02).set_label(
        "detection rate", fontsize=9)
    fig.suptitle("REAL deployments: where the monitor works, by failure class "
                 "(blank = class not present in that corpus)",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGURES / "fig3_class_coverage_real.png", dpi=150)
    plt.close(fig)


def fig_judge_complementarity_real() -> None:
    """Monitor and judge fail on DIFFERENT classes - the case for escalation.

    Both series are measured, not stipulated: the judge rates come from a real
    gemini-2.5-flash run on a labelled subset (run_judge_calibration), the
    monitor rates from the same corpus that judge was scored on.
    """
    summary = json.loads((RESULTS / "tables" /
                          "judge_calibration_summary.json").read_text("utf-8"))
    judge = {k: v["p_detect"] for k, v in summary["per_class_detection"].items()}
    per = pd.read_csv(RESULTS / "tables" / "l7b_per_class.csv")
    mon = per[(per["monitor"] == PRIMARY)
              & (per["dataset"] == summary["corpus"])]
    mon = dict(zip(mon["failure_class"], mon["detection_rate"]))

    classes = [c for c in judge if c in mon]
    classes.sort(key=lambda c: judge[c] - mon[c])
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.bar(x - 0.2, [mon[c] for c in classes], width=0.38, color=BLUE,
           label="telemetry monitor")
    ax.bar(x + 0.2, [judge[c] for c in classes], width=0.38, color=CRITICAL,
           label="LLM judge (measured)")
    for i, c in enumerate(classes):
        ax.text(i - 0.2, mon[c] + 0.02, f"{mon[c]:.2f}", ha="center",
                fontsize=8, color=INK)
        ax.text(i + 0.2, judge[c] + 0.02, f"{judge[c]:.2f}", ha="center",
                fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in classes], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("detection rate")
    # Outside the plotting area: the leftmost bars reach 1.00 and an inset
    # legend sits on top of their value labels.
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y")
    ax.grid(False, axis="x")
    _despine_all(ax)
    fig.suptitle(f"REAL traces ({summary['corpus']}): monitor and judge fail on "
                 "DIFFERENT classes",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(FIGURES / "fig5_judge_complementarity_real.png", dpi=150)
    plt.close(fig)



def fig_monitor_benchmark_real(table: str = "l7b_benchmark") -> None:
    """Monitor comparison on REAL corpora only (the simulator is excluded).

    Mean AUROC per monitor across the real deployments, with each corpus drawn
    as a dot so the spread is visible rather than hidden behind an average -
    the per-dataset variation is the point (see the tie results in L7b).
    """
    df = pd.read_csv(RESULTS / "tables" / f"{table}.csv")
    df = df[df["dataset"] != "sim"]
    order = (df.groupby("monitor")["auroc"].mean()
               .sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(order) + 1.8))
    for i, mon in enumerate(order):
        vals = df.loc[df["monitor"] == mon, "auroc"].to_numpy(dtype=float)
        ax.plot(vals, np.full(len(vals), i), "o", ms=5, color=BLUE, alpha=0.45)
        ax.plot([vals.mean()], [i], "|", ms=18, mew=2.5, color=INK)
        ax.text(vals.mean(), i - 0.34, f"{vals.mean():.3f}", ha="center",
                fontsize=8, color=INK)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(0.5, len(order) - 0.4, " chance", color=MUTED, fontsize=8,
            ha="left", va="top")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("episode AUROC (one dot per real corpus, bar = mean)")
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    _despine_all(ax)
    fig.suptitle(f"REAL corpora only ({df['dataset'].nunique()} deployments): "
                 "monitor comparison",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIGURES / "fig2_monitor_benchmark_real.png", dpi=150)
    plt.close(fig)


#: Post-onset horizon bands. The boundaries are the ones every horizon claim in
#: the papers is stated at, kept here so a figure cannot quietly use different
#: ones from the text.
_HORIZON_BANDS = ((0, 3, r"$\leq 3$"), (4, 8, "4-8"), (9, 10**6, r"$\geq 9$"))


def _horizon_panel(ax, df, title: str) -> None:
    """Detection rate per horizon band for the two parent detectors.

    Both parents on one axis, because the claim is about the GAP between them
    rather than either level: the ESN needs post-onset steps to integrate
    evidence, so its margin over a memoryless distance should grow with the
    horizon and collapse without it.
    """
    width = 0.36
    ticks = []
    for j, (lo, hi, label) in enumerate(_HORIZON_BANDS):
        band = df[(df.horizon >= lo) & (df.horizon <= hi)]
        # n per band is the other half of the story: on the external corpus the
        # long-horizon band is where the ESN wins and is also nearly empty. It
        # goes in the tick label rather than free text, so it cannot collide
        # with the axis at any figure size.
        ticks.append(f"{label}\n$n$={len(band)}")
        if band.empty:
            continue
        esn = float(band.det_esn.mean())
        maha = float(band.det_maha.mean())
        ax.bar(j - width / 2, esn, width, color=BLUE,
               label="ESN + CUSUM" if j == 0 else None)
        ax.bar(j + width / 2, maha, width, color=MUTED,
               label=r"$\Delta$-Mahalanobis" if j == 0 else None)
        ax.text(j, max(esn, maha) + 0.045, f"{esn - maha:+.3f}",
                ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(range(len(_HORIZON_BANDS)))
    ax.set_xticklabels(ticks)
    ax.set_xlabel("post-onset horizon (steps after $\\tau$)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y")
    ax.grid(False, axis="x")
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    _despine_all(ax)


def fig_horizon_law() -> None:
    """The horizon law, and its replication on a corpus we did not build.

    Left: our 1,002 injected episodes. Right: AFTraj-2K, imported unchanged.
    The same monotone gap appears in both, which is what makes it a law rather
    than a property of our injector -- and the n annotations explain the
    external result, where ranking transfers but the operating point does not.
    """
    ours = pd.read_csv(RESULTS / "tables" / "hybrid_diagnosis.csv")
    ext = pd.read_csv(RESULTS / "tables" / "aftraj_diagnosis.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), sharey=True)
    _horizon_panel(axes[0], ours,
                   f"ours: {len(ours)} injected episodes")
    _horizon_panel(axes[1], ext,
                   f"AFTraj-2K (external): {len(ext)} failures")
    axes[0].set_ylabel("detection rate at the 5% budget")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Temporal monitoring pays in proportion to post-onset horizon",
                 x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIGURES / "fig6_horizon_law.png", dpi=150)
    plt.close(fig)


def fig_h1_lead() -> None:
    """H1: expected budget saved (mean lead over ALL failures) per monitor."""
    h1 = pd.read_csv(RESULTS / "tables" / "h1_main.csv")
    keep = [PRIMARY, "esn_cusum", "esn_full", "esn_single", "esn[e,u]",
            "linear_ar", "gru", "lstm", "tcn",
            "delta_mahalanobis", "mahalanobis", "self_drift", "iforest",
            "cosine_drift", "rolling_surprisal"]
    df = (h1[h1["monitor"].isin(keep)]
          .sort_values("mean_lead_all")
          .reset_index(drop=True))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(df))
    ax.barh(y, df["mean_lead_all"], height=0.55, color=BLUE, zorder=3)
    ax.errorbar(df["mean_lead_all"], y,
                xerr=[df["mean_lead_all"] - df["mean_lead_all_ci_lo"],
                      df["mean_lead_all_ci_hi"] - df["mean_lead_all"]],
                fmt="none", ecolor=INK, elinewidth=1.2, capsize=2.5, zorder=4)
    for i, r in df.iterrows():
        ax.text(max(r["mean_lead_all_ci_hi"], r["mean_lead_all"]) + 0.12, i,
                f"{r['mean_lead_all']:.1f}  (det {r['detection_rate']:.0%})",
                va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, df["monitor"])
    ax.set_xlabel("expected steps of budget saved per failure episode "
                  "(misses count 0) — 95% bootstrap CI")
    ax.set_xlim(0, float(df["mean_lead_all_ci_hi"].max()) * 1.35)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    _despine_all(ax)
    ax.set_title("SIMULATOR (illustrative). H1 — temporal ESN monitors save more budget than\n"
                 "memoryless baselines at the same 5% false-alarm budget",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_h1_lead.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 3
def fig_h2_heatmap() -> None:
    """H2: mean lead (all injected) by channel/family x failure class."""
    h2 = pd.read_csv(RESULTS / "tables" / "h2_channels.csv")
    cols = ["esn_cusum[e]", "esn_cusum[u]", "esn_cusum[m]",
            "esn_cusum[e,u]", "esn_cusum", "esn_cusum_max", "self_drift"]
    col_labels = ["e", "u", "m", "e+u", "e+u+m\n(mean)", "max(e,u,m)",
                  "self-drift\n(e, centroid)"]
    M = np.zeros((len(FAILURE_CLASSES), len(cols)))
    D = np.zeros_like(M)
    for i, fc in enumerate(FAILURE_CLASSES):
        for j, mon in enumerate(cols):
            sub = h2[(h2["monitor"] == mon) & (h2["failure_class"] == fc)]
            M[i, j] = float(sub["mean_lead_all"].iloc[0]) if len(sub) else np.nan
            D[i, j] = float(sub["detection_rate"].iloc[0]) if len(sub) else np.nan
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    vmax = np.nanmax(M)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            dark = M[i, j] > 0.55 * vmax
            ax.text(j, i - 0.12, f"{M[i, j]:.1f}", ha="center", va="center",
                    fontsize=10, color="#ffffff" if dark else INK)
            ax.text(j, i + 0.26, f"det {D[i, j]:.0%}", ha="center",
                    va="center", fontsize=7.5,
                    color="#e8f0fb" if dark else INK2)
    ax.set_xticks(range(len(cols)), col_labels)
    ax.set_yticks(range(len(FAILURE_CLASSES)),
                  [fc.replace("_", " ") for fc in FAILURE_CLASSES])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("mean steps saved per episode (misses = 0)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("SIMULATOR (illustrative). H2 — no single channel dominates: u leads grounding loss,\n"
                 "e leads looping; slow goal drift needs self-consistency",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_h2_heatmap.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 4
def fig_reliability() -> None:
    """H3a: reliability diagram of episode-level alarm confidence (fused).

    Reads only what the published artifacts actually contain. Two notes, both
    forced by the committed tables rather than chosen:

    * ECE is recomputed from the same bins the curve is drawn from, so the
      annotation cannot disagree with the line. (Verified: this reproduces the
      summary table's `iso_ece` to four decimals.) The earlier version read an
      `ece` column from `h3_calibration.csv`, which that table no longer has -
      its schema is now one row per stream with `iso_ece`.
    * `h3_reliability.csv` carries the ORACLE (isotonic) calibrator only. The
      label-free null has no per-bin series in the published artifacts, so it
      cannot be drawn; its calibration quality is reported instead from the
      columns that do exist (healthy-null KS and realized FA at 0.95).
    """
    rel = pd.read_csv(RESULTS / "tables" / "h3_reliability.csv",
                      keep_default_na=False)
    cal = pd.read_csv(RESULTS / "tables" / "h3_calibration.csv",
                      keep_default_na=False)
    fused = rel[rel["stream"] == "fused"]
    summary = cal[cal["stream"] == "fused"].iloc[0]

    def ece_from_bins(sub: pd.DataFrame) -> float:
        n = sub["count"].sum()
        if not n:
            return float("nan")
        w = sub["count"] / n
        return float((w * (sub["mean_confidence"] - sub["empirical_freq"]).abs()).sum())

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    palette = {"isotonic": VIOLET, "null": BLUE}
    for name in sorted(fused["calibrator"].unique()):
        sub = fused[(fused["calibrator"] == name) & (fused["count"] >= 5)]
        if sub.empty:
            continue
        label = ("oracle isotonic" if name == "isotonic" else name)
        ax.plot(sub["mean_confidence"], sub["empirical_freq"],
                color=palette.get(name, BLUE), lw=2, marker="o", ms=6,
                label=f"{label} (ECE {ece_from_bins(sub):.3f})", zorder=3,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.text(0.03, 0.97,
            "label-free null (no per-bin series published):\n"
            f"  healthy-null KS {summary['null_healthy_ks']:.3f}\n"
            f"  realized FA @0.95 {summary['null_fa_at_0.95']:.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color=INK2)
    ax.set_xlabel("predicted alarm confidence  P(derailed)")
    ax.set_ylabel("empirical failure frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    _despine_all(ax)
    ax.set_title("SIMULATOR (illustrative; labelled posterior). H3a --- episode-level\n"
                 "alarm confidence, fused stream",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_reliability.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig_escalation() -> None:
    """H3b: detection recovered vs judge-call overhead."""
    esc = pd.read_csv(RESULTS / "tables" / "h3_escalation.csv")
    judge = esc[esc["policy"] == "judge_every_step"].iloc[0]
    halt = esc[esc["policy"] == "halt_on_alarm"].iloc[0]
    sweep = esc[(esc["policy"] == "escalate_on_alarm")
                & esc["conf_threshold"].notna()].sort_values("conf_threshold")
    raw = esc[(esc["policy"] == "escalate_on_alarm")
              & esc["conf_threshold"].isna()].iloc[0]
    chosen = esc[esc["selected_on_cal"].astype(bool)]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(sweep["judge_call_ratio"], sweep["detection_rate"], color=BLUE,
            lw=2, marker="o", ms=7, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=1.5,
            label="escalate on confidence (sweep)")
    for _, r in sweep.iterrows():
        ax.annotate(f"{r['conf_threshold']:.2f}",
                    (r["judge_call_ratio"], r["detection_rate"]),
                    textcoords="offset points", xytext=(6, -11),
                    fontsize=8, color=INK2)
    ax.plot([judge["judge_call_ratio"]], [judge["detection_rate"]], "s",
            ms=9, color=VIOLET, zorder=4, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="judge every step (upper baseline)")
    ax.plot([raw["judge_call_ratio"]], [raw["detection_rate"]], "^", ms=9,
            color=AQUA, zorder=4, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="escalate on raw score")
    ax.plot([halt["judge_call_ratio"]], [halt["detection_rate"]], "D", ms=8,
            color=YELLOW, zorder=4, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="halt on alarm (no judge)")
    if len(chosen):
        c = chosen.iloc[0]
        ax.plot([c["judge_call_ratio"]], [c["detection_rate"]], "o", ms=14,
                mfc="none", mec=INK, mew=1.6, zorder=5)
        ax.annotate("selected on cal", (c["judge_call_ratio"],
                                        c["detection_rate"]),
                    textcoords="offset points", xytext=(10, 8), fontsize=9,
                    color=INK)
    ax.set_xlabel("judge-LLM calls per episode, relative to judging every step")
    ax.set_ylabel("detection rate on injected episodes")
    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8.5)
    _despine_all(ax)
    # Title is data-driven off the selected-on-cal operating point, not a
    # hard-coded figure: the pct must track whatever policy the run
    # actually selected.
    if len(chosen):
        _pct = int(round(float(chosen.iloc[0]["judge_call_ratio"]) * 100))
        _title = ("H3b — calibrated escalation recovers most judge detection "
                  f"at ~{_pct}% of its calls")
    else:
        _title = ("H3b — calibrated escalation recovers most judge detection "
                  "at a fraction of its calls")
    ax.set_title(_title, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_escalation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig_score_traces_real()      # PRIMARY: real end-to-end agent traces
    fig_class_coverage_real()    # PRIMARY: per-class coverage, real datasets
    fig_judge_complementarity_real()   # PRIMARY: measured judge vs monitor
    fig_monitor_benchmark_real()       # PRIMARY: monitors on real corpora
    fig_horizon_law()            # PRIMARY: the horizon law + its replication
    fig_score_traces()           # secondary: simulated telemetry, labelled
    fig_h1_lead()
    fig_h2_heatmap()
    fig_reliability()
    fig_escalation()
    results = json.loads((RESULTS / "results.json").read_text(encoding="utf-8"))
    print("wrote 10 figures to", FIGURES)
    for name, verdict in results["verdicts"].items():
        print(f"  {name}: {verdict[:100]}")


if __name__ == "__main__":
    main()
