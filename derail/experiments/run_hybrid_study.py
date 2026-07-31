"""Hybrid ESN + Mahalanobis study (exp/hybrid-fusion).

Answers three questions raised by the dataset expansion:

1. WHY does DeltaMahalanobis beat the ESN on real_research7b? The script
   records, per injected episode, the post-onset horizon (T - 1 - tau) and
   each detector's outcome, then correlates horizon with the ESN's detection
   advantage (the temporal-information hypothesis).
2. WHICH failure classes are state-based (favor the memoryless distance) vs
   temporal (favor the reservoir)? Per-class detection tables across every
   dataset.
3. DOES a hybrid beat both? Four fusion variants (weighted, max, gated,
   logistic — see derail.monitor.hybrid) are evaluated against the two
   standalone monitors on the simulator plus four real datasets, under the
   frozen evaluation protocol (same splits via rng_for(0, "real-split"),
   same 5% FA healthy-val-quantile thresholds, same metrics), with paired
   significance tests and bootstrap CIs.

Supervision discipline: HybridLogistic needs labeled steps. On the simulator
it trains on the `cal` split (disjoint from test by construction). On real
datasets, injected test episodes are 2-fold cross-fit (stratified by class):
each fold is scored by the model trained on the other fold; healthy episodes
are scored by the mean logit of the two fold models. No episode is ever
scored by a model that saw it in training.

Run:  py -m derail.experiments.run_hybrid_study [--skip-sim] [--datasets ...]
Writes results/tables/hybrid_{benchmark,per_class,stats,diagnosis}.csv.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from derail.common import (
    DatasetConfig,
    Episode,
    OnlineMonitor,
    SimConfig,
    Standardizer,
    rng_for,
)
from derail.evaluation.metrics import (
    episode_auc,
    evaluate_alarms,
    length_confound_report,
    pick_threshold,
    summarize,
)
from derail.evaluation.protocol import (
    cross_fit_scores,
    full_model,
    holm_bonferroni,
)
from derail.evaluation.stats import (
    mcnemar_test,
    paired_permutation_test,
    wilcoxon_signed_rank,
)
from derail.monitor.hybrid import HybridLogistic, make_hybrids
from derail.telemetry.adapter import load_trace_jsonl
from derail.telemetry.generator import make_dataset

TRACES_DIR = Path(__file__).resolve().parents[2] / "traces"
TABLES_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"
FA_BUDGET = 0.05
MIN_T = 4

REAL_DATASETS: dict[str, Path] = {
    "gemini": TRACES_DIR,
    "autogen7b": TRACES_DIR / "autogen7b",
    "ollama7b": TRACES_DIR / "ollama7b",
    "langgraph7b": TRACES_DIR / "langgraph7b",   # held out for the
    # generalization study: never used to develop the hybrids
    "real_research7b": TRACES_DIR / "real_research7b",
    "real_research7b_long": TRACES_DIR / "real_research7b_long",
    "real_research3b": TRACES_DIR / "real_research3b",  # T6c model transfer
    # L4: a second model FAMILY collected on the ollama7b task /
    # tool / injector plan, so qwen7b -> llama8b is a cross-FAMILY transfer
    # rather than a smaller sibling of the same model.
    "ollama_llama8b": TRACES_DIR / "ollama_llama8b",
    # L5: Gemini on the SAME long research task as
    # real_research7b_long, replacing the 18-episode/1-positive `real` set with
    # one that has post-onset horizon. Provider is the only difference.
    "real_gemini_long": TRACES_DIR / "real_gemini_long",
    # L7b: the long research task grown to 13 injected per class
    # (n=91) so its paired comparisons can reach 80% power. ADDITIVE sibling -
    # `real_research7b_long` stays frozen at the 72 episodes every published
    # table was computed from. Unlike that pre-v5 corpus, every episode here
    # carries the v5 provenance fingerprint and trace checksum.
    "real_research7b_long_ext": TRACES_DIR / "real_research7b_long_ext",
}

#: Datasets the PUBLISHED results/tables/hybrid_*.csv are computed from - the
#: set the last full regeneration ran and verified. Corpora added afterwards
#: are deliberately excluded so a regeneration reproduces the published scope
#: rather than quietly widening it.
PUBLISHED_DATASETS: frozenset[str] = frozenset({
    "gemini", "autogen7b", "ollama7b", "langgraph7b",
    "real_research7b", "real_research7b_long", "real_research3b",
})


# ---------------------------------------------------------------- loading
def load_real(traces_dir: Path, grounding: bool = False,
              use_st: bool = False
              ) -> tuple[dict[str, list[Episode]], tuple[str, ...]]:
    """Load a real dataset with the exact run_real_traces protocol.

    grounding=True loads v4 episodes (content-grounding channel);
    default False keeps the published 51-dim behavior unchanged.
    use_st=True switches ALL semantic dims (e channel + grounding
    cosines) to MiniLM — an explicit opt-in, never inferred (T3 probe).
    """
    manifest = json.loads((traces_dir / "manifest.json").read_text("utf-8"))
    episodes = []
    for entry in manifest:
        if entry["T"] < MIN_T:
            continue
        episodes.append(load_trace_jsonl(
            traces_dir / entry["file"], episode_id=entry["episode_id"],
            tau=entry["tau"], failure_class=entry["failure_class"],
            severity=None if entry["tau"] is None else 0.5,
            use_sentence_transformers=use_st, extended=True,
            grounding=grounding))
    healthy = [ep for ep in episodes if ep.is_healthy]
    injected = [ep for ep in episodes if not ep.is_healthy]
    perm = rng_for(0, "real-split").permutation(len(healthy))
    n_train = int(round(0.6 * len(healthy)))
    n_val = int(round(0.2 * len(healthy)))
    data = {
        "train": [healthy[i] for i in perm[:n_train]],
        "val": [healthy[i] for i in perm[n_train:n_train + n_val]],
        "cal": [],  # no injected calibration split on real data -> cross-fit
        "test": [healthy[i] for i in perm[n_train + n_val:]] + injected,
    }
    has_lp = sum(bool(e.get("has_logprobs")) for e in manifest)
    base = ("e", "u", "m") if has_lp >= 0.9 * len(manifest) else ("e", "m")
    return data, base + ("x",)


def load_sim(seed: int = 0) -> tuple[dict[str, list[Episode]],
                                     tuple[str, ...]]:
    """Full-size simulator dataset (the frozen study's master config)."""
    data = make_dataset(DatasetConfig(master_seed=seed), SimConfig())
    return data, ("e", "u", "m")


# ---------------------------------------------------------------- helpers
def _footprint_mb(mon: OnlineMonitor) -> float:
    """Approximate resident numpy bytes reachable from the monitor."""
    seen: set[int] = set()

    def walk(obj) -> int:
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        n = 0
        if isinstance(obj, dict):
            n += sum(walk(v) for v in obj.values())
        elif isinstance(obj, (list, tuple, set)):
            n += sum(walk(v) for v in obj)
        elif hasattr(obj, "__dict__"):
            n += walk(vars(obj))
        return n

    return walk(mon) / 1e6


def _stratified_folds(episodes: list[Episode], seed_label: str,
                      seed: int = 0) -> tuple[list[Episode], list[Episode]]:
    """Deterministic 2-fold class-stratified split of injected episodes."""
    rng = rng_for(seed, "hybrid-folds", seed_label)
    by_class: dict[str, list[Episode]] = {}
    for ep in sorted(episodes, key=lambda e: e.episode_id):
        by_class.setdefault(ep.failure_class, []).append(ep)
    folds: tuple[list[Episode], list[Episode]] = ([], [])
    for cls in sorted(by_class):
        eps = by_class[cls]
        order = rng.permutation(len(eps))
        for i, j in enumerate(order):
            folds[i % 2].append(eps[j])
    return folds


def _bootstrap_dauc(test: list[Episode], scores_a: dict, scores_b: dict,
                    n_boot: int = 1000, seed: int = 0
                    ) -> tuple[float, float]:
    """Percentile CI of AUC(a) - AUC(b) over episode resamples."""
    rng = rng_for(seed, "hybrid-dauc")
    max_a = np.array([np.max(scores_a[ep.episode_id]) for ep in test])
    max_b = np.array([np.max(scores_b[ep.episode_id]) for ep in test])
    y = np.array([0 if ep.is_healthy else 1 for ep in test])
    from sklearn.metrics import roc_auc_score
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(test), size=len(test))
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue
        deltas.append(roc_auc_score(yy, max_a[idx])
                      - roc_auc_score(yy, max_b[idx]))
    d = np.asarray(deltas)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


# ---------------------------------------------------------------- evaluation
def evaluate_dataset(name: str, data: dict[str, list[Episode]],
                     channels: tuple[str, ...], seed: int = 0) -> dict:
    """Fit + evaluate all monitors on one dataset; return all result rows.

    `seed` varies the monitor-side randomness only (ESN reservoir seeds and
    the logistic cross-fit fold assignment) for multiseed stability runs;
    the data splits stay frozen per the evaluation protocol. seed=0 is the
    published configuration.
    """
    train, val, cal, test = (data["train"], data["val"], data["cal"],
                             data["test"])
    injected = [ep for ep in test if not ep.is_healthy]
    print(f"[{name}] channels={channels} train={len(train)} val={len(val)} "
          f"cal={len(cal)} test={len(test)} (injected {len(injected)})")

    std = Standardizer().fit(train)
    esn, maha, hybrids = make_hybrids(std, channels=channels,
                                      seed=1300 + seed)
    fit_times: dict[str, float] = {}
    t0 = time.perf_counter()
    esn.fit(train)
    fit_times[esn.name] = time.perf_counter() - t0
    t0 = time.perf_counter()
    maha.fit(train)
    fit_times[maha.name] = time.perf_counter() - t0
    for h in hybrids:
        t0 = time.perf_counter()
        h.fit(train)
        fit_times[h.name] = (time.perf_counter() - t0
                             + fit_times[esn.name] + fit_times[maha.name])

    # -- logistic supervision (never on episodes it will score) ------------
    logistic = next(h for h in hybrids if isinstance(h, HybridLogistic))
    log_scores: dict[str, np.ndarray] = {}
    log_val: list[np.ndarray] = []
    cal_injected = [ep for ep in cal if not ep.is_healthy]
    if cal_injected:                       # simulator: dedicated cal split
        logistic.fit_supervised([ep for ep in cal if ep.is_healthy],
                                cal_injected)
        log_val = [logistic.score_episode(ep) for ep in val]
        log_scores = {ep.episode_id: logistic.score_episode(ep)
                      for ep in test}
        log_note = f"cal-split ({len(cal_injected)} injected)"
    elif len(injected) >= 4:               # real data: out-of-fold cross-fit
        # ONE scoring rule for every test episode, healthy and injected
        # alike: each episode is assigned to a fold by its id, never by its
        # label, and scored by the model trained without that fold. Any scheme
        # in which the class selects the scoring algorithm changes the score
        # variance by class and inflates the separation being measured.
        def _make_log():
            return HybridLogistic(esn, maha, std, subs_prefit=True)

        cf = cross_fit_scores(test, _make_log, train, k=2,
                              salt=f"{name}:{seed}")
        log_scores = cf.scores
        logistic.coef_ = cf.mean_coef                     # for reporting only
        logistic.intercept_ = cf.mean_intercept
        # The validation cohort is disjoint from test, so a model trained on
        # ALL injected episodes may score it (used only to pick a threshold).
        val_model = full_model(_make_log, train, injected)
        log_val = [val_model.score_episode(ep) for ep in val]
        n0 = sum(1 for f in cf.folds.values() if f == 0)
        log_note = (f"out-of-fold cross-fit (k=2, folds "
                    f"{n0}/{len(cf.folds) - n0} by id); "
                    f"mean coef={np.asarray(cf.mean_coef).round(3)}")
    else:
        log_note = "insufficient injected episodes; one-class fallback"

    # -- score, threshold, metrics -----------------------------------------
    rows, per_class_rows, det_flags, lead_all, all_scores = [], [], {}, {}, {}
    outcomes: dict[str, dict[str, str]] = {}
    thetas: dict[str, float] = {}
    monitors: list[OnlineMonitor] = [esn, maha] + hybrids
    for mon in monitors:
        if mon is logistic and log_scores:
            val_scores, scores = log_val, log_scores
            elapsed = None                 # cross-fit timing is not comparable
        else:
            val_scores = [mon.score_episode(ep) for ep in val]
            t0 = time.perf_counter()
            scores = {ep.episode_id: mon.score_episode(ep) for ep in test}
            elapsed = time.perf_counter() - t0
        n_steps = sum(ep.T for ep in test)
        theta = float(pick_threshold(val_scores, fa_budget=FA_BUDGET))
        df = evaluate_alarms(test, scores, theta)
        summ = summarize(df)
        y = np.array([0 if ep.is_healthy else 1 for ep in test])
        mx = np.array([float(np.max(scores[ep.episode_id])) for ep in test])
        rows.append({
            "dataset": name, "monitor": mon.name,
            "auroc": float(episode_auc(test, scores)),
            "auprc": float(average_precision_score(y, mx)),
            "detection_rate": summ["detection_rate"],
            "healthy_fa_rate": summ["healthy_fa_rate"],
            "early_alarm_rate": summ["early_alarm_rate"],
            "mean_lead_all": summ["mean_lead_all"],
            "median_delay": summ["median_delay"],
            "fit_seconds": round(fit_times.get(mon.name, float("nan")), 3),
            "step_latency_us": (float("nan") if elapsed is None
                                else round(1e6 * elapsed / n_steps, 1)),
            "footprint_mb": round(_footprint_mb(mon), 2),
        })
        for fc, v in summ["per_class"].items():
            per_class_rows.append({
                "dataset": name, "monitor": mon.name, "failure_class": fc,
                "detection_rate": v["detection_rate"],
                "median_delay": v["median_delay"],
                "mean_lead_all": v["mean_lead_all"],
            })
        inj_df = df[~df["is_healthy"].astype(bool)]
        det_flags[mon.name] = (inj_df["outcome"] == "true_alarm").to_numpy()
        lead_all[mon.name] = inj_df["lead"].fillna(0.0).to_numpy()
        all_scores[mon.name] = scores
        outcomes[mon.name] = dict(zip(df["episode_id"], df["outcome"]))
        thetas[mon.name] = theta
        r = rows[-1]
        print(f"  {mon.name:>18s}: auroc={r['auroc']:.3f} "
              f"auprc={r['auprc']:.3f} det={r['detection_rate']:.2f} "
              f"fa={r['healthy_fa_rate']:.2f} lead={r['mean_lead_all']:.2f}")
    print(f"  [logistic] {log_note}")

    # -- per-episode diagnosis records (objective 1) ------------------------
    diag_rows = []
    inj_sorted = [ep for ep in test if not ep.is_healthy]
    for ep in inj_sorted:
        diag_rows.append({
            "dataset": name, "episode_id": ep.episode_id,
            "failure_class": ep.failure_class, "T": ep.T, "tau": ep.tau,
            "horizon": ep.T - 1 - ep.tau,
            "det_esn": outcomes[esn.name][ep.episode_id] == "true_alarm",
            "det_maha": outcomes[maha.name][ep.episode_id] == "true_alarm",
            "det_logistic": (outcomes[logistic.name][ep.episode_id]
                             == "true_alarm"),
        })

    # -- per-episode explain records: the hybrid's 2-D feature view ---------
    # One scoring path: the SAME calibrated, clipped [z_esn, z_maha] features
    # the logistic consumes, plus the step that decides the episode (argmax
    # of the logistic score stream) and every monitor's alarm outcome.
    explain_rows = []
    for ep in test:
        F = logistic.step_features(ep)          # (T, 2), model-independent
        s = np.asarray(all_scores[logistic.name][ep.episode_id])
        t_star = int(np.argmax(s))
        explain_rows.append({
            "dataset": name, "episode_id": ep.episode_id,
            "is_healthy": ep.is_healthy, "failure_class": ep.failure_class,
            "T": ep.T, "tau": ep.tau,
            "horizon": float("nan") if ep.is_healthy else ep.T - 1 - ep.tau,
            "z_esn_at_alarm": float(F[t_star, 0]),
            "z_maha_at_alarm": float(F[t_star, 1]),
            "z_esn_max": float(F[:, 0].max()),
            "z_maha_max": float(F[:, 1].max()),
            "logit_max": float(s.max()),
            "theta_logistic": thetas[logistic.name],
            "coef_esn": float(logistic.coef_[0]),
            "coef_maha": float(logistic.coef_[1]),
            "intercept": float(logistic.intercept_),
            "outcome_esn": outcomes[esn.name][ep.episode_id],
            "outcome_maha": outcomes[maha.name][ep.episode_id],
            "outcome_logistic": outcomes[logistic.name][ep.episode_id],
        })

    # -- paired statistics: EVERY hybrid vs each standalone -----------------
    # The old code selected the maximum-test-AUROC hybrid and then computed its
    # p-values on that SAME test set, which is post-selection inference
    #. Instead every hybrid-vs-standalone comparison is pre-specified
    # and reported, and the whole family of McNemar p-values is corrected with
    # Holm-Bonferroni. No comparison is chosen after seeing the test
    # scores, so there is nothing to over-fit.
    hybrid_names = [r["monitor"] for r in rows if r["monitor"].startswith("hybrid")]
    stats_rows = []
    mcnemar_family: dict[str, float] = {}
    for hyb in hybrid_names:
        for ref in (esn.name, maha.name):
            mc = mcnemar_test(det_flags[hyb], det_flags[ref])
            pp = paired_permutation_test(lead_all[hyb], lead_all[ref], seed=11)
            wx = wilcoxon_signed_rank(lead_all[hyb], lead_all[ref])
            lo, hi = _bootstrap_dauc(test, all_scores[hyb], all_scores[ref],
                                     seed=13)
            key = f"{hyb}_vs_{ref}"
            mcnemar_family[key] = float(mc["p_value"])
            stats_rows.append({
                "dataset": name, "hybrid": hyb, "vs": ref,
                "dauc_ci_lo": round(lo, 4), "dauc_ci_hi": round(hi, 4),
                "mcnemar_n10": mc["n10"], "mcnemar_n01": mc["n01"],
                "mcnemar_p": mc["p_value"],
                "perm_mean_dlead": round(pp["mean_diff"], 3),
                "perm_p": pp["p_value"], "wilcoxon_p": wx["p_value"],
                "_key": key,
            })
    # Holm correction over this dataset's detection-comparison family.
    corrected = holm_bonferroni(mcnemar_family)
    for row in stats_rows:
        adj = corrected[row.pop("_key")]
        row["mcnemar_p_holm"] = round(adj["p_holm"], 5)
        row["mcnemar_reject_holm"] = adj["reject"]
        row["family_size"] = len(mcnemar_family)
    # -- length-confound diagnostic: is the offline AUROC inflated by
    # episode length, and how does it change under length matching?
    length_rows = [{"dataset": name, "monitor": mon,
                    **length_confound_report(test, all_scores[mon])}
                   for mon in (esn.name, maha.name, logistic.name)
                   if mon in all_scores]
    return {"rows": rows, "per_class": per_class_rows, "stats": stats_rows,
            "diagnosis": diag_rows, "explain": explain_rows,
            "length": length_rows}


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.run_hybrid_study")
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help=f"subset of {['sim'] + list(REAL_DATASETS)}")
    parser.add_argument("--all-datasets", action="store_true",
                        help="include the Phase 10 corpora too (changes the "
                             "scope of every table - use --out-prefix with it)")
    parser.add_argument("--out-prefix", default="hybrid",
                        help="table filename prefix (default: hybrid)")
    parser.add_argument("--seed", type=int, default=0,
                        help="monitor-side seed (ESN reservoirs, cross-fit "
                             "folds) and sim master seed; 0 = published "
                             "configuration")
    args = parser.parse_args(argv)

    # The PUBLISHED tables cover the datasets the last full regeneration ran
    # and verified. Three corpora were added afterwards
    # (ollama_llama8b, real_gemini_long, real_research7b_long_ext); they are
    # reported in their own sections with their own --out-prefix tables, and
    # folding them in here silently would change the scope of every published
    # table without anyone asking for it. Ask for them explicitly with
    # --datasets, or --all-datasets to sweep everything the code knows.
    if args.datasets:
        wanted = list(args.datasets)
    elif args.all_datasets:
        wanted = ["sim"] + list(REAL_DATASETS)
    else:
        wanted = ["sim"] + [d for d in REAL_DATASETS if d in PUBLISHED_DATASETS]
    if args.skip_sim and "sim" in wanted:
        wanted.remove("sim")

    all_rows, all_pc, all_stats, all_diag, all_explain = [], [], [], [], []
    all_length = []
    for name in wanted:
        if name == "sim":
            data, channels = load_sim(seed=args.seed)
        else:
            data, channels = load_real(REAL_DATASETS[name])
        out = evaluate_dataset(name, data, channels, seed=args.seed)
        all_rows += out["rows"]
        all_pc += out["per_class"]
        all_stats += out["stats"]
        all_diag += out["diagnosis"]
        all_explain += out["explain"]
        all_length += out["length"]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    p = args.out_prefix
    pd.DataFrame(all_rows).to_csv(TABLES_DIR / f"{p}_benchmark.csv",
                                  index=False)
    pd.DataFrame(all_pc).to_csv(TABLES_DIR / f"{p}_per_class.csv",
                                index=False)
    pd.DataFrame(all_stats).to_csv(TABLES_DIR / f"{p}_stats.csv", index=False)
    diag = pd.DataFrame(all_diag)
    diag.to_csv(TABLES_DIR / f"{p}_diagnosis.csv", index=False)
    pd.DataFrame(all_explain).to_csv(TABLES_DIR / f"{p}_explain.csv",
                                     index=False)
    pd.DataFrame(all_length).to_csv(TABLES_DIR / f"{p}_length_confound.csv",
                                    index=False)

    # -- horizon analysis (objective 1) -------------------------------------
    if len(diag):
        print("\n[diagnosis] ESN vs Mahalanobis detection by post-onset "
              "horizon (T-1-tau):")
        bins = [(0, 3, "<=3"), (4, 8, "4-8"), (9, 10**9, ">=9")]
        for lo, hi, label in bins:
            sub = diag[(diag["horizon"] >= lo) & (diag["horizon"] <= hi)]
            if not len(sub):
                continue
            print(f"  horizon {label:>4s}: n={len(sub):4d}  "
                  f"ESN det={sub['det_esn'].mean():.2f}  "
                  f"Maha det={sub['det_maha'].mean():.2f}  "
                  f"(ESN-Maha {sub['det_esn'].mean() - sub['det_maha'].mean():+.2f})")
        d = diag
        adv = d["det_esn"].astype(float) - d["det_maha"].astype(float)
        if adv.std() > 0 and d["horizon"].std() > 0:
            r = float(np.corrcoef(d["horizon"], adv)[0, 1])
            print(f"  corr(horizon, ESN-advantage) = {r:+.3f} over "
                  f"{len(d)} injected episodes")
    print(f"\n[hybrid] wrote {p}_benchmark/_per_class/_stats/_diagnosis.csv "
          f"to {TABLES_DIR}")


if __name__ == "__main__":
    main()
