"""Content-grounding channel study (exp/grounding-channel).

Does the g channel (telemetry v4) close the content-corruption gap that
behavioral (ESN), statistical (Mahalanobis), and 2-way hybrid monitors all
share? Compares, on every real dataset under the frozen protocol:

    esn_cusum_max, delta_mahalanobis,
    hybrid_weighted50, hybrid_logistic          (the merged 2-way hybrids)
    grounding                                   (the new channel alone)
    hybrid_weighted_g, hybrid_logistic_g        (3-way grounded hybrids)

plus a per-dim ablation of the nine grounding signals. The simulator is
excluded by construction: its telemetry is synthetic vectors with no step
text or tool results, so the g channel does not exist there. gemini (v1
text format, results never recorded) is included deliberately — its g dims
are inert zeros, and showing that the grounded monitors degrade cleanly to
the ungrounded ones there is part of the contract.

Supervision for the logistic variants: 2-fold class-stratified cross-fit,
identical to run_hybrid_study (no episode is scored by a model that saw
it). Statistics: paired McNemar / permutation / Wilcoxon + bootstrap
dAUC CI of hybrid_logistic_g vs hybrid_logistic (the marginal value of
grounding) per dataset.

Run:  py -m derail.experiments.run_grounding_study
Writes results/tables/grounding_{benchmark,per_class,stats,ablation}.csv.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from derail.common import Episode, Standardizer
from derail.evaluation.metrics import (
    episode_auc,
    evaluate_alarms,
    pick_threshold,
    summarize,
)
from derail.evaluation.stats import (
    mcnemar_test,
    paired_permutation_test,
    wilcoxon_signed_rank,
)
from derail.experiments.run_hybrid_study import (
    FA_BUDGET,
    REAL_DATASETS,
    TABLES_DIR,
    _bootstrap_dauc,
    _footprint_mb,
    load_real,
)
from derail.monitor.grounding import (
    GRD_DIM_NAMES,
    GroundingMonitor,
    HybridAdaptive,
    HybridContentGate,
    HybridLogisticG,
    HybridWeightedG,
)
from derail.evaluation.protocol import (cross_fit_scores, full_model,
                                        holm_bonferroni)
from derail.monitor.hybrid import HybridLogistic, HybridWeighted, make_hybrids

CONTENT_CLASSES = ("context_corruption", "wrong_document", "malformed_json")

#: Datasets the PUBLISHED results/tables/grounding_*.csv are computed from.
#: The same guard `run_hybrid_study.PUBLISHED_DATASETS` provides, and for the
#: same reason: defaulting to all of `REAL_DATASETS` lets a corpus added later
#: widen the scope of every grounding table on the next regeneration, without
#: anyone asking. Request anything outside this set explicitly with --datasets.
GROUNDING_PUBLISHED_DATASETS: tuple[str, ...] = (
    "gemini", "autogen7b", "ollama7b", "langgraph7b",
    "real_research7b", "real_research7b_long", "real_research3b",
    "ollama_llama8b", "real_gemini_long", "real_research7b_long_ext",
)


def _crossfit_logistic(make, train, test, name_label, seed=0):
    """Out-of-fold scores for EVERY test episode under one rule + a val scorer.

    Each episode (healthy or injected) is assigned to a fold by its id and
    scored by the model trained without its fold. One rule for every episode:
    scoring injected episodes with a single opposite-fold model while AVERAGING
    two models for healthy ones would let the true class choose the scoring
    rule. A model trained on all injected scores the disjoint validation
    cohort.
    """
    injected = [ep for ep in test if not ep.is_healthy]
    cf = cross_fit_scores(test, make, train, k=2, salt=f"{name_label}:{seed}")
    val_model = full_model(make, train, injected)
    return cf.scores, val_model.score_episode, cf.mean_coef


def _view51(episodes: list[Episode]) -> list[Episode]:
    """Published 51-dim view of 60-dim episodes (drop the g dims).

    The ungrounded reference monitors must see EXACTLY the published
    telemetry: DeltaMahalanobis consumes the full vector, so scoring it on
    60-dim episodes would leak grounding information into the baseline and
    corrupt the attribution. Same ids, same labels, first 51 dims.
    """
    from derail.common import D_TOTAL_EXT
    return [Episode(X=ep.X[:, :D_TOTAL_EXT].copy(),
                    episode_id=ep.episode_id, is_healthy=ep.is_healthy,
                    failure_class=ep.failure_class, tau=ep.tau,
                    t_fail=ep.t_fail, severity=ep.severity)
            for ep in episodes]


def evaluate_dataset(name: str, data, channels, seed: int = 0) -> dict:
    train, val, test = data["train"], data["val"], data["test"]
    injected = [ep for ep in test if not ep.is_healthy]
    print(f"[{name}] channels={channels} train={len(train)} "
          f"val={len(val)} test={len(test)} (injected {len(injected)})")

    # -- ungrounded references on the published 51-dim view -----------------
    train51, val51, test51 = _view51(train), _view51(val), _view51(test)
    std51 = Standardizer().fit(train51)
    esn, maha, hybrids2 = make_hybrids(std51, channels=channels,
                                       seed=1300 + seed)
    esn.fit(train51)
    maha.fit(train51)
    weighted = next(h for h in hybrids2 if isinstance(h, HybridWeighted))
    weighted.fit(train51)

    # -- grounded monitors: behavioural submodels on the 51-dim view --------
    # The behavioural ESN/Mahalanobis must NOT see the grounding dims, which
    # the explicit grounding stream already covers; fitting them on the full
    # 60-dim telemetry double-counted grounding inside the behavioural distance.
    # They are fit on the 51-dim view and the grounded hybrids mask
    # scoring to 51 via behav_slice.
    std56 = Standardizer().fit(train51)
    esn56, maha56, _ = make_hybrids(std56, channels=channels,
                                    seed=1300 + seed)
    esn56.fit(train51)
    maha56.fit(train51)
    grd = GroundingMonitor()          # standalone row: all dims incl. lex
    grd.fit(train)
    # Hybrids calibrate their continuous grounding stream WITHOUT the
    # binary lex dim (its {0,1} z would inflate the trip and swallow
    # context evidence); lex reaches them via the override/feature path.
    grd_cont = GroundingMonitor(dims=GRD_DIM_NAMES[:-1],
                                name="grounding_cont")
    grd_cont.fit(train)
    wg = HybridWeightedG(esn56, maha56, grd_cont, std56, subs_prefit=True)
    wg.fit(train)
    gate = HybridContentGate(esn56, maha56, grd_cont, std56,
                             subs_prefit=True)
    gate.fit(train)
    adap = HybridAdaptive(esn56, maha56, grd_cont, std56, subs_prefit=True)
    adap.fit(train)

    # (monitor, prescored, val_eps, test_eps)
    monitors: list[tuple] = [
        (esn, None, val51, test51), (maha, None, val51, test51),
        (weighted, None, val51, test51), (grd, None, val, test),
        (wg, None, val, test), (gate, None, val, test),
        (adap, None, val, test)]

    # cross-fit logistics (2-way on the view, grounded 3-way on full)
    logi2 = HybridLogistic(esn, maha, std51, subs_prefit=True)
    logi2.fit(train51)
    logi3 = HybridLogisticG(esn56, maha56, grd_cont, std56,
                            subs_prefit=True)
    logi3.fit(train)
    cf = {}
    if len(injected) >= 4:
        for mon, make, tr, t_eps, v_eps in (
            (logi2, lambda: HybridLogistic(esn, maha, std51,
                                           subs_prefit=True),
             train51, test51, val51),
            (logi3, lambda: HybridLogisticG(esn56, maha56, grd_cont, std56,
                                            subs_prefit=True),
             train, test, val),
        ):
            # OOF scores cover every test episode, healthy and injected;
            # val_fn is a model trained on all injected, for the disjoint val.
            scores, val_fn, coef = _crossfit_logistic(
                make, tr, t_eps, name, seed=seed)
            mon.coef_, cf[mon.name] = coef, (scores, val_fn)
            monitors.append((mon, cf[mon.name][0], v_eps, t_eps))
        print(f"  [logistic] coefs 2-way={logi2.coef_.round(3)} "
              f"3-way={logi3.coef_.round(3)}")

    rows, pc_rows, det_flags, lead_all, all_scores = [], [], {}, {}, {}
    n_steps = sum(ep.T for ep in test)
    for mon, prescored, v_eps, t_eps in monitors:
        if prescored is not None:
            val_fn = cf[mon.name][1]
            val_scores = [val_fn(ep) for ep in v_eps]
            # `prescored` already holds OOF scores for every test episode,
            # healthy and injected, under one rule - no per-class re-scoring.
            scores = dict(prescored)
            elapsed = None
        else:
            val_scores = [mon.score_episode(ep) for ep in v_eps]
            t0 = time.perf_counter()
            scores = {ep.episode_id: mon.score_episode(ep) for ep in t_eps}
            elapsed = time.perf_counter() - t0
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
            "mean_lead_all": summ["mean_lead_all"],
            "median_delay": summ["median_delay"],
            "step_latency_us": (float("nan") if elapsed is None
                                else round(1e6 * elapsed / n_steps, 1)),
            "footprint_mb": round(_footprint_mb(mon), 2),
        })
        for fc, v in summ["per_class"].items():
            pc_rows.append({"dataset": name, "monitor": mon.name,
                            "failure_class": fc,
                            "detection_rate": v["detection_rate"],
                            "median_delay": v["median_delay"]})
        inj_df = df[~df["is_healthy"].astype(bool)]
        det_flags[mon.name] = (inj_df["outcome"] == "true_alarm").to_numpy()
        lead_all[mon.name] = inj_df["lead"].fillna(0.0).to_numpy()
        all_scores[mon.name] = scores
        r = rows[-1]
        print(f"  {r['monitor']:>18s}: auroc={r['auroc']:.3f} "
              f"det={r['detection_rate']:.2f} fa={r['healthy_fa_rate']:.2f} "
              f"lead={r['mean_lead_all']:.2f}")

    # -- joint-budget deployment: OR of the two streams, calibrated to the
    # FULL false-alarm budget on validation.
    #
    # The previous "dual budget" claimed FA/2 per stream but did not implement
    # it: the behavioural stream kept the whole budget, the grounding stream
    # tripped at its healthy-train maximum, and the lexical override was
    # unbudgeted, so OR-ing the three could exceed the 5% guarantee.
    # FA/2 per stream would not compose to FA anyway unless the streams were
    # independent, which they are not.
    #
    # Honest replacement: build the SAME per-step decision statistic used at
    # test time - max over the behavioural, grounding and (clean-null) lexical
    # streams, each in per-stream-trip units - and pick the ONE threshold that
    # spends the full FA budget on the healthy validation cohort. The realized
    # joint healthy FA is then <= FA_BUDGET by construction, and the table
    # reports it.
    from derail.common import IDX_GRD_LEX_MISS

    def _joint_stream(ep):
        b, g = wg.score_episode_streams(ep)
        s = np.maximum(b, g)
        if wg._lex_clean:
            s = np.maximum(s, ep.X[:, IDX_GRD_LEX_MISS])
        return s

    theta_joint = float(pick_threshold([_joint_stream(ep) for ep in val],
                                       fa_budget=FA_BUDGET))
    dual_scores = {ep.episode_id: _joint_stream(ep) for ep in test}
    df = evaluate_alarms(test, dual_scores, theta_joint)
    summ = summarize(df)
    y = np.array([0 if ep.is_healthy else 1 for ep in test])
    mx = np.array([float(np.max(dual_scores[ep.episode_id])) for ep in test])
    rows.append({
        "dataset": name, "monitor": "joint_budget",
        "auroc": float(episode_auc(test, dual_scores)),
        "auprc": float(average_precision_score(y, mx)),
        "detection_rate": summ["detection_rate"],
        "healthy_fa_rate": summ["healthy_fa_rate"],
        "mean_lead_all": summ["mean_lead_all"],
        "median_delay": summ["median_delay"],
        "step_latency_us": float("nan"),
        "footprint_mb": round(_footprint_mb(wg), 2)})
    for fc, v in summ["per_class"].items():
        pc_rows.append({"dataset": name, "monitor": "joint_budget",
                        "failure_class": fc,
                        "detection_rate": v["detection_rate"],
                        "median_delay": v["median_delay"]})
    inj_df = df[~df["is_healthy"].astype(bool)]
    det_flags["joint_budget"] = (inj_df["outcome"] == "true_alarm").to_numpy()
    lead_all["joint_budget"] = inj_df["lead"].fillna(0.0).to_numpy()
    all_scores["joint_budget"] = dual_scores
    r = rows[-1]
    print(f"  {'joint_budget':>18s}: auroc={r['auroc']:.3f} "
          f"det={r['detection_rate']:.2f} fa={r['healthy_fa_rate']:.2f} "
          f"lead={r['mean_lead_all']:.2f}")

    # marginal value of grounding: 3-way logistic vs 2-way logistic
    stats_rows = []
    if "hybrid_logistic_g" in det_flags:
        for ref in ("hybrid_logistic", "hybrid_weighted50"):
            mine = "hybrid_logistic_g" if ref == "hybrid_logistic" \
                else "hybrid_weighted_g"
            mc = mcnemar_test(det_flags[mine], det_flags[ref])
            pp = paired_permutation_test(lead_all[mine], lead_all[ref],
                                         seed=17)
            wx = wilcoxon_signed_rank(lead_all[mine], lead_all[ref])
            lo, hi = _bootstrap_dauc(test, all_scores[mine],
                                     all_scores[ref], seed=19)
            stats_rows.append({
                "dataset": name, "grounded": mine, "vs": ref,
                "dauc_ci_lo": round(lo, 4), "dauc_ci_hi": round(hi, 4),
                "mcnemar_n10": mc["n10"], "mcnemar_n01": mc["n01"],
                "mcnemar_p": mc["p_value"],
                "perm_mean_dlead": round(pp["mean_diff"], 3),
                "perm_p": pp["p_value"], "wilcoxon_p": wx["p_value"]})

    # per-episode detection records: the success criterion is per-CLASS
    # (content classes must improve, behavioral classes must not degrade),
    # so class-level paired tests need episode-level flags.
    diag_rows = []
    for i, ep in enumerate(injected):
        diag_rows.append({
            "dataset": name, "episode_id": ep.episode_id,
            "failure_class": ep.failure_class,
            "is_content": ep.failure_class in CONTENT_CLASSES,
            **{f"det_{k}": bool(det_flags[k][i]) for k in det_flags}})

    # per-dim ablation, focused on where the channel is supposed to work
    abl_rows = []
    for dim in GRD_DIM_NAMES:
        m = GroundingMonitor(dims=(dim,))
        m.fit(train)
        val_scores = [m.score_episode(ep) for ep in val]
        theta = float(pick_threshold(val_scores, fa_budget=FA_BUDGET))
        scores = {ep.episode_id: m.score_episode(ep) for ep in test}
        summ = summarize(evaluate_alarms(test, scores, theta))
        abl_rows.append({
            "dataset": name, "dim": dim,
            "auroc": float(episode_auc(test, scores)),
            "detection_rate": summ["detection_rate"],
            "healthy_fa_rate": summ["healthy_fa_rate"],
            **{f"det[{fc}]": summ["per_class"].get(fc, {}).get(
                "detection_rate", float("nan")) for fc in CONTENT_CLASSES}})

    return {"rows": rows, "per_class": pc_rows, "stats": stats_rows,
            "ablation": abl_rows, "diagnosis": diag_rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="py -m derail.experiments.run_grounding_study")
    parser.add_argument("--datasets", nargs="*",
                        default=list(GROUNDING_PUBLISHED_DATASETS))
    parser.add_argument("--out-prefix", default="grounding")
    parser.add_argument("--st", action="store_true",
                        help="MiniLM for ALL semantic dims (explicit "
                             "opt-in; sets HF offline mode)")
    parser.add_argument("--seed", type=int, default=0,
                        help="monitor-side seed (ESN reservoirs, cross-"
                             "fit folds); splits frozen; 0 = published")
    args = parser.parse_args(argv)
    if args.st:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    all_rows, all_pc, all_stats, all_abl, all_diag = [], [], [], [], []
    for name in args.datasets:
        data, channels = load_real(REAL_DATASETS[name], grounding=True,
                                   use_st=args.st)
        out = evaluate_dataset(name, data, channels, seed=args.seed)
        all_rows += out["rows"]
        all_pc += out["per_class"]
        all_stats += out["stats"]
        all_abl += out["ablation"]
        all_diag += out["diagnosis"]

    p = args.out_prefix
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(TABLES_DIR / f"{p}_benchmark.csv",
                                  index=False)
    pd.DataFrame(all_pc).to_csv(TABLES_DIR / f"{p}_per_class.csv",
                                index=False)
    pd.DataFrame(all_stats).to_csv(TABLES_DIR / f"{p}_stats.csv",
                                   index=False)
    pd.DataFrame(all_abl).to_csv(TABLES_DIR / f"{p}_ablation.csv",
                                 index=False)
    diag = pd.DataFrame(all_diag)
    diag.to_csv(TABLES_DIR / f"{p}_diagnosis.csv", index=False)

    # -- THE success criterion: content classes improve, behavioral don't
    # degrade. Grounded vs ungrounded paired McNemar.
    #
    # pooling framework/model datasets into ONE McNemar mixes
    # heterogeneous populations and reports uncorrected p-values over many
    # monitor/group comparisons. Fixed by (a) STRATIFYING the paired counts by
    # dataset (a Cochran-Mantel-Haenszel-style pooled discordant-pair test that
    # respects the strata) and (b) Holm-correcting the whole family of
    # comparison p-values. The pooled descriptive rate is still shown, labelled
    # as descriptive.
    if len(diag):
        print("\n[criterion] dataset-STRATIFIED paired detection, grounded vs "
              "ungrounded (Holm-corrected):")
        crit_rows, family = [], {}
        for grounded, ref in (("det_hybrid_logistic_g", "det_hybrid_logistic"),
                              ("det_hybrid_weighted_g", "det_hybrid_weighted50"),
                              ("det_hybrid_content_gate", "det_hybrid_weighted50"),
                              ("det_hybrid_adaptive", "det_hybrid_weighted50"),
                              ("det_joint_budget", "det_hybrid_weighted50")):
            if grounded not in diag.columns:
                continue
            for label, sub in (("content", diag[diag.is_content]),
                               ("behavioral", diag[~diag.is_content])):
                # Pool discordant pairs ACROSS datasets (strata), not the raw
                # episodes: n10/n01 summed per dataset, one exact test on the
                # totals - homogeneous within each stratum.
                n10 = n01 = 0
                for _, s in sub.groupby("dataset", sort=False):
                    n10 += int(((s[grounded]) & (~s[ref].astype(bool))).sum())
                    n01 += int(((~s[grounded].astype(bool)) & (s[ref])).sum())
                mc = mcnemar_test(np.r_[np.ones(n10), np.zeros(n01)],
                                  np.zeros(n10 + n01)) if (n10 + n01) else \
                    {"p_value": 1.0}
                key = f"{grounded[4:]}_vs_{ref[4:]}_{label}"
                family[key] = float(mc["p_value"])
                crit_rows.append((key, grounded, ref, label, len(sub),
                                  float(sub[ref].mean()),
                                  float(sub[grounded].mean()), n10, n01))
        holm = holm_bonferroni(family)
        for key, grounded, ref, label, n, r0, r1, n10, n01 in crit_rows:
            adj = holm[key]
            print(f"  {grounded[4:]:>18s} vs {ref[4:]:<18s} "
                  f"[{label:>10s}] n={n:4d}  det {r0:.2f} -> {r1:.2f}  "
                  f"(+only {n10}, -only {n01}, "
                  f"p={adj['p_raw']:.2e}, Holm={adj['p_holm']:.2e}"
                  f"{' *' if adj['reject'] else ''})")

    pc = pd.DataFrame(all_pc)
    focus = pc[pc.failure_class.isin(CONTENT_CLASSES)]
    if len(focus):
        pv = focus.pivot_table(index="failure_class", columns="monitor",
                               values="detection_rate", aggfunc="mean")
        print("\n[grounding] content-class detection "
              "(mean across datasets):")
        print(pv.round(2).to_string())
    print(f"\n[grounding] wrote {p}_benchmark/_per_class/_stats/"
          f"_ablation.csv to {TABLES_DIR}")


if __name__ == "__main__":
    main()
