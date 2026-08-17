"""Post-onset horizon vs the ESN's advantage over the memoryless baseline.

`run_hybrid_study` records, per injected episode, the post-onset horizon
(T - 1 - tau) and whether each detector alarmed, then reports the
ESN-minus-Mahalanobis detection gap inside three horizon bands. That report
pools every episode from every corpus into one mean per band, which is only a
statement about horizon if the corpora are otherwise comparable. They are not:
each corpus is a different model, framework, task and injector, and each one
occupies a different part of the horizon range. Pooled band means therefore mix
the horizon effect with the question of WHICH corpora happen to populate each
band, and a correlation over pooled episodes treats rows clustered by corpus as
independent.

This study estimates the same relationship without that confound:

1. POOLED, reproducing the pooled band means exactly, and reporting alongside
   them the composition that produces them - how much of each band is
   simulator, and which corpora contribute.
2. WITHIN corpus, the fixed-effects estimator: horizon and gap are centred
   inside each corpus before being correlated, so a corpus that is simply
   better or longer than another contributes nothing. Significance comes from
   `stratified_permutation_test`, which shuffles horizon inside each corpus.
3. WITHIN corpus AND failure class, because classes differ in both detectability
   and typical length, so corpus alone does not remove the composition effect.
4. PER corpus, one row per corpus and band, which is the only view that shows
   the direction each deployment actually moves in.

Every real corpus the repository holds is scored, including the ones outside
`run_hybrid_study.PUBLISHED_DATASETS`, because the published scope reaches the
top band with 10 real episodes and the question is precisely what happens
there. Sim rows are loaded and reported but never pooled with real ones: they
are labelled `provenance`, and the headline estimates are real-only.

Run:  py -m derail.experiments.run_horizon_study
Writes results/tables/horizon_{pooled,by_dataset,within,contrasts}.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from derail.evaluation.stats import (
    stratified_permutation_test,
    within_stratum_corr,
)

TABLES_DIR = Path(__file__).resolve().parents[2] / "results" / "tables"

#: Bands the published horizon report uses, kept identical so the pooled view
#: reproduces it rather than re-cutting the axis.
BANDS: tuple[tuple[str, int, int], ...] = (
    ("<=3", 0, 3),
    ("4-8", 4, 8),
    (">=9", 9, 10 ** 6),
)

#: Episode-level diagnosis tables to read, widest first. Every one carries the
#: same columns; later tables only contribute episodes not already seen, keyed
#: by (dataset, episode_id), so a corpus scored in two runs is counted once.
#: `l7b` is a strict superset of `hybrid` on the seven published corpora and is
#: verified to agree with it episode for episode by
#: test_horizon_sources_agree_where_they_overlap.
#:
#: The corpora the LIVE serving path runs on (`live_*`, `live_ext_*`) are
#: deliberately absent. The law is checked against that deployment in
#: `results/horizon_report.md` §6, and a law estimated partly on the corpus it
#: is then tested against is not being tested.
SOURCES: tuple[str, ...] = (
    "l7b_diagnosis.csv",          # 10 corpora incl. sim, the widest run
    "hybrid_diagnosis.csv",       # the published scope
    "l7bx_diagnosis.csv",         # real_research7b_long_ext
    "drift_diagnosis.csv",        # real_research7b_long_drift (goal_drift)
    "aftraj_diagnosis.csv",       # AFTraj-2K, external
    "gemini_long_diagnosis.csv",  # real_gemini_long
)


def _band_of(horizon: pd.Series) -> pd.Series:
    edges = [BANDS[0][1] - 1] + [hi for _, _, hi in BANDS]
    return pd.cut(horizon, edges, labels=[b for b, _, _ in BANDS])


def load_records(tables_dir: Path = TABLES_DIR) -> pd.DataFrame:
    """Every scored injected episode, one row each, deduplicated by key.

    Raises if a table that IS present disagrees with an earlier one about an
    episode both scored: that means two runs of the same code produced two
    answers for one episode, which has to be resolved before anything is
    estimated from either.
    """
    frames, seen = [], {}
    for fname in SOURCES:
        path = tables_dir / fname
        if not path.exists():          # optional corpora (aftraj is not committed)
            continue
        d = pd.read_csv(path)
        d = d[["dataset", "episode_id", "failure_class", "T", "tau",
               "horizon", "det_esn", "det_maha"]].copy()
        d["source"] = fname
        keys = list(zip(d.dataset, d.episode_id))
        fresh = []
        for k, row in zip(keys, d.itertuples()):
            prev = seen.get(k)
            if prev is None:
                seen[k] = (row.det_esn, row.det_maha, fname)
                fresh.append(True)
                continue
            fresh.append(False)
            if (prev[0], prev[1]) != (row.det_esn, row.det_maha):
                raise ValueError(
                    f"{fname} disagrees with {prev[2]} on {k}: "
                    f"det_esn/det_maha {(row.det_esn, row.det_maha)} vs "
                    f"{(prev[0], prev[1])}")
        frames.append(d[np.array(fresh, dtype=bool)])
    if not frames:
        raise FileNotFoundError(f"no diagnosis tables under {tables_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["provenance"] = np.where(out.dataset == "sim", "sim", "real")
    out["band"] = _band_of(out.horizon)
    out["gap"] = out.det_esn.astype(int) - out.det_maha.astype(int)
    return out


def pooled_table(d: pd.DataFrame) -> pd.DataFrame:
    """Band means over pooled episodes, with the composition behind each one."""
    rows = []
    for arm, sub in (("all", d), ("real", d[d.provenance == "real"]),
                     ("sim", d[d.provenance == "sim"])):
        for band, _, _ in BANDS:
            b = sub[sub.band == band]
            if b.empty:
                continue
            rows.append({
                "arm": arm, "band": band, "n": len(b),
                "det_esn": round(float(b.det_esn.mean()), 4),
                "det_maha": round(float(b.det_maha.mean()), 4),
                "gap": round(float(b.gap.mean()), 4),
                "sim_share": round(float((b.dataset == "sim").mean()), 4),
                "n_datasets": int(b.dataset.nunique()),
                "top_dataset": b.dataset.value_counts().idxmax(),
                "top_dataset_share": round(
                    float(b.dataset.value_counts(normalize=True).iloc[0]), 4),
            })
    return pd.DataFrame(rows)


def by_dataset_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per corpus and band - the view that shows per-deployment direction."""
    rows = []
    for ds, sub in d.groupby("dataset", sort=True):
        for band, _, _ in BANDS:
            b = sub[sub.band == band]
            if b.empty:
                continue
            rows.append({
                "dataset": ds,
                "provenance": sub.provenance.iloc[0],
                "band": band, "n": len(b),
                "det_esn": round(float(b.det_esn.mean()), 4),
                "det_maha": round(float(b.det_maha.mean()), 4),
                "gap": round(float(b.gap.mean()), 4),
                "n_classes": int(b.failure_class.nunique()),
            })
    return pd.DataFrame(rows)


def contrasts_table(d: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    """Adjacent-band gap deltas measured INSIDE each corpus, then combined.

    A corpus contributes a contrast only when both bands hold at least `min_n`
    episodes, so a band represented by one or two episodes cannot carry a
    direction. The combined row is the unweighted mean over corpora, which
    treats each deployment as one observation - the unit the claim generalises
    over - and its p-value is a sign-flip permutation over corpora.
    """
    from derail.evaluation.stats import paired_permutation_test

    pairs = (("<=3", "4-8"), ("4-8", ">=9"), ("<=3", ">=9"))
    rows = []
    for lo, hi in pairs:
        deltas, names = [], []
        for ds, sub in d.groupby("dataset", sort=True):
            a = sub[sub.band == lo]
            b = sub[sub.band == hi]
            if len(a) < min_n or len(b) < min_n:
                continue
            deltas.append(float(b.gap.mean() - a.gap.mean()))
            names.append(ds)
        for ds, delta in zip(names, deltas):
            rows.append({"contrast": f"{lo} -> {hi}", "dataset": ds,
                         "provenance": "sim" if ds == "sim" else "real",
                         "delta_gap": round(delta, 4), "n_datasets": 1})
        real = [x for x, n in zip(deltas, names) if n != "sim"]
        if real:
            arr = np.asarray(real, dtype=float)
            t = paired_permutation_test(arr, np.zeros_like(arr), seed=0)
            rows.append({
                "contrast": f"{lo} -> {hi}", "dataset": "COMBINED(real)",
                "provenance": "real",
                "delta_gap": round(float(arr.mean()), 4),
                "n_datasets": len(real),
                "n_positive": int((arr > 0).sum()),
                "n_negative": int((arr < 0).sum()),
                "p_value": round(float(t["p_value"]), 4),
            })
    return pd.DataFrame(rows)


def within_table(d: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    """Pooled vs within-corpus vs within-corpus-and-class correlation.

    The three rows answer the same question at three levels of control, so the
    drop between them IS the confound, measured.
    """
    rows = []
    for arm, sub in (("all", d), ("real", d[d.provenance == "real"])):
        if sub.empty:
            continue
        h, g = sub.horizon.to_numpy(float), sub.gap.to_numpy(float)
        pooled = (float(np.corrcoef(h, g)[0, 1])
                  if h.std() > 0 and g.std() > 0 else float("nan"))
        one = np.zeros(len(sub), dtype=int)
        pooled_t = stratified_permutation_test(h, g, one, n_perm=n_perm,
                                               seed=seed)
        ds = sub.dataset.to_numpy()
        ds_t = stratified_permutation_test(h, g, ds, n_perm=n_perm, seed=seed)
        dc = np.array([f"{a}|{b}" for a, b in
                       zip(sub.dataset, sub.failure_class.fillna("?"))])
        dc_t = stratified_permutation_test(h, g, dc, n_perm=n_perm, seed=seed)
        rows += [
            {"arm": arm, "control": "none (pooled)", "r": round(pooled, 4),
             "p_value": round(pooled_t["p_value"], 5), "n": len(sub),
             "n_strata": 1},
            {"arm": arm, "control": "dataset", "r": round(ds_t["r_within"], 4),
             "p_value": round(ds_t["p_value"], 5), "n": len(sub),
             "n_strata": ds_t["n_strata"]},
            {"arm": arm, "control": "dataset x class",
             "r": round(dc_t["r_within"], 4),
             "p_value": round(dc_t["p_value"], 5), "n": len(sub),
             "n_strata": dc_t["n_strata"]},
        ]
    return pd.DataFrame(rows)


def robustness_table(d: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-corpus-out on the real within-corpus estimate.

    A within-corpus correlation is still a cross-corpus claim if one corpus
    supplies most of the rows. Each row here drops one corpus and re-estimates,
    so a result that depends on a single corpus shows up as a row that moves.
    """
    r = d[d.provenance == "real"]
    rows = [{"dropped": "(none)", "n": len(r),
             "r_within_dataset": round(within_stratum_corr(
                 r.horizon.to_numpy(float), r.gap.to_numpy(float),
                 r.dataset.to_numpy()), 4),
             "gap_ge9": round(float(r[r.band == ">=9"].gap.mean()), 4),
             "n_ge9": int((r.band == ">=9").sum())}]
    for ds in sorted(r.dataset.unique()):
        s = r[r.dataset != ds]
        top = s[s.band == ">=9"]
        rows.append({
            "dropped": ds, "n": len(s),
            "r_within_dataset": round(within_stratum_corr(
                s.horizon.to_numpy(float), s.gap.to_numpy(float),
                s.dataset.to_numpy()), 4),
            "gap_ge9": round(float(top.gap.mean()), 4) if len(top) else float("nan"),
            "n_ge9": len(top)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="horizon")
    args = ap.parse_args()

    d = load_records()
    print(f"[horizon] {len(d)} injected episodes over "
          f"{d.dataset.nunique()} corpora "
          f"({(d.provenance == 'real').sum()} real, "
          f"{(d.provenance == 'sim').sum()} sim)")

    pooled = pooled_table(d)
    by_ds = by_dataset_table(d)
    contr = contrasts_table(d)
    within = within_table(d, n_perm=args.n_perm, seed=args.seed)
    robust = robustness_table(d)

    print("\n[pooled] band means and what populates them")
    print(pooled.to_string(index=False))
    print("\n[within] the same relationship at three levels of control")
    print(within.to_string(index=False))
    print("\n[contrasts] adjacent-band deltas measured inside each corpus")
    print(contr.to_string(index=False))
    print("\n[robustness] real estimate with one corpus dropped at a time")
    print(robust.to_string(index=False))

    p = args.out_prefix
    for name, frame in (("pooled", pooled), ("by_dataset", by_ds),
                        ("contrasts", contr), ("within", within),
                        ("robustness", robust)):
        frame.to_csv(TABLES_DIR / f"{p}_{name}.csv", index=False)
    print(f"\n[horizon] wrote {p}_"
          "{pooled,by_dataset,contrasts,within,robustness}.csv to "
          + str(TABLES_DIR))


if __name__ == "__main__":
    main()
