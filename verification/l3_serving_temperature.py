"""L3 - Does organic detection survive at the SERVING temperature?

Every organic detection number in this
repository was measured at sampling temperature 0.9; the demo and the shipped
monitor serve at 0.2, and 0.9 is a failure-PROVOKING setting that also makes
small models emit junk tokens and leak tool syntax. So "the monitor detects
organic failure" and "the monitor detects high-temperature degradation that
co-occurs with failure" currently fit the evidence equally well.

This contrasts the two arms - same 120 task seeds, same model, same toolset,
same labeller, same served monitor and FA budget, no injection - varying only
temperature. Each arm is scored against its OWN temperature-matched cross-fit
null (a shared null would reintroduce the confound).

Run (after collecting and scoring both arms):
  py -m verification.l3_serving_temperature
Writes results/tables/serving_temperature.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
HOT_CSV = TABLES / "organic_hallucination_ext.csv"      # temperature 0.9
COLD_CSV = TABLES / "organic_hallucination_cold.csv"    # temperature 0.2
LABELS = ("healthy", "arithmetic_error", "hallucinated", "other",
          "incomplete")
MIN_N = 10          # pre-registered per-class power floor


def _counts(df: pd.DataFrame, label: str) -> tuple[int, int]:
    d = df[df.label == label]
    return len(d), int(d.alarmed.sum())


def _fisher(a_hit: int, a_n: int, b_hit: int, b_n: int, alt: str) -> float:
    return float(sps.fisher_exact(
        [[a_hit, a_n - a_hit], [b_hit, b_n - b_hit]], alternative=alt)[1])


def main() -> None:
    for path, temp in ((HOT_CSV, "0.9"), (COLD_CSV, "0.2")):
        if not path.exists():
            raise SystemExit(
                f"{path.name} not found - score the temperature-{temp} arm "
                f"first.")
    hot, cold = pd.read_csv(HOT_CSV), pd.read_csv(COLD_CSV)

    rows = []
    for arm, df in (("0.9 (provoking)", hot), ("0.2 (serving)", cold)):
        n_h, k_h = _counts(df, "healthy")
        for lab in LABELS:
            n, k = _counts(df, lab)
            if not n:
                continue
            rows.append({
                "temperature": arm, "label": lab, "n": n, "alarmed": k,
                "rate": round(k / n, 3),
                "vs_healthy_p": ("" if lab == "healthy" or n < MIN_N
                                 else round(_fisher(k, n, k_h, n_h,
                                                    "greater"), 4)),
                "powered": "" if lab == "healthy" else n >= MIN_N})
    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "serving_temperature.csv", index=False)

    print("[L3] organic detection at provoking (0.9) vs serving (0.2) "
          "temperature\n")
    print(table.to_string(index=False))

    # ---- base rate and failure mix, reported BEFORE any detection claim ----
    print("\n[L3] organic failure base rate and mix")
    mix = {}
    for arm, df in (("0.9", hot), ("0.2", cold)):
        n_fail = int((df.label != "healthy").sum())
        print(f"  T={arm}: {n_fail}/{len(df)} = {n_fail / len(df):.0%} failed"
              f"   mix: " + ", ".join(
                  f"{lab}={int((df.label == lab).sum())}"
                  for lab in LABELS if lab != "healthy"))
        mix[arm] = [int((df.label == lab).sum())
                    for lab in LABELS if lab != "healthy"]
    obs = list(mix.values())          # one row per arm, one column per class
    try:
        chi2, p_mix, _, _ = sps.chi2_contingency(obs)
        print(f"  failure-mix shift across arms: chi2={chi2:.2f}, "
              f"p={p_mix:.4f}"
              + ("  <- mix differs, so read the PER-CLASS rows"
                 if p_mix < 0.05 else "  <- mix comparable"))
    except ValueError as exc:                         # a class is empty
        print(f"  failure-mix test not computable ({exc})")

    # ---- primary comparison: all-failure detection across arms -----------
    def _all_fail(df):
        d = df[df.label != "healthy"]
        return len(d), int(d.alarmed.sum())

    n_hot, k_hot = _all_fail(hot)
    n_cold, k_cold = _all_fail(cold)
    nh_hot, kh_hot = _counts(hot, "healthy")
    nh_cold, kh_cold = _counts(cold, "healthy")
    print(f"\n[L3] all-failure detection   T=0.9: {k_hot}/{n_hot} = "
          f"{k_hot / max(n_hot, 1):.0%}   (healthy FA {kh_hot / max(nh_hot, 1):.0%})")
    print(f"[L3] all-failure detection   T=0.2: {k_cold}/{n_cold} = "
          f"{k_cold / max(n_cold, 1):.0%}   (healthy FA "
          f"{kh_cold / max(nh_cold, 1):.0%})")

    if n_cold < MIN_N:
        print(f"\n[VERDICT] NO VERDICT - the serving arm produced only "
              f"{n_cold} failures (pre-registered floor {MIN_N}).")
        print(f"\n[L3] wrote {TABLES / 'serving_temperature.csv'}")
        return

    p_arm = _fisher(k_cold, n_cold, k_hot, n_hot, "two-sided")
    print(f"[L3] across-arm Fisher (two-sided) p = {p_arm:.4g}")

    # ---- confound-free per-class check: arithmetic_error ------------------
    na_c, ka_c = _counts(cold, "arithmetic_error")
    na_h, ka_h = _counts(hot, "arithmetic_error")
    print(f"\n[L3] arithmetic_error only   T=0.9: {ka_h}/{na_h}"
          f"   T=0.2: {ka_c}/{na_c}")
    if na_c < MIN_N:
        print(f"[L3] arithmetic_error UNDERPOWERED at 0.2 (n={na_c} < "
              f"{MIN_N}) - no per-class claim.")
        p_cls = None
        cold_beats_own_fa = None
    else:
        p_cls = _fisher(ka_c, na_c, ka_h, na_h, "two-sided")
        cold_beats_own_fa = _fisher(ka_c, na_c, kh_cold, nh_cold, "greater")
        print(f"[L3] arithmetic_error across-arm p = {p_cls:.4g}; "
              f"at 0.2 vs its own healthy FA p = {cold_beats_own_fa:.4g}")

    # ---- pre-declared verdicts -------------------------------------------
    print()
    drop = (k_cold / n_cold) < (k_hot / n_hot)
    if p_arm < 0.05 and drop and p_cls is not None and p_cls < 0.05:
        print("[VERDICT] (B) SUPPORTED - detection is TEMPERATURE-CARRIED: it "
              "falls significantly at the serving temperature and the drop "
              "persists within arithmetic_error alone, so it is not a "
              "failure-mix artefact.")
    elif cold_beats_own_fa is not None and cold_beats_own_fa < 0.05:
        print("[VERDICT] (A) SUPPORTED - detection is REAL at the serving "
              "temperature: arithmetic_error alarms above the cold arm's own "
              "healthy false-alarm rate.")
    elif p_arm < 0.05 and drop:
        print("[VERDICT] PARTIAL - overall detection drops significantly at "
              "the serving temperature, but the per-class check did not "
              "confirm it; the failure mix may account for part of the drop.")
    else:
        print("[VERDICT] INCONCLUSIVE - neither pre-registered criterion met.")

    print(f"\n[L3] wrote {TABLES / 'serving_temperature.csv'}")


if __name__ == "__main__":
    main()
