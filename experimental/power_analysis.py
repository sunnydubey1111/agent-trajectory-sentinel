"""L7 - Statistical power of the paired detection comparisons at current n.

The real-agent corpora are small (18-171 injected episodes per dataset), so a
significant/non-significant McNemar result must be read against the *power* the
sample actually has. This script quantifies that, deterministically, from the
per-episode detection records the study already writes
(``results/tables/hybrid_diagnosis.csv``): for each dataset and each primary
paired comparison it reports

  - n (injected episodes) and the two detection rates,
  - the observed discordant pairs (b = only-A detects, c = only-B detects) and
    the exact two-sided McNemar p,
  - the ACHIEVED POWER at the current n for the observed discordance, and
  - the MINIMUM DETECTABLE EFFECT (MDE): the smallest marginal detection-rate
    difference that this n could detect at 80% power / alpha 0.05, holding the
    discordance rate at its observed level.

Power and MDE are estimated by Monte-Carlo simulation of the exact McNemar
test, seeded (MASTER_SEED) so the table is bit-reproducible. No new data, no
network, no model calls.

Run:  py -m experimental.power_analysis
Writes results/tables/power_analysis.csv
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from derail.common import MASTER_SEED


def _stable_seed(*parts: object) -> int:
    """Deterministic 31-bit seed from the arguments (Python's built-in hash()
    is per-process randomized, so it cannot be used for reproducible tables)."""
    h = hashlib.sha256("::".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF

TABLES = Path(__file__).resolve().parents[1] / "results" / "tables"
ALPHA = 0.05
TARGET_POWER = 0.80
N_SIM = 4000

# (label, column A, column B) - A is the "advantaged" monitor in the paper's
# claim; b counts episodes only A detects, c only B detects.
COMPARISONS = [
    ("logistic_vs_esn", "det_logistic", "det_esn"),
    ("esn_vs_maha", "det_esn", "det_maha"),
]


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * sps.binom.cdf(min(b, c), n, 0.5)))


def _power(n: int, p_a: float, p_b: float, rng: np.random.Generator,
           n_sim: int = N_SIM) -> float:
    """MC power of the exact McNemar test at alpha, given per-episode
    probabilities of the discordant cells (only-A = p_a, only-B = p_b)."""
    if p_a + p_b <= 0 or n <= 0:
        return 0.0
    # Each episode is only-A / only-B / concordant.
    draws = rng.choice(3, size=(n_sim, n), p=[p_a, p_b, 1.0 - p_a - p_b])
    b = (draws == 0).sum(axis=1)
    c = (draws == 1).sum(axis=1)
    rejects = 0
    for bi, ci in zip(b.tolist(), c.tolist()):
        if _exact_mcnemar_p(bi, ci) < ALPHA:
            rejects += 1
    return rejects / n_sim


def _mde(n: int, pi_d: float, rng: np.random.Generator) -> float:
    """Smallest marginal detection-rate difference delta detectable at
    TARGET_POWER, holding the discordance rate pi_d fixed and splitting it
    asymmetrically as p_a = (pi_d + delta)/2, p_b = (pi_d - delta)/2."""
    if pi_d <= 0:
        return float("nan")
    for delta in np.round(np.arange(0.01, min(pi_d, 1.0) + 1e-9, 0.01), 3):
        p_a = (pi_d + delta) / 2.0
        p_b = (pi_d - delta) / 2.0
        if p_b < 0:
            break
        if _power(n, p_a, p_b, rng) >= TARGET_POWER:
            return float(delta)
    return float("nan")


#: Beyond this, "collect more episodes" stops being an action a person can
#: take on a local model in reasonable time - report it as infeasible instead.
N_FEASIBLE_MAX = 2000


def _n_for_power(p_a: float, p_b: float, rng: np.random.Generator) -> float:
    """Smallest n reaching TARGET_POWER at the OBSERVED discordance rates.

    This is L7b's actual question: not "is this underpowered?" but "how many
    more positives would fix it?" - and, just as importantly, when the honest
    answer is that no feasible number would, because the two monitors are tied
    rather than separated by an effect too small to see.
    """
    if p_a + p_b <= 0:
        return float("nan")
    lo, hi = 10, 100
    while hi <= N_FEASIBLE_MAX:
        if _power(hi, p_a, p_b, rng, n_sim=1500) >= TARGET_POWER:
            break
        lo, hi = hi, hi * 2
    else:
        return float("nan")                     # not reachable feasibly
    while lo < hi:                              # bisect to the smallest n
        mid = (lo + hi) // 2
        if _power(mid, p_a, p_b, rng, n_sim=1500) >= TARGET_POWER:
            hi = mid
        else:
            lo = mid + 1
    return float(lo)


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnosis", default="hybrid_diagnosis",
                    help="table stem under results/tables (default: the "
                         "published hybrid_diagnosis)")
    ap.add_argument("--out", default="power_analysis",
                    help="output table stem under results/tables")
    args = ap.parse_args(argv)
    diag = pd.read_csv(TABLES / f"{args.diagnosis}.csv")
    rows = []
    for dataset, sub in diag.groupby("dataset"):
        n = len(sub)
        for label, col_a, col_b in COMPARISONS:
            a = sub[col_a].astype(int).to_numpy()
            b_ = sub[col_b].astype(int).to_numpy()
            b = int(((a == 1) & (b_ == 0)).sum())   # only A
            c = int(((a == 0) & (b_ == 1)).sum())   # only B
            pi_d = (b + c) / n if n else 0.0
            # Seed per (dataset, comparison) for reproducibility + independence.
            seed = _stable_seed(MASTER_SEED, dataset, label)
            rng = np.random.default_rng(seed)
            power_obs = _power(n, b / n if n else 0.0, c / n if n else 0.0, rng)
            mde = _mde(n, pi_d, np.random.default_rng(seed + 1))
            rows.append({
                "dataset": dataset, "comparison": label, "n": n,
                "det_a": round(float(a.mean()), 3),
                "det_b": round(float(b_.mean()), 3),
                "discordant_b": b, "discordant_c": c,
                "mcnemar_p": round(_exact_mcnemar_p(b, c), 4),
                "achieved_power": round(power_obs, 3),
                "mde_80pct": round(mde, 3) if mde == mde else np.nan,
                # L7b: the collection target. NaN = no feasible n reaches 80%
                # at this discordance, i.e. the monitors are tied here and more
                # episodes would not change the verdict.
                "n_for_80pct": _n_for_power(b / n if n else 0.0,
                                            c / n if n else 0.0,
                                            np.random.default_rng(seed + 2)),
            })
    table = pd.DataFrame(rows)
    out_path = TABLES / f"{args.out}.csv"
    table.to_csv(out_path, index=False)
    print("[power] paired detection comparisons, alpha=0.05, target power=0.80,"
          f" {N_SIM} MC sims/seed={MASTER_SEED}")
    print(table.to_string(index=False))
    print(f"\n[power] wrote {out_path}")
    # Honest one-line reading.
    under = table[(table.achieved_power < TARGET_POWER)]
    print(f"\n[power] {len(under)}/{len(table)} comparisons are UNDERPOWERED "
          f"(<{TARGET_POWER:.0%}) at their current n.")


if __name__ == "__main__":
    main()
