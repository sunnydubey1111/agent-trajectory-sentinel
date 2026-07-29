"""Paired significance tests for monitor comparisons (review weakness C).

Bootstrap CIs (metrics.bootstrap_ci) quantify uncertainty of a single
monitor's statistic; the tests here compare TWO monitors on the SAME test
episodes, which is the question reviewers actually ask ("is the primary
better than this baseline, or is it seed noise?"):

  - paired_permutation_test: two-sided sign-flip permutation test on the
    mean of per-episode paired differences (used on lead_all values, where
    a missed episode contributes 0 — the survivorship-free convention).
  - wilcoxon_signed_rank: the classical rank-based paired test on the same
    differences (zero_method="pratt" — ties at zero are common here).
  - mcnemar_test: exact two-sided McNemar test on paired binary detection
    outcomes (which episodes each monitor detected).

Both are distribution-free and respect the pairing by episode, so
between-episode difficulty variation (severity, class, length) cancels.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps

from derail.common import rng_for


def paired_permutation_test(a: np.ndarray, b: np.ndarray,
                            n_perm: int = 20000, seed: int = 0) -> dict:
    """Two-sided sign-flip permutation test on mean(a - b), paired.

    a, b: per-episode values for the two monitors over the SAME episodes in
    the same order. Under H0 (no difference) each paired difference is
    symmetric around 0, so its sign is exchangeable; the null distribution
    is the mean under random sign flips. Returns {"mean_diff", "p_value"}.
    The p-value uses the add-one (permutation-inclusive) estimator, so its
    floor is 1/(n_perm + 1) — it is never exactly 0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1 and a.size > 0
    d = a - b
    obs = float(d.mean())
    rng = rng_for(seed, "paired-perm")
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, d.size))
    null = (signs * d).mean(axis=1)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return {"mean_diff": obs, "p_value": p}


def wilcoxon_signed_rank(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired values.

    Uses zero_method="pratt" because ties at zero difference are common here
    (episodes missed by BOTH monitors contribute lead_all 0 - 0 = 0).
    Degenerate case (all differences zero) returns p = 1.0.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1
    d = a - b
    if not np.any(d != 0.0):
        return {"statistic": float("nan"), "p_value": 1.0}
    res = sps.wilcoxon(a, b, zero_method="pratt", alternative="two-sided")
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


def mcnemar_test(detected_a: np.ndarray, detected_b: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired binary outcomes.

    detected_a/b: boolean per-episode detection flags over the same
    episodes. Only discordant pairs are informative: n10 = a-only,
    n01 = b-only; under H0 each discordant pair is a-only with p=0.5.
    Returns {"n10", "n01", "p_value"} (p=1.0 when there are no discordant
    pairs).
    """
    a = np.asarray(detected_a, dtype=bool)
    b = np.asarray(detected_b, dtype=bool)
    assert a.shape == b.shape and a.ndim == 1
    n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b))
    n = n10 + n01
    p = 1.0 if n == 0 else float(sps.binomtest(n10, n, 0.5).pvalue)
    return {"n10": n10, "n01": n01, "p_value": p}


if __name__ == "__main__":
    rng = rng_for(0, "stats-smoke")

    # Permutation test: shifted pairs -> significant; identical -> not.
    base = rng.normal(size=400)
    shifted = base + 0.3 + 0.1 * rng.normal(size=400)
    r = paired_permutation_test(shifted, base, seed=1)
    assert r["mean_diff"] > 0.25 and r["p_value"] < 0.001, r
    same = base + 0.001 * rng.normal(size=400)
    r0 = paired_permutation_test(same, base, seed=2)
    assert r0["p_value"] > 0.05, r0
    r1 = paired_permutation_test(shifted, base, seed=1)
    assert r1 == r, "not deterministic"

    # Wilcoxon: shifted pairs -> significant; identical -> degenerate p=1.
    w = wilcoxon_signed_rank(shifted, base)
    assert w["p_value"] < 0.001, w
    assert wilcoxon_signed_rank(base, base)["p_value"] == 1.0

    # McNemar: strongly one-sided discordance -> significant.
    det_a = np.concatenate([np.ones(60, bool), np.zeros(40, bool)])
    det_b = np.concatenate([np.ones(30, bool), np.zeros(70, bool)])
    m = mcnemar_test(det_a, det_b)
    assert m["n10"] == 30 and m["n01"] == 0 and m["p_value"] < 1e-6, m
    m0 = mcnemar_test(det_a, det_a)
    assert m0["p_value"] == 1.0 and m0["n10"] == m0["n01"] == 0

    print(f"PASS stats smoke test | perm p={r['p_value']:.5f} (shifted), "
          f"{r0['p_value']:.3f} (null) | mcnemar p={m['p_value']:.2e}")
