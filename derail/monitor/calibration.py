"""Alarm-confidence calibration + calibration-quality metrics (DESIGN.md Module 4).

Episode-level confidence is computed from the per-episode running/final MAX of a
monitor's score stream:

  - ``NullCalibrator``: label-free. Fit on healthy-VAL per-episode max scores;
    confidence(x) = 1 - p-value of x under the healthy-max ECDF (rank-based,
    Hazen plotting position, clipped away from 0/1).
  - ``IsotonicCalibrator``: the oracle upper bound. Isotonic regression from
    max score to P(injected), fit on the labeled CAL split.

``ece`` and ``reliability_curve`` measure calibration quality with standard
equal-width bins on [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from derail.common import rng_for


class NullCalibrator:
    """Label-free confidence: split-conformal upper-tail p-value on healthy val.

    Confidence of a running-max score x is ``1 - p(x)`` where ``p(x)`` is the
    split-conformal upper-tail p-value under the healthy-val max-score null:

        p(x) = (1 + #{healthy reference >= x}) / (n + 1)

    This is TIE-AWARE with an explicit, conservative convention: a reference
    score equal to x counts toward the null tail (``>= x``), so a test score
    sitting AT the healthy level is not treated as an exceedance.  The previous
    Hazen rank used ``searchsorted(side="right")``, which counted ties as
    exceedances - with a degenerate all-zeros null, a test score of 0 got
    confidence 0.996 and would escalate almost every healthy episode; a NaN
    score sorted to the top and got the maximal confidence.

    ``p(x)`` lies in ``[1/(n+1), 1]`` so confidence lies in ``[0, n/(n+1)]`` and
    never reaches exactly 1.  Non-finite inputs are rejected.
    """

    def __init__(self) -> None:
        self.sorted_: np.ndarray | None = None
        self.n_: int = 0

    def fit(self, val_healthy_max_scores: np.ndarray) -> "NullCalibrator":
        """Store the sorted healthy-val per-episode max scores.

        Parameters
        ----------
        val_healthy_max_scores : (n,) array of max_t s_t over healthy val
            episodes. Must be non-empty and finite.
        """
        s = np.asarray(val_healthy_max_scores, dtype=float).ravel()
        # Raised rather than asserted: these run under `python -O` too, where
        # an assert is compiled out and a null quietly built from NaNs would
        # produce a threshold that no score can ever exceed.
        if s.size < 1:
            raise ValueError("NullCalibrator.fit needs at least one score")
        if not np.all(np.isfinite(s)):
            raise FloatingPointError("non-finite healthy max scores")
        self.sorted_ = np.sort(s)
        self.n_ = int(s.size)
        return self

    def confidence(self, running_max_score: float | np.ndarray) -> np.ndarray:
        """1 - split-conformal p-value of the score(s), in ``[0, n/(n+1)]``.

        Accepts a scalar or an array of any shape; returns an ``np.ndarray`` of
        the same shape (0-d for scalar input). Rejects non-finite inputs, which
        would otherwise sort to the top of the reference and receive maximal
        confidence.
        """
        assert self.sorted_ is not None, "NullCalibrator not fitted"
        x = np.asarray(running_max_score, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("NullCalibrator.confidence got a non-finite score")
        # #{reference >= x}: n minus the count strictly less than x. Using
        # side="left" makes ties count toward the null tail (conservative).
        n_ge = self.n_ - np.searchsorted(self.sorted_, x, side="left")
        p = (1.0 + n_ge) / (self.n_ + 1.0)
        return np.asarray(1.0 - p, dtype=float)


class IsotonicCalibrator:
    """Oracle upper bound: isotonic regression fit on the labeled cal split.

    Maps per-episode max scores to P(injected) with sklearn's
    ``IsotonicRegression(increasing=True, out_of_bounds="clip")`` and outputs
    constrained to [0, 1].
    """

    def __init__(self) -> None:
        self.iso_: IsotonicRegression | None = None

    def fit(self, cal_max_scores: np.ndarray, cal_labels: np.ndarray) -> "IsotonicCalibrator":
        """Fit score -> P(injected) on the labeled calibration split.

        Parameters
        ----------
        cal_max_scores : (n,) per-episode max scores on the cal split.
        cal_labels : (n,) binary labels (1 = injected, 0 = healthy).
        """
        s = np.asarray(cal_max_scores, dtype=float).ravel()
        y = np.asarray(cal_labels, dtype=float).ravel()
        assert s.shape == y.shape, "scores/labels length mismatch"
        assert s.size >= 2, "IsotonicCalibrator.fit needs at least two points"
        self.iso_ = IsotonicRegression(
            y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip"
        )
        self.iso_.fit(s, y)
        return self

    def confidence(self, max_scores: float | np.ndarray) -> np.ndarray:
        """Calibrated P(injected) for scalar or array input, as ``np.ndarray``.

        Output has the same shape as the input (0-d for scalar) and lies in
        [0, 1]; scores outside the fitted range are clipped to the boundary
        values (``out_of_bounds="clip"``).
        """
        assert self.iso_ is not None, "IsotonicCalibrator not fitted"
        x = np.asarray(max_scores, dtype=float)
        pred = self.iso_.predict(np.atleast_1d(x).ravel())
        return np.asarray(pred, dtype=float).reshape(x.shape)


def _bin_stats(
    confidences: np.ndarray, labels: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per equal-width bin on [0,1]: (count, mean confidence, empirical freq).

    Bins are [i/n_bins, (i+1)/n_bins); confidence 1.0 falls in the last bin.
    Empty bins have count 0 and NaN means. Vectorized with bincount.
    """
    conf = np.asarray(confidences, dtype=float).ravel()
    lab = np.asarray(labels, dtype=float).ravel()
    assert conf.shape == lab.shape, "confidences/labels length mismatch"
    assert conf.size > 0, "empty confidence array"
    idx = np.clip(np.floor(conf * n_bins).astype(int), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(float)
    sum_conf = np.bincount(idx, weights=conf, minlength=n_bins)
    sum_lab = np.bincount(idx, weights=lab, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.where(count > 0, sum_conf / np.maximum(count, 1.0), np.nan)
        emp_freq = np.where(count > 0, sum_lab / np.maximum(count, 1.0), np.nan)
    return count, mean_conf, emp_freq


def ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1].

    ECE = sum_b (n_b / N) * |empirical_freq_b - mean_confidence_b| over the
    non-empty bins b.
    """
    count, mean_conf, emp_freq = _bin_stats(confidences, labels, n_bins)
    nonempty = count > 0
    weights = count[nonempty] / count.sum()
    return float(np.sum(weights * np.abs(emp_freq[nonempty] - mean_conf[nonempty])))


def reliability_curve(
    confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Reliability diagram data: one row per NON-EMPTY equal-width bin.

    Columns: ``bin_center`` (midpoint of the bin on [0,1]), ``mean_confidence``
    (mean predicted confidence in the bin), ``empirical_freq`` (fraction of
    label-1 episodes in the bin), ``count`` (episodes in the bin).
    """
    count, mean_conf, emp_freq = _bin_stats(confidences, labels, n_bins)
    nonempty = count > 0
    centers = (np.arange(n_bins) + 0.5) / n_bins
    return pd.DataFrame(
        {
            "bin_center": centers[nonempty],
            "mean_confidence": mean_conf[nonempty],
            "empirical_freq": emp_freq[nonempty],
            "count": count[nonempty].astype(int),
        }
    )


if __name__ == "__main__":
    # Smoke test: healthy max scores ~ N(0,1), injected ~ N(2,1).
    rng = rng_for(0, "calibration", "smoke")
    n_val, n_cal_h, n_cal_i, n_test_h, n_test_i = 300, 300, 300, 400, 400

    val_healthy = rng.normal(0.0, 1.0, n_val)
    cal_scores = np.concatenate(
        [rng.normal(0.0, 1.0, n_cal_h), rng.normal(2.0, 1.0, n_cal_i)]
    )
    cal_labels = np.concatenate([np.zeros(n_cal_h), np.ones(n_cal_i)])
    test_scores = np.concatenate(
        [rng.normal(0.0, 1.0, n_test_h), rng.normal(2.0, 1.0, n_test_i)]
    )
    test_labels = np.concatenate([np.zeros(n_test_h), np.ones(n_test_i)])

    # NullCalibrator: array + scalar inputs, ndarray output, conformal range.
    null_cal = NullCalibrator().fit(val_healthy)
    conf_null = null_cal.confidence(test_scores)
    assert isinstance(conf_null, np.ndarray) and conf_null.shape == test_scores.shape
    assert conf_null.min() >= 0.0 and conf_null.max() <= n_val / (n_val + 1.0) + 1e-12
    conf_scalar = null_cal.confidence(1.7)
    assert isinstance(conf_scalar, np.ndarray) and conf_scalar.shape == ()
    assert 0.0 < float(conf_scalar) < 1.0
    # Monotone in the score.
    grid_conf = null_cal.confidence(np.linspace(-3.0, 5.0, 50))
    assert np.all(np.diff(grid_conf) >= 0.0)

    # regression: a DEGENERATE all-zeros null must not escalate a
    # healthy-level score, and must reject NaN instead of ranking it at the top.
    degen = NullCalibrator().fit(np.zeros(120))
    assert float(degen.confidence(0.0)) <= 0.05, "score at the null escalated"
    assert float(degen.confidence(1.0)) >= 0.95, "score above the null not flagged"
    try:
        degen.confidence(np.nan)
        raise AssertionError("NaN score was not rejected")
    except ValueError:
        pass

    # IsotonicCalibrator: fit on labeled cal, scalar + array inputs.
    iso_cal = IsotonicCalibrator().fit(cal_scores, cal_labels)
    conf_iso = iso_cal.confidence(test_scores)
    assert isinstance(conf_iso, np.ndarray) and conf_iso.shape == test_scores.shape
    assert conf_iso.min() >= 0.0 and conf_iso.max() <= 1.0
    assert iso_cal.confidence(2.5).shape == ()
    assert float(iso_cal.confidence(10.0)) >= float(iso_cal.confidence(-10.0))

    ece_null = ece(conf_null, test_labels)
    ece_iso = ece(conf_iso, test_labels)
    assert np.isfinite(ece_null) and 0.0 <= ece_null <= 1.0
    assert ece_iso < ece_null, f"isotonic ECE {ece_iso:.3f} !< null ECE {ece_null:.3f}"

    # Reliability curve on the oracle-calibrated confidences: well-formed and
    # monotone-ish. Bins with very few episodes carry pure sampling noise
    # (a count-1 bin has empirical_freq in {0, 1}), so the monotonicity check
    # only considers bins with a meaningful population.
    curve = reliability_curve(conf_iso, test_labels)
    assert list(curve.columns) == ["bin_center", "mean_confidence", "empirical_freq", "count"]
    assert int(curve["count"].sum()) == test_scores.size
    solid = curve[curve["count"] >= 20]
    freq = solid["empirical_freq"].to_numpy()
    assert np.all(np.diff(freq) > -0.1), "reliability curve not monotone-ish"
    assert freq[-1] - freq[0] > 0.5

    print(
        "PASS calibration: "
        f"ECE(null)={ece_null:.3f}, ECE(isotonic)={ece_iso:.3f}, "
        f"reliability bins={len(curve)}, freq range "
        f"[{freq[0]:.2f}, {freq[-1]:.2f}]"
    )
