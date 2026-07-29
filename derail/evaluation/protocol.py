"""Deployable evaluation protocol: folds, out-of-fold scoring, multiplicity.

found that several evaluations were not deployable
inference rules dressed up as ones:

  * the *true class* of an episode chose how it was scored - injected episodes
    were scored by one opposite-fold model while healthy episodes were scored
    by averaging two folds, so score variance differed by class;
  * a monitor was fit on a healthy set and then tested on a subset of that same
    set;
  * the best model was selected on the test set and its p-values computed on
    the same test set;
  * families of tests were reported without any multiplicity correction.

This module is the single source of the primitives that make those evaluations
honest.  A deployed monitor cannot know an episode's label, so every rule here
is a function of the episode's *identity* only.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from derail.common import Episode


# --------------------------------------------------------------- fold assignment
def fold_of(episode_id: str, k: int, salt: str = "") -> int:
    """Deterministic fold index for an episode, independent of its label.

    A hash of the episode id (not its class, not its position) decides the
    fold, so the assignment a deployed system could reproduce is exactly the
    one used here.  `salt` varies the assignment across seeds/datasets without
    ever consulting the label.
    """
    digest = hashlib.sha256(f"{salt}:{episode_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % int(k)


def assign_folds(episodes: Sequence[Episode], k: int = 2,
                 salt: str = "") -> dict[str, int]:
    """{episode_id: fold} for every episode, label-independent."""
    if k < 2:
        raise ValueError(f"need at least 2 folds, got {k}")
    return {ep.episode_id: fold_of(ep.episode_id, k, salt) for ep in episodes}


@dataclass
class CrossFitResult:
    """Out-of-fold scores plus the per-fold models (for reporting only)."""

    scores: dict[str, np.ndarray]          # episode_id -> per-step score array
    folds: dict[str, int]                  # episode_id -> fold
    models: list[object]                   # model per fold (fold k omitted its own)
    mean_coef: np.ndarray | None           # mean of per-fold coefficients, if any
    mean_intercept: float | None


def cross_fit_scores(
    test: Sequence[Episode],
    make_model: Callable[[], object],
    fit_unsupervised: Sequence[Episode],
    k: int = 2,
    salt: str = "",
    supervised: bool = True,
) -> CrossFitResult:
    """Score every test episode out-of-fold under ONE rule.

    Each episode is assigned to a fold by its id.  Model *k* is trained without
    fold *k*'s injected episodes and then scores every episode in fold *k* -
    healthy and injected alike, with the same model and the same code path.
    There is no averaging for one class and single-model scoring for another.

    `fit_unsupervised` is the one-class healthy training set (disjoint from
    `test`); `make_model()` must return a fresh model exposing ``fit`` and,
    when ``supervised``, ``fit_supervised(healthy, injected)`` plus
    ``score_episode(ep) -> np.ndarray``.
    """
    folds = assign_folds(test, k, salt)
    injected = [ep for ep in test if not ep.is_healthy]

    scores: dict[str, np.ndarray] = {}
    models: list[object] = []
    coefs: list[np.ndarray] = []
    intercepts: list[float] = []
    for fold in range(k):
        model = make_model()
        model.fit(list(fit_unsupervised))
        if supervised:
            train_injected = [ep for ep in injected
                              if folds[ep.episode_id] != fold]
            if train_injected:
                model.fit_supervised(list(fit_unsupervised), train_injected)
        for ep in test:
            if folds[ep.episode_id] == fold:
                scores[ep.episode_id] = np.asarray(model.score_episode(ep))
        models.append(model)
        if getattr(model, "coef_", None) is not None:
            coefs.append(np.asarray(model.coef_))
            intercepts.append(float(getattr(model, "intercept_", 0.0)))

    mean_coef = np.mean(coefs, axis=0) if coefs else None
    mean_intercept = float(np.mean(intercepts)) if intercepts else None
    return CrossFitResult(scores, folds, models, mean_coef, mean_intercept)


def full_model(make_model: Callable[[], object],
               fit_unsupervised: Sequence[Episode],
               injected: Sequence[Episode], supervised: bool = True):
    """A model trained on ALL injected episodes - for scoring a DISJOINT set
    (e.g. the validation healthy cohort used to pick a threshold).  Never use
    it to score `test`; that would be in-sample."""
    model = make_model()
    model.fit(list(fit_unsupervised))
    if supervised and injected:
        model.fit_supervised(list(fit_unsupervised), list(injected))
    return model


# --------------------------------------------------------------- multiplicity
def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05
                    ) -> dict[str, dict]:
    """Holm-Bonferroni step-down correction over a family of tests.

    Returns, per key, the raw p, the Holm-adjusted p, and the reject decision
    at family-wise level `alpha`.  Controls the family-wise error rate without
    assuming independence.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(items):
        adjusted = min(1.0, (m - rank) * p)
        running_max = max(running_max, adjusted)   # enforce monotonicity
        out[key] = {"p_raw": float(p), "p_holm": float(running_max),
                    "reject": bool(running_max <= alpha)}
    return out


def benjamini_hochberg(pvalues: dict[str, float], alpha: float = 0.05
                       ) -> dict[str, dict]:
    """Benjamini-Hochberg FDR control over a family of tests.

    Use where controlling the false-discovery rate is more appropriate than
    the family-wise error rate (many exploratory comparisons).
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        key, p = items[rank]
        adjusted = min(running_min, p * m / (rank + 1))
        running_min = adjusted
        out[key] = {"p_raw": float(p), "p_bh": float(adjusted),
                    "reject": bool(adjusted <= alpha)}
    return out


# --------------------------------------------------------------- nested selection
@dataclass
class Selection:
    chosen: str
    selection_metric: dict[str, float]     # what selection saw (val/inner)
    basis: str                             # human note on the split used


def select_on_validation(candidate_val_metric: dict[str, float],
                         higher_is_better: bool = True,
                         basis: str = "validation split") -> Selection:
    """Pick the winning candidate on a held-out metric, NOT on the test set.

    The caller then reports the winner's TEST metric and runs inference on the
    test set once, so selection and evaluation never share data.
    """
    if not candidate_val_metric:
        raise ValueError("no candidates to select from")
    chosen = (max if higher_is_better else min)(
        candidate_val_metric, key=candidate_val_metric.get)
    return Selection(chosen, dict(candidate_val_metric), basis)


# --------------------------------------------------------------- smoke test
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    class _ToyModel:
        """A model whose score depends on which injected episodes it saw, so
        the OOF discipline is observable."""

        def __init__(self) -> None:
            self.coef_ = None
            self.intercept_ = 0.0
            self._bias = 0.0

        def fit(self, healthy):
            self._base = np.mean([ep.X.mean() for ep in healthy])

        def fit_supervised(self, healthy, injected):
            self._bias = float(len(injected))
            self.coef_ = np.array([self._bias])
            self.intercept_ = 0.1 * self._bias

        def score_episode(self, ep):
            return ep.X.mean(axis=1) + self._bias

    from derail.common import D_TOTAL

    def _ep(eid, healthy):
        X = rng.normal(size=(6, D_TOTAL))
        return Episode(X=X, episode_id=eid, is_healthy=healthy,
                       failure_class=None if healthy else "looping",
                       tau=None if healthy else 2,
                       t_fail=None if healthy else 5, severity=None if healthy else 0.5)

    healthy_train = [_ep(f"h-train-{i}", True) for i in range(8)]
    test = ([_ep(f"h-{i}", True) for i in range(10)]
            + [_ep(f"inj-{i}", False) for i in range(10)])

    # Fold assignment ignores the label: shuffling healthy<->injected ids
    # must not change a given id's fold.
    folds = assign_folds(test, k=2, salt="s")
    assert folds == assign_folds(list(reversed(test)), k=2, salt="s")

    res = cross_fit_scores(test, _ToyModel, healthy_train, k=2, salt="s")
    assert set(res.scores) == {ep.episode_id for ep in test}
    # Every episode is scored by the model of ITS fold - the same rule for
    # healthy and injected. Check an injected and a healthy episode in the same
    # fold were scored by the same model (identical bias term).
    for fold in range(2):
        ids = [e for e, f in res.folds.items() if f == fold]
        biases = {float(res.scores[i][0] - dict(
            (ep.episode_id, ep) for ep in test)[i].X.mean(axis=1)[0])
            for i in ids}
        assert len(biases) == 1, f"fold {fold} scored episodes inconsistently"

    # Holm and BH behave at the extremes.
    fam = {"a": 0.001, "b": 0.2, "c": 0.04}
    holm = holm_bonferroni(fam)
    assert holm["a"]["reject"] and not holm["b"]["reject"]
    assert holm["a"]["p_holm"] >= holm["a"]["p_raw"]
    bh = benjamini_hochberg(fam)
    assert bh["a"]["reject"]
    # A single test corrects to itself.
    assert abs(holm_bonferroni({"only": 0.03})["only"]["p_holm"] - 0.03) < 1e-12

    sel = select_on_validation({"m1": 0.7, "m2": 0.9, "m3": 0.6})
    assert sel.chosen == "m2"

    print("PASS protocol.py smoke test | label-independent OOF scoring, "
          "Holm/BH multiplicity, nested selection")
