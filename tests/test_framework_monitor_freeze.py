"""The freeze artifact's integrity hashes must describe the corpus and the
code, not the machine that took the freeze.

Both hashes it stores were originally bit-exact over representations that
differ between platforms, and both rejected an identical checkout on Linux CI:
the source-file hash over CRLF-vs-LF bytes, and the feature-array hash over
float64 bits whose last place libm is free to disagree on.
"""
from __future__ import annotations

import numpy as np

from derail.experiments import framework_monitor_freeze as fz


class _Ep:
    """Minimal stand-in: `_hash_feature_arrays` only reads `.X`."""

    def __init__(self, X):
        self.X = X


def _features(seed: int = 0, rows: int = 40, dims: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # log/log1p are the inexact paths the real features run through, and the
    # reason the raw hash was platform-dependent in the first place.
    return np.log1p(rng.uniform(0.0, 50.0, size=(rows, dims)))


def test_last_place_arithmetic_noise_does_not_change_the_feature_hash():
    """A 1-ULP disagreement is what a different platform's libm produces."""
    X = _features()
    nudged = np.nextafter(X, np.inf)          # every value moved one ULP up
    assert not np.array_equal(X, nudged), "the nudge must actually change bits"
    assert np.max(np.abs(nudged - X)) < 1e-12, "a nudge this size is pure noise"

    assert (fz._hash_feature_arrays([_Ep(X)])
            == fz._hash_feature_arrays([_Ep(nudged)])), (
        "one ULP changed the frozen feature hash -- the check is measuring "
        "the platform's floating point, not the feature construction")


def test_a_real_feature_change_still_changes_the_hash():
    """Tolerance must not blunt what the hash is for."""
    X = _features()
    changed = X.copy()
    changed[3, 2] += 1e-3                     # far above the 1e-6 quantum
    assert fz._hash_feature_arrays([_Ep(X)]) != fz._hash_feature_arrays([_Ep(changed)])

    reordered = [_Ep(X[:20]), _Ep(X[20:])]
    assert (fz._hash_feature_arrays(reordered)
            != fz._hash_feature_arrays(list(reversed(reordered)))), \
        "episode order must still be part of the hash"


def test_negative_zero_hashes_as_zero():
    """`np.round` turns small negatives into -0.0, whose bits differ from
    +0.0 even though the two compare equal -- a value hovering at zero would
    otherwise flip the hash on sign alone."""
    plus = np.zeros((4, 3))
    minus = np.full((4, 3), -0.0)
    assert np.array_equal(plus, minus)
    assert (np.signbit(minus).any() and not np.signbit(plus).any())
    assert fz._hash_feature_arrays([_Ep(plus)]) == fz._hash_feature_arrays([_Ep(minus)])


def test_the_source_file_hash_ignores_line_endings(tmp_path):
    """The other half of the same defect: git hands a Windows checkout CRLF
    for files it stores as LF, and line endings never reach a parsed step."""
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    body = b'{"text": "a"}\n{"text": "b"}\n{"text": "c"}'
    lf.write_bytes(body)
    crlf.write_bytes(body.replace(b"\n", b"\r\n"))
    assert lf.read_bytes() != crlf.read_bytes()
    assert fz._sha256_file(lf) == fz._sha256_file(crlf)

    edited = tmp_path / "edited.jsonl"
    edited.write_bytes(body.replace(b'"c"', b'"z"'))
    assert fz._sha256_file(edited) != fz._sha256_file(lf), \
        "normalising line endings must not blunt content detection"
