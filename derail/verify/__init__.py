"""Deterministic answer verification — checks, not anomaly detection.

The behavioural monitors ask "is this run unusual?", which needs a healthy
reference distribution and therefore a fresh calibration corpus for every
(model, decoding config, toolset, task, framework) combination. This package
asks "is this answer consistent with what the tools returned?", which needs no
reference distribution at all: no null, no threshold, no calibration, and
nothing to recollect when the deployment changes.

See `derail.verify.checks`.
"""
