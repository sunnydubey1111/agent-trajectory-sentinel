"""Rollback-and-retry: does acting on a detection improve the outcome?

Repair is a shipped capability, not an experiment. `derail.experiments.demo`
calls this module live: a run whose checks fail is rolled back and re-run before
its answer is delivered.

`evaluate_repair_policies` is the measurement harness for it — it replays the same
code over committed traces to compare repair policies and report what each is
worth. The capability and its evaluation are the same code path; only the
harness is offline.

Detection and localization come from `derail.verify.checks` rather than the
behavioural monitor. At the served temperature the two have comparable recall,
but the checks raise no false alarm, need no calibration corpus, and — the part
a rollback depends on — say *what* is wrong. See `derail.intervene.rollback`
and DESIGN.md Module 9.
"""
