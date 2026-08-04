<!-- Thanks for contributing. Delete any section that does not apply. -->

## What this changes

<!-- One or two sentences. Link the issue it closes, if there is one. -->

## Why

<!-- The problem or the result that motivated it. -->

## How it was checked

<!-- Paste the command you ran and what it printed. -->

- [ ] `python -m pytest -m "not slow and not network and not ollama"` passes
- [ ] Slow gate run, if this touches detection, verification, repair or `results/`
- [ ] New behaviour has a test

## Does this move a published number?

- [ ] No — no result, table or figure changes.
- [ ] Yes — the before/after is below, and the derived documents were
      regenerated in this PR (`claims_ledger`, `data_card`, `artifact_manifest`,
      `behavior_snapshot`, each with `--check` passing).

<!-- If yes: which number, from what to what, and the command that produced it. -->
