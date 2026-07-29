"""Does detection actually IMPROVE the agent? Paired, live, end to end.

Every failed episode of the serving-temperature arm is rolled back to the same
checkpoint and re-run under each repair rung, so the arms are paired on the
identical prefix and the identical task. Only the repair differs.

  none      the committed run, untouched (the status quo)
  resample  rollback + fresh sampling, identical context  <- the control
  generic   rollback + a task-independent "re-check your work"
  specific  rollback + the check's own finding, in words  <- uses localization
  adaptive  specific when the answer is wrong, generic when only completeness is

Outcome is graded by the objective labeller against the task's ground truth,
which the intervention never sees: the repair prompt is a function of the
checks alone, and the checks read only what the agent observed. A run counts as
correct only if it states the right total AND performs the work the task asks
for, so fixing the sum by abandoning a required lookup is not scored as a win.

Also reported, because a detector that fires on already-correct runs costs more
than it saves: interventions triggered on runs that needed none.

Run (needs Ollama + qwen2.5:7b; ~20 min at --parallel 4):
  py -m derail.intervene.evaluate_repair_policies --parallel 4
Writes results/tables/repair_policies.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats as sps

from derail.intervene.rollback import RUNGS, retry_from_checkpoint
from derail.verify.checks import BOOKING_SPEC, stated_total, verify

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
CORPUS = ROOT / "traces" / "organic_demo7b_cold"      # the SERVING arm


def _correct_via_checks_parser(steps: list[dict], expected: float) -> bool:
    """Grade with the CHECKS' parser. Kept only as a cross-check."""
    said = None
    for s in reversed(steps):
        said = stated_total(str(s.get("text", "")))
        if said is not None:
            break
    return said is not None and abs(said - expected) < 0.5


def _correct(steps: list[dict], expected: float, seed: int) -> bool:
    """Oracle grading, via the objective labeller rather than the checks' parser.

    Sharing one parser between "was the answer right?" and "is the run
    self-consistent?" would let a single parsing bug move both together and
    stay invisible, so the grade uses the independent labeller. The
    checks-parser verdict is recorded alongside as `now_correct_checks_parser`
    and any disagreement is reported.
    """
    import verification.organic_hallucination as oh
    label, _ = oh.label(steps, expected,
                        required_weather=oh.required_weather_for(seed))
    return label == "healthy"


def _labels() -> list[dict]:
    os.environ["AGENTWATCH_ORGANIC_DIR"] = str(CORPUS)
    import importlib

    import verification.organic_hallucination as oh
    importlib.reload(oh)
    return oh.label_all()


def _one(job: tuple) -> dict:
    entry, rung, model, temperature, rep = job
    steps = [json.loads(x) for x in
             (CORPUS / entry["file"]).read_text("utf-8").splitlines() if x]
    att = retry_from_checkpoint(steps, int(entry["seed"]), rung,
                                model=model, temperature=temperature)
    exp = float(entry["expected_total"])
    # Persist the retried trace so the grade can be re-derived independently
    # later; a study that keeps only booleans cannot be re-audited.
    if rung != "none":
        out_dir = CORPUS.parent / f"{CORPUS.name}_retry" / f"{rung}-r{rep}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{entry['episode_id']}.jsonl").write_text(
            "\n".join(json.dumps(s) for s in att.steps), "utf-8")
    return {"episode_id": entry["episode_id"], "rung": rung, "rep": rep,
            "label": entry["label"],
            "was_correct": _correct(steps, exp, int(entry["seed"])),
            "now_correct": _correct(att.steps, exp, int(entry["seed"])),
            "now_correct_checks_parser": _correct_via_checks_parser(
                att.steps, exp),
            "resumed_from": att.resumed_from,
            "steps_before": len(steps), "steps_after": len(att.steps),
            "model_calls": att.model_calls,
            "checks_flag_before": bool(att.findings_before),
            "checks_flag_after": bool(att.findings_after)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="py -m derail.intervene.evaluate_repair_policies")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of episodes (smoke runs)")
    ap.add_argument("--from-csv", action="store_true",
                    help="re-analyse the committed table without re-running "
                         "the agent (no Ollama needed)")
    ap.add_argument("--resume", action="store_true",
                    help="keep (episode, rung) rows already in the table and "
                         "run only what is missing")
    ap.add_argument("--regrade", action="store_true",
                    help="with --from-csv, recompute every outcome from the "
                         "persisted retry traces (use after a grader change)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent retries per (episode, rung); >1 gives a "
                         "variance estimate for the recovery rate")
    ap.add_argument("--corpus", default=None,
                    help="trace dir to study (default: the serving arm)")
    args = ap.parse_args(argv)
    if args.corpus:
        globals()["CORPUS"] = ROOT / "traces" / args.corpus

    labelled = _labels()
    by_id = {r["episode_id"]: r for r in labelled}

    # The intervention fires on what the CHECKS flag, never on the label.
    flagged = []
    for r in labelled:
        steps = [json.loads(x) for x in
                 (CORPUS / r["file"]).read_text("utf-8").splitlines() if x]
        if verify(steps, BOOKING_SPEC).failed:
            flagged.append(r)
    if args.limit:
        flagged = flagged[:args.limit]
    n_healthy = sum(1 for r in flagged if r["label"] == "healthy")
    print(f"[intervene] checks flagged {len(flagged)} of {len(labelled)} "
          f"episodes ({n_healthy} of them label-healthy -> candidate "
          f"unnecessary interventions)")

    jobs = [(e, rung, args.model, args.temperature, rep)
            for e in flagged for rung in RUNGS
            for rep in range(1, args.repeats + 1)]
    rows = []
    # Resumable: rows already committed for an (episode, rung) pair are kept as
    # they are, so adding a rung costs only that rung's runs and never silently
    # re-rolls a result that has been reported.
    csv_path = TABLES / "repair_policies.csv"
    if args.resume and csv_path.exists() and not args.from_csv:
        prev = pd.read_csv(csv_path)
        have = {(r.episode_id, r.rung, getattr(r, "rep", 1))
                for r in prev.itertuples()}
        rows = prev.to_dict("records")
        jobs = [j for j in jobs
                if (j[0]["episode_id"], j[1], j[4]) not in have]
        print(f"[intervene] resuming: {len(rows)} rows kept, "
              f"{len(jobs)} to run")
    if args.from_csv:
        df = pd.read_csv(TABLES / "repair_policies.csv")
        if args.regrade:
            # Re-derive every outcome from the persisted retry traces. Needed
            # when the grader changes after a run: the stored booleans are
            # then stale, and re-running the agent would answer a different
            # question (fresh samples) rather than the same one.
            by_id = {r["episode_id"]: r for r in labelled}
            n = 0
            for row in df.itertuples():
                ent = by_id.get(row.episode_id)
                if ent is None:
                    continue
                exp, seed = float(ent["expected_total"]), int(ent["seed"])
                if row.rung == "none":
                    steps = [json.loads(x) for x in
                             (CORPUS / ent["file"]).read_text("utf-8")
                             .splitlines() if x]
                else:
                    rep = getattr(row, "rep", 1)
                    f = (CORPUS.parent / f"{CORPUS.name}_retry" /
                         f"{row.rung}-r{rep}" / f"{row.episode_id}.jsonl")
                    if not f.exists():
                        continue
                    steps = [json.loads(x) for x in
                             f.read_text("utf-8").splitlines() if x]
                df.at[row.Index, "was_correct"] = _correct(
                    [json.loads(x) for x in (CORPUS / ent["file"])
                     .read_text("utf-8").splitlines() if x], exp, seed)
                df.at[row.Index, "now_correct"] = _correct(steps, exp, seed)
                df.at[row.Index, "now_correct_checks_parser"] =                     _correct_via_checks_parser(steps, exp)
                n += 1
            print(f"[intervene] regraded {n}/{len(df)} rows from persisted "
                  f"traces (no agent re-run)")
        else:
            print(f"[intervene] re-analysing {len(df)} committed rows "
                  f"(no agent re-run)")
        rows = df.to_dict("records")
    elif args.parallel > 1:
        with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for i, res in enumerate(pool.map(_one, jobs), 1):
                rows.append(res)
                if i % 20 == 0:
                    print(f"  [{i}/{len(jobs)}]", flush=True)
    else:
        for i, j in enumerate(jobs, 1):
            rows.append(_one(j))
            if i % 20 == 0:
                print(f"  [{i}/{len(jobs)}]", flush=True)

    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "repair_policies.csv", index=False)

    if "now_correct_checks_parser" in df.columns:
        # Rows carried over from an earlier run predate this column; compare
        # only where both verdicts exist.
        both = df[df.now_correct_checks_parser.notna()]
        disagree = both[both.now_correct != both.now_correct_checks_parser]
        # The two graders answer different questions: the parser asks whether
        # the stated total is right, the labeller whether the task was done.
        # Disagreement on `incomplete` runs is therefore expected; anywhere
        # else it would indicate a parsing defect.
        by_label = disagree.groupby("label").size().to_dict()
        unexpected = {k: v for k, v in by_label.items() if k != "incomplete"}
        print(f"[intervene] grader cross-check: {len(disagree)}/{len(both)} "
              f"rows disagree, by label {by_label or '{}'}"
              + ("" if not unexpected else
                 f"  <- UNEXPECTED outside `incomplete`: {unexpected}"))

    # ---- outcome, on the episodes that were genuinely wrong ---------------
    # With repeats there are R rows per (episode, rung). Every quantity below
    # is computed WITHIN a repeat and then summarised across repeats, so a
    # single episode never counts as R independent observations.
    wrong = df[~df.was_correct]
    reps = sorted(wrong.rep.unique()) if "rep" in wrong.columns else [1]
    if "rep" not in wrong.columns:
        wrong = wrong.assign(rep=1)
    n_eps = wrong[wrong.rep == reps[0]].episode_id.nunique()
    print()
    print(f"[intervene] task success after intervention "
          f"(n={n_eps} genuinely-wrong episodes, paired"
          + (f", {len(reps)} repeats)" if len(reps) > 1 else ")"))
    print()
    print(f"{'rung':<10}{'correct':>9}{'rate':>8}{'vs none':>10}"
          f"{'vs resample':>13}{'extra calls':>13}")

    def _paired_p(rung: str, ref: str) -> float:
        """Median exact McNemar p across repeats (each repeat is one pairing)."""
        ps = []
        for rep in reps:
            a = wrong[(wrong.rung == rung) & (wrong.rep == rep)]                 .set_index("episode_id")
            b = wrong[(wrong.rung == ref) & (wrong.rep == rep)]                 .set_index("episode_id")
            ids = a.index.intersection(b.index)
            disc_a = int(sum(bool(a.loc[i, "now_correct"])
                             and not bool(b.loc[i, "now_correct"]) for i in ids))
            disc_b = int(sum(bool(b.loc[i, "now_correct"])
                             and not bool(a.loc[i, "now_correct"]) for i in ids))
            n_disc = disc_a + disc_b
            ps.append(sps.binomtest(disc_a, n_disc, 0.5).pvalue
                      if n_disc else 1.0)
        return float(np.median(ps))

    for rung in RUNGS:
        sub = wrong[wrong.rung == rung]
        if sub.empty:
            continue
        per_rep = [g.now_correct.mean() for _, g in sub.groupby("rep")]
        rate = float(np.mean(per_rep))
        line = f"{rung:<10}{rate * n_eps:>9.0f}{rate:>8.0%}"
        for ref in ("none", "resample"):
            w = 10 if ref == "none" else 13
            if rung == ref:
                line += f"{'':>{w}}"
            else:
                line += f"{_paired_p(rung, ref):>{w}.4f}"
        line += f"{sub.model_calls.mean():>13.1f}"
        print(line)
    print("  (paired exact McNemar within each repeat, median across repeats; "
          "'vs none' = beats doing nothing, 'vs resample' = beats retry luck)")

    # ---- cost of firing on runs that were already correct -----------------
    ok = df[df.was_correct]
    if len(ok):
        print(f"\n[intervene] unnecessary interventions: "
              f"{len(ok[ok.rung == 'none'])} episodes were already correct")
        for rung in RUNGS:
            s = ok[ok.rung == rung]
            if not len(s):
                continue
            broke = int((~s.now_correct).sum())
            print(f"  {rung:<10} still correct "
                  f"{int(s.now_correct.sum())}/{len(s)}  (broken: {broke})")
    else:
        print("\n[intervene] unnecessary interventions: none — the checks "
              "flagged no already-correct episode")

    # ---- net effect on the whole corpus ----------------------------------
    # What a deployment actually gets: recovered failures minus correct runs
    # the intervention broke, over every episode including the ones the checks
    # never flagged (which are unchanged by construction).
    n_total = len(labelled)
    n_correct_base = sum(
        _correct([json.loads(x) for x in
                  (CORPUS / r["file"]).read_text("utf-8").splitlines() if x],
                 float(r["expected_total"]), int(r["seed"]))
        for r in labelled)
    print(f"\n[intervene] NET task success over all {n_total} episodes "
          f"(unflagged episodes are untouched)")
    print(f"{'policy':<10}{'correct':>9}{'rate':>8}{'recovered':>11}"
          f"{'broken':>8}")
    print(f"{'none':<10}{n_correct_base:>9}"
          f"{n_correct_base / n_total:>8.0%}{'':>11}{'':>8}")
    reps_all = sorted(df.rep.unique()) if "rep" in df.columns else [1]
    for rung in RUNGS:
        if rung == "none":
            continue
        sel = df[df.rung == rung]
        if sel.empty:
            continue
        # Counted WITHIN each repeat and averaged: summing across repeats would
        # credit one episode several times and can push the rate past 100%.
        rec = float(np.mean([
            (g[~g.was_correct].now_correct.sum())
            for _, g in sel.groupby("rep")])) if len(reps_all) else 0.0
        brk = float(np.mean([
            ((~g[g.was_correct].now_correct).sum())
            for _, g in sel.groupby("rep")])) if len(reps_all) else 0.0
        net = n_correct_base + rec - brk
        print(f"{rung:<10}{net:>9.0f}{net / n_total:>8.0%}"
              f"{rec:>11.1f}{brk:>8.1f}")

    # ---- what the repair costs ------------------------------------------
    # Extra model calls come from the study rows; per-step latency is measured
    # from the retried traces themselves, so the wall-clock figure is observed
    # rather than assumed.
    lat: dict[str, list[float]] = {}
    retry_root = CORPUS.parent / f"{CORPUS.name}_retry"
    if retry_root.exists():
        for sub in retry_root.iterdir():
            if not sub.is_dir():
                continue
            rung = sub.name.split("-r")[0]
            for f in sub.glob("*.jsonl"):
                for line in f.read_text("utf-8").splitlines():
                    if not line.strip():
                        continue
                    v = json.loads(line).get("latency_s")
                    if v is not None:
                        lat.setdefault(rung, []).append(float(v))
    print()
    print(f"[intervene] cost of intervening "
          f"(fires on {len(flagged)}/{n_total} runs)")
    print(f"{'rung':<10}{'calls':>7}{'s/step':>8}{'added s':>9}"
          f"{'calls/recovery':>16}")
    for rung in RUNGS:
        if rung == "none":
            continue
        sub = wrong[wrong.rung == rung]
        if sub.empty:
            continue
        calls = float(sub.model_calls.mean())
        med = float(np.median(lat[rung])) if lat.get(rung) else float("nan")
        per_rep = [g.now_correct.mean() for _, g in sub.groupby("rep")]
        rate = float(np.mean(per_rep))
        per_rec = calls / rate if rate else float("nan")
        print(f"{rung:<10}{calls:>7.2f}{med:>8.2f}{calls * med:>9.1f}"
              f"{per_rec:>16.1f}")
    print("  (added s = extra model calls x measured median step latency; "
          "calls/recovery = what one recovered failure costs)")

    # A single measurement of a stochastic retry is not a result; with
    # repeats the recovery rate is reported with its spread across
    # independent repeats.
    if "rep" in df.columns and df.rep.nunique() > 1:
        print()
        print(f"[intervene] recovery rate across {df.rep.nunique()} "
              f"independent repeats (genuinely-wrong episodes)")
        print(f"{'rung':<10}{'mean':>8}{'min':>8}{'max':>8}{'spread':>9}")
        w = df[~df.was_correct]
        for rung in RUNGS:
            per = [g.now_correct.mean()
                   for _, g in w[w.rung == rung].groupby("rep")]
            if len(per) > 1:
                print(f"{rung:<10}{sum(per) / len(per):>8.0%}"
                      f"{min(per):>8.0%}{max(per):>8.0%}"
                      f"{max(per) - min(per):>9.1%}")

    print(f"\n[intervene] wrote {TABLES / 'repair_policies.csv'}")


if __name__ == "__main__":
    main()
