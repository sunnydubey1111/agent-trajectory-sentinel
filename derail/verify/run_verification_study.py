"""Do deterministic checks catch what the behavioural monitor cannot?

Head-to-head on the SAME episodes, the SAME objective labels and the SAME
arms as the serving-temperature study, so the two approaches are directly
comparable:

  monitor  = shipped grounded content gate, nested out-of-fold theta at the
             served 10% FA budget (numbers imported from the tables
             `score_organic_halluc` writes)
  checks   = derail.verify.checks, no null and no threshold at all

Both arms are real qwen2.5:7b runs with no injection. Labels come from the
objective labeller and never from either detector.

The checks were written by inspecting failures in the serving arm, so that arm
cannot also be their test set; `--holdout` scores them, frozen, on a corpus
collected afterwards at disjoint task seeds.

Run:  py -m derail.verify.run_verification_study
      py -m derail.verify.run_verification_study --holdout organic_demo7b_holdout
Writes results/tables/verification_vs_monitor.csv
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from scipy import stats as sps

from derail.verify.checks import BOOKING_SPEC, verify

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"
ARMS = {
    "T=0.9 (provoking)": ("organic_demo7b_ext",
                          "organic_hallucination_ext.csv"),
    "T=0.2 (serving)": ("organic_demo7b_cold",
                        "organic_hallucination_cold.csv"),
}
LABELS = ("healthy", "arithmetic_error", "hallucinated", "other",
          "incomplete")


def _labels_for(corpus: str) -> list[dict]:
    """Objective labels for one arm, via the frozen labeller."""
    os.environ["AGENTWATCH_ORGANIC_DIR"] = str(ROOT / "traces" / corpus)
    import importlib

    import verification.organic_hallucination as oh
    importlib.reload(oh)
    return oh.label_all()


def _fisher(a_hit: int, a_n: int, b_hit: int, b_n: int) -> float:
    return float(sps.fisher_exact([[a_hit, a_n - a_hit],
                                   [b_hit, b_n - b_hit]],
                                  alternative="greater")[1])


def holdout(corpus: str) -> None:
    """Score the frozen checks, unchanged, on a corpus they were never
    designed against (see module docstring for why this arm is needed)."""
    labelled = _labels_for(corpus)
    src = ROOT / "traces" / corpus
    by_label: dict[str, list[tuple[bool, bool]]] = {}
    rows = []
    for r in labelled:
        steps = [json.loads(x) for x in
                 (src / r["file"]).read_text("utf-8").splitlines() if x]
        res = verify(steps, BOOKING_SPEC)
        money = any(f.check == "total_consistency" for f in res.findings)
        cover = any(f.check == "required_coverage" for f in res.findings)
        by_label.setdefault(r["label"], []).append((money, cover))
        # Every corpus numbers its own episodes from zero, so `episode_id` is
        # unique only within one of them. `dataset` is what makes a row
        # traceable to the corpus it was measured on, and what makes the key
        # (dataset, episode_id) unique across the whole results tree.
        rows.append({"dataset": corpus,
                     "episode_id": r["episode_id"], "label": r["label"],
                     "total_consistency": money, "required_coverage": cover})
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / f"verification_{corpus.replace('organic_demo7b_', '')}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"\n[holdout] {corpus}: {len(labelled)} episodes, checks frozen\n")
    print(f"{'label':<18}{'n':>4}{'total_consistency':>19}{'+coverage':>11}")
    for lab in LABELS:
        v = by_label.get(lab, [])
        if not v:
            continue
        m = sum(1 for a, _ in v if a)
        b = sum(1 for a, c in v if a or c)
        print(f"{lab:<18}{len(v):>4}{m:>13} = {m / len(v):>3.0%}"
              f"{b:>6} = {b / len(v):>3.0%}")

    h = by_label.get("healthy", [])
    fails = [x for lab, v in by_label.items() if lab != "healthy" for x in v]
    if h and fails:
        fa = sum(1 for a, _ in h if a)
        det = sum(1 for a, _ in fails if a)
        print(f"\n[holdout] failures caught {det}/{len(fails)} = "
              f"{det / len(fails):.0%} at {fa}/{len(h)} = {fa / len(h):.0%} "
              f"false positives (totals question)")
        cov_h = sum(1 for _, c in h if c)
        print(f"[holdout] coverage findings on label-healthy runs: "
              f"{cov_h}/{len(h)} — required work skipped on a run whose total "
              f"was still right; the labeller grades only the total")
        both = sum(1 for a, c in fails if a or c)
        print(f"[holdout] with coverage: {both}/{len(fails)} = "
              f"{both / len(fails):.0%}")
    print(f"[holdout] wrote {out}")


def main() -> None:
    rows = []
    for arm, (corpus, monitor_csv) in ARMS.items():
        labelled = _labels_for(corpus)
        src = ROOT / "traces" / corpus
        # `total_consistency` is the SCOPE-MATCHED check: the objective label
        # grades only the stated total, and so does it. `required_coverage`
        # answers a question the label does not ask (did the agent do all the
        # work?), so its hits on label-healthy runs are ungraded findings, not
        # false positives — they are reported separately, never netted in.
        money: dict[str, bool] = {}
        cover: dict[str, bool] = {}
        unchecked = 0
        for r in labelled:
            steps = [json.loads(x) for x in
                     (src / r["file"]).read_text("utf-8").splitlines() if x]
            res = verify(steps, BOOKING_SPEC)
            money[r["episode_id"]] = any(f.check == "total_consistency"
                                         for f in res.findings)
            cover[r["episode_id"]] = any(f.check == "required_coverage"
                                         for f in res.findings)
            unchecked += not res.checked
        # A totals check that observed no price is not a pass. Reported so the
        # false-positive rate below cannot be inflated by episodes the check
        # never actually ran on.
        print(f"[verify] {arm}: totals check ran on "
              f"{len(labelled) - unchecked}/{len(labelled)} episodes"
              + (f" — {unchecked} had no priced result to reconcile"
                 if unchecked else ""))

        mon = pd.read_csv(TABLES / monitor_csv).set_index("episode_id")
        by_label: dict[str, list[str]] = {}
        for r in labelled:
            by_label.setdefault(r["label"], []).append(r["episode_id"])

        h_ids = by_label.get("healthy", [])
        chk_fa = sum(money[i] for i in h_ids)
        mon_fa = int(mon.loc[h_ids, "alarmed"].sum())
        for lab in LABELS:
            ids = by_label.get(lab, [])
            if not ids:
                continue
            c_hit = sum(money[i] for i in ids)
            both = sum(money[i] or cover[i] for i in ids)
            m_hit = int(mon.loc[ids, "alarmed"].sum())
            rows.append({
                "dataset": corpus,
                "arm": arm, "label": lab, "n": len(ids),
                "total_consistency": c_hit,
                "total_consistency_rate": round(c_hit / len(ids), 3),
                "with_coverage": both,
                "with_coverage_rate": round(both / len(ids), 3),
                "monitor_alarmed": m_hit,
                "monitor_rate": round(m_hit / len(ids), 3),
                "checks_p_vs_healthy": ("" if lab == "healthy" else
                                        round(_fisher(c_hit, len(ids),
                                                      chk_fa, len(h_ids)), 6)),
                "monitor_p_vs_healthy": ("" if lab == "healthy" else
                                         round(_fisher(m_hit, len(ids),
                                                       mon_fa, len(h_ids)), 6)),
            })
        print(f"  [{arm}] coverage findings on label-HEALTHY runs: "
              f"{sum(cover[i] for i in h_ids)}/{len(h_ids)} — required work "
              f"skipped on a run whose total was still right (the label grades "
              f"only the total, so these are ungraded, not false alarms)")

    table = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES / "verification_vs_monitor.csv", index=False)

    print("\n[verify] deterministic checks vs the behavioural monitor "
          "(same episodes, same labels)\n")
    print(table.to_string(index=False))

    print("\n[verify] headline — scope-matched: total_consistency vs the "
          "monitor, on the question the labels actually grade")
    for arm in ARMS:
        a = table[table.arm == arm]
        h = a[a.label == "healthy"]
        f = a[a.label != "healthy"]
        if h.empty or f.empty:
            continue
        c_hit = int(f.total_consistency.sum())
        b_hit = int(f.with_coverage.sum())
        m_hit = int(f.monitor_alarmed.sum())
        n_f = int(f.n.sum())
        print(f"  {arm}")
        print(f"     checks  {c_hit}/{n_f} = {c_hit / n_f:.0%} of failures at "
              f"{int(h.total_consistency.iloc[0])}/{int(h.n.iloc[0])} = "
              f"{h.total_consistency_rate.iloc[0]:.0%} false positives "
              f"(adding coverage: {b_hit}/{n_f} = {b_hit / n_f:.0%})")
        print(f"     monitor {m_hit}/{n_f} = {m_hit / n_f:.0%} at "
              f"{h.monitor_rate.iloc[0]:.0%} false alarms")
    print(f"\n[verify] wrote {TABLES / 'verification_vs_monitor.csv'}")


def contract_coverage() -> None:
    """Score `tool_contract` on every injected corpus in the repository.

    The check ships without a null, so the question that decides whether it is
    usable is not how much it catches but whether a healthy run can trip it.
    This scores every labelled corpus at once — the injected classes it is not
    aimed at are the control, and they should stay near zero as firmly as the
    healthy runs do.

    Two scope rules, both load-bearing for the false-positive claim:

    * Corpora whose directory starts with `_` are skipped. Those are imported
      from other projects, so counting their episodes would restate someone
      else's runs as evidence about this check.
    * The per-label DENOMINATORS are written to
      `tool_contract_denominators.csv`, not just printed. The headline here is
      a false-positive rate, and a rate whose denominator exists only in a
      runner's stdout cannot be checked: the published "0 of 1,825 healthy"
      was copied from one such run and stayed in five documents while the
      corpus grew underneath it.
    """
    from derail.verify.checks import tool_contract

    per_label: dict[str, list[int]] = {}
    rows = []
    for directory in sorted((ROOT / "traces").iterdir()):
        manifest = directory / "manifest.json"
        if not directory.is_dir() or not manifest.exists():
            continue
        if directory.name.startswith("_"):
            continue
        for entry in json.loads(manifest.read_text("utf-8")):
            path = directory / entry.get("file", f"{entry['episode_id']}.jsonl")
            if not path.exists():
                continue
            steps = [json.loads(x) for x in
                     path.read_text("utf-8").splitlines() if x.strip()]
            findings = tool_contract(steps, BOOKING_SPEC)
            label = entry.get("failure_class") or "healthy"
            per_label.setdefault(label, []).append(bool(findings))
            if findings:
                onset = entry.get("tau")
                rows.append({"dataset": directory.name,
                             "episode_id": entry["episode_id"],
                             "label": label, "tau": onset,
                             "first_violation_step": findings[0].step,
                             "detail": findings[0].terse})
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "tool_contract_coverage.csv",
                              index=False)
    pd.DataFrame([{"label": label, "flagged": int(sum(v)), "n": len(v),
                   "rate": round(sum(v) / len(v), 4)}
                  for label, v in sorted(per_label.items())]
                 ).to_csv(TABLES / "tool_contract_denominators.csv", index=False)

    print("\n[contract] episodes with a tool-boundary contract violation\n")
    print(f"{'label':<22}{'flagged':>9}{'n':>8}{'rate':>8}")
    for label in sorted(per_label):
        v = per_label[label]
        print(f"{label:<22}{sum(v):>9}{len(v):>8}{sum(v) / len(v):>8.1%}")
    healthy = per_label.get("healthy", [])
    print(f"\n[contract] false positives on healthy runs: "
          f"{sum(healthy)}/{len(healthy)}")
    lead = [r for r in rows if r["tau"] is not None]
    if lead:
        at_onset = sum(1 for r in lead
                       if r["first_violation_step"] <= int(r["tau"]) + 1)
        print(f"[contract] flagged within one step of onset: "
              f"{at_onset}/{len(lead)}")
    print(f"[contract] wrote {TABLES / 'tool_contract_coverage.csv'}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        prog="py -m derail.verify.run_verification_study")
    ap.add_argument("--holdout", default=None,
                    help="score the frozen checks on a corpus they were not "
                         "designed against (e.g. organic_demo7b_holdout)")
    ap.add_argument("--contract-coverage", action="store_true",
                    help="score tool_contract across every labelled corpus")
    a = ap.parse_args()
    if a.holdout:
        holdout(a.holdout)
    elif a.contract_coverage:
        contract_coverage()
    else:
        main()
