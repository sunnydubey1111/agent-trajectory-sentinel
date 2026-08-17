"""One derivation of every episode total this project quotes.

Several totals circulate and they do not add up to each other: 3,226 committed
episodes, 2,823 as of arXiv v1, 2,080 healthy, 2,248 collection attempts with
541 discarded, 1,002 in the behavioural study, 874 in the grounding study,
1,825 healthy in the false-positive claim. Each is correct for what it counts.
The failure mode is arithmetic performed ACROSS them — most memorably
1,825 + 1,002 = 2,827, which lands four short of 2,823 and looks like a
rounding error, but is a coincidence between three incommensurable quantities.

This module derives all of them from source artifacts and states the identities
that DO hold, so a reader can check the accounting instead of trusting a
sentence. It reads manifests and `rejected.json` files for the corpus side and
the study tables for the population side; it computes nothing itself.

Four axes, and totals may only be added along one of them:

  ownership   project-owned corpora vs corpora imported from other projects
              (`traces/_*`), which are never counted as ours
  label       healthy vs injected, which partition a corpus exactly
  admission   accepted (in the manifest) vs rejected (in `rejected.json`);
              attempted = accepted + rejected, per corpus
  version     as of the arXiv v1 snapshot vs the current tree

Study populations are a fifth thing and are NOT an axis: they overlap each
other, they subset the committed corpora, and one of them (the behavioural
study) contains generated simulator episodes that are not committed traces at
all. They are reported here with what they draw from, and never summed.

    py -m devtools.episode_accounting            # print the accounting
    py -m devtools.episode_accounting --write    # + episode_accounting.csv
    py -m devtools.episode_accounting --check    # verify every identity
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACES = REPO_ROOT / "traces"
TABLES = REPO_ROOT / "results" / "tables"
OUT = TABLES / "episode_accounting.csv"


#: The corpus that lives at the TOP of `traces/` rather than in a subdirectory.
#: Every count in this project enumerates corpora with `traces/*/manifest.json`,
#: which matches subdirectories only, so this one is invisible to the data card,
#: the claims ledger and the Hugging Face export. That is recorded in
#: DATA_CARD.md and is deliberate: the published totals were all computed
#: against the glob scope, so restating them would move numbers to fix
#: bookkeeping. This module therefore counts it SEPARATELY - the gap becomes a
#: derived number instead of a paragraph, without changing any denominator.
ROOT_CORPUS = "gemini (traces root)"


def _manifests(owned: bool = True) -> list[pathlib.Path]:
    """Corpus manifests, ours or imported. A leading `_` marks not-ours."""
    return [m for m in sorted(TRACES.glob("*/manifest.json"))
            if m.parent.name.startswith("_") is not owned]


def corpus_rows() -> list[dict]:
    """One row per corpus: accepted split by label, plus what was rejected."""
    from devtools.claims_ledger import ADDED_AFTER_V1

    rows = []
    for owned in (True, False):
        for m in _manifests(owned):
            entries = json.loads(m.read_text("utf-8"))
            injected = sum(1 for e in entries if e.get("failure_class"))
            rejected_path = m.parent / "rejected.json"
            rejected = (len(json.loads(rejected_path.read_text("utf-8")))
                        if rejected_path.exists() else 0)
            rows.append({
                "corpus": m.parent.name,
                "owned": owned,
                "in_glob_scope": True,
                "in_v1": owned and m.parent.name not in ADDED_AFTER_V1,
                "healthy": len(entries) - injected,
                "injected": injected,
                "accepted": len(entries),
                "rejected": rejected,
                "attempted": len(entries) + rejected,
            })

    root = TRACES / "manifest.json"
    if root.exists():
        entries = json.loads(root.read_text("utf-8"))
        injected = sum(1 for e in entries if e.get("failure_class"))
        rejected_path = TRACES / "rejected.json"
        rejected = (len(json.loads(rejected_path.read_text("utf-8")))
                    if rejected_path.exists() else 0)
        rows.append({
            "corpus": ROOT_CORPUS, "owned": True, "in_glob_scope": False,
            "in_v1": False,
            "healthy": len(entries) - injected, "injected": injected,
            "accepted": len(entries), "rejected": rejected,
            "attempted": len(entries) + rejected,
        })
    return rows


def totals(rows: list[dict]) -> dict[str, int]:
    """Every quoted total, derived once."""
    # `own` is the GLOB SCOPE - the corpora every published total is computed
    # over. The root corpus is ours too, but no published number counts it, so
    # it is reported beside them rather than folded in.
    own = [r for r in rows if r["owned"] and r["in_glob_scope"]]
    root = [r for r in rows if r["owned"] and not r["in_glob_scope"]]
    v1 = [r for r in own if r["in_v1"]]
    imported = [r for r in rows if not r["owned"]]
    rejecting = [r for r in own if r["rejected"]]

    def s(rs, k):
        return sum(r[k] for r in rs)

    return {
        "owned_corpora": len(own),
        "owned_episodes": s(own, "accepted"),
        "owned_healthy": s(own, "healthy"),
        "owned_injected": s(own, "injected"),
        "root_corpus_episodes": s(root, "accepted"),
        "root_corpus_healthy": s(root, "healthy"),
        "root_corpus_injected": s(root, "injected"),
        "committed_episodes_all": s(own, "accepted") + s(root, "accepted"),
        "v1_corpora": len(v1),
        "v1_episodes": s(v1, "accepted"),
        "v1_healthy": s(v1, "healthy"),
        "v1_injected": s(v1, "injected"),
        "added_after_v1_episodes": s(own, "accepted") - s(v1, "accepted"),
        "imported_corpora": len(imported),
        "imported_episodes": s(imported, "accepted"),
        "rejecting_corpora": len(rejecting),
        "attempted_where_recorded": s(rejecting, "attempted"),
        "rejected_where_recorded": s(rejecting, "rejected"),
        "accepted_where_recorded": s(rejecting, "accepted"),
    }


def study_rows() -> list[dict]:
    """Study populations, with what each draws from. Never summed."""
    def n(name):
        return len(pd.read_csv(TABLES / name))

    hybrid = pd.read_csv(TABLES / "hybrid_diagnosis.csv")
    sim = int((hybrid.dataset == "sim").sum())
    grounding = n("grounding_diagnosis.csv")
    contract = pd.read_csv(TABLES / "tool_contract_denominators.csv")
    healthy_contract = int(contract.loc[contract.label == "healthy", "n"].iloc[0])
    return [
        {"study": "behavioural (hybrid)", "n": len(hybrid),
         "committed_episodes": len(hybrid) - sim, "generated_episodes": sim,
         "draws_from": "injected episodes of 7 owned corpora, plus simulator"},
        {"study": "grounding", "n": grounding, "committed_episodes": grounding,
         "generated_episodes": 0,
         "draws_from": "injected episodes of 10 owned corpora"},
        {"study": "contract false-positive", "n": healthy_contract,
         "committed_episodes": healthy_contract, "generated_episodes": 0,
         "draws_from": "healthy episodes of every owned corpus"},
    ]


def _injected_keys() -> set[tuple[str, str]]:
    """Every committed injected episode, keyed as the study tables key them.

    The root corpus is keyed `gemini` by the studies even though its manifest
    sits at the top of `traces/`, so it is included here under that name -
    otherwise its episodes look like study rows with no corpus behind them.
    """
    keys = set()
    for m in _manifests(owned=True):
        for e in json.loads(m.read_text("utf-8")):
            if e.get("failure_class"):
                keys.add((m.parent.name, e["episode_id"]))
    root = TRACES / "manifest.json"
    if root.exists():
        for e in json.loads(root.read_text("utf-8")):
            if e.get("failure_class"):
                keys.add(("gemini", e["episode_id"]))
    return keys


def coverage_rows() -> list[dict]:
    """Which committed injected episodes each study scores, and which none do.

    Two failure modes this makes visible. Summing study populations
    double-counts, because the behavioural study's real half is a subset of the
    grounding study rather than a disjoint set. And a corpus can sit committed
    and unscored, which is legitimate but should be a number rather than an
    assumption.
    """
    committed = _injected_keys()
    h = pd.read_csv(TABLES / "hybrid_diagnosis.csv")
    g = pd.read_csv(TABLES / "grounding_diagnosis.csv")
    hk = {(d, i) for d, i in zip(h.dataset, h.episode_id) if d != "sim"}
    gk = set(zip(g.dataset, g.episode_id))
    unscored = committed - (hk | gk)
    by_corpus: dict[str, int] = {}
    for d, _ in unscored:
        by_corpus[d] = by_corpus.get(d, 0) + 1
    return [
        {"quantity": "committed injected (owned, incl. root corpus)",
         "n": len(committed), "note": "the population studies draw from"},
        {"quantity": "scored by the behavioural study (real half)",
         "n": len(hk), "note": "excludes its 400 generated simulator episodes"},
        {"quantity": "scored by the grounding study", "n": len(gk),
         "note": "all committed"},
        {"quantity": "scored by BOTH", "n": len(hk & gk),
         "note": "double-counted if the two populations are added"},
        {"quantity": "union of the two studies", "n": len(hk | gk),
         "note": "the behavioural real half is a subset of the grounding set"},
        {"quantity": "committed but scored by neither", "n": len(unscored),
         "note": "; ".join(f"{k}={v}" for k, v in sorted(by_corpus.items()))},
        {"quantity": "study rows with no committed episode behind them",
         "n": len((hk | gk) - committed),
         "note": "must be 0: a scored episode has to exist in a manifest"},
    ]


def identities(t: dict[str, int], studies: list[dict]) -> list[dict]:
    """The equations that hold, and the famous one that does not.

    Each row is checkable arithmetic. `holds` is what `--check` enforces; a row
    with `holds=False` is documented as NOT an identity, so that nobody
    reconstructs it later and assumes it should balance.
    """
    hybrid = next(s for s in studies if s["study"].startswith("behavioural"))
    rows = [
        {"identity": "owned = healthy + injected",
         "lhs": t["owned_episodes"],
         "rhs": t["owned_healthy"] + t["owned_injected"], "is_identity": True},
        {"identity": "v1 = v1 healthy + v1 injected",
         "lhs": t["v1_episodes"],
         "rhs": t["v1_healthy"] + t["v1_injected"], "is_identity": True},
        {"identity": "owned = v1 + added after v1",
         "lhs": t["owned_episodes"],
         "rhs": t["v1_episodes"] + t["added_after_v1_episodes"],
         "is_identity": True},
        {"identity": "all committed = glob scope + root corpus",
         "lhs": t["committed_episodes_all"],
         "rhs": t["owned_episodes"] + t["root_corpus_episodes"],
         "is_identity": True},
        {"identity": "attempted = accepted + rejected (where recorded)",
         "lhs": t["attempted_where_recorded"],
         "rhs": t["accepted_where_recorded"] + t["rejected_where_recorded"],
         "is_identity": True},
        {"identity": "behavioural study = committed + generated",
         "lhs": hybrid["n"],
         "rhs": hybrid["committed_episodes"] + hybrid["generated_episodes"],
         "is_identity": True},
        # The arithmetic that started this: it mixes a healthy COUNT with a
        # study population that is neither all-committed nor a superset of
        # anything, so the near-match with the corpus total means nothing.
        {"identity": "v1 healthy + behavioural study =/= v1 total",
         "lhs": t["v1_healthy"] + hybrid["n"], "rhs": t["v1_episodes"],
         "is_identity": False},
    ]
    for r in rows:
        r["holds"] = (r["lhs"] == r["rhs"])
    return rows


def build() -> tuple[pd.DataFrame, dict[str, int], list[dict], list[dict]]:
    rows = corpus_rows()
    t = totals(rows)
    studies = study_rows()
    return pd.DataFrame(rows), t, studies, identities(t, studies)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    frame, t, studies, ids = build()
    cover = coverage_rows()

    print("[accounting] totals derived from manifests and rejected.json\n")
    for k, v in t.items():
        print(f"  {k:28s} {v:6d}")
    print("\n[accounting] study populations - overlapping, never summed\n")
    print(pd.DataFrame(studies).to_string(index=False))
    print("\n[accounting] identities\n")
    print(pd.DataFrame(ids).to_string(index=False))

    orphans = next(r["n"] for r in cover if r["quantity"].startswith("study rows"))
    broken = [r for r in ids if r["is_identity"] and not r["holds"]]
    if orphans:
        broken.append({"identity": "every scored episode exists in a manifest",
                       "lhs": orphans, "rhs": 0})
    coincidence = [r for r in ids if not r["is_identity"] and r["holds"]]
    if args.write:
        TABLES.mkdir(parents=True, exist_ok=True)
        frame.to_csv(OUT, index=False)
        print(f"\n[accounting] wrote {OUT}")
    if args.check:
        for r in broken:
            print(f"BROKEN {r['identity']}: {r['lhs']} != {r['rhs']}")
        for r in coincidence:
            print(f"UNEXPECTED {r['identity']}: both sides are {r['lhs']}; "
                  "this is documented as NOT an identity")
        if broken or coincidence:
            return 1
        print("\n[accounting] every identity holds and no false one does")
    return 0


if __name__ == "__main__":
    sys.exit(main())
