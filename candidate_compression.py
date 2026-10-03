"""
candidate_compression.py -- meta-blocking / adaptive candidate budgets.

This is the stage that makes candidate_pairs.tsv small WITHOUT throwing away
true matches, and it is deliberately separated from the matcher so that
candidate_pairs.tsv can be written as EXACTLY the set the model scores
(competition rule: candidate_pairs.tsv must be the last filtering stage, not
an early blocking pass that is later narrowed).

WHY THIS STAGE EXISTS
---------------------
Measured on the India validation fold, raw union retrieval sits on this
frontier (3,000 queries, full 4.13M corpus):

    hash + n5(20) + addr(20)      recall 0.826   avg  48.8   p95  87
    hash + n5(50) + addr(50)      recall 0.864   avg  97.5   p95 142
    hash + n5(200) + addr(200)    recall 0.910   avg 354.1   p95 401

Recall keeps creeping up but candidate count explodes. Because the competition
metric is F_0.5 -- precision weighted 2x -- feeding 354 candidates per entity
into the matcher is actively harmful: it is 7x the distractors for +8 points of
ceiling. The job here is to retrieve WIDE (high k, high recall ceiling) and
then rank and cut, keeping the recall but not the bulk.

ADAPTIVE BUDGET (Part 9 of the brief)
-------------------------------------
A fixed K is wasteful. An entity whose normalized name matched EXACTLY in the
corpus needs almost no budget; an entity with a missing address and a
transliterated name needs a wide one. `adaptive_budget` assigns each query to a
confidence tier from cheap evidence that is already computed, and gives each
tier its own K. Tier thresholds are tuned on validation, never hard-coded to a
country or an id.
"""
import math

import numpy as np
from rapidfuzz import fuzz

# Tier budgets -- tune with tune_budgets() on your own validation fold.
TIER_BUDGETS = dict(exact=12, strong=20, medium=30, weak=50)


def _jw(a, b):
    if not a or not b:
        return 0.0
    return fuzz.WRatio(a, b) / 100.0


def cheap_features(q, cand_idx, ev, corpus_text):
    """Cheap re-ranking features. Deliberately NOT the expensive feature set --
    this runs on every retrieved candidate, so it must stay O(1)-ish per pair.

    q            : (eid, name_alnum, name_core, addr, house, postal)
    ev           : evidence dict from BlockingIndexes.retrieve for this cand
    corpus_text  : dict of preloaded corpus columns (lists indexed by cand_idx)
    """
    _, q_na, q_nc, q_ad, q_ho, q_po = q
    c_nc = corpus_text["name_core"][cand_idx] or ""
    c_ad = corpus_text["addr_street"][cand_idx] or ""
    c_ho = corpus_text["house_number"][cand_idx] or ""
    c_po = corpus_text["postal_code"][cand_idx] or ""

    name_sim = _jw(q_nc, c_nc)
    addr_sim = _jw(q_ad, c_ad) if (q_ad and c_ad) else 0.0
    n_channels = sum(1 for k in ("A", "B", "C", "D", "E", "F", "G", "J") if k in ev)

    return dict(
        name_sim=name_sim,
        addr_sim=addr_sim,
        exact_name=1.0 if (q_nc and q_nc == c_nc) else 0.0,
        n_channels=n_channels,
        d_score=ev.get("D", 0.0),
        e_score=ev.get("E", 0.0),
        d_rank=ev.get("Dr", 999),
        e_rank=ev.get("Er", 999),
        j_score=ev.get("J", 0.0),
        house_match=1.0 if (q_ho and q_ho == c_ho) else 0.0,
        postal_match=1.0 if (q_po and q_po == c_po) else 0.0,
        addr_missing=1.0 if not c_ad else 0.0,
    )


def rank_score(f):
    """Cheap monotone ranking score used to order candidates before the cut.

    Hand-weighted rather than learned on purpose: it only has to ORDER
    candidates well enough that the true match lands above the cut, and a
    learned ranker here would need its own leakage-controlled fold. The
    weights reflect the measured channel strengths (address is the strongest
    single channel; multi-channel agreement is the strongest cheap signal).
    """
    return (
        2.0 * f["name_sim"]
        + 2.0 * f["addr_sim"]
        + 1.5 * f["exact_name"]
        + 0.8 * f["n_channels"]
        + 1.0 * f["house_match"]
        + 1.0 * f["postal_match"]
        + 0.5 * f["d_score"]
        + 0.5 * f["e_score"]
        + 1.5 * f["j_score"]
    )


def adaptive_budget(q, ranked, budgets=None):
    """Pick K for THIS query from the evidence actually retrieved.

    Rationale: candidate count is dominated by the easy majority. If an entity
    has an exact normalized-name hit with a corroborating address, three
    candidates is plenty and anything beyond that is pure false-positive risk
    under F_0.5. The hard tail is where a wide budget pays for itself.
    """
    b = {**TIER_BUDGETS, **(budgets or {})}
    if not ranked:
        return b["weak"]
    top = ranked[0][2]  # feature dict of best-ranked candidate

    if top["exact_name"] and (top["addr_sim"] > 0.80 or top["postal_match"]
                              or top["house_match"]):
        return b["exact"]
    if top["exact_name"] or top["name_sim"] > 0.92:
        return b["strong"]
    if top["name_sim"] > 0.75 or top["addr_sim"] > 0.85:
        return b["medium"]
    return b["weak"]


PROTECT_TOP = 5     # name (D) / address (E) retriever top-N always kept
PROTECT_JOINT = 12  # joint (J) retriever top-N always kept


def compress(q, retrieved, corpus_text, ids, budgets=None, fixed_k=None):
    """retrieved: {cand_idx: evidence} -> list of (entity_id, score, features, ev).

    Returns the FINAL candidate list for this query -- exactly what gets
    written to candidate_pairs.tsv and exactly what the matcher scores.

    Candidates a retriever ranked highly are never cut, whatever the cheap
    rank_score says. Measured motivation: on India, the budget cut alone left
    recall at 0.681 because transliterated true matches get low cheap name
    similarity and sort below look-alike distractors, even though the
    retrievers themselves had ranked them highly.
    """
    scored = []
    for cand_idx, ev in retrieved.items():
        f = cheap_features(q, cand_idx, ev, corpus_text)
        scored.append((cand_idx, rank_score(f), f, ev))
    scored.sort(key=lambda r: -r[1])

    k = fixed_k if fixed_k is not None else adaptive_budget(q, scored, budgets)
    cut = scored[:k]
    for r in scored[k:]:
        ev = r[3]
        if (ev.get("Dr", 999) < PROTECT_TOP or ev.get("Er", 999) < PROTECT_TOP
                or ev.get("Jr", 999) < PROTECT_JOINT or "R" in ev):
            # "R": reverse-retrieval candidates (already capped per S1 in
            # reverse.py) are always kept -- they are the highest-recall channel
            cut.append(r)
    cut.sort(key=lambda r: -r[1])
    return [(ids[ci], sc, f, ev) for ci, sc, f, ev in cut]


def tune_budgets(eval_fn, grid=None):
    """Sweep tier budgets against a validation scorer.

    eval_fn(budgets) -> dict with at least candidate_recall and avg_candidates.
    Choose the smallest budget set whose recall is within `tol` of the best --
    this is the Pareto-knee selection the brief asks for, made explicit rather
    than eyeballed.
    """
    grid = grid or [
        dict(exact=2, strong=5, medium=12, weak=25),
        dict(exact=3, strong=8, medium=20, weak=40),
        dict(exact=3, strong=10, medium=30, weak=60),
        dict(exact=5, strong=15, medium=40, weak=80),
    ]
    rows = []
    for b in grid:
        m = eval_fn(b)
        rows.append((b, m))
        print(f"  budgets={b} recall={m['candidate_recall']:.4f} "
              f"avg={m['avg_candidates']:.1f} p95={m['p95_candidates']}")
    best_recall = max(m["candidate_recall"] for _, m in rows)
    tol = 0.005
    viable = [(b, m) for b, m in rows
              if m["candidate_recall"] >= best_recall - tol]
    return min(viable, key=lambda r: r[1]["avg_candidates"])
