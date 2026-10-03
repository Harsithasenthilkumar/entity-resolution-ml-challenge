"""
metrics.py — exact replica of the competition scorer (Part 20).

F_0.5 = (1.25 * P * R) / (0.25 * P + R), computed PER Source-1 entity and then
macro-averaged over ALL Source-1 entities in the evaluation set.

Edge cases, per the official rules:
  - true set empty & predicted empty  -> 1.0  (correct singleton)
  - true set empty & predicted non-empty -> 0.0
  - true set non-empty & predicted empty -> 0.0
  - otherwise P=|T&P|/|P|, R=|T&P|/|T|; if P=R=0 -> 0.0
"""

def f_beta_05(true_set: set, pred_set: set) -> float:
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    p = tp / len(pred_set)
    r = tp / len(true_set)
    return (1.25 * p * r) / (0.25 * p + r)


def macro_f05(truth: dict, preds: dict) -> dict:
    """truth/preds: {s1_id: set(matched_ids)}. Scored over all keys of `truth`."""
    tot = 0.0
    n = 0
    tp_all = fp_all = fn_all = 0
    singleton_correct = singleton_total = 0
    for s1, tset in truth.items():
        pset = preds.get(s1, set())
        tot += f_beta_05(tset, pset)
        n += 1
        tp_all += len(tset & pset)
        fp_all += len(pset - tset)
        fn_all += len(tset - pset)
        if not tset:
            singleton_total += 1
            if not pset:
                singleton_correct += 1
    micro_p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) else 0.0
    micro_r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) else 0.0
    return {
        "macro_f05": tot / n if n else 0.0,
        "n_entities": n,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "tp": tp_all, "fp": fp_all, "fn": fn_all,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total else None,
        "singleton_total": singleton_total,
    }


def candidate_metrics(truth: dict, cands: dict, corpus_size: int) -> dict:
    """Blocking-quality metrics (Part 10)."""
    import statistics
    recovered = 0
    total_true = 0
    sizes = []
    for s1, tset in truth.items():
        cset = cands.get(s1, set())
        sizes.append(len(cset))
        total_true += len(tset)
        recovered += len(tset & cset)
    sizes.sort()
    n = len(sizes)
    def pct(q):
        if n == 0: return 0
        return sizes[min(n - 1, int(q * n))]
    total_cands = sum(sizes)
    possible = n * corpus_size
    return {
        "candidate_recall": recovered / total_true if total_true else 0.0,
        "true_pairs_total": total_true,
        "true_pairs_recovered": recovered,
        "total_candidates": total_cands,
        "avg_candidates": total_cands / n if n else 0,
        "median_candidates": pct(0.50),
        "p90_candidates": pct(0.90),
        "p95_candidates": pct(0.95),
        "p99_candidates": pct(0.99),
        "max_candidates": sizes[-1] if sizes else 0,
        "candidate_precision": recovered / total_cands if total_cands else 0.0,
        "reduction_ratio": 1 - (total_cands / possible) if possible else 0.0,
        "zero_candidate_entities": sum(1 for s in sizes if s == 0),
    }
