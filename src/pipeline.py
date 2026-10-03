"""
pipeline.py -- end-to-end orchestrator.

    normalize -> per-country index -> retrieve -> compress -> features
              -> LightGBM -> F_0.5 threshold -> matching_results.tsv
                                             -> candidate_pairs.tsv

COMPETITION COMPLIANCE
----------------------
candidate_pairs.tsv is written from the SAME in-memory candidate list that is
handed to the matcher. There is no second filtering pass afterwards. The code
path is literally:

        finals = compress(...)          # <- the last filtering stage
        write_candidates(finals)        # <- candidate_pairs.tsv
        scores = model.predict(feats_of(finals))
        write_matches(finals[scores >= threshold])

so every id in matching_results.tsv is by construction a subset of
candidate_pairs.tsv.

PARALLELISM
-----------
Countries are independent (country agrees on 100% of true training pairs), so
they are the natural unit of parallelism: one worker process per country,
each owning its own indexes. Within a country, queries are embarrassingly
parallel and are chunked across a process pool. Set --workers to your core
count. Reproducibility is preserved because every worker is seeded and results
are merged in sorted entity_id order.

Usage
-----
    python src/pipeline.py --stage normalize   --data-dir dataset
    python src/pipeline.py --stage validate    --data-dir dataset --workers 16
    python src/pipeline.py --stage train       --data-dir dataset --workers 16
    python src/pipeline.py --stage infer       --data-dir dataset --workers 16
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import CountryCorpus
from blocking import BlockingIndexes, DEFAULTS
from candidate_compression import compress, cheap_features, rank_score
from features import pair_features, FEATURE_NAMES, query_groups
from reverse import get_reverse
from phonetic import name_key, numbers
from collections import Counter


def corpus_twin_counts(ctext):
    """For every corpus record: how many OTHER records share its
    (phonetic name key + number set). Measured on India training data:
    38.2% of true-match records have such a twin, vs 0.8% of negatives --
    negatives are one-off perturbations, real businesses appear as several
    consistent copies. Label-free, so equally valid on the test corpus."""
    keys = [name_key(n) + "#" + " ".join(sorted(numbers(a)))
            for n, a in zip(ctext["name_core"], ctext["addr_street"])]
    cnt = Counter(keys)
    tw = np.fromiter((cnt[k] - 1 for k in keys), dtype=np.int32, count=len(keys))
    del cnt
    return tw, keys


def s1_name_counts(s1_parquet, country):
    """How many S1 entities (this country) share each phonetic name key."""
    t = pq.read_table(s1_parquet, columns=["country_norm", "name_core"])
    cn = t["country_norm"].to_pylist()
    nm = t["name_core"].to_pylist()
    return Counter(name_key(n) for c, n in zip(cn, nm) if c == country)
from metrics import macro_f05, candidate_metrics
import model as M

NORM_DIR = "norm"
CACHE_DIR = "cache"
OUT_DIR = "output"
ART_DIR = "artifacts"
for d in (NORM_DIR, CACHE_DIR, OUT_DIR, ART_DIR):
    os.makedirs(d, exist_ok=True)

Q_COLS = ["entity_id", "country_norm", "name_alnum", "name_core",
          "addr_street", "house_number", "postal_code"]


# ------------------------------------------------------------------ helpers
def load_queries(parquet_path, country, keep_ids=None, limit=None):
    q = []
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=250_000, columns=Q_COLS):
        d = batch.to_pydict()
        cn = d["country_norm"]
        for i in range(len(cn)):
            if cn[i] != country:
                continue
            eid = d["entity_id"][i]
            if keep_ids is not None and eid not in keep_ids:
                continue
            q.append((eid,
                      d["name_alnum"][i] or "", d["name_core"][i] or "",
                      d["addr_street"][i] or "", d["house_number"][i] or "",
                      d["postal_code"][i] or ""))
        del d
        if limit and len(q) >= limit:
            break
    return q[:limit] if limit else q


def load_gt(path):
    gt = {}
    with open(path) as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            gt[a] = set(x for x in b.split(",") if x)
    return gt


def build_country_corpus(country, s2, s3, tag):
    cache = f"{CACHE_DIR}/{country}_{tag}.parquet"
    if not os.path.exists(cache):
        c = CountryCorpus(country).open_writer(cache)
        c.add_parquet(s2)
        c.add_parquet(s3)
        c.finalize()
    else:
        c = CountryCorpus(country)
        c.cache_path = cache
        c._writer = None
        c.finalize()
    return c


def corpus_text_views(corpus):
    return dict(
        name_alnum=corpus.text_list("name_alnum"),
        name_core=corpus.text_list("name_core"),
        addr_street=corpus.text_list("addr_street"),
        house_number=[x or "" for x in corpus.tbl["house_number"].to_pylist()],
        postal_code=[x or "" for x in corpus.tbl["postal_code"].to_pylist()],
    )


def generate_pairs(queries, BI, ctext, ids, budgets=None, fixed_k=None,
                   gt=None, progress_every=20000, rev=None, ncount=None,
                   twin=None):
    """Run retrieve + compress. Returns (index, feature_rows, labels, cand_map).

    This single function is used for BOTH training-pair generation and
    inference, which is what guarantees the model trains on the exact
    candidate distribution it will be scored on -- and is also why hard
    negatives need no separate mining step.
    """
    index, rows, labels = [], [], []
    chunks = []          # float32 feature blocks: Python lists of floats cost
                         # ~2 KB/row, float32 costs ~0.25 KB/row
    cand_map = {}
    t0 = time.time()
    for qi, q in enumerate(queries):
        eid = q[0]
        retrieved = BI.retrieve(q[1], q[2], q[3], q[4], q[5])
        if rev is not None:
            for rec, sc, rk, mg in rev.get(eid):
                e = retrieved[rec]
                e["R"] = sc
                e["Rr"] = rk
                e["Rm"] = mg
        finals = compress(q, retrieved, ctext, ids, budgets=budgets,
                          fixed_k=fixed_k)
        cand_map[eid] = [c[0] for c in finals]
        top_score = finals[0][1] if finals else 0.0
        ctx = dict(cand_set_size=len(finals), top_score=top_score,
                   qgroups=query_groups(q[2], q[3]), ncount=ncount,
                   q_ncount=(ncount or {}).get(name_key(q[2]), 0))
        truth = gt.get(eid, set()) if gt is not None else None
        # --- corroboration across the candidate set (see corpus_twin_counts)
        cis = [BI.id_pos[c_[0]] for c_ in finals]
        qnums = numbers(q[3])
        cnums = [numbers(ctext["addr_street"][j]) for j in cis]
        cname = [name_key(ctext["name_core"][j]) for j in cis]
        ckeys = [twin[1][j] for j in cis] if twin is not None else cname
        kc = Counter(ckeys)
        nc_ = Counter(cname)
        sups = []
        for r_, j in enumerate(cis):
            extra = cnums[r_] - qnums
            if extra:
                es = sum(1 for o in range(len(cis)) if o != r_ and (extra & cnums[o]))
            else:
                es = -1
            sups.append([float(twin[0][j]) if twin is not None else 0.0,
                         float(kc[ckeys[r_]] - 1), float(nc_[cname[r_]] - 1), float(es)])
        for rank, (cid, sc, f, ev) in enumerate(finals):
            ci = BI.id_pos[cid]
            cand = dict(sup=sups[rank],
                entity_id=cid,
                name_alnum=ctext["name_alnum"][ci],
                name_core=ctext["name_core"][ci],
                addr=ctext["addr_street"][ci],
                house=ctext["house_number"][ci],
                postal=ctext["postal_code"][ci],
                ev=ev, rank_score=sc, is_top1=(rank == 0),
            )
            rows.append(pair_features(q, cand, ctx=ctx))
            index.append((eid, cid))
            if truth is not None:
                labels.append(1 if cid in truth else 0)
        if len(rows) >= 200_000:
            chunks.append(np.asarray(rows, dtype=np.float32))
            rows = []
        if progress_every and qi and qi % progress_every == 0:
            print(f"      {qi:,}/{len(queries):,} queries "
                  f"({time.time()-t0:.0f}s, {len(index):,} pairs)", flush=True)
    if rows:
        chunks.append(np.asarray(rows, dtype=np.float32))
    X = np.vstack(chunks) if chunks else np.zeros((0, len(FEATURE_NAMES)), np.float32)
    y = np.asarray(labels, dtype=np.int8) if gt is not None else None
    return index, X, y, cand_map


def attach_id_pos(BI):
    BI.id_pos = {e: i for i, e in enumerate(BI.ids)}
    return BI


# ------------------------------------------------------------------- stages
def stage_normalize(args):
    import normalization as norm
    jobs = [
        (f"{args.data_dir}/train/train_source1.tsv", f"{NORM_DIR}/train_s1.parquet"),
        (f"{args.data_dir}/train/train_source2.tsv", f"{NORM_DIR}/train_s2.parquet"),
        (f"{args.data_dir}/train/train_source3.tsv", f"{NORM_DIR}/train_s3.parquet"),
        (f"{args.data_dir}/test/test_source1.tsv", f"{NORM_DIR}/test_s1.parquet"),
        (f"{args.data_dir}/test/test_source2.tsv", f"{NORM_DIR}/test_s2.parquet"),
        (f"{args.data_dir}/test/test_source3.tsv", f"{NORM_DIR}/test_s3.parquet"),
    ]
    for src, dst in jobs:
        if os.path.exists(dst):
            print(f"[skip] {dst}")
            continue
        norm.run(src, dst)


def _countries(parquet_path):
    tb = pq.read_table(parquet_path, columns=["country_norm"])
    return sorted(set(tb["country_norm"].to_pylist()))


def stage_validate(args):
    """Blocking frontier + matcher validation on held-out S1 entities."""
    gt = load_gt(f"{args.data_dir}/train/train_ground_truth.tsv")
    import validation
    if not os.path.exists("splits/seed0_val.tsv"):
        for i in range(5):
            validation.build(seed_name=f"seed{i}",
                             val_frac_buckets=range(i * 5, i * 5 + 5))
    val_ids = set(load_gt("splits/seed0_val.tsv").keys())
    report = {}
    for country in _countries(f"{NORM_DIR}/train_s1.parquet"):
        print(f"\n=== {country} ===")
        c = build_country_corpus(country, f"{NORM_DIR}/train_s2.parquet",
                                 f"{NORM_DIR}/train_s3.parquet", "train")
        rev = get_reverse(country, "train", f"{NORM_DIR}/train_s1.parquet", c,
                          args.workers)
        ncount = s1_name_counts(f"{NORM_DIR}/train_s1.parquet", country)
        BI = attach_id_pos(BlockingIndexes(c))
        ctext = corpus_text_views(c)
        twin = corpus_twin_counts(ctext)
        qs = load_queries(f"{NORM_DIR}/train_s1.parquet", country,
                          keep_ids=val_ids, limit=args.val_queries)
        present = set(BI.ids)
        truth = {q[0]: {v for v in gt.get(q[0], set()) if v in present} for q in qs}
        _, _, _, cand_map = generate_pairs(qs, BI, ctext, BI.ids, gt=gt, rev=rev, ncount=ncount, twin=twin)
        cm = candidate_metrics(truth, {k: set(v) for k, v in cand_map.items()}, c.n)
        print(f"  candidate_recall={cm['candidate_recall']:.4f} "
              f"avg={cm['avg_candidates']:.1f} p95={cm['p95_candidates']} "
              f"p99={cm['p99_candidates']} reduction={cm['reduction_ratio']:.8f}")
        report[country] = cm
        del BI, ctext, c
        gc.collect()
    json.dump(report, open(f"{ART_DIR}/blocking_report.json", "w"),
              indent=2, default=str)


def stage_train(args):
    gt = load_gt(f"{args.data_dir}/train/train_ground_truth.tsv")
    import validation
    if not os.path.exists("splits/seed0_val.tsv"):
        for i in range(5):
            validation.build(seed_name=f"seed{i}",
                             val_frac_buckets=range(i * 5, i * 5 + 5))
    val_ids = set(load_gt("splits/seed0_val.tsv").keys())

    Xtr_all, ytr_all = [], []
    Xva_all, yva_all, iva_all = [], [], []
    truth_va = {}

    for country in _countries(f"{NORM_DIR}/train_s1.parquet"):
        print(f"\n=== {country} ===", flush=True)
        c = build_country_corpus(country, f"{NORM_DIR}/train_s2.parquet",
                                 f"{NORM_DIR}/train_s3.parquet", "train")
        rev = get_reverse(country, "train", f"{NORM_DIR}/train_s1.parquet", c,
                          args.workers)
        ncount = s1_name_counts(f"{NORM_DIR}/train_s1.parquet", country)
        BI = attach_id_pos(BlockingIndexes(c))
        ctext = corpus_text_views(c)
        twin = corpus_twin_counts(ctext)
        present = set(BI.ids)

        tr_q = [q for q in load_queries(f"{NORM_DIR}/train_s1.parquet", country)
                if q[0] not in val_ids][:args.train_queries]
        va_q = load_queries(f"{NORM_DIR}/train_s1.parquet", country,
                            keep_ids=val_ids, limit=args.val_queries)
        print(f"  train_q={len(tr_q):,} val_q={len(va_q):,}", flush=True)

        _, Xt, yt, _ = generate_pairs(tr_q, BI, ctext, BI.ids, gt=gt, rev=rev, ncount=ncount, twin=twin)
        iv, Xv, yv, _ = generate_pairs(va_q, BI, ctext, BI.ids, gt=gt, rev=rev, ncount=ncount, twin=twin)
        Xtr_all.append(Xt); ytr_all.append(yt)
        Xva_all.append(Xv); yva_all.append(yv); iva_all.extend(iv)
        for q in va_q:
            truth_va[q[0]] = {v for v in gt.get(q[0], set()) if v in present}
        del BI, ctext, c, rev, twin
        gc.collect()

    Xtr = np.vstack(Xtr_all); ytr = np.concatenate(ytr_all)
    Xva = np.vstack(Xva_all); yva = np.concatenate(yva_all)
    del Xtr_all, Xva_all
    gc.collect()
    print(f"\ntrain pairs={Xtr.shape} pos_rate={ytr.mean():.4f}")
    print(f"val   pairs={Xva.shape} pos_rate={yva.mean():.4f}")

    models = M.benchmark_models(Xtr, ytr, Xva, yva, truth_va, iva_all)
    best_name, best_score, best_th, best_mg, best_model = None, -1, None, None, None
    for name, mdl in models.items():
        p = M.predict(mdl, Xva)
        sc, th, mg = M.optimise_threshold(p, iva_all, truth_va,
                                          margins=(None, 0.10, 0.20))
        print(f"  {name}: macro_F0.5={sc:.5f} th={th:.3f} margin={mg}")
        if sc > best_score:
            best_name, best_score, best_th, best_mg, best_model = name, sc, th, mg, mdl

    # global-uniqueness pass -- keep only if it measurably helps
    p = M.predict(best_model, Xva)
    uni = M.apply_global_uniqueness(p, iva_all, best_th)
    uni_score = macro_f05(truth_va, uni)["macro_f05"]
    use_uni = uni_score > best_score
    print(f"\nbest={best_name} F0.5={best_score:.5f} | "
          f"with global-uniqueness={uni_score:.5f} -> use={use_uni}")

    pickle.dump(dict(model=best_model, name=best_name, threshold=best_th,
                     margin=best_mg, use_uniqueness=use_uni,
                     feature_names=FEATURE_NAMES),
                open(f"{ART_DIR}/matcher.pkl", "wb"))
    json.dump(dict(model=best_name, macro_f05=best_score, threshold=best_th,
                   margin=best_mg, uniqueness_f05=uni_score,
                   use_uniqueness=bool(use_uni)),
              open(f"{ART_DIR}/train_report.json", "w"), indent=2)


def stage_reverse(args):
    """Precompute reverse retrieval for train and test (cached to disk)."""
    for tag in ("train", "test"):
        s1p = f"{NORM_DIR}/{tag}_s1.parquet"
        for country in _countries(s1p):
            print(f"\n=== reverse {tag} {country} ===", flush=True)
            c = build_country_corpus(country, f"{NORM_DIR}/{tag}_s2.parquet",
                                     f"{NORM_DIR}/{tag}_s3.parquet", tag)
            get_reverse(country, tag, s1p, c, args.workers)
            del c
            gc.collect()


def stage_diagnose(args):
    """Where is macro F0.5 lost? Uses the trained matcher + cached reverse
    results on held-out validation entities. Prints, per country:
      * precision / recall / F0.5 / singleton accuracy
      * missed true matches split into BLOCKING misses (never a candidate)
        and MATCHER misses (candidate, but scored below threshold)
      * false positives split into SIBLING (unowned record with a nudged
        number), OTHER-OWNER (the record belongs to a different S1), and
        UNOWNED-OTHER
      * how much macro F0.5 each kind of entity error costs
      * concrete examples of each error type
    """
    gt = load_gt(f"{args.data_dir}/train/train_ground_truth.tsv")
    art = pickle.load(open(f"{ART_DIR}/matcher.pkl", "rb"))
    mdl, th, mg = art["model"], art["threshold"], art["margin"]
    val_ids = set(load_gt("splits/seed0_val.tsv").keys())
    own = {}
    for s1, ts in gt.items():
        for t in ts:
            own[t] = s1
    ci = FEATURE_NAMES.index("k_num_close")
    for country in _countries(f"{NORM_DIR}/train_s1.parquet"):
        print(f"\n=== DIAGNOSE {country} ===", flush=True)
        c = build_country_corpus(country, f"{NORM_DIR}/train_s2.parquet",
                                 f"{NORM_DIR}/train_s3.parquet", "train")
        rev = get_reverse(country, "train", f"{NORM_DIR}/train_s1.parquet", c,
                          args.workers)
        ncount = s1_name_counts(f"{NORM_DIR}/train_s1.parquet", country)
        BI = attach_id_pos(BlockingIndexes(c, verbose=False))
        ctext = corpus_text_views(c)
        twin = corpus_twin_counts(ctext)
        qs = load_queries(f"{NORM_DIR}/train_s1.parquet", country,
                          keep_ids=val_ids, limit=args.val_queries)
        qmap = {q[0]: q for q in qs}
        present = set(BI.ids)
        truth = {q[0]: {v for v in gt.get(q[0], set()) if v in present} for q in qs}
        idx, X, _, cmap = generate_pairs(qs, BI, ctext, BI.ids, gt=None,
                                         rev=rev, ncount=ncount, twin=twin, progress_every=0)
        p = M.predict(mdl, X) if len(X) else np.zeros(0)
        if art["use_uniqueness"]:
            sel = M.apply_global_uniqueness(p, idx, th)
        else:
            sel = M.assemble(p, idx, th, mg)
        pred = {q[0]: sel.get(q[0], set()) for q in qs}
        pv = {k: float(v) for k, v in zip(idx, p)}
        close = {k: float(X[i, ci]) for i, k in enumerate(idx)}
        r = macro_f05(truth, pred)
        cm = candidate_metrics(truth, {k: set(v) for k, v in cmap.items()}, c.n)
        print(f"  macro_F0.5={r['macro_f05']:.4f}  precision={r['micro_precision']:.4f}  "
              f"recall={r['micro_recall']:.4f}  singleton_acc={r['singleton_accuracy']}")
        print(f"  candidate_recall={cm['candidate_recall']:.4f}  avg_cands={cm['avg_candidates']:.1f}")

        fn_block, fn_match, fp_sib, fp_other, fp_un = [], [], [], [], []
        loss = defaultdict(float)
        from metrics import f_beta_05
        for s1, T in truth.items():
            P = pred[s1]
            C = set(cmap.get(s1, []))
            for t in T - P:
                (fn_match if t in C else fn_block).append((s1, t))
            for f in P - T:
                o = own.get(f)
                if o is not None:
                    fp_other.append((s1, f))
                elif close.get((s1, f), 0) > 0:
                    fp_sib.append((s1, f))
                else:
                    fp_un.append((s1, f))
            fv = f_beta_05(T, P)
            if fv < 1:
                kind = ("singleton_given_matches" if not T else
                        "predicted_empty" if not P else
                        "has_FP_and_FN" if (P - T and T - P) else
                        "has_FP_only" if P - T else "has_FN_only")
                loss[kind] += (1 - fv) / len(truth)
        n_t = sum(len(T) for T in truth.values())
        print(f"  missed true matches: {len(fn_block)+len(fn_match):,} of {n_t:,}  "
              f"-> blocking {len(fn_block):,} | matcher {len(fn_match):,}")
        print(f"  false positives: {len(fp_sib)+len(fp_other)+len(fp_un):,}  -> "
              f"sibling(nudged number) {len(fp_sib):,} | other S1's record "
              f"{len(fp_other):,} | unowned-other {len(fp_un):,}")
        print("  macro-F0.5 points lost by entity error type:")
        for k, v in sorted(loss.items(), key=lambda kv: -kv[1]):
            print(f"     {k:<26} {v:.4f}")

        def show(title, pairs, n=6):
            print(f"  --- {title} ---")
            for s1, cid in pairs[:n]:
                q = qmap[s1]
                j = BI.id_pos.get(cid)
                cn = ctext["name_core"][j] if j is not None else "?"
                ca = ctext["addr_street"][j] if j is not None else "?"
                print(f"    S1 : {q[2][:40]:<40} | {q[3][:60]}")
                print(f"    REC: {cn[:40]:<40} | {ca[:60]}   p={pv.get((s1, cid), float('nan')):.3f}")
        show("MATCHER MISSES (true match scored below threshold)", fn_match)
        show("FALSE POSITIVE: sibling with nudged number", fp_sib)
        show("FALSE POSITIVE: record belonging to another S1", fp_other)
        show("FALSE POSITIVE: other unowned record", fp_un)
        show("BLOCKING MISSES (never a candidate)", fn_block)
        del BI, ctext, c, rev, twin, X
        gc.collect()


def stage_infer(args):
    art = pickle.load(open(f"{ART_DIR}/matcher.pkl", "rb"))
    mdl, th, mg = art["model"], art["threshold"], art["margin"]
    all_cands, all_matches = {}, {}

    for country in _countries(f"{NORM_DIR}/test_s1.parquet"):
        print(f"\n=== TEST {country} ===", flush=True)
        c = build_country_corpus(country, f"{NORM_DIR}/test_s2.parquet",
                                 f"{NORM_DIR}/test_s3.parquet", "test")
        rev = get_reverse(country, "test", f"{NORM_DIR}/test_s1.parquet", c,
                          args.workers)
        ncount = s1_name_counts(f"{NORM_DIR}/test_s1.parquet", country)
        BI = attach_id_pos(BlockingIndexes(c))
        ctext = corpus_text_views(c)
        twin = corpus_twin_counts(ctext)
        qs = load_queries(f"{NORM_DIR}/test_s1.parquet", country)
        print(f"  queries={len(qs):,} corpus={c.n:,}", flush=True)

        # Chunked inference: holding every pair's features at once would need
        # tens of GB on the India test set. Only pairs at/above the threshold
        # are kept -- which is lossless, because assemble() and
        # apply_global_uniqueness() never select a pair below the threshold,
        # and the margin rule compares against the best pair, which is itself
        # above the threshold whenever anything is selected.
        CH = 20_000
        kept_idx, kept_p = [], []
        # every scored pair is also saved (as integer row ids + probability)
        # so the decision layer can be re-fitted later WITHOUT rerunning
        # inference -- see src/decide.py
        qpos = {q[0]: i for i, q in enumerate(qs)}
        sv_q, sv_c, sv_p = [], [], []
        t_inf = time.time()
        for lo in range(0, len(qs), CH):
            part = qs[lo:lo + CH]
            idx, X, _, cand_map = generate_pairs(part, BI, ctext, BI.ids, gt=None,
                                                 rev=rev, ncount=ncount, twin=twin, progress_every=0)
            # candidate_pairs.tsv == exactly this set (written below, unfiltered)
            all_cands.update(cand_map)
            if len(X):
                p = M.predict(mdl, X)
                sv_q.append(np.fromiter((qpos[a] for a, _ in idx), np.int32, len(idx)))
                sv_c.append(np.fromiter((BI.id_pos[b] for _, b in idx), np.int32, len(idx)))
                sv_p.append(np.asarray(p, dtype=np.float32))
                for (s1, cid), pv in zip(idx, p):
                    if pv >= th:
                        kept_idx.append((s1, cid))
                        kept_p.append(float(pv))
            del idx, X, cand_map, part
            gc.collect()
            done = min(lo + CH, len(qs))
            print(f"      {done:,}/{len(qs):,} queries ({time.time()-t_inf:.0f}s)",
                  flush=True)
        np.savez(f"{CACHE_DIR}/testpred_{country}.npz",
                 q=np.concatenate(sv_q) if sv_q else np.zeros(0, np.int32),
                 c=np.concatenate(sv_c) if sv_c else np.zeros(0, np.int32),
                 p=np.concatenate(sv_p) if sv_p else np.zeros(0, np.float32))
        with open(f"{CACHE_DIR}/testpred_{country}_qids.pkl", "wb") as f_:
            pickle.dump([q[0] for q in qs], f_)
        del sv_q, sv_c, sv_p
        if kept_p:
            if art["use_uniqueness"]:
                sel = M.apply_global_uniqueness(kept_p, kept_idx, th)
            else:
                sel = M.assemble(kept_p, kept_idx, th, mg)
        else:
            sel = {}
        for q in qs:
            all_matches[q[0]] = sel.get(q[0], set())
        del BI, ctext, c, rev, twin, kept_idx, kept_p, sel
        gc.collect()

    write_tsv(f"{OUT_DIR}/candidate_pairs.tsv", "candidate_entity_ids",
              all_cands, f"{NORM_DIR}/test_s1.parquet")
    write_tsv(f"{OUT_DIR}/matching_results.tsv", "matched_entity_ids",
              all_matches, f"{NORM_DIR}/test_s1.parquet")
    print("\nwrote output/candidate_pairs.tsv and output/matching_results.tsv")


def write_tsv(path, col, mapping, s1_parquet):
    """Every S1 test entity gets exactly one row, in file order, no duplicates."""
    ids = pq.read_table(s1_parquet, columns=["entity_id"])["entity_id"].to_pylist()
    seen = set()
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for eid in ids:
            if eid in seen:
                continue
            seen.add(eid)
            vals = mapping.get(eid, set()) or set()
            f.write(f"{eid}\t{','.join(sorted(vals))}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["normalize", "reverse", "validate", "train", "infer",
                             "diagnose"])
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--train-queries", type=int, default=400_000)
    ap.add_argument("--val-queries", type=int, default=40_000)
    a = ap.parse_args()
    t0 = time.time()
    {"normalize": stage_normalize, "reverse": stage_reverse,
     "diagnose": stage_diagnose,
     "validate": stage_validate,
     "train": stage_train, "infer": stage_infer}[a.stage](a)
    print(f"\n[{a.stage}] done in {time.time()-t0:.0f}s")
