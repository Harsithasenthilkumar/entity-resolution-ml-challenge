"""
ce.py -- second-stage CROSS-ENCODER re-ranker on top of the LightGBM matcher.

WHY
---
The LightGBM matcher (validation macro F0.5 0.961) only sees the 69
hand-designed features. A cross-encoder reads BOTH records as text and learns
the noise model directly from labelled pairs (transliteration, typos,
'6th'/'sixth', 'az'/'arizona', reordered words). It is used only where the
LightGBM model is uncertain (LO <= p <= HI), and the combination is kept ONLY
if it beats LightGBM alone on held-out validation entities.

MODEL / LICENCE
---------------
cross-encoder/ms-marco-MiniLM-L6-v2 -- Apache-2.0, ~22M parameters
(competition limit: MIT/Apache-2.0 and <= 8B parameters). Fine-tuned here on
the challenge training data only. No external data is looked up: the model
only ever sees the two records it is comparing.

LEAKAGE CONTROL
---------------
* Cross-encoder training pairs come from S1 entities NOT in the seed0
  validation fold (the same fold LightGBM's threshold was chosen on).
* The LightGBM/cross-encoder combination and its threshold are fitted on
  validation predictions, and the go/no-go decision uses a 2-fold estimate
  (fit on one half of the validation entities, score the other half).

COMMANDS (run from code/business_entity_resolution)
---------------------------------------------------
  python src/ce.py build-train            # CPU, ~5 min, safe during inference
  python src/ce.py train --smoke          # GPU, ~2 min: catches setup errors
  python src/ce.py train                  # GPU, ~45 min
  python src/ce.py valpred                # CPU, ~15 min (after inference)
  python src/ce.py score                  # GPU, scores uncertain val+test pairs
  python src/ce.py decide                 # CPU, ~5 min: writes new matches
                                          # only if validation improves
"""
import argparse
import gc
import math
import os
import pickle
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NORM = "norm"
CACHE = "cache"
ART = "artifacts"
OUT = "output"
CE_DIR = f"{ART}/ce_model"
BASE_MODELS = ("cross-encoder/ms-marco-MiniLM-L6-v2",
               "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Countries are DISCOVERED FROM THE DATA, never hard-coded (competition rule):
#   training countries = labels present in train_s1 (here: india, us)
#   test countries     = labels present in test_s1  (here: + france)
# A country with no training data (France here) has no labelled validation
# pairs to fit a blend on; the blend fitted on the training countries is
# applied to it only when it is listed in --countries.


def _distinct_countries(tag):
    t = pq.read_table(f"{NORM}/{tag}_s1.parquet", columns=["country_norm"])
    return sorted(set(t["country_norm"].to_pylist()) - {"", None})


def train_countries():
    return _distinct_countries("train")


def test_countries():
    return _distinct_countries("test")


def _resolve(arg):
    """--countries value -> list. 'all' = every train or test country."""
    if arg in (None, "", "all"):
        return sorted(set(train_countries()) | set(test_countries()))
    return [c.strip() for c in arg.split(",") if c.strip()]
LO, HI = 0.02, 0.98              # LightGBM uncertainty band re-scored by CE
MAX_LEN = 128
GT_PATH = "dataset/train/train_ground_truth.tsv"
VAL_SPLIT = "splits/seed0_val.tsv"


# ---------------------------------------------------------------- helpers
def fmt(name, addr):
    return f"{name or ''} ; {addr or ''}"


def ensure_splits():
    """Create the validation split if missing (it is normally written by the
    pipeline's train stage; reproducing with the shipped models skips that).
    Deterministic: MD5 of the S1 id, so it is identical to the training split."""
    if not os.path.exists(VAL_SPLIT):
        import validation
        for i in range(5):
            validation.build(seed_name=f"seed{i}", val_frac_buckets=range(i * 5, i * 5 + 5),
                             gt_path=GT_PATH)


def load_ids(path):
    ids = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            ids.add(line.split("\t", 1)[0])
    return ids


def s1_text_map(tag, country):
    t = pq.read_table(f"{NORM}/{tag}_s1.parquet",
                      columns=["entity_id", "country_norm", "name_alnum", "addr_street"])
    t = t.filter(pc.equal(t["country_norm"], country))
    return {e: fmt(n, a) for e, n, a in zip(t["entity_id"].to_pylist(),
                                            t["name_alnum"].to_pylist(),
                                            t["addr_street"].to_pylist())}


def corpus_table(tag, country):
    return pq.read_table(f"{CACHE}/{country}_{tag}.parquet",
                         columns=["entity_id", "name_alnum", "addr_street"],
                         memory_map=True)


def corpus_texts(tbl, rows):
    sub = tbl.take(pa.array(np.asarray(rows, dtype=np.int64)))
    return [fmt(n, a) for n, a in zip(sub["name_alnum"].to_pylist(),
                                      sub["addr_street"].to_pylist())]


def load_model(path_or_names):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    names = [path_or_names] if isinstance(path_or_names, str) else list(path_or_names)
    last = None
    for nm in names:
        try:
            tok = AutoTokenizer.from_pretrained(nm)
            model = AutoModelForSequenceClassification.from_pretrained(nm, num_labels=1)
            print(f"  loaded model: {nm}", flush=True)
            return tok, model.to("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as e:  # try the next name
            last = e
            print(f"  could not load {nm}: {type(e).__name__}: {e}", flush=True)
    raise RuntimeError(f"no model could be loaded: {last}")


def score_texts(model, tok, A, B, batch=1024):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = np.empty(len(A), dtype=np.float32)
    order = np.argsort(np.fromiter((len(a) + len(b) for a, b in zip(A, B)),
                                   dtype=np.int64, count=len(A)))
    with torch.no_grad():
        for s in range(0, len(A), batch):
            ii = order[s:s + batch]
            enc = tok([A[i] for i in ii], [B[i] for i in ii], truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                enabled=dev.type == "cuda"):
                lg = model(**enc).logits.squeeze(-1)
            out[ii] = lg.float().cpu().numpy()
    return out


# ------------------------------------------------------------ build-train
def cmd_build_train(args):
    """Training pairs = reverse-retrieval links (record -> one of its top-3
    S1 owners), labelled from ground truth. These are exactly the hard
    candidates the matcher has to separate. Validation-fold S1 excluded."""
    t0 = time.time()
    ensure_splits()
    val = load_ids(VAL_SPLIT)
    rng = np.random.default_rng(0)
    A, B, Y = [], [], []
    for country in train_countries():
        if not os.path.exists(f"{CACHE}/rev_{country}_train.pkl"):
            print(f"  skip {country}: no reverse cache")
            continue
        d = pickle.load(open(f"{CACHE}/rev_{country}_train.pkl", "rb"))
        s1_ids = d["s1_ids"]
        not_val = np.fromiter((e not in val for e in s1_ids), dtype=bool, count=len(s1_ids))
        idx = np.nonzero(not_val[d["s1"]])[0]
        n = min(args.per_country, len(idx))
        sel = np.sort(rng.choice(idx, size=n, replace=False))
        rec, s1 = d["rec"][sel], d["s1"][sel]
        del d, idx
        gc.collect()
        need = {s1_ids[j] for j in np.unique(s1)}
        gt = {}
        with open(GT_PATH, encoding="utf-8") as f:
            next(f)
            for line in f:
                a, _, b = line.rstrip("\n").partition("\t")
                if a in need:
                    gt[a] = set(x for x in b.split(",") if x)
        tbl = corpus_table("train", country)
        rec_ids = tbl["entity_id"].take(pa.array(rec.astype(np.int64))).to_pylist()
        s1map = s1_text_map("train", country)
        for k in range(len(sel)):
            e1 = s1_ids[s1[k]]
            A.append(s1map[e1])
            Y.append(1 if rec_ids[k] in gt.get(e1, ()) else 0)
        B.extend(corpus_texts(tbl, rec))
        print(f"  {country}: {n:,} pairs, positive rate "
              f"{np.mean(Y[-n:]):.3f} ({time.time()-t0:.0f}s)", flush=True)
        del s1map, gt, rec_ids, tbl
        gc.collect()
    pq.write_table(pa.table({"a": A, "b": B, "label": np.asarray(Y, dtype=np.int8)}),
                   f"{CACHE}/ce_train.parquet")
    print(f"wrote {CACHE}/ce_train.parquet: {len(Y):,} pairs ({time.time()-t0:.0f}s)")


# ------------------------------------------------------------------ train
def cmd_train(args):
    import torch
    torch.manual_seed(0)          # dropout / data order -> repeatable runs
    torch.cuda.manual_seed_all(0)
    t0 = time.time()
    d = pq.read_table(f"{CACHE}/ce_train.parquet").to_pydict()
    A, B = d["a"], d["b"]
    Y = np.asarray(d["label"], dtype=np.float32)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Y))
    hold, tr = perm[:20000], perm[20000:]
    if args.smoke:
        tr, hold = tr[:args.batch * 40], hold[:4000]
    tok, model = load_model(args.model or BASE_MODELS)
    dev = next(model.parameters()).device
    print(f"  device: {dev}  train pairs: {len(tr):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(tr) / args.batch)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.02, (steps - s) / max(1, steps - warm)))
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    tl = time.time()
    run = 0.0
    for st in range(steps):
        ii = tr[st * args.batch:(st + 1) * args.batch]
        enc = tok([A[i] for i in ii], [B[i] for i in ii], truncation=True,
                  max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        yb = torch.from_numpy(Y[ii]).to(dev)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                            enabled=dev.type == "cuda"):
            logits = model(**enc).logits.squeeze(-1)
        loss = lossf(logits.float(), yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * float(loss) if st else float(loss)
        if st % 100 == 0 or st == steps - 1:
            el = time.time() - tl
            rate = (st + 1) * args.batch / max(el, 1e-6)
            eta = (steps - st - 1) * args.batch / max(rate, 1e-6) / 60
            print(f"    step {st+1:,}/{steps:,} loss {run:.4f}  {rate:,.0f} pairs/s  "
                  f"eta {eta:.1f} min", flush=True)
        if (time.time() - tl) / 60 > args.max_minutes:
            print(f"  time cap of {args.max_minutes} min reached at step {st+1:,}; stopping",
                  flush=True)
            break
    os.makedirs(CE_DIR, exist_ok=True)
    model.save_pretrained(CE_DIR)
    tok.save_pretrained(CE_DIR)
    s = score_texts(model, tok, [A[i] for i in hold], [B[i] for i in hold])
    yh = Y[hold]
    acc = float(((s > 0) == (yh > 0.5)).mean())
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yh, s)
    except Exception:
        auc = float("nan")
    t_s = time.time()
    _ = score_texts(model, tok, [A[i] for i in hold[:4000]], [B[i] for i in hold[:4000]])
    sps = 4000 / max(time.time() - t_s, 1e-6)
    print(f"  holdout: AUC {auc:.4f}  accuracy {acc:.4f}  | scoring speed {sps:,.0f} pairs/s")
    print(f"saved {CE_DIR} ({(time.time()-t0)/60:.1f} min)"
          + ("   [SMOKE TEST ONLY - run again without --smoke]" if args.smoke else ""))


# ---------------------------------------------------------------- valpred
def cmd_valpred(args):
    """LightGBM predictions for held-out validation entities, saved exactly
    like inference saves test predictions."""
    import pipeline as P
    import model as M
    art = pickle.load(open(f"{ART}/matcher.pkl", "rb"))
    gt = P.load_gt(GT_PATH)
    ensure_splits()
    val_ids = set(P.load_gt(VAL_SPLIT).keys())
    for country in train_countries():
        t0 = time.time()
        print(f"\n=== valpred {country} ===", flush=True)
        c = P.build_country_corpus(country, f"{NORM}/train_s2.parquet",
                                   f"{NORM}/train_s3.parquet", "train")
        rev = P.get_reverse(country, "train", f"{NORM}/train_s1.parquet", c, args.workers)
        ncount = P.s1_name_counts(f"{NORM}/train_s1.parquet", country)
        BI = P.attach_id_pos(P.BlockingIndexes(c, verbose=False))
        ctext = P.corpus_text_views(c)
        twin = P.corpus_twin_counts(ctext)
        qs = P.load_queries(f"{NORM}/train_s1.parquet", country,
                            keep_ids=val_ids, limit=args.val_queries)
        qpos = {q[0]: i for i, q in enumerate(qs)}
        idx, X, y, _ = P.generate_pairs(qs, BI, ctext, BI.ids, gt=gt, rev=rev,
                                        ncount=ncount, twin=twin, progress_every=0)
        p = M.predict(art["model"], X)
        qa = np.fromiter((qpos[a] for a, _ in idx), np.int32, count=len(idx))
        ca = np.fromiter((BI.id_pos[b] for _, b in idx), np.int32, count=len(idx))
        truth = [np.fromiter((BI.id_pos[t] for t in gt.get(q[0], ()) if t in BI.id_pos),
                             np.int32) for q in qs]
        np.savez(f"{CACHE}/valpred_{country}.npz", q=qa, c=ca,
                 p=np.asarray(p, np.float32), y=np.asarray(y, np.int8))
        with open(f"{CACHE}/valpred_{country}_meta.pkl", "wb") as f:
            pickle.dump(dict(qids=[q[0] for q in qs], truth=truth), f)
        print(f"  {len(qs):,} entities, {len(idx):,} pairs ({time.time()-t0:.0f}s)", flush=True)
        del BI, ctext, c, rev, twin, X
        gc.collect()


# ------------------------------------------------------------------ score
def cmd_score(args):
    t0 = time.time()
    if args.fake:
        print("  FAKE scorer (plumbing test only)")
    else:
        tok, model = load_model(CE_DIR)
    lo_b, hi_b = args.lo, args.hi
    jobs = []
    for country in _resolve(args.countries):
        if os.path.exists(f"{CACHE}/valpred_{country}.npz"):
            jobs.append(("val", "train", country, f"{CACHE}/valpred_{country}.npz",
                         f"{CACHE}/valpred_{country}_meta.pkl"))
        if os.path.exists(f"{CACHE}/testpred_{country}.npz"):
            jobs.append(("test", "test", country, f"{CACHE}/testpred_{country}.npz",
                         f"{CACHE}/testpred_{country}_qids.pkl"))
    for kind, tag, country, npz, meta in jobs:
        d = np.load(npz)
        p = d["p"]
        out_path = f"{CACHE}/ce_{kind}_{country}.npy"
        ce = np.full(len(p), np.nan, dtype=np.float32)
        if os.path.exists(out_path):
            old_ce = np.load(out_path)
            if len(old_ce) == len(p):
                ce = old_ce            # reuse: only score pairs not scored yet
        in_band = (p >= lo_b) & (p <= hi_b)
        rows = np.nonzero(in_band & np.isnan(ce))[0]
        m = pickle.load(open(meta, "rb"))
        qids = m["qids"] if isinstance(m, dict) else m
        print(f"  {kind} {country}: band [{lo_b},{hi_b}] has {int(in_band.sum()):,} of "
              f"{len(p):,} pairs; {len(rows):,} still to score", flush=True)
        s1map = s1_text_map(tag, country)
        tbl = corpus_table(tag, country)
        CH = 400_000
        for lo in range(0, len(rows), CH):
            r = rows[lo:lo + CH]
            if args.fake:
                rs = np.random.default_rng(lo)
                z = np.log(p[r] / (1 - p[r]))
                if "y" in d.files:
                    z = z + 2.0 * (d["y"][r] * 2 - 1)
                ce[r] = z + rs.normal(0, 1.0, len(r))
            else:
                A = [s1map[qids[q]] for q in d["q"][r]]
                B = corpus_texts(tbl, d["c"][r])
                ce[r] = score_texts(model, tok, A, B)
            print(f"    {min(lo+CH, len(rows)):,}/{len(rows):,} ({time.time()-t0:.0f}s)",
                  flush=True)
        np.save(out_path, ce)
        del s1map, tbl
        gc.collect()
    print(f"scoring done ({(time.time()-t0)/60:.1f} min)")


# ----------------------------------------------------------------- decide
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _combine(p, ce, coef):
    s = p.astype(np.float64).copy()
    m = ~np.isnan(ce)
    if m.any():
        z = coef[0] * _logit(p[m]) + coef[1] * ce[m] + coef[2]
        s[m] = 1 / (1 + np.exp(-z))
    return s


def _fit(p, ce, y):
    from sklearn.linear_model import LogisticRegression
    m = ~np.isnan(ce)
    Xf = np.column_stack([_logit(p[m]), ce[m]])
    lr = LogisticRegression(C=10.0, max_iter=500).fit(Xf, y[m])
    return (float(lr.coef_[0][0]), float(lr.coef_[0][1]), float(lr.intercept_[0]))


def _select(s, index, th, uni):
    import model as M
    return M.apply_global_uniqueness(s, index, th) if uni else M.assemble(s, index, th, None)


def _tune(s, index, truth):
    """Best (threshold, uniqueness) by macro F0.5 on the given entities."""
    from metrics import macro_f05
    best = (-1.0, 0.5, False)
    for uni in (False, True):
        for th in np.arange(0.30, 0.96, 0.025):
            sel = _select(s, index, th, uni)
            f = macro_f05(truth, {k: sel.get(k, set()) for k in truth})["macro_f05"]
            if f > best[0]:
                best = (f, float(th), uni)
    return best


def _band(p, ce, lo_b, hi_b):
    """Use cross-encoder scores only inside [lo_b, hi_b] of the LightGBM
    probability, so any band <= the scored one can be reproduced exactly."""
    ce = ce.copy()
    ce[(p < lo_b) | (p > hi_b)] = np.nan
    return ce


def cmd_decide(args):
    from metrics import macro_f05
    import pipeline as P
    t0 = time.time()
    ce_countries = set(_resolve(args.countries))
    art = pickle.load(open(f"{ART}/matcher.pkl", "rb"))
    base_th, base_mg, base_uni = art["threshold"], art["margin"], art["use_uniqueness"]

    # ---- validation arrays (keys offset per country so they never collide)
    P_, C_, Y_, idx_, truth = [], [], [], [], {}
    for ci, country in enumerate(train_countries()):
        if not os.path.exists(f"{CACHE}/valpred_{country}.npz"):
            continue
        d = np.load(f"{CACHE}/valpred_{country}.npz")
        m = pickle.load(open(f"{CACHE}/valpred_{country}_meta.pkl", "rb"))
        ce = _band(d["p"], np.load(f"{CACHE}/ce_val_{country}.npy"), args.lo, args.hi)
        off = (ci + 1) * 100_000_000
        P_.append(d["p"]); C_.append(ce); Y_.append(d["y"])
        idx_.extend(zip((d["q"] + off).tolist(), (d["c"] + off).tolist()))
        for qi, t in enumerate(m["truth"]):
            truth[qi + off] = set((t + off).tolist())
    p = np.concatenate(P_).astype(np.float64)
    ce = np.concatenate(C_)
    y = np.concatenate(Y_)
    print(f"  validation: {len(truth):,} entities, {len(p):,} pairs, "
          f"{int((~np.isnan(ce)).sum()):,} re-scored by the cross-encoder")

    # ---- 2-fold estimate: fit/tune on one half of the entities, score the other
    fold = {k: (hash_int(k) & 1) for k in truth}
    pf = np.fromiter((fold[a] for a, _ in idx_), np.int8, count=len(idx_))
    res = {"lightgbm": [], "lightgbm+ce": []}
    for f in (0, 1):
        tr_m, te_m = pf != f, pf == f
        tr_idx = [idx_[i] for i in np.nonzero(tr_m)[0]]
        te_idx = [idx_[i] for i in np.nonzero(te_m)[0]]
        tr_truth = {k: v for k, v in truth.items() if fold[k] != f}
        te_truth = {k: v for k, v in truth.items() if fold[k] == f}
        # LightGBM alone
        _, th, uni = _tune(p[tr_m], tr_idx, tr_truth)
        sel = _select(p[te_m], te_idx, th, uni)
        res["lightgbm"].append(macro_f05(te_truth, {k: sel.get(k, set()) for k in te_truth})["macro_f05"])
        # LightGBM + cross-encoder
        coef = _fit(p[tr_m], ce[tr_m], y[tr_m])
        s_tr = _combine(p[tr_m], ce[tr_m], coef)
        _, th, uni = _tune(s_tr, tr_idx, tr_truth)
        sel = _select(_combine(p[te_m], ce[te_m], coef), te_idx, th, uni)
        res["lightgbm+ce"].append(macro_f05(te_truth, {k: sel.get(k, set()) for k in te_truth})["macro_f05"])
        print(f"  fold {f}: lightgbm {res['lightgbm'][-1]:.5f} | "
              f"lightgbm+ce {res['lightgbm+ce'][-1]:.5f}", flush=True)
    base_cv, ce_cv = np.mean(res["lightgbm"]), np.mean(res["lightgbm+ce"])
    print(f"\n  HELD-OUT macro F0.5:  LightGBM alone {base_cv:.5f}  |  "
          f"LightGBM + cross-encoder {ce_cv:.5f}  ({ce_cv-base_cv:+.5f})")
    if ce_cv <= base_cv + 0.0005 and not args.force:
        print("  -> no meaningful gain: KEEPING the current matching_results.tsv")
        return

    coef = _fit(p, ce, y)
    s_all = _combine(p, ce, coef)
    best_f, th, uni = _tune(s_all, idx_, truth)
    print(f"  final combination coef={tuple(round(x, 3) for x in coef)} "
          f"threshold={th:.3f} uniqueness={uni} (in-sample {best_f:.5f})")

    # ---- apply to test
    mapping = {}
    for country in P._countries(f"{NORM}/test_s1.parquet"):
        d = np.load(f"{CACHE}/testpred_{country}.npz")
        qids = pickle.load(open(f"{CACHE}/testpred_{country}_qids.pkl", "rb"))
        cids = pq.read_table(f"{CACHE}/{country}_test.parquet",
                             columns=["entity_id"])["entity_id"].to_pylist()
        index = list(zip(d["q"].tolist(), d["c"].tolist()))
        if country in ce_countries and os.path.exists(f"{CACHE}/ce_test_{country}.npy"):
            s = _combine(d["p"], _band(d["p"], np.load(f"{CACHE}/ce_test_{country}.npy"),
                                       args.lo, args.hi), coef)
            sel = _select(s, index, th, uni)
            how = "LightGBM + cross-encoder"
        else:
            import model as M
            s = d["p"].astype(np.float64)
            sel = (M.apply_global_uniqueness(s, index, base_th) if base_uni
                   else M.assemble(s, index, base_th, base_mg))
            how = "LightGBM (unchanged)"
        n_m = 0
        for qi, e in enumerate(qids):
            got = {cids[c] for c in sel.get(qi, ())}
            mapping[e] = got
            n_m += len(got)
        print(f"  test {country}: {len(qids):,} entities, {n_m:,} matches  [{how}]")
    dst = args.out
    default = f"{OUT}/matching_results.tsv"
    if os.path.normpath(dst) == os.path.normpath(default) and os.path.exists(default) \
            and not os.path.exists(f"{OUT}/matching_results_lightgbm_only.tsv"):
        os.replace(default, f"{OUT}/matching_results_lightgbm_only.tsv")
    P.write_tsv(dst, "matched_entity_ids", mapping, f"{NORM}/test_s1.parquet")
    print(f"\nwrote {dst} ({time.time()-t0:.0f}s)")


# ============================================================ STAGE 2
# Corroboration at the TEXT level. Measured on training data: negatives are
# one-off perturbations (0.8% have a twin), true matches arrive as several
# consistent copies (38.2%). A borderline record is therefore compared not
# only with the S1 entity but with that entity's CONFIDENT matches (anchors),
# using the cross-encoder. A small second-stage model then decides with the
# whole candidate group as context.
ANCHOR_MIN = 0.90      # blended score for a candidate to act as an anchor
N_ANCHORS = 3


def _val_coef(lo_b, hi_b):
    P_, C_, Y_ = [], [], []
    for country in train_countries():
        if not os.path.exists(f"{CACHE}/valpred_{country}.npz"):
            continue
        d = np.load(f"{CACHE}/valpred_{country}.npz")
        P_.append(d["p"]); Y_.append(d["y"])
        C_.append(_band(d["p"], np.load(f"{CACHE}/ce_val_{country}.npy"), lo_b, hi_b))
    return _fit(np.concatenate(P_), np.concatenate(C_), np.concatenate(Y_))


def _groups(q, s):
    """order pairs by (query, score desc); return order, group starts, ends."""
    order = np.lexsort((-s, q))
    qs = q[order]
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    ends = np.r_[starts[1:], len(qs)]
    return order, starts, ends


def _jobs():
    out = []
    for country in sorted(set(train_countries()) | set(test_countries())):
        if os.path.exists(f"{CACHE}/valpred_{country}.npz"):
            out.append(("val", "train", country, f"{CACHE}/valpred_{country}.npz"))
        if os.path.exists(f"{CACHE}/testpred_{country}.npz") and \
                os.path.exists(f"{CACHE}/ce_test_{country}.npy"):
            out.append(("test", "test", country, f"{CACHE}/testpred_{country}.npz"))
    return out


def cmd_anchors(args):
    t0 = time.time()
    coef = _val_coef(args.lo, args.hi)
    print(f"  blend coef {tuple(round(x, 3) for x in coef)}", flush=True)
    if not args.fake:
        tok, model = load_model(CE_DIR)
    for kind, tag, country, npz in _jobs():
        d = np.load(npz)
        p, q, c = d["p"], d["q"], d["c"]
        ce = _band(p, np.load(f"{CACHE}/ce_{kind}_{country}.npy"), args.lo, args.hi)
        s = _combine(p, ce, coef)
        order, starts, ends = _groups(q, s)
        tx, ax = [], []
        for a, b in zip(starts.tolist(), ends.tolist()):
            g = order[a:b]
            anchors = [i for i in g[:args.n_anchors + 1] if s[i] >= ANCHOR_MIN]
            if not anchors:
                continue
            for i in g:
                if args.t_lo <= s[i] <= args.t_hi:
                    k = 0
                    for an in anchors:
                        if an != i and k < args.n_anchors:
                            tx.append(i); ax.append(an); k += 1
        tx = np.asarray(tx, dtype=np.int64)
        ax = np.asarray(ax, dtype=np.int64)
        print(f"  {kind} {country}: {len(np.unique(tx)):,} uncertain candidates, "
              f"{len(tx):,} record-vs-anchor comparisons", flush=True)
        sc = np.empty(len(tx), dtype=np.float32)
        tbl = corpus_table(tag, country)
        CH = 400_000
        for lo in range(0, len(tx), CH):
            t_, a_ = tx[lo:lo + CH], ax[lo:lo + CH]
            if args.fake:
                z = _logit(s[t_]) + np.random.default_rng(lo).normal(0, 1, len(t_))
                if "y" in d.files:
                    z = z + 1.5 * (d["y"][t_] * 2 - 1)
                sc[lo:lo + CH] = z
            else:
                sc[lo:lo + CH] = score_texts(model, tok, corpus_texts(tbl, c[t_]),
                                             corpus_texts(tbl, c[a_]))
            print(f"    {min(lo+CH, len(tx)):,}/{len(tx):,} ({time.time()-t0:.0f}s)", flush=True)
        amax = np.full(len(p), np.nan, np.float32)
        asum = np.zeros(len(p), np.float32)
        an = np.zeros(len(p), np.int8)
        if len(tx):
            np.fmax.at(amax, tx, sc)
            np.add.at(asum, tx, sc)
            np.add.at(an, tx, 1)
        np.savez(f"{CACHE}/anc_{kind}_{country}.npz", amax=amax, amean=asum / np.maximum(an, 1), n=an)
        del tbl
        gc.collect()
    print(f"anchors done ({(time.time()-t0)/60:.1f} min)")


def _s2_features(p, ce, s, q, anc):
    order, starts, ends = _groups(q, s)
    n = len(p)
    rank = np.empty(n, np.float32)
    gmax = np.empty(n, np.float32)
    gsize = np.empty(n, np.float32)
    so = s[order]
    lens = (ends - starts)
    gid = np.repeat(np.arange(len(starts)), lens)
    pos = np.arange(n) - np.repeat(starts, lens)
    rank[order] = pos
    gmax[order] = np.repeat(so[starts], lens)
    gsize[order] = np.repeat(lens, lens)
    gsum = np.add.reduceat(so, starts)
    g05 = np.add.reduceat((so > 0.5).astype(np.float64), starts)
    gs = np.empty(n, np.float32); gs[order] = np.repeat(gsum, lens)
    gc5 = np.empty(n, np.float32); gc5[order] = np.repeat(g05, lens)
    # second-best in group (for a margin feature)
    sec = np.where(lens > 1, so[np.minimum(starts + 1, n - 1)], 0.0)
    g2 = np.empty(n, np.float32); g2[order] = np.repeat(sec, lens)
    has_ce = ~np.isnan(ce)
    has_a = ~np.isnan(anc["amax"])
    return np.column_stack([
        _logit(p), np.where(has_ce, ce, 0.0), has_ce, s, gmax, s - gmax, g2, rank,
        gsize, gs, gc5, np.where(has_a, anc["amax"], 0.0), np.where(has_a, anc["amean"], 0.0),
        anc["n"], has_a]).astype(np.float32)


def cmd_stage2(args):
    import lightgbm as lgb
    from metrics import macro_f05
    import pipeline as P
    t0 = time.time()
    coef = _val_coef(args.lo, args.hi)
    X_, y_, s_, idx_, truth = [], [], [], [], {}
    for ci, country in enumerate(train_countries()):
        if not os.path.exists(f"{CACHE}/valpred_{country}.npz"):
            continue
        d = np.load(f"{CACHE}/valpred_{country}.npz")
        m = pickle.load(open(f"{CACHE}/valpred_{country}_meta.pkl", "rb"))
        ce = _band(d["p"], np.load(f"{CACHE}/ce_val_{country}.npy"), args.lo, args.hi)
        s = _combine(d["p"], ce, coef)
        anc = np.load(f"{CACHE}/anc_val_{country}.npz")
        off = (ci + 1) * 100_000_000
        X_.append(_s2_features(d["p"], ce, s, d["q"], anc))
        y_.append(d["y"]); s_.append(s)
        idx_.extend(zip((d["q"] + off).tolist(), (d["c"] + off).tolist()))
        for qi, t in enumerate(m["truth"]):
            truth[qi + off] = set((t + off).tolist())
    X = np.vstack(X_); y = np.concatenate(y_); s_blend = np.concatenate(s_)
    fold = {k: (hash_int(k) & 1) for k in truth}
    pf = np.fromiter((fold[a] for a, _ in idx_), np.int8, count=len(idx_))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=0, deterministic=True, num_threads=-1)
    oof = np.zeros(len(y))
    for f in (0, 1):
        m_ = lgb.train(params, lgb.Dataset(X[pf != f], y[pf != f]), num_boost_round=300)
        oof[pf == f] = m_.predict(X[pf == f])
    res = {"blend (v6)": [], "stage2": []}
    for f in (0, 1):
        tr_m, te_m = pf != f, pf == f
        tr_idx = [idx_[i] for i in np.nonzero(tr_m)[0]]
        te_idx = [idx_[i] for i in np.nonzero(te_m)[0]]
        tr_t = {k: v for k, v in truth.items() if fold[k] != f}
        te_t = {k: v for k, v in truth.items() if fold[k] == f}
        for name, sc in (("blend (v6)", s_blend), ("stage2", oof)):
            _, th, uni = _tune(sc[tr_m], tr_idx, tr_t)
            sel = _select(sc[te_m], te_idx, th, uni)
            res[name].append(macro_f05(te_t, {k: sel.get(k, set()) for k in te_t})["macro_f05"])
        print(f"  fold {f}: blend {res['blend (v6)'][-1]:.5f} | stage2 {res['stage2'][-1]:.5f}",
              flush=True)
    b_cv, s_cv = np.mean(res["blend (v6)"]), np.mean(res["stage2"])
    print(f"\n  HELD-OUT macro F0.5:  v6 blend {b_cv:.5f}  |  stage-2 {s_cv:.5f}  ({s_cv-b_cv:+.5f})")
    if s_cv <= b_cv + 0.001 and not args.force:
        print("  -> no meaningful gain: nothing written")
        return
    final = lgb.train(params, lgb.Dataset(X, y), num_boost_round=300)
    _, th, uni = _tune(oof, idx_, truth)
    print(f"  stage-2 threshold={th:.3f} uniqueness={uni}")

    # base decisions for every entity (France etc. come from here)
    mapping = {}
    with open(args.base, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            mapping[a] = set(x for x in b.split(",") if x)
    for country in _resolve(args.countries):
        if not (os.path.exists(f"{CACHE}/testpred_{country}.npz")
                and os.path.exists(f"{CACHE}/ce_test_{country}.npy")):
            continue
        d = np.load(f"{CACHE}/testpred_{country}.npz")
        qids = pickle.load(open(f"{CACHE}/testpred_{country}_qids.pkl", "rb"))
        cids = pq.read_table(f"{CACHE}/{country}_test.parquet", columns=["entity_id"])["entity_id"].to_pylist()
        ce = _band(d["p"], np.load(f"{CACHE}/ce_test_{country}.npy"), args.lo, args.hi)
        s = _combine(d["p"], ce, coef)
        anc = np.load(f"{CACHE}/anc_test_{country}.npz")
        pr = final.predict(_s2_features(d["p"], ce, s, d["q"], anc))
        sel = _select(pr, list(zip(d["q"].tolist(), d["c"].tolist())), th, uni)
        n_m = 0
        for qi, e in enumerate(qids):
            mapping[e] = {cids[c_] for c_ in sel.get(qi, ())}
            n_m += len(mapping[e])
        print(f"  test {country}: {len(qids):,} entities, {n_m:,} matches  [stage-2]")
    P.write_tsv(args.out, "matched_entity_ids", mapping, f"{NORM}/test_s1.parquet")
    print(f"\nwrote {args.out}  (other countries copied from {args.base}) ({time.time()-t0:.0f}s)")


def hash_int(k):
    return (k * 2654435761) >> 7


# ------------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-train", "train", "valpred", "score", "decide",
                                    "anchors", "stage2"])
    ap.add_argument("--per-country", type=int, default=750_000)
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--max-minutes", type=float, default=45)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--val-queries", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--lo", type=float, default=LO)
    ap.add_argument("--hi", type=float, default=HI)
    ap.add_argument("--countries", default="all",
                    help="score: countries to score; decide: countries whose test "
                         "matches use the cross-encoder blend")
    ap.add_argument("--out", default=f"{OUT}/matching_results.tsv")
    ap.add_argument("--base", default=f"{OUT}/matching_results.tsv",
                    help="stage2: file whose decisions are kept for non-CE countries")
    ap.add_argument("--n-anchors", dest="n_anchors", type=int, default=N_ANCHORS)
    ap.add_argument("--t-lo", dest="t_lo", type=float, default=0.02)
    ap.add_argument("--t-hi", dest="t_hi", type=float, default=0.98)
    ap.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    {"build-train": cmd_build_train, "train": cmd_train, "valpred": cmd_valpred,
     "score": cmd_score, "decide": cmd_decide, "anchors": cmd_anchors,
     "stage2": cmd_stage2}[a.cmd](a)
