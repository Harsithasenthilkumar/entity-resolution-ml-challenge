"""
reverse.py -- REVERSE retrieval: for every noisy S2/S3 record, find which S1
entity it belongs to.

WHY (measured on 20,000 true match records per country, full S1 index):
  The ground truth is strictly bipartite -- no S2/S3 record ever belongs to
  more than one S1 entity (0 violations over 7,638,365 edges). So instead of
  only asking "given S1, which of 4-6M noisy records are its copies?", we
  also ask each record "which of the ~1M CLEAN, deduplicated S1 entities do
  you belong to?". A truncated record such as '#94/26, Bangalore, KA' has few
  tokens, but against a clean index those few tokens are decisive.

                         owner in top-1   top-3   top-10
      India                   0.901       0.935    0.956
      US                      0.952       0.973    0.983

  Keeping each record's top-3 owners gives ~14 candidates per S1 entity
  (3 x records / S1 entities) -- the same budget as the forward pipeline,
  whose candidate recall at that budget was India 0.681 / US 0.864.

SCALE: index build is over S1 only (~1M rows/country); querying is one
bounded posting walk per record, embarrassingly parallel. Workers are
started with the 'spawn' method, which is what Windows uses anyway, so the
behaviour is identical on every OS. Results are cached to disk so train and
infer never recompute them.
"""
import gc
import os
import pickle
import time

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from multiprocessing import get_context

from indexing import InvertedIndex
from phonetic import Tok4

REV_K = 3            # owners kept per record
REV_DF = 0.01        # document-frequency cap for the S1 index
REV_MQT = 20         # rarest query tokens expanded
REV_CAP = 40         # max reverse candidates kept per S1 entity
CHUNK = 50_000

_IX = None


def _init(path):
    global _IX
    with open(path, "rb") as f:
        _IX = pickle.load(f)


def _work(args):
    lo, texts = args
    n = len(texts)
    rec = np.empty(n * REV_K, np.int32)
    s1 = np.empty(n * REV_K, np.int32)
    sc = np.empty(n * REV_K, np.float32)
    rk = np.empty(n * REV_K, np.int8)
    mg = np.empty(n * REV_K, np.float32)
    m = 0
    for i, q in enumerate(texts):
        docs, scs = _IX.query_topk(q, k=REV_K, max_query_tokens=REV_MQT)
        if len(docs) == 0:
            continue
        top = float(scs[0])
        second = float(scs[1]) if len(scs) > 1 else 0.0
        for r in range(len(docs)):
            rec[m] = lo + i
            s1[m] = docs[r]
            sc[m] = scs[r]
            rk[m] = r
            # rank 0: how dominant the best owner is; others: gap to the best
            mg[m] = (top - second) if r == 0 else (float(scs[r]) - top)
            m += 1
    return rec[:m], s1[:m], sc[:m], rk[:m], mg[:m]


def _s1_texts(s1_parquet, country):
    t = pq.read_table(s1_parquet, columns=["entity_id", "country_norm",
                                           "name_core", "addr_street"])
    t = t.filter(pc.equal(t["country_norm"], country))
    ids = t["entity_id"].to_pylist()
    txt = [(a or "") + "|" + (b or "")
           for a, b in zip(t["name_core"].to_pylist(), t["addr_street"].to_pylist())]
    return ids, txt


class ReverseMap:
    """CSR map: S1 entity -> its reverse candidates (corpus row indices)."""

    def __init__(self, s1_ids, rec, s1, sc, rk, mg):
        order = np.lexsort((-sc, s1))          # by S1, best score first
        self.rec, self.s1 = rec[order], s1[order]
        self.sc, self.rk, self.mg = sc[order], rk[order], mg[order]
        counts = np.bincount(self.s1, minlength=len(s1_ids))
        self.indptr = np.concatenate([[0], np.cumsum(counts)])
        self.pos = {e: i for i, e in enumerate(s1_ids)}

    def get(self, s1_eid):
        j = self.pos.get(s1_eid)
        if j is None:
            return ()
        a, b = self.indptr[j], min(self.indptr[j + 1], self.indptr[j] + REV_CAP)
        return zip(self.rec[a:b].tolist(), self.sc[a:b].tolist(),
                   self.rk[a:b].tolist(), self.mg[a:b].tolist())


def get_reverse(country, tag, s1_parquet, corpus, workers, cache_dir="cache"):
    """Load cached reverse results or compute them in parallel."""
    out = f"{cache_dir}/rev_{country}_{tag}.pkl"
    if os.path.exists(out):
        with open(out, "rb") as f:
            d = pickle.load(f)
        print(f"    [reverse] loaded {out}", flush=True)
        return ReverseMap(d["s1_ids"], d["rec"], d["s1"], d["sc"], d["rk"], d["mg"])

    t0 = time.time()
    s1_ids, s1_txt = _s1_texts(s1_parquet, country)
    ix = InvertedIndex(max_df_frac=REV_DF, tokenizer=Tok4()).build(s1_txt, verbose=False)
    del s1_txt
    ix_path = f"{cache_dir}/rev_ix_{country}_{tag}.pkl"
    with open(ix_path, "wb") as f:
        pickle.dump(ix, f, protocol=4)
    del ix
    gc.collect()
    print(f"    [reverse] S1 index over {len(s1_ids):,} ({time.time()-t0:.0f}s)", flush=True)

    nc = corpus.text_list("name_core")
    ad = corpus.text_list("addr_street")
    texts = [a + "|" + b for a, b in zip(nc, ad)]
    del nc, ad
    gc.collect()
    jobs = [(lo, texts[lo:lo + CHUNK]) for lo in range(0, len(texts), CHUNK)]
    del texts
    parts = []
    workers = max(1, int(workers))
    ctx = get_context("spawn")
    with ctx.Pool(workers, initializer=_init, initargs=(ix_path,)) as pool:
        for i, res in enumerate(pool.imap_unordered(_work, jobs)):
            parts.append(res)
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"      reverse {i+1}/{len(jobs)} chunks "
                      f"({time.time()-t0:.0f}s)", flush=True)
    del jobs
    rec = np.concatenate([p[0] for p in parts])
    s1 = np.concatenate([p[1] for p in parts])
    sc = np.concatenate([p[2] for p in parts])
    rk = np.concatenate([p[3] for p in parts])
    mg = np.concatenate([p[4] for p in parts])
    del parts
    with open(out, "wb") as f:
        pickle.dump(dict(s1_ids=s1_ids, rec=rec, s1=s1, sc=sc, rk=rk, mg=mg), f, protocol=4)
    try:
        os.remove(ix_path)
    except OSError:
        pass
    print(f"    [reverse] {len(rec):,} links for {country}/{tag} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ReverseMap(s1_ids, rec, s1, sc, rk, mg)
