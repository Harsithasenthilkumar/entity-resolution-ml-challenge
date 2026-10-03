"""
indexing.py — memory-compact, scalable retrieval structures.

Two primitives, both designed to hold multi-million-row corpora inside a
small RAM budget and to remain valid designs at billion scale:

  HashBlock      exact-key blocking without a Python dict. Keys are hashed to
                 int64 and stored in a SORTED numpy array alongside an int32
                 payload of record ids; lookup is np.searchsorted (O(log n),
                 ~48 bits/entry). A Python dict over 4M string keys cost ~400MB
                 and OOM-killed the 4GB sandbox; this costs ~48MB. 64-bit hash
                 collisions between distinct keys are vanishingly rare and, if
                 they occur, merely inject one extra candidate that the
                 downstream matcher rejects -- they cannot lose a true match.

  InvertedIndex  Sparkly-style (Paulsen et al., PVLDB 2023) top-k TF/IDF
                 retrieval over whitespace tokens, in CSR layout. Sparkly's key
                 empirical results drive two design choices: (1) top-k
                 retrieval rather than threshold blocking; (2) cosine over
                 tf-idf vectors, applying IDF to BOTH query and document sides,
                 which they show beats BM25 for entity matching (BM25 weights
                 only the document side).
                 Built in two streaming passes over the text with NO
                 per-document Python token list retained -- materialising one
                 cost ~4GB at 4M docs.

Query cost is bounded by MAX_DF and by max_query_tokens, so it is independent
of corpus size; index build is a single linear scan. Both shard cleanly by
country, which is the natural partition key here (the audit found country
agrees on 100% of true training pairs).
"""
import numpy as np
import time, gc

_MASK64 = (1 << 63) - 1


def hash_key(s):
    """Stable 63-bit hash of a string (Python's hash() is salted per process)."""
    h = 1469598103934665603
    for ch in s.encode("utf-8", "ignore"):
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h & _MASK64


def hash_keys(strings):
    return np.fromiter((hash_key(s) for s in strings), dtype=np.int64,
                       count=len(strings))


class HashBlock:
    """Exact-key blocking index: hashed key -> record ids, numpy-only."""

    def build(self, keys, valid_mask=None):
        n = len(keys)
        idx = np.arange(n, dtype=np.int32)
        h = np.empty(n, dtype=np.int64)
        keep = np.zeros(n, dtype=bool)
        for i, k in enumerate(keys):
            if k:
                h[i] = hash_key(k)
                keep[i] = True
        if valid_mask is not None:
            keep &= valid_mask
        h = h[keep]
        idx = idx[keep]
        order = np.argsort(h, kind="stable")
        self.h = h[order]
        self.idx = idx[order]
        return self

    def get(self, key):
        if not key:
            return np.empty(0, np.int32)
        hv = hash_key(key)
        lo = np.searchsorted(self.h, hv, "left")
        hi = np.searchsorted(self.h, hv, "right")
        return self.idx[lo:hi]


def word_tokens(txt):
    return set(txt.split(" ")) - {""}


class CharNGrams:
    """Picklable character n-gram tokenizer (a closure cannot be pickled, and
    indexes must be cacheable to disk to survive process restarts)."""
    __slots__ = ("n",)
    def __init__(self, n): self.n = n
    def __call__(self, txt):
        s = txt.replace(" ", "")
        n = self.n
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i + n] for i in range(len(s) - n + 1)}
    def __reduce__(self): return (CharNGrams, (self.n,))


def make_char_ngrams(n):
    """Character n-grams over the whitespace-stripped string.

    Rationale (measured, not assumed): the India validation fold showed
    word-token name retrieval reaching only 0.227 recall while address
    retrieval alone reached 0.482. The cause is transliteration -- an S1 name
    'swastik solutions' appears in S2/S3 as Telugu script, which Unidecode
    folds to 'svstik solyushns'. That shares ZERO whole tokens with the S1
    name but still shares character n-grams ('sti', 'tik', ...). Character
    n-grams are therefore the correct retrieval unit for this corpus.
    """
    return CharNGrams(n)


class InvertedIndex:
    def __init__(self, max_df_frac=0.001, max_df_abs=50000, tokenizer=None):
        self.max_df_frac = max_df_frac
        self.max_df_abs = max_df_abs
        self.tokenizer = tokenizer or word_tokens

    def build(self, texts, verbose=True):
        t0 = time.time()
        n_docs = len(texts)

        # ---- pass 1: vocabulary + document frequency ----
        vocab = {}
        df_list = []
        tokenize = self.tokenizer
        for txt in texts:
            if not txt:
                continue
            for tok in tokenize(txt):
                tid = vocab.get(tok, -1)
                if tid < 0:
                    vocab[tok] = len(df_list)
                    df_list.append(1)
                else:
                    df_list[tid] += 1
        df = np.asarray(df_list, dtype=np.int64)
        del df_list
        n_vocab = len(vocab)
        idf = np.log(1.0 + n_docs / (1.0 + df)).astype(np.float32)
        df_cap = min(self.max_df_abs, max(1, int(self.max_df_frac * n_docs)))
        indexable = df <= df_cap
        if verbose:
            print(f"      vocab={n_vocab:,} df_cap={df_cap} "
                  f"indexable={int(indexable.sum()):,} ({time.time()-t0:.0f}s)")

        # ---- pass 2: posting counts + document norms ----
        counts = np.zeros(n_vocab + 1, dtype=np.int64)
        doc_norm = np.ones(n_docs, dtype=np.float32)
        for d, txt in enumerate(texts):
            if not txt:
                continue
            sq = 0.0
            for tok in tokenize(txt):
                tid = vocab.get(tok, -1)
                if tid < 0:
                    continue
                w = idf[tid]
                sq += w * w
                if indexable[tid]:
                    counts[tid + 1] += 1
            if sq > 0:
                doc_norm[d] = np.sqrt(sq)
        indptr = np.cumsum(counts)
        del counts
        total_post = int(indptr[-1])
        postings = np.empty(total_post, dtype=np.int32)
        cursor = indptr[:-1].copy()

        # ---- pass 3: fill posting lists ----
        for d, txt in enumerate(texts):
            if not txt:
                continue
            for tok in tokenize(txt):
                tid = vocab.get(tok, -1)
                if tid >= 0 and indexable[tid]:
                    postings[cursor[tid]] = d
                    cursor[tid] += 1
        del cursor

        self.vocab = vocab
        self.df = df
        self.idf = idf
        self.indexable = indexable
        self.indptr = indptr
        self.postings = postings
        self.doc_norm = doc_norm
        self.n_docs = n_docs
        self.df_cap = df_cap
        gc.collect()
        if verbose:
            print(f"      postings={total_post:,} ({time.time()-t0:.0f}s)")
        return self

    def query_topk(self, text, k=10, max_query_tokens=8):
        if not text:
            return np.empty(0, np.int32), np.empty(0, np.float32)
        uniq_toks = self.tokenizer(text)
        if not uniq_toks:
            return np.empty(0, np.int32), np.empty(0, np.float32)

        tids = []
        qnorm_sq = 0.0
        vocab = self.vocab
        for t in uniq_toks:
            tid = vocab.get(t, -1)
            if tid < 0:
                continue
            w = self.idf[tid]
            qnorm_sq += w * w
            if self.indexable[tid]:
                tids.append(tid)
        if not tids:
            return np.empty(0, np.int32), np.empty(0, np.float32)
        qnorm = np.sqrt(qnorm_sq) if qnorm_sq > 0 else 1.0

        # rarest (most informative) first; ties broken by the token string so
        # the choice never depends on set iteration order (which varies per
        # process because Python salts string hashes) -> reproducible runs
        inv = {tid: t for t in uniq_toks for tid in (vocab.get(t, -1),) if tid >= 0}
        tids.sort(key=lambda t_: (self.df[t_], inv[t_]))
        tids = tids[:max_query_tokens]

        segs, wts, lens = [], [], []
        for tid in tids:
            s, e = self.indptr[tid], self.indptr[tid + 1]
            if e > s:
                segs.append(self.postings[s:e])
                wts.append(float(self.idf[tid]) ** 2)
                lens.append(e - s)
        if not segs:
            return np.empty(0, np.int32), np.empty(0, np.float32)

        docs = np.concatenate(segs)
        w = np.repeat(np.asarray(wts, dtype=np.float32),
                      np.asarray(lens, dtype=np.int64))
        order = np.argsort(docs, kind="stable")
        docs_sorted = docs[order]
        uniq, start_idx = np.unique(docs_sorted, return_index=True)
        acc = np.add.reduceat(w[order], start_idx)
        scores = acc / (qnorm * self.doc_norm[uniq])
        if len(uniq) > k:
            part = np.argpartition(-scores, k)[:k]
            uniq, scores = uniq[part], scores[part]
        srt = np.argsort(-scores)
        return uniq[srt].astype(np.int32), scores[srt].astype(np.float32)
