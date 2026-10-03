"""
blocking.py -- multi-channel candidate generation.

DESIGN, AND THE EVIDENCE BEHIND IT
==================================
Every channel below earned its place in a measured ablation on a 3,000-query
India validation sample against the full 4,133,346-record S2+S3 corpus
(numbers in README.md / docs/METHODOLOGY.md). A true match only has to
be found by ONE channel, so the system is a union-of-retrievers rather than a
single "best" representation.

  A  exact_name     hash on name_alnum                     O(log n) probe
  B  exact_core     hash on name_core (legal suffix off)   O(log n) probe
  C  core_sorted    hash on sorted core tokens             O(log n) probe
                    -> recovers word-order transpositions ("Laa Inc"/"Inc Laa")
  D  name_ngram     top-k TF/IDF over character 5-grams    bounded posting walk
                    -> THE transliteration channel. Word-token name retrieval
                       measured only 0.227 recall because an S1 name
                       "swastik solutions" appears in S2/S3 as Telugu script,
                       which Unidecode folds to "svstik solyushns" -- zero
                       shared whole tokens, but shared character n-grams.
                       Switching to char 5-grams lifted name-only recall to
                       0.580 @ k=50.
  E  addr_word      top-k TF/IDF over address word tokens  bounded posting walk
                    -> the single strongest channel (0.482 recall ALONE at
                       k=5). Many true pairs have an unrelated DBA/trade name
                       ("Guru Logistics Private Limited" -> "Vantageumbra")
                       where no name channel can ever succeed; address is the
                       only way to reach them.
  F  addr_ngram     top-k TF/IDF over address char 5-grams (optional)
                    -> for transliterated/abbreviated address variants.
  G  postal_house   hash on postal_code|house_number       O(log n) probe
                    -> cheap insurance for the hardest pairs: the audit found
                       ~85% of pairs with BOTH low name and low address
                       similarity still share a digit token.

Complexity: no channel ever forms an N x M product. Each is a hash probe
(O(log n) via searchsorted) or a walk over posting lists whose length is
capped by MAX_DF and whose count is capped by max_query_tokens. Total query
cost is therefore independent of corpus size, and index build is a single
linear pass. Everything is partitioned by country first -- the audit found
country agrees on 100% of true training pairs, making it a free, exact,
recall-preserving reduction and the natural shard key for a distributed
deployment.
"""
import time
import gc
from collections import defaultdict

from indexing import InvertedIndex, HashBlock, CharNGrams, word_tokens
from jointtok import JointTok


def sorted_tokens(s):
    return " ".join(sorted(s.split(" "))) if s else ""


# Tuned on validation. See README "Tuning knobs".
DEFAULTS = dict(
    name_ngram_n=5,
    addr_ngram_n=5,
    name_df_frac=0.005,
    addr_df_frac=0.005,
    addr_ngram_df_frac=0.002,
    k_name=50,
    k_addr=50,
    k_addr_ngram=0,
    max_query_tokens_name=12,
    max_query_tokens_addr=10,
    # J: joint name-skeleton + address index. Measured on India validation
    # (3,000 queries, full 4.13M corpus): 0.816 recall @ 15 candidates, vs
    # 0.681 @ 14.4 for the separate-channel pipeline at the same budget.
    joint_df_frac=0.002,
    k_joint=20,
    max_query_tokens_joint=16,
)


class BlockingIndexes:
    """All retrieval structures for one country's S2+S3 corpus. Built once."""

    def __init__(self, corpus, cfg=None, verbose=True, build_addr_ngram=False):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        cfg = self.cfg
        t0 = time.time()
        self.corpus = corpus
        self.n = corpus.n
        self.ids = corpus.tbl["entity_id"].to_pylist()

        na = corpus.text_list("name_alnum")
        self.b_name = HashBlock().build(na)
        del na
        gc.collect()

        nc = corpus.text_list("name_core")
        self.b_core = HashBlock().build(nc)
        self.b_coresort = HashBlock().build([sorted_tokens(x) for x in nc])
        if verbose:
            print(f"    [block] name hashes ({time.time()-t0:.0f}s)")
        self.ix_name = InvertedIndex(
            max_df_frac=cfg["name_df_frac"],
            tokenizer=CharNGrams(cfg["name_ngram_n"]),
        ).build(nc, verbose)
        del nc
        gc.collect()

        po = corpus.tbl["postal_code"].to_pylist()
        ho = corpus.tbl["house_number"].to_pylist()
        self.b_posthouse = HashBlock().build(
            [(p + "|" + h) if (p and h) else "" for p, h in zip(po, ho)]
        )
        del po, ho
        gc.collect()

        ad = corpus.text_list("addr_street")
        self.ix_addr = InvertedIndex(
            max_df_frac=cfg["addr_df_frac"], tokenizer=word_tokens
        ).build(ad, verbose)
        self.ix_addr_ng = None
        if build_addr_ngram or cfg["k_addr_ngram"]:
            self.ix_addr_ng = InvertedIndex(
                max_df_frac=cfg["addr_ngram_df_frac"],
                tokenizer=CharNGrams(cfg["addr_ngram_n"]),
            ).build(ad, verbose)
        del ad
        gc.collect()

        # joint channel J: one index scoring name AND address together.
        # Names in this corpus are heavily reused (exact-core-name alone
        # returns ~18 records per query), so name-only top-k fills with
        # same-name businesses at other addresses and address-only top-k
        # fills with other businesses on the same street. The true match
        # shares BOTH, and only a joint score ranks it above both kinds of
        # distractor.
        nc = corpus.text_list("name_core")
        ad = corpus.text_list("addr_street")
        jt = [a + "|" + b for a, b in zip(nc, ad)]
        del nc, ad
        gc.collect()
        self.ix_joint = InvertedIndex(
            max_df_frac=cfg["joint_df_frac"], tokenizer=JointTok()
        ).build(jt, verbose)
        del jt
        gc.collect()
        if verbose:
            print(f"    [block] ALL indexes ready ({time.time()-t0:.0f}s)")

    # ---------------------------------------------------------------- query
    def retrieve(self, na, nc, ad, ho, po, k_name=None, k_addr=None,
                 k_addr_ng=None, channels="ABCDEFGJ"):
        """One S1 query -> {corpus_idx: evidence dict}.

        The evidence dict is deliberately rich: which channels fired, each
        channel's score, and each channel's rank. Those become meta-blocking
        features in candidate_compression.py and matcher features in
        features.py -- "how many independent retrievers agreed" is one of the
        strongest cheap signals available.
        """
        cfg = self.cfg
        k_name = cfg["k_name"] if k_name is None else k_name
        k_addr = cfg["k_addr"] if k_addr is None else k_addr
        k_addr_ng = cfg["k_addr_ngram"] if k_addr_ng is None else k_addr_ng
        out = defaultdict(dict)

        if "A" in channels and na:
            for i in self.b_name.get(na):
                out[int(i)]["A"] = 1.0
        if "B" in channels and nc:
            for i in self.b_core.get(nc):
                out[int(i)]["B"] = 1.0
        if "C" in channels and nc:
            for i in self.b_coresort.get(sorted_tokens(nc)):
                out[int(i)]["C"] = 1.0
        if "G" in channels and po and ho:
            for i in self.b_posthouse.get(po + "|" + ho):
                out[int(i)]["G"] = 1.0
        if "D" in channels and nc and k_name:
            docs, sc = self.ix_name.query_topk(
                nc, k=k_name, max_query_tokens=cfg["max_query_tokens_name"])
            for r in range(len(docs)):
                e = out[int(docs[r])]
                e["D"] = float(sc[r])
                e["Dr"] = r
        if "E" in channels and ad and k_addr:
            docs, sc = self.ix_addr.query_topk(
                ad, k=k_addr, max_query_tokens=cfg["max_query_tokens_addr"])
            for r in range(len(docs)):
                e = out[int(docs[r])]
                e["E"] = float(sc[r])
                e["Er"] = r
        if "F" in channels and ad and k_addr_ng and self.ix_addr_ng is not None:
            docs, sc = self.ix_addr_ng.query_topk(
                ad, k=k_addr_ng, max_query_tokens=cfg["max_query_tokens_addr"])
            for r in range(len(docs)):
                e = out[int(docs[r])]
                e["F"] = float(sc[r])
                e["Fr"] = r
        if "J" in channels and cfg["k_joint"] and (nc or ad):
            docs, sc = self.ix_joint.query_topk(
                nc + "|" + ad, k=cfg["k_joint"],
                max_query_tokens=cfg["max_query_tokens_joint"])
            for r in range(len(docs)):
                e = out[int(docs[r])]
                e["J"] = float(sc[r])
                e["Jr"] = r
        return out
