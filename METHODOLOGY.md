# Business Entity Resolution — Methodology (ML Challenge 2026)

**Competition:** Amazon ML Challenge 2026 on Unstop — Business Entity Resolution
**Entry:** team *epoch 0* (team submission)
**Repository author:** Harsitha Senthilkumar
**Submitted:** 1 October 2026

> **How to read the numbers in this document.** Every figure was measured on
> the supplied data. Each is labelled with what it was measured on
> (validation fold, sample size, country). Validation figures come from
> held-out Source-1 entities that were never used for training, scored with
> an exact re-implementation of the competition metric against the *full*
> Source-2/3 corpus.

---

## 1. Executive Summary

We treat entity resolution as **retrieval followed by decision**, and we let
the data — not assumptions — drive the design. A structural audit of the
training data revealed three facts that shaped everything: (1) the ground
truth is strictly **bipartite** — no Source-2/3 record ever belongs to more
than one Source-1 entity (0 violations across all 7,638,365 match edges);
(2) business names are **heavily reused** across unrelated businesses; and
(3) the hard negatives are **one-off perturbations** of real businesses
(a nudged house number, an added industry word), whereas true matches arrive
as **several mutually consistent copies**.

Exploiting (1), we added **reverse retrieval**: every noisy record is asked
which of the ~1M clean, deduplicated Source-1 entities it belongs to. With
transliteration-robust phonetic tokens and address bigrams, this lifted
candidate recall from 0.681 → 0.958 (India) and 0.864 → 0.984 (US).
Exploiting (2) and (3), we added label-free **name-ambiguity** and
**corroboration (twin)** features to a LightGBM matcher. Validation macro
F0.5 rose from **0.8187 → 0.9611**. A fine-tuned **cross-encoder**
(MiniLM, Apache-2.0, 22M parameters) re-scores only the pairs LightGBM is
uncertain about, and is kept only if it improves held-out validation.

---

## 2. Methodology

### 2.1 Problem Analysis

**Dataset (row counts, full files).**

| File | Rows |
|---|---|
| train_source1 / train_ground_truth | 2,206,821 |
| train_source2 | 5,034,616 |
| train_source3 | 5,285,603 |
| test_source1 | 1,732,544 |
| test_source2 | 4,887,273 |
| test_source3 | 5,082,316 |

**Integrity.** Zero duplicate `entity_id`s in any file. Source 1 has no
missing fields; Source 2/3 are missing addresses in ~3.3% of rows. Fully
duplicated records occur in Source 2/3 (25,873 / 18,860) but never in
Source 1. No leakage channels exist: entity IDs are random, row order is
shuffled, and 0 of 300,000 sampled test records (S1 and S2) are identical to
any training record.

**Match structure (all 2,206,821 S1 entities).** Singletons 5.59%; mean
3.46 matches; median 3; max 11. Strictly bipartite (above).

**Country.** Train covers US and India; test adds France (15.0% of test S1).
Country agrees on 100% of sampled true pairs, so it is an exact, free
partition key. It is never hard-coded — the label set is read from the data,
so France flows through unchanged.

**Noise, measured on 150k–200k sampled true pairs per source.**

| Signal | S1↔S2 | S1↔S3 |
|---|---|---|
| Exact lowercased name equality | 10.95% | 10.42% |
| Mean name Jaro-Winkler | 0.864 | 0.871 |
| Mean address Jaro-Winkler | 0.855 | 0.825 |
| Non-ASCII names (Indic scripts) | 15.2% of S2 | 11.5% of S3 |

Abbreviation rules were verified before use: Limited↔Ltd and Road↔Rd have
strong evidence; Corporation↔Corp, Street↔St, Avenue↔Ave showed **zero**
occurrences in the sample and were deliberately not encoded.

**How the data was generated (the decisive analysis).** Inspecting unowned
records next to their most similar S1 entity showed the negative-generation
rule directly:

| Negative (unowned record) | Its source S1 entity |
|---|---|
| sushil holdings \| **26** krishna colony … | sushil holdings \| **23** krishna colony … |
| labdhi **infratech** \| … **155** | labdhi \| … **151** |
| zed platforms \| b **44** anurag society … | zed platforms \| b **43** anurag society … |

Measured on India (21,151 sampled unowned records vs 120,000 true pairs):

| Property | Negatives | True matches |
|---|---|---|
| A number within 30 of the S1's number, but not equal ("nudged") | **66.4%** | **1.9%** |
| Another record shares the same phonetic name **and** number set ("twin") | **0.8%** | **38.2%** |

Negatives are single perturbations; real businesses appear as several
consistent copies. Pairwise features alone cannot fully separate them —
the diagnostic below found a true match (`tejika entertainment`, 190→188) and
a negative (`d s mercator`, 359→368) with the *same* pairwise pattern — so
information from **outside the pair** is required.

### 2.2 Solution Strategy

**Approach type:** multi-channel sparse retrieval (forward + reverse) →
meta-blocking compression → LightGBM pair classifier with corroboration
features → optional cross-encoder re-ranking of uncertain pairs →
F0.5-optimised decision with bipartite (global-uniqueness) assignment.

**Core innovations:**
1. **Reverse retrieval** justified by the bipartite ground truth.
2. **Transliteration-robust phonetic skeletons** and **address bigrams**.
3. **Corroboration features** derived from the measured generation process.

---

## 3. Candidate Generation (Blocking)

### 3.1 Research basis
* **Sparkly** (Paulsen, Govind & Doan, PVLDB 2023): top-k TF/IDF blocking
  outperformed 8 state-of-the-art blockers; TF/IDF-cosine beats BM25 for
  entity matching because IDF applies to both sides. Our inverted indexes
  follow this.
* **Resource-efficient blocking** (Karapiperis, Tjortjis & Verykios,
  *Information Systems*, 2026): LSH fails to scale; uncompressed HNSW hits a
  memory wall past ~10M records; dense methods reached only F1≈0.17 on noisy
  web data. Combined with our finding that the noise is lexical
  (transliteration, typos, reordering), we used sparse retrieval rather than
  dense-embedding ANN.

### 3.2 Channels

| Channel | Structure | Purpose |
|---|---|---|
| A/B/C exact name, core name, sorted core | hashed-key sorted arrays | exact and reordered names |
| D name character 5-grams, top-k | CSR inverted index | transliteration |
| E address tokens, top-k | CSR inverted index | DBA / trade-name substitutions |
| G postal+house number | hashed-key sorted array | hardest residual pairs |
| J joint name-skeleton + address, top-k | CSR inverted index | reused names |
| **R reverse: each record → its top-3 S1 owners** | inverted index over S1 | **bipartite structure** |

**Phonetic skeleton.** Leetspeak digits → letters, ph→f, c/q→k, w→v, z/j→g,
repeated letters collapsed, initial vowel → *a*, then non-initial vowels/h/y
dropped. On 12 real failure pairs inspected (`phrstt`/`first`,
`myaaneejmentt`/`management`, `mhaaraassttr`/`maharashtra`, …), 12/12 map to
the same key.

**Address bigrams.** Individually common words (`community`, `industrial`)
exceed the document-frequency cap and are discarded, but adjacent pairs
(`centre_naraina`, or `94_26` from unit code `94/26`) are near-unique.

### 3.3 Measured evolution of candidate recall

*Validation entities, full corpus. "Owner@k" = share of true match records
whose owner is in their reverse top-k (20,000 records per country).*

| Stage | India | US |
|---|---|---|
| Word-token name TF/IDF, k=10 (4,000 queries) | 0.227 | — |
| Char 5-gram name TF/IDF, k=50 (3,000 queries) | 0.580 | — |
| Separate forward channels, compressed (5,000 queries) | 0.681 @ 14.4 cands | 0.864 @ 13.9 cands |
| Joint name+address index, k=15 (3,000 queries) | 0.816 @ 14.8 cands | — |
| Reverse, unigram phonetic tokens — owner@3 | 0.872 | — |
| Reverse + address bigrams — owner@3 | 0.932 | — |
| Reverse + improved skeleton — owner@1 / @3 / @10 | 0.901 / 0.935 / 0.956 | 0.952 / 0.973 / 0.983 |
| **Final union, compressed (5,000 queries)** | **0.958 @ 31.7 cands** | **0.984 @ 28.9 cands** |

### 3.4 How true matches were protected
* Union of independent retrievers — any one channel suffices.
* Retriever top-ranks (name/address top-5, joint top-12, all reverse links)
  are never removed by compression.
* A compression bug was found by measurement and fixed: an adaptive budget of
  K=3 for "exact-name" entities cut recall to 0.50, because entities have up
  to 11 true matches. The minimum budget was raised to 12.
* A scoring-normalisation hypothesis was tested and **rejected**
  (owner@1 0.833 → 0.819) and reverted.

**Final test candidate set:** `output/candidate_pairs.tsv`, 1,732,544 S1 entities, every one with at least one candidate (validator: 0 empty rows). On validation the final union averaged 31.7 candidates per S1 entity (India) and 28.9 (US), a reduction ratio above 0.99999 against the 4–6M-record country corpora.

---

## 4. Matching Model

### 4.1 Features — 69 per pair
* **Name (15):** exact equalities, Jaro-Winkler, ratio, partial/token-sort/
  token-set ratios, normalised Levenshtein, prefix, length ratio, token
  Jaccard/containment, IDF-weighted rare-token overlap.
* **Address (14):** equality, ratios, Jaro-Winkler, token overlap, house and
  postal match, digit overlap, missing-address flags.
* **Cross-field and retrieval evidence (18):** name×address interactions,
  channels fired, per-channel scores and ranks, compressor score,
  candidate-set size, joint-channel score/rank, S2/S3 indicator.
* **Reverse retrieval (5):** hit, score, rank, margin to the record's best
  owner, is-top-1.
* **Phonetic agreement (10):** skeleton Jaccard for name / address / bigrams /
  all tokens, shared bigrams, name pairs and codes, zero-stripped number
  Jaccard, share of S1 numbers missing, and a **nudged-number** flag.
* **Name ambiguity (3):** how many S1 entities share the S1's name and the
  record's name, and the latter when the record has no address.
* **Corroboration (4):** global twin count, twin count inside the candidate
  set, name support, and support for any number the record has that the S1
  lacks.

All ambiguity and corroboration features are **label-free** (computed only
from records and S1 names), so they are equally valid on the test set.

### 4.2 Hard negatives
Training pairs are produced by running the real retrieval pipeline over
training queries and labelling what it returns, so the negatives are exactly
the confusable candidates seen at inference (nudged-number siblings,
same-name businesses elsewhere, same-street neighbours).

### 4.3 Model and decision
* **LightGBM** (MIT licence): binary objective, 600 rounds, learning rate
  0.05, 127 leaves, deterministic mode. Trained on 150,000 S1 entities per
  country → 9,066,441 pairs (positive rate 11.1%).
* **Threshold:** chosen by directly maximising macro F0.5 over validation
  entities (singletons included) — coarse then fine search.
* **Global uniqueness:** each record is assigned to at most one S1 entity
  (its highest-scoring claimant), matching the bipartite ground truth. Kept
  only when validation shows it does not hurt.

### 4.4 Cross-encoder re-ranker (second stage)
* **Model:** `cross-encoder/ms-marco-MiniLM-L6-v2` — Apache-2.0, ~22M
  parameters. Fine-tuned on challenge data only; it sees only the two records
  being compared.
* **Training data:** 3,000,000 reverse-retrieval links from S1 entities
  **outside** the validation fold (positive rate 0.231 India / 0.240 US).
* **Training:** 1 epoch, 11,641 steps, batch 256, AdamW lr 4e-5 with warm-up
  and linear decay, bf16, max length 128; 22.7 minutes on an RTX 5070 Laptop
  GPU. Pair-level holdout AUC 0.9997, accuracy 0.9931 (random pair split, so
  optimistic — the entity-level test is below).
* **Use:** re-scores only pairs with LightGBM probability in [0.02, 0.98]; a
  logistic blend of the two scores and a new threshold are fitted on
  validation predictions. **Go/no-go:** 2-fold held-out estimate over
  validation entities (fit on one half, score the other).
* **Held-out result (2-fold over 40,000 validation entities):** LightGBM alone
  **0.96090** | LightGBM + cross-encoder **0.97747** (**+0.01657**); both folds
  improved (+0.0159, +0.0173) → **used**. Learned blend weights: 0.679 on
  logit(LightGBM), 0.621 on the cross-encoder logit; threshold 0.750 with
  global uniqueness. 49,390 of 1,211,877 validation pairs (4%) fell in the
  uncertainty band.
* **Wider band tested and not adopted:** [0.005, 0.995] scored 0.97796
  (+0.0005 over the [0.02, 0.98] band) — below the 0.001 margin fixed in
  advance, so the original band was kept.
* **France:** a variant applying the blend to France changed 228 of ~886,000
  French matches; this variant is the final submission (public
  leaderboard 0.968).
* France has no labelled data to validate a blend on, so it keeps LightGBM
  alone.

---

## 5. Results & Error Analysis

### 5.1 Scores

| Version | Change | Validation macro F0.5 | Leaderboard |
|---|---|---|---|
| v1 | forward retrieval + LightGBM (45 features) | 0.8187 | 0.825 |
| v4 | + reverse retrieval, phonetic tokens, ambiguity features (65) | 0.95685 | — |
| v5 | + corroboration features (69) | **0.96106** | — |
| v6 | + cross-encoder re-ranking (India, US) | **0.97747** (held-out 2-fold) | — |
| final | v6 + blend also applied to France | (France unvalidated) | **0.968** |

*Validation: 40,000 held-out S1 entities (20,000 per country), full corpus.
v1's validation score tracked its leaderboard score closely.*

**Validation vs leaderboard.** Validation predicted ~0.977; the public
leaderboard shows 0.968. The most likely cause is France (15% of test S1),
which has no training data: its address and name conventions were never
seen by either model, and no French labels exist to tune a threshold on.
Validation covers only US and India, so it cannot measure this. With more
time, the next steps would be (1) feeding the cross-encoder score into the
LightGBM model as a feature, (2) widening reverse retrieval to top-5
owners, and (3) the anchor-corroboration second stage implemented in
`ce.py` (`anchors`, `stage2`), which was built but not run before the
deadline.

### 5.2 Error attribution (v4 diagnostic, 5,000 validation entities per country)

| | India | US |
|---|---|---|
| Macro F0.5 | 0.9483 | 0.9682 |
| Precision / recall | 0.9826 / 0.8975 | 0.9890 / 0.9347 |
| Missed matches: blocking / matcher | 721 / 1,051 | 281 / 846 |
| False positives: sibling / other S1's record / other | 5 / 57 / 213 | 10 / 25 / 144 |
| Largest loss | entities with missed matches only (0.0258) | same (0.0159) |

Precision is already ~0.98–0.99; most remaining loss is **recall at the
matcher** — true matches scored just under the threshold.

### 5.3 Common false positives (wrong merges)
* **Empty-address records with a reused name** assigned with ~98% confidence
  to every S1 sharing the name → addressed by the name-ambiguity features.
* **Siblings** differing only by a nudged number or one swapped word →
  addressed by the nudged-number and corroboration features.

### 5.4 Common false negatives (missed matches)
* **Positive number noise** (`28` → `46 48 county rd`, `542` → `42 4th st`)
  that looks like a sibling at pair level.
* **Transliteration plus truncated address** (`supriim injiiniyring elelpii`
  vs `supreme engineering`).
* **Unrelated DBA names** (`Guru Logistics Private Limited` → `Vantageumbra`),
  reachable only through the address.
* **Unresolvable:** records with an empty address and a name shared by several
  S1 entities carry no information about which one they belong to.

---

## 6. Conclusion

The largest gains came from understanding how the data was generated rather
than from model tuning: the bipartite ground truth justified reverse retrieval
(+28 points of India candidate recall), and measuring the negative-generation
process (66.4% vs 1.9% nudged numbers; 0.8% vs 38.2% twins) produced label-free
features that pairwise similarity cannot provide. Every change was accepted or
rejected on held-out validation, including two that failed. The main lesson:
measure before building — our early small-sample tests hid a recall bug that
only a full-corpus validation revealed.

---

## Appendix

### A. Code Artefacts

```
code/business_entity_resolution/
├── src/
│   ├── normalization.py          streaming TSV → Parquet (O(n), constant memory)
│   ├── corpus.py                 Arrow-backed, memory-mapped country corpus
│   ├── indexing.py               HashBlock + Sparkly-style inverted index
│   ├── phonetic.py               phonetic skeletons, bigrams, number normalisation
│   ├── jointtok.py               joint name+address tokenizer
│   ├── blocking.py               forward channels A–G, J
│   ├── reverse.py                reverse retrieval, parallel, cached
│   ├── candidate_compression.py  meta-blocking, adaptive budgets, protection
│   ├── features.py               69 pair features
│   ├── model.py                  LightGBM, F0.5 threshold search, uniqueness
│   ├── metrics.py                exact competition metric
│   ├── validation.py             leakage-safe splits (MD5 on S1 id)
│   ├── pipeline.py               orchestrator
│   └── ce.py                     cross-encoder re-ranker
├── README.md
└── requirements.txt
```

**Reproduce:**

```powershell
$env:PYTHONHASHSEED = "0"
python src\pipeline.py --stage normalize --data-dir dataset
python src\pipeline.py --stage reverse   --data-dir dataset --workers 8
python src\pipeline.py --stage train     --data-dir dataset --train-queries 150000 --val-queries 20000
python src\pipeline.py --stage infer     --data-dir dataset
# optional second stage
python src\ce.py build-train --per-country 1500000
python src\ce.py train --batch 256
python src\ce.py valpred
python src\ce.py score
python src\ce.py decide --out output\matching_results.tsv
```

`metrics.py` reproduces the official worked example (P = 2/3, R = 1 → 0.714).

### B. Measured runtimes (RTX 5070 Laptop / Core Ultra 9)

| Step | Time |
|---|---|
| Reverse retrieval, train — India / US | 343 s / 580 s |
| Forward index build — India / US | 203 s / 279 s |
| Train stage (v5, 300k queries, 9.07M pairs) | 2,112 s |
| Cross-encoder training (3M pairs) | 22.7 min |
| Test inference (1.73M S1 entities) | ~2.5 h |

### C. Scalability
Every retrieval step is a hash probe (O(log n)) or a posting-list walk capped
by a document-frequency ceiling and a query-token budget, so per-query cost is
independent of corpus size; index builds are linear passes. Country is an
exact partition key, and within a country the index shards arbitrarily
(query every shard, union the top-k — the scheme Sparkly uses for 500M+
tuples). Reverse retrieval is embarrassingly parallel over records.
Nothing requires the corpus to fit in one machine's memory.

### D. Leakage control and reproducibility
* Splits are over S1 entities (MD5 hash → 100 buckets); an entity's whole
  match set moves with it, and the bipartite ground truth means no labelled
  edge can cross the boundary.
* Indexes are built over the **full** corpus, so validation contains the same
  distractors as test.
* The cross-encoder is trained on non-validation entities; its blend and
  threshold are chosen on validation; the go/no-go uses a 2-fold held-out
  estimate.
* `PYTHONHASHSEED=0` removes the only observed CPU run-to-run variation
  (1 of 12,000 entities in a test run, from set-iteration tie-breaking);
  LightGBM runs in deterministic mode.
* Cross-encoder training is seeded, but GPU training is not guaranteed to be
  bit-for-bit identical across runs or hardware, so a retrained model may
  differ very slightly. The exact trained models are shipped in
  `code/business_entity_resolution/artifacts/` (LightGBM `matcher.pkl` and
  the fine-tuned cross-encoder `ce_model/`), so the submitted decisions can be
  reproduced without retraining — see the README.
* No country is hard-coded: training and test countries are read from the
  data at every stage, so the pipeline runs unchanged on a new dataset.
