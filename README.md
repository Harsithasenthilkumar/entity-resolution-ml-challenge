# Business Entity Resolution at Scale

Matching noisy business records across three data sources — **24 million
records**, names in multiple Indian scripts, abbreviated and truncated
addresses — built for the **Amazon ML Challenge 2026** (Unstop) as part of
team *epoch 0*.

**Task.** For each of 1.73M reference businesses (Source 1), find every
matching record among ~10M noisy records (Sources 2 and 3), or none.
Scored by macro-averaged F0.5 (precision weighted 2x).

**Result.** Public leaderboard **0.825 → 0.968** over the course of the
challenge; held-out validation **0.9775**.

## What made the difference

| Idea | Effect (measured on held-out validation) |
|---|---|
| **Reverse retrieval** — the ground truth is strictly one-owner-per-record, so each noisy record is also asked *which clean business it belongs to* | candidate recall India **0.681 → 0.958**, US **0.864 → 0.984** |
| **Phonetic skeletons + address bigrams** — fold transliteration (`phrstt helthkeyr` = `first healthcare`) and rescue individually-common words (`centre_naraina`) | India owner@1 **0.833 → 0.901** |
| **Reverse-engineering the negatives** — fake matches were copies of real businesses with a nudged house number (66% vs 1.9% of true matches) and appear only once (0.8% vs 38% for real copies) | label-free "corroboration" features for the matcher |
| **LightGBM matcher, 69 features** | macro F0.5 **0.8187 → 0.9611** |
| **Fine-tuned cross-encoder** (MiniLM, Apache-2.0) on my own GPU, re-scoring only uncertain pairs, kept only because it won on held-out data | **0.9609 → 0.9775** |

Full write-up, including what *didn't* work: [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## Stack

Python · NumPy · PyArrow · LightGBM · RapidFuzz · PyTorch · Hugging Face
Transformers · custom sparse inverted indexes (no full N×M comparison
anywhere — every retrieval step is a hash probe or a bounded posting-list walk).

## Trained models

The trained models are attached to this repository's **Releases** page
(not stored in the repo, to keep it small):

| File | What it is | Size |
|---|---|---|
| `matcher.pkl` | LightGBM matcher, 69 features, decision threshold 0.71 | 8.5 MB |
| `ce_model.zip` | fine-tuned cross-encoder (MiniLM-L6, Apache-2.0, see `NOTICE.txt`) | ~85 MB |

To use them, place them as:

```
artifacts/matcher.pkl
artifacts/ce_model/          <- contents of ce_model.zip
```

and follow *Reproducing the submitted decisions with the shipped models* below.

## Data

The competition dataset is **not** included (it belongs to the organisers).
To run the pipeline, place the competition files under `dataset/train/` and
`dataset/test/` as described below.

---

## Pipeline details

Multi-channel retrieval → adaptive candidate compression → LightGBM → F_0.5-optimised decision.

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the competition data so that these paths exist:

```
dataset/train/train_source1.tsv   train_source2.tsv   train_source3.tsv   train_ground_truth.tsv
dataset/test/test_source1.tsv     test_source2.tsv    test_source3.tsv
```

## 2. Run (in order)

For bit-for-bit reproducible runs, fix Python's per-process string-hash salt
first (otherwise ~0.01% of entities can differ between runs because of
set-iteration tie-breaking):

```powershell
$env:PYTHONHASHSEED = "0"      # PowerShell
export PYTHONHASHSEED=0        # bash
```


```bash
# 1. raw TSV -> normalized parquet   (~6 min, constant memory)
python src/pipeline.py --stage normalize --data-dir dataset

# 1b. reverse retrieval for train + test, parallel, cached to cache/rev_*.pkl
python src/pipeline.py --stage reverse --data-dir dataset --workers 8

# 2. blocking quality on held-out S1 entities — READ THIS OUTPUT BEFORE TRAINING
python src/pipeline.py --stage validate --data-dir dataset --val-queries 40000

# 3. train matcher + pick threshold on validation
python src/pipeline.py --stage train --data-dir dataset \
        --train-queries 400000 --val-queries 40000

# 4. full test inference -> output/
python src/pipeline.py --stage infer --data-dir dataset

# 5. competition validator
python3 utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
```

Stages cache to `norm/`, `cache/`, `artifacts/` and are safely re-runnable.

## 3. Hardware notes (RTX 5070 / Core Ultra 9)

This code was developed under a hard 1-CPU / 4 GB constraint, so it is
aggressively memory-disciplined: Arrow-backed memory-mapped corpora, numpy
`searchsorted` hash blocks instead of Python dicts, CSR posting lists, and
three-pass streaming index builds. **On your machine those limits are gone**,
and the following changes are the high-value ones:

| Knob | Where | Sandbox | Your laptop |
|---|---|---|---|
| `--train-queries` | CLI | 400k | **all** (drop the cap) |
| `k_name`, `k_addr` | `blocking.DEFAULTS` | 20 / 20 | **50 / 50** |
| `TIER_BUDGETS` | `candidate_compression.py` | 3/8/20/40 | sweep via `tune_budgets()` |
| `num_round` | `model.train_lightgbm` | 600 | 1500–3000 + early stopping |
| `k_addr_ngram` | `blocking.DEFAULTS` | 0 (off) | **turn on** (recall channel F) |
| `num_threads` | already `-1` | 1 | uses all cores |

The index build was the sandbox bottleneck (char-5-gram name index over the
4.1M-record India corpus produced 26.8M postings and OOM-killed a 4 GB box).
With 32 GB+ it builds comfortably and you can raise `max_df_frac` — which is
currently set low purely to survive memory pressure and is **costing you
recall**.

### GPU (optional, measure before keeping)
The RTX 5070 makes a dense channel feasible (~10M records through
`sentence-transformers/all-MiniLM-L6-v2`, Apache-2.0, 22M params — well inside
the 8B cap). Add it as channel **H** in `blocking.py`, index with
`faiss-gpu` IVFPQ, and union it in. **But**: the measured evidence says be
sceptical. A 2026 *Information Systems* design-space study found every dense
method scored F1≈0.17 on noisy web data, and the failure mode here is
transliteration and DBA-name substitution — a semantic model will not map
"Guru Logistics Private Limited" → "Vantageumbra" either. Keep it only if
`--stage validate` shows it lifting candidate recall.

## 4. Module map

| File | Role |
|---|---|
| `normalization.py` | streaming TSV→Parquet; Unicode folding, legal-suffix stripping, digit/postal extraction |
| `corpus.py` | Arrow-backed, memory-mapped, country-partitioned corpus |
| `indexing.py` | `HashBlock` (numpy exact-key) + `InvertedIndex` (Sparkly-style top-k TF/IDF) |
| `blocking.py` | forward retrieval channels A–G + joint channel J |
| `phonetic.py` | transliteration-robust skeletons, address bigrams, number normalisation |
| `reverse.py` | reverse retrieval: each S2/S3 record -> its top-3 S1 owners (parallel) |
| `candidate_compression.py` | meta-blocking rank + **adaptive per-entity K** |
| `features.py` | 45 pairwise features (name / address / cross-field / retrieval evidence) |
| `model.py` | LightGBM + XGBoost/CatBoost benchmark, F_0.5 threshold search, global-uniqueness pass |
| `metrics.py` | exact competition scorer (reproduces the official 0.714 worked example) |
| `validation.py` | 5 leakage-safe splits, hashed on S1 entity id |
| `pipeline.py` | orchestrator |

## 5. Competition compliance

- **`candidate_pairs.tsv` is the true final candidate set.** In `stage_infer`,
  `compress()` returns `finals`, that exact list is written to
  `candidate_pairs.tsv`, and the same list is featurised and scored. There is
  no post-filter. Matches are a subset by construction.
- **No external data.** Only the provided TSVs are read. Unidecode is
  algorithmic transliteration, not a lookup table of businesses.
- **Licensing.** LightGBM (MIT), RapidFuzz (MIT), Unidecode (GPL-compatible
  Artistic/GPL — swap for `text-unidecode` (Artistic 2.0) if your team needs a
  permissive-only tree), PyArrow/NumPy (Apache-2.0/BSD).
- **Country is never hard-coded.** `_countries()` discovers the label set from
  the data, so France flows through with no code change.

## 6. Known gaps — where the remaining points are

1. **Blocking recall is the binding constraint** (0.78–0.83 measured).
   Everything downstream is capped by it. Raise `max_df_frac`, raise `k`, and
   enable channel F first.
2. **`tune_budgets()` is written but not swept** — do this, it is cheap and
   directly trades candidate count against recall.
3. **Only seed0 was used.** Run all 5 splits for a variance estimate.
4. **Address char-n-gram channel (F) is implemented but off** — it targets the
   truncated/transliterated addresses that the error analysis identified as a
   top failure mode.

## Second stage: cross-encoder re-ranker (`src/ce.py`, GPU)

PyTorch must be installed first from the CUDA 12.8 index (required for
RTX 50-series GPUs), then the rest of the requirements:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Run after `--stage infer` (which saves every scored pair to `cache/testpred_*`):

```powershell
python src/ce.py build-train --per-country 1500000   # CPU: training pairs
python src/ce.py train --batch 256                   # GPU: fine-tune MiniLM (~23 min)
python src/ce.py valpred                             # CPU: validation predictions
python src/ce.py score                               # GPU: re-score uncertain pairs
python src/ce.py decide --out output/matching_results.tsv
```

`decide` prints a 2-fold held-out comparison (LightGBM alone vs
LightGBM + cross-encoder) and only replaces the matches if the blend wins.
The submitted file was produced with the command above
(held-out macro F0.5 0.96090 -> 0.97747).

Optional, implemented but not run before the deadline:
`python src/ce.py anchors` and `python src/ce.py stage2 --base <file>`
(anchor-corroboration second stage; see docs/METHODOLOGY.md, section 5.1).

## Running on a new dataset

Nothing is tied to this dataset's countries or IDs. Put new data in the same
layout (`dataset/train/*.tsv`, `dataset/test/*.tsv`, same column names) and
run the same commands. Country labels are read from the data at every stage:
`pipeline.py` partitions by whatever countries appear, and `ce.py` takes its
training countries from `train_source1` and its test countries from
`test_source1` (`--countries all`, the default). A test country with no
training data (France here) is still matched; the blend fitted on the training
countries is applied to it, exactly as in the submitted run.

## Reproducibility

* Set `PYTHONHASHSEED=0` before every command (see above).
* CPU stages (normalisation, retrieval, LightGBM with `deterministic=True`,
  threshold search, `decide`) are deterministic given the same inputs and models.
* Cross-encoder training is seeded (`torch.manual_seed(0)`), but GPU training
  is not guaranteed to be bit-for-bit identical across runs or hardware
  (bf16 kernels, dropout ordering), so a retrained model can differ very
  slightly. Expect validation scores within a few ten-thousandths of the
  reported ones, not identical files. Reusing the same trained models
  (`artifacts/matcher.pkl`, `artifacts/ce_model/`) and re-running
  `infer -> valpred -> score -> decide` reproduces the decisions.
* The pretrained model is downloaded from the Hugging Face Hub on first use
  (`cross-encoder/ms-marco-MiniLM-L6-v2`, Apache-2.0); no other external
  resource is used, and no business data is looked up anywhere.

## Reproducing the submitted decisions with the shipped models

`artifacts/` contains the exact trained models behind the submission:

```
artifacts/matcher.pkl          LightGBM matcher (69 features, threshold 0.71)
artifacts/train_report.json    its validation report (macro F0.5 0.96106)
artifacts/ce_model/            fine-tuned cross-encoder (MiniLM-L6, Apache-2.0)
```

To reproduce the submitted matches WITHOUT retraining (skips the two
training steps, and with them the only GPU non-determinism):

```powershell
$env:PYTHONHASHSEED = "0"
python src\pipeline.py --stage normalize --data-dir dataset
python src\pipeline.py --stage reverse   --data-dir dataset
python src\pipeline.py --stage infer     --data-dir dataset
python src\ce.py valpred
python src\ce.py score
python src\ce.py decide --out output\matching_results.tsv
```

Do NOT run `--stage train` or `ce.py train` in this mode: they overwrite the
shipped models. `valpred` needs only the shipped `matcher.pkl` (it builds
the validation split itself on first use).
