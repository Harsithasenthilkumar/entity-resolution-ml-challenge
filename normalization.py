"""
normalize_stream.py
===================
Single-pass streaming normalizer: raw TSV -> Parquet.

Why streaming instead of DuckDB LIMIT/OFFSET chunking:
  An earlier implementation paged through the source table with
  `ORDER BY entity_id LIMIT n OFFSET k`. That re-sorts the whole table on every
  chunk (O(n log n) x n_chunks) and materialises large intermediates, which
  OOM-killed the 4 GB sandbox twice at ~1-3.4M rows. Reading the source TSV
  line-by-line with the stdlib csv reader and flushing fixed-size row batches to
  a Parquet writer is O(n) total with O(batch) memory, and is exactly the access
  pattern that scales to billions of records (sequential scan, no global sort,
  trivially shardable across workers).
"""
import csv, gc, os, re, sys, time
import pyarrow as pa
import pyarrow.parquet as pq
from unidecode import unidecode

csv.field_size_limit(10_000_000)

LEGAL_SUFFIXES = sorted(set([
    "private limited", "pvt ltd", "pvt. ltd.", "private ltd", "pvt limited",
    "limited liability company", "limited liability partnership",
    "corporation", "incorporated", "limited", "company",
    "llp", "llc", "inc", "corp", "ltd", "co", "plc", "sarl", "sas", "sasu", "sa",
]), key=len, reverse=True)
_LEGAL_RE = re.compile(r"(?<!\w)(" + "|".join(re.escape(s) for s in LEGAL_SUFFIXES) + r")(?!\w)")

STREET_ABBR = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy", "square": "sq", "place": "pl",
    "apartment": "apt", "building": "bldg", "floor": "fl", "suite": "ste",
}
_punct_re = re.compile(r"[^a-z0-9\s]")
_ws_re = re.compile(r"\s+")
_digit_re = re.compile(r"\d+")

_ascii_cache = {}

def to_ascii(s):
    if not s:
        return ""
    if s.isascii():
        return s
    v = _ascii_cache.get(s)
    if v is None:
        v = unidecode(s)
        if len(_ascii_cache) < 400_000:
            _ascii_cache[s] = v
    return v

def alnum_only(s):
    if not s:
        return ""
    return _ws_re.sub(" ", _punct_re.sub(" ", s.lower())).strip()

def strip_legal_suffix(name_alnum):
    if not name_alnum:
        return ""
    s = _ws_re.sub(" ", _LEGAL_RE.sub(" ", name_alnum)).strip()
    return s if s else name_alnum

def norm_street(addr_alnum):
    if not addr_alnum:
        return ""
    return " ".join(STREET_ABBR.get(t, t) for t in addr_alnum.split(" "))

SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("country_norm", pa.string()),
    ("name_alnum", pa.string()),       # ascii-folded, punctuation-stripped, lowercased
    ("name_core", pa.string()),        # name_alnum minus legal suffixes
    ("addr_alnum", pa.string()),
    ("addr_street", pa.string()),      # addr_alnum with street-type abbrevs unified
    ("house_number", pa.string()),
    ("postal_code", pa.string()),
    ("digits", pa.string()),           # space-joined digit groups
])
# name_compact / *_sorted_tokens are derived on demand (cheap str ops) rather
# than stored: storing them tripled parquet size for no retrieval benefit.

def run(src_tsv, out_parquet, batch_size=100_000):
    t0 = time.time()
    writer = pq.ParquetWriter(out_parquet, SCHEMA, compression="snappy")
    cols = {f.name: [] for f in SCHEMA}
    n = 0

    def flush():
        writer.write_table(pa.Table.from_pydict(cols, schema=SCHEMA))
        for k in cols:
            cols[k].clear()

    with open(src_tsv, "r", encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(rd)
        assert header[:4] == ["entity_id", "business_name", "business_address", "country"], header
        for row in rd:
            if len(row) < 4:
                row = row + [""] * (4 - len(row))
            eid, bname, baddr, ctry = row[0], row[1], row[2], row[3]

            na = to_ascii(bname)
            aa = to_ascii(baddr)
            nal = alnum_only(na)
            aal = alnum_only(aa)
            core = strip_legal_suffix(nal)
            dg = _digit_re.findall(baddr) if baddr else []
            pcs = [g for g in dg if len(g) in (5, 6)]

            cols["entity_id"].append(eid)
            cols["country_norm"].append(ctry.strip().lower() if ctry else "")
            cols["name_alnum"].append(nal)
            cols["name_core"].append(core)
            cols["addr_alnum"].append(aal)
            cols["addr_street"].append(norm_street(aal))
            cols["house_number"].append(dg[0] if dg else None)
            cols["postal_code"].append(pcs[-1] if pcs else None)
            cols["digits"].append(" ".join(dg))

            n += 1
            if len(cols["entity_id"]) >= batch_size:
                flush()
                if n % 1_000_000 == 0:
                    gc.collect()
                    print(f"   {os.path.basename(out_parquet)}: {n:,} rows ({time.time()-t0:.0f}s)", flush=True)
    if cols["entity_id"]:
        flush()
    writer.close()
    print(f"[done] {out_parquet}: {n:,} rows in {time.time()-t0:.0f}s", flush=True)
    return n

# Entry point is src/pipeline.py --stage normalize, which calls run() with
# paths derived from --data-dir. This module is intentionally not runnable
# standalone so that paths can never drift from the pipeline's.
