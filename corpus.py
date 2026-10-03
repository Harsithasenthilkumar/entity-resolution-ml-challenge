"""
corpus.py — country-partitioned corpus storage backed by Arrow buffers.

Why Arrow and not Python lists:
  A CountryCorpus of 4.1M India records held as six Python `list[str]` measured
  2.46 GB resident -- CPython pays ~49 bytes of object header per string, which
  dwarfs the ~25 bytes of actual text. The same data in Arrow StringArrays
  (one contiguous char buffer + int32 offsets) is ~10x smaller. Strings are
  materialised into Python only one chunk at a time, inside the index build
  loop, and are released immediately after. This is what makes a 10M-record
  corpus fit in a 4 GB box, and it is the same columnar layout a distributed
  deployment would shard.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import gc, time

COLS = ["entity_id", "country_norm", "name_alnum", "name_core",
        "addr_street", "house_number", "postal_code"]

CHUNK = 250_000


class CountryCorpus:
    def __init__(self, country):
        self.country = country
        self._tables = []
        self.tbl = None

    def open_writer(self, cache_path):
        """Stream country rows from many parquet files into ONE cache file."""
        self.cache_path = cache_path
        self._writer = None
        return self

    def add_parquet(self, path, verbose=True):
        """Filter to this country and append straight to the cache file.
        Nothing accumulates on the heap: peak RSS stays at one batch."""
        t0 = time.time()
        pf = pq.ParquetFile(path)
        n = 0
        for batch in pf.iter_batches(batch_size=250_000, columns=COLS):
            t = pa.Table.from_batches([batch])
            t = t.filter(pc.equal(t["country_norm"], self.country)).drop(["country_norm"])
            if t.num_rows:
                if self._writer is None:
                    self._writer = pq.ParquetWriter(self.cache_path, t.schema,
                                                    compression="snappy")
                self._writer.write_table(t)
                n += t.num_rows
            del t
        gc.collect()
        if verbose:
            print(f"    +{path.split('/')[-1]}: +{n:,} rows ({time.time()-t0:.0f}s)")

    def finalize(self):
        """Close the cache and MEMORY-MAP it. Memory-mapped Arrow buffers are
        paged by the OS and do not count against the Python heap, so a corpus
        far larger than RAM can be scanned column-by-column."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        self.tbl = pq.read_table(self.cache_path, memory_map=True)
        self.n = self.tbl.num_rows
        gc.collect()
        return self

    def col(self, name):
        return self.tbl[name]

    def iter_text(self, name):
        """Yield Python strings for `name`, one chunk at a time (never all)."""
        arr = self.tbl[name]
        for chunk in arr.chunks if hasattr(arr, "chunks") else [arr]:
            for v in chunk.to_pylist():
                yield v if v is not None else ""

    def text_list(self, name):
        """Full Python list -- use only for short-lived index construction."""
        return [v if v is not None else "" for v in self.tbl[name].to_pylist()]

    def ids(self):
        return self.tbl["entity_id"]
