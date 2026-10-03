"""
validation.py -- held-out validation splits over Source-1 entities.

The split is over S1 ENTITIES; each entity's whole match set moves with it.
The ground truth is strictly bipartite (no S2/S3 record ever matches two S1
entities -- 0 violations across 7,638,365 edges), so no labelled edge can
straddle the fold boundary. MD5 is used instead of hash() because Python's
hash() is salted per process and splits would not be reproducible.
"""
import os
import hashlib

DEFAULT_GT = "dataset/train/train_ground_truth.tsv"


def stable_hash_bucket(eid: str, nbuckets: int = 100) -> int:
    return int(hashlib.md5(eid.encode()).hexdigest()[:8], 16) % nbuckets


def load_ground_truth(gt_path=DEFAULT_GT):
    rows = []
    with open(gt_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            eid, _, matched = line.rstrip("\n").partition("\t")
            if eid:
                rows.append((eid, matched))
    return rows


def build(seed_name="seed0", val_frac_buckets=range(0, 5), out_dir="splits",
          gt_path=DEFAULT_GT):
    os.makedirs(out_dir, exist_ok=True)
    rows = load_ground_truth(gt_path)
    vb = set(val_frac_buckets)
    val, trn = [], []
    for eid, matched in rows:
        (val if stable_hash_bucket(eid) in vb else trn).append((eid, matched))
    print(f"{seed_name}: train={len(trn):,} val={len(val):,}")
    for nm, data in (("train", trn), ("val", val)):
        with open(f"{out_dir}/{seed_name}_{nm}.tsv", "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for eid, m in data:
                f.write(f"{eid}\t{m}\n")
    return len(trn), len(val)


if __name__ == "__main__":
    for i in range(5):
        build(seed_name=f"seed{i}", val_frac_buckets=range(i * 5, i * 5 + 5))
