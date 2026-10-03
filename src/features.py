"""
features.py -- pairwise features for the matching model.

Split into three groups (name / address / cross-field + retrieval evidence).
All string similarities come from RapidFuzz (MIT licensed, SIMD-accelerated C++)
rather than pure-Python Levenshtein -- at ~50M candidate pairs the difference
is hours.

A deliberate choice: retrieval evidence (which channels fired, their scores and
ranks) is fed to the matcher as features. This is "supervised meta-blocking" in
the Papadakis sense -- the number of independent retrievers that agreed on a
pair is one of the most informative cheap signals available, and it costs
nothing because blocking already computed it.
"""
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

from phonetic import Tok4, group, name_key

_TOK = Tok4()

FEATURE_NAMES = [
    # --- name ---
    "n_exact_alnum", "n_exact_core", "n_exact_compact", "n_sorted_eq",
    "n_jaro_winkler", "n_ratio", "n_partial_ratio", "n_token_sort",
    "n_token_set", "n_lev_norm", "n_prefix4", "n_len_ratio",
    "n_tok_jaccard", "n_tok_containment", "n_rare_tok_overlap",
    # --- address ---
    "a_exact", "a_ratio", "a_token_sort", "a_token_set", "a_jaro_winkler",
    "a_tok_jaccard", "a_tok_containment", "a_len_ratio",
    "a_house_match", "a_postal_match", "a_digit_jaccard", "a_digit_overlap_n",
    "a_missing_q", "a_missing_c",
    # --- cross-field & retrieval evidence ---
    "x_name_x_addr", "x_name_plus_addr", "x_both_strong", "x_either_strong",
    "r_n_channels", "r_exact_hash_hit", "r_d_score", "r_d_rank",
    "r_e_score", "r_e_rank", "r_g_hit", "r_rank_score",
    "r_cand_set_size", "r_score_margin_top", "r_is_top1",
    "s_is_s3",
    "r_j_score", "r_j_rank",
    # --- reverse retrieval evidence (see reverse.py) ---
    "rv_hit", "rv_score", "rv_rank", "rv_margin", "rv_is_top1",
    # --- transliteration-robust agreement (see phonetic.py) ---
    "k_name_jac", "k_addr_jac", "k_bigram_jac", "k_bigram_shared",
    "k_namepair_shared", "k_code_shared", "k_all_jac",
    # --- number agreement, zeros stripped; k_num_close targets the
    #     adversarial siblings in the data (true '1013 Eastridge Dr' vs
    #     negative '1026 Eastridge Dr') ---
    "k_num_jac", "k_num_s1_missing", "k_num_close",
    # --- name ambiguity: names are reused across different businesses, so a
    #     record with an empty address and a name shared by k S1 entities is
    #     right for at most one of them. Label-free (counts S1 names only). ---
    "amb_s1_name_count", "amb_rec_name_count", "amb_rec_count_if_noaddr",
    # --- corroboration: real businesses appear as several consistent copies,
    #     negatives are one-off perturbations (38.2% vs 0.8% twin rate) ---
    "cor_global_twin", "cor_local_twin", "cor_name_support", "cor_extra_num_support",
]


def _j(a, b):
    if not a and not b:
        return -1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _num_close(n1, n2):
    """1 if a number unique to one side is within 20 of a number unique to
    the other side -- the signature of a nudged house/unit number."""
    a = [int(x) for x in (n1 - n2) if len(x) <= 6]
    b = [int(x) for x in (n2 - n1) if len(x) <= 6]
    for x in a:
        for y in b:
            if 0 < abs(x - y) <= 20:
                return 1.0
    return 0.0


def phonetic_features(qg, cand_name, cand_addr):
    cg = group(_TOK((cand_name or "") + "|" + (cand_addr or "")))
    qa = qg["n"] | qg["a"] | qg["d"] | qg["c"] | qg["b"] | qg["p"]
    ca = cg["n"] | cg["a"] | cg["d"] | cg["c"] | cg["b"] | cg["p"]
    qn, cn = qg["d"], cg["d"]
    miss = (len(qn - cn) / len(qn)) if qn else -1.0
    return [
        _j(qg["n"], cg["n"]),
        _j(qg["a"], cg["a"]),
        _j(qg["b"], cg["b"]),
        float(len(qg["b"] & cg["b"])),
        float(len(qg["p"] & cg["p"])),
        float(len(qg["c"] & cg["c"])),
        _j(qa, ca),
        _j(qn, cn),
        miss,
        _num_close({t[2:] for t in qn}, {t[2:] for t in cn}),
    ]


def query_groups(name_core, addr):
    return group(_TOK((name_core or "") + "|" + (addr or "")))


def _safe_ratio(a, b):
    if not a or not b:
        return 0.0
    return len(a) / len(b) if len(a) < len(b) else len(b) / len(a)


def _jac(sa, sb):
    if not sa or not sb:
        return 0.0, 0.0
    inter = len(sa & sb)
    return inter / len(sa | sb), inter / min(len(sa), len(sb))


def pair_features(q, cand, idf_lookup=None, ctx=None):
    """q    : (eid, name_alnum, name_core, addr, house, postal)
    cand : dict with keys name_alnum, name_core, addr, house, postal,
           entity_id, plus 'f' (cheap features) and 'ev' (retrieval evidence)
    ctx  : per-query context (candidate set size, top score, etc.)
    """
    _, q_na, q_nc, q_ad, q_ho, q_po = q
    c_na = cand.get("name_alnum") or ""
    c_nc = cand.get("name_core") or ""
    c_ad = cand.get("addr") or ""
    c_ho = cand.get("house") or ""
    c_po = cand.get("postal") or ""
    ev = cand.get("ev", {})
    ctx = ctx or {}

    qnt = set(q_nc.split(" ")) - {""}
    cnt = set(c_nc.split(" ")) - {""}
    qat = set(q_ad.split(" ")) - {""}
    cat = set(c_ad.split(" ")) - {""}

    n_jac, n_con = _jac(qnt, cnt)
    a_jac, a_con = _jac(qat, cat)

    # rare-token overlap: shared tokens weighted by inverse corpus frequency.
    rare = 0.0
    if idf_lookup is not None:
        for t in (qnt & cnt):
            rare += idf_lookup(t)

    q_dig = set(d for d in q_ad.split(" ") if d.isdigit())
    c_dig = set(d for d in c_ad.split(" ") if d.isdigit())
    d_jac, _ = _jac(q_dig, c_dig)

    n_jw = JaroWinkler.similarity(q_nc, c_nc) if (q_nc and c_nc) else 0.0
    a_jw = JaroWinkler.similarity(q_ad, c_ad) if (q_ad and c_ad) else 0.0
    n_ratio = fuzz.ratio(q_nc, c_nc) / 100.0 if (q_nc and c_nc) else 0.0
    a_ratio = fuzz.ratio(q_ad, c_ad) / 100.0 if (q_ad and c_ad) else 0.0
    lev = (1.0 - Levenshtein.normalized_distance(q_nc, c_nc)) if (q_nc and c_nc) else 0.0

    name_strong = 1.0 if n_jw > 0.90 else 0.0
    addr_strong = 1.0 if a_jw > 0.85 else 0.0

    return [
        1.0 if (q_na and q_na == c_na) else 0.0,
        1.0 if (q_nc and q_nc == c_nc) else 0.0,
        1.0 if q_nc.replace(" ", "") == c_nc.replace(" ", "") and q_nc else 0.0,
        1.0 if (qnt and qnt == cnt) else 0.0,
        n_jw,
        n_ratio,
        fuzz.partial_ratio(q_nc, c_nc) / 100.0 if (q_nc and c_nc) else 0.0,
        fuzz.token_sort_ratio(q_nc, c_nc) / 100.0 if (q_nc and c_nc) else 0.0,
        fuzz.token_set_ratio(q_nc, c_nc) / 100.0 if (q_nc and c_nc) else 0.0,
        lev,
        1.0 if q_nc[:4] == c_nc[:4] and q_nc else 0.0,
        _safe_ratio(q_nc, c_nc),
        n_jac, n_con, rare,
        1.0 if (q_ad and q_ad == c_ad) else 0.0,
        a_ratio,
        fuzz.token_sort_ratio(q_ad, c_ad) / 100.0 if (q_ad and c_ad) else 0.0,
        fuzz.token_set_ratio(q_ad, c_ad) / 100.0 if (q_ad and c_ad) else 0.0,
        a_jw, a_jac, a_con, _safe_ratio(q_ad, c_ad),
        1.0 if (q_ho and q_ho == c_ho) else 0.0,
        1.0 if (q_po and q_po == c_po) else 0.0,
        d_jac, float(len(q_dig & c_dig)),
        1.0 if not q_ad else 0.0,
        1.0 if not c_ad else 0.0,
        n_jw * a_jw,
        n_jw + a_jw,
        name_strong * addr_strong,
        max(name_strong, addr_strong),
        float(sum(1 for k in "ABCDEFGJ" if k in ev)),
        1.0 if ("A" in ev or "B" in ev or "C" in ev) else 0.0,
        float(ev.get("D", 0.0)),
        float(ev.get("Dr", 999)),
        float(ev.get("E", 0.0)),
        float(ev.get("Er", 999)),
        1.0 if "G" in ev else 0.0,
        float(cand.get("rank_score", 0.0)),
        float(ctx.get("cand_set_size", 0)),
        float(cand.get("rank_score", 0.0) - ctx.get("top_score", 0.0)),
        1.0 if cand.get("is_top1") else 0.0,
        1.0 if str(cand.get("entity_id", "")).startswith("S3") else 0.0,
        float(ev.get("J", 0.0)),
        float(ev.get("Jr", 999)),
        1.0 if "R" in ev else 0.0,
        float(ev.get("R", 0.0)),
        float(ev.get("Rr", 99)),
        float(ev.get("Rm", -1.0)),
        1.0 if ev.get("Rr", 99) == 0 else 0.0,
    ] + phonetic_features(ctx["qgroups"], c_nc, c_ad) + _amb(ctx, c_nc, c_ad) \
      + list(cand.get("sup", [0.0, 0.0, 0.0, -1.0]))


def _amb(ctx, c_nc, c_ad):
    nc = ctx.get("ncount") or {}
    rc = float(nc.get(name_key(c_nc), 0))
    return [float(ctx.get("q_ncount", 0)), rc, 0.0 if c_ad else rc]


def build_matrix(rows):
    return np.asarray(rows, dtype=np.float32)
