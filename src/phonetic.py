"""
phonetic.py -- transliteration-robust tokens for reverse retrieval + features.

WHY THIS EXISTS (all figures measured on training data):
  * S2/S3 names/addresses are often Indic script, which Unidecode folds to
    spellings like 'phrstt helthkeyr praaivett limittedd' for
    'first healthcare private limited'. skel() maps both to the same key.
    On the 12 real failure pairs inspected, 12/12 collapse correctly.
  * Names are heavily REUSED across different businesses, and individual
    address words ('community', 'industrial', 'delhi') are too common to
    survive a document-frequency cap. Adjacent address BIGRAMS
    ('centre_naraina', '94_26' = unit code 94/26) are near-unique.
    Adding bigrams lifted India owner@1 from 0.833 to 0.897.
"""
import re

_REP = re.compile(r"(.)\1+")
_NUM = re.compile(r"\d+")
_NONVOW = re.compile(r"[aeiouyh]")
_LEET = str.maketrans({'5': 's', '0': 'o', '1': 'i', '3': 'e', '4': 'a',
                       '7': 't', '8': 'b', '@': 'a', '$': 's'})
_PH = [('ph', 'f'), ('ck', 'k'), ('x', 'ks'), ('q', 'k'), ('c', 'k'),
       ('w', 'v'), ('z', 'j'), ('j', 'g')]
_cache = {}


def skel(t):
    """Phonetic skeleton tuned to Indic->Latin transliteration noise."""
    v = _cache.get(t)
    if v is None:
        s = t
        if not s.isdigit() and any(ch.isdigit() for ch in s):
            s = s.translate(_LEET)
        for a, b in _PH:
            s = s.replace(a, b)
        s = _REP.sub(r"\1", s)
        if s and s[0] in 'aeiou':
            s = 'a' + s[1:]
        v = _REP.sub(r"\1", s[:1] + _NONVOW.sub("", s[1:]))
        if len(_cache) < 3_000_000:
            _cache[t] = v
    return v


def numbers(addr):
    """Numbers in an address with leading zeros stripped ('0055' == '55')."""
    return {x.lstrip('0') or '0' for x in _NUM.findall(addr or '')}


def _addr_units(ad):
    out = []
    for t in ad.split():
        if t.isdigit():
            out.append(t.lstrip('0') or '0')
        elif t.isalpha():
            out.append(skel(t))
        else:
            out.append(t)
    return out


class Tok4:
    """Input 'name_core|addr_street'. Emits
       n: name-token skeletons        a: alphabetic address skeletons
       d: numbers (zeros stripped)    c: mixed codes ('hd020', '1013b')
       b: adjacent address bigrams    p: unordered name-token pairs
    A class (not a closure) so indexes built with it can be pickled and
    shipped to worker processes."""

    def __call__(self, txt):
        nm, _, ad = txt.partition('|')
        s = set()
        nt = []
        for t in nm.split():
            if len(t) > 1:
                k = skel(t)
                s.add('n:' + k)
                nt.append(k)
        for t in ad.split():
            if t.isalpha():
                if len(t) > 1:
                    s.add('a:' + skel(t))
            elif not t.isdigit():
                s.add('c:' + t)
        for x in _NUM.findall(ad):
            s.add('d:' + (x.lstrip('0') or '0'))
        u = _addr_units(ad)
        for a, b in zip(u, u[1:]):
            s.add('b:' + a + '_' + b)
        nt = sorted(set(nt))
        for i in range(len(nt)):
            for j in range(i + 1, len(nt)):
                s.add('p:' + nt[i] + '_' + nt[j])
        return s


def group(tokset):
    """Split a Tok4 token set by namespace -> dict prefix -> set."""
    g = {'n': set(), 'a': set(), 'd': set(), 'c': set(), 'b': set(), 'p': set()}
    for t in tokset:
        g[t[0]].add(t)
    return g


def name_key(name):
    """Order-free phonetic key for a business name (for ambiguity counts)."""
    return " ".join(sorted({skel(t) for t in (name or "").split() if len(t) > 1}))
