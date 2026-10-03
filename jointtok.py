"""
jointtok.py -- tokenizer for the joint name+address retrieval channel (J).

Input is "name_core|addr_street". Name tokens become phonetic SKELETONS:
repeated letters collapsed, then vowels/h/w/y dropped after the first letter.
This folds the Unidecode output of Indic-script transliterations back onto the
English spelling -- 'limittedd' -> 'lmtd' == 'limited' -> 'lmtd';
'maarketting' -> 'mrktng' == 'marketing'. Measured on 120k true training
pairs: sharing >=1 name token covers 77.3% of India pairs; sharing >=1 name
skeleton covers 90.7%.

Name and address tokens are namespaced (n:/a:) so a word appearing in a name
never collides with the same word in an address.

A class (not a closure) so indexes built with it can be pickled.
"""
import re

_REP = re.compile(r"(.)\1+")
_VOW = re.compile(r"[aeiouyhw]")


def skel(t):
    t = _REP.sub(r"\1", t)
    return t[:1] + _VOW.sub("", t[1:])


class JointTok:
    def __call__(self, txt):
        nm, _, ad = txt.partition("|")
        s = {"n:" + skel(t) for t in nm.split() if len(t) > 1}
        s |= {"a:" + t for t in ad.split() if t}
        return s
