from collections import defaultdict

from src.boolean_query import OPERATORS
from src.edit_distance import damerau_levenshtein


class SpellChecker:
    """Suggest the closest vocabulary word for a misspelling.

    Shortlist: same first letter and similar length (bucketed), then edit distance.
    Best = smallest distance; ties go to the more frequent word.
    Limitation: a typo in the FIRST letter is not corrected.
    """

    def __init__(self, vocab: dict, max_dist: int = 2):
        self.vocab = vocab
        self.max_dist = max_dist
        self.buckets = defaultdict(list)          # (first letter, length) -> words
        for w in vocab:
            self.buckets[(w[0], len(w))].append(w)

    def _limit(self, word: str) -> int:
        return 1 if len(word) <= 4 else self.max_dist     # short words: stricter

    def correct(self, word: str):
        """Best correction, or None if nothing is close enough."""
        word = word.lower()
        if word in self.vocab:
            return word
        if len(word) < 3 or not word.isalpha():
            return None
        limit = self._limit(word)
        best = None
        for length in range(len(word) - limit, len(word) + limit + 1):
            for cand in self.buckets.get((word[0], length), ()):
                d = damerau_levenshtein(word, cand, limit)
                if d <= limit:
                    key = (d, -self.vocab[cand], cand)
                    if best is None or key < best:
                        best = key
        return best[2] if best else None


def _is_known(word, index, preprocess_fn) -> bool:
    tokens = preprocess_fn(word)
    if not tokens:                       # stopword: nothing to check
        return True
    return all(index.doc_freq(t) > 0 for t in tokens)


def did_you_mean(query: str, index, checker: SpellChecker, preprocess_fn):
    """Return a corrected query string, or None if nothing needs fixing."""
    out, changed = [], False
    for tok in query.split():
        if tok in OPERATORS:
            out.append(tok)
            continue
        core = tok.strip('"')
        if not core:
            out.append(tok)
            continue
        lead = tok[: len(tok) - len(tok.lstrip('"'))]
        trail = tok[len(tok.rstrip('"')):]
        if core.isalpha() and not _is_known(core, index, preprocess_fn):
            fix = checker.correct(core)
            if fix and fix != core.lower():
                core, changed = fix, True
        out.append(lead + core + trail)
    return " ".join(out) if changed else None
