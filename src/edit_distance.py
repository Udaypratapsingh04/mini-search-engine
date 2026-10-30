"""Edit distances. Both accept max_dist for early exit: if the true distance
is larger than max_dist, they return max_dist + 1 (a cheap "too far")."""


def _clamp(d, max_dist):
    return d if max_dist is None else min(d, max_dist + 1)


def levenshtein(a: str, b: str, max_dist=None) -> int:
    """Minimum inserts, deletes and replaces. O(m*n) time, O(min(m,n)) space."""
    return _clamp(_levenshtein(a, b, max_dist), max_dist)


def _levenshtein(a, b, max_dist):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a                       # b is the shorter string
    if not b:
        return len(a)
    if max_dist is not None and len(a) - len(b) > max_dist:
        return max_dist + 1

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        if max_dist is not None and min(cur) > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


def damerau_levenshtein(a: str, b: str, max_dist=None) -> int:
    """Like Levenshtein, but swapping two adjacent letters costs 1
    (optimal string alignment variant): 'teh' -> 'the' = 1."""
    return _clamp(_damerau(a, b, max_dist), max_dist)


def _damerau(a, b, max_dist):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if max_dist is not None and abs(len(a) - len(b)) > max_dist:
        return max_dist + 1

    prev2 = None
    prev = list(range(len(b) + 1))
    prev_min = 0
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
        cur_min = min(cur)
        # a transposition looks two rows back, so stop only if both rows are too far
        if max_dist is not None and cur_min > max_dist and prev_min > max_dist:
            return max_dist + 1
        prev2, prev, prev_min = prev, cur, cur_min
    return prev[-1]
