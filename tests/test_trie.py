from src.trie import Trie
from src.autocomplete import Autocomplete

VOCAB = {"machine": 50, "machinery": 5, "mach": 2, "match": 30, "mat": 10,
         "learning": 40, "learn": 20}


def make():
    t = Trie()
    for w, f in VOCAB.items():
        t.insert(w, f)
    return t


def test_contains_and_size():
    t = make()
    assert t.size == len(VOCAB)
    assert t.contains("machine") and t.contains("mat")
    assert not t.contains("machin") and not t.contains("zzz")


def test_multiple_words_for_prefix_ranked_by_freq():
    words = [w for w, _ in make().autocomplete("mach", 10, use_cache=False)]
    assert words == ["machine", "machinery", "mach"]
    assert set(words) == {"machine", "machinery", "mach"}
    assert words[0] == "machine"


def test_exact_word_is_returned():
    assert ("learn", 20) in make().autocomplete("learn", 5)


def test_unknown_prefix():
    assert make().autocomplete("zzz") == []
    assert make().words_with_prefix("zzz") == []


def test_limit_respected():
    assert len(make().autocomplete("ma", 2)) == 2


def test_cache_matches_no_cache():
    t = make()
    expected = {p: t.autocomplete(p, 3, use_cache=False)
                for p in ["m", "ma", "mac", "mach", "l", "lea", "learn", "x"]}
    t.build_cache(5)
    for p, exp in expected.items():
        assert t.autocomplete(p, 3, use_cache=True) == exp


def test_insert_invalidates_cache():
    t = make()
    t.build_cache(5)
    t.insert("machinist", 100)
    assert t.cache_k is None
    assert t.autocomplete("mach", 1)[0][0] == "machinist"


def test_ties_are_alphabetical():
    t = Trie()
    t.insert("bat", 5); t.insert("bar", 5); t.insert("baz", 5)
    assert [w for w, _ in t.autocomplete("ba", 3)] == ["bar", "bat", "baz"]


def test_autocomplete_class_min_prefix_and_case():
    ac = Autocomplete(VOCAB)
    assert ac.suggest("m") == []                  # below min prefix length
    assert ac.suggest("MACH")[0] == "machine"     # case-insensitive
    assert ac.suggest("mach", limit=2) == ["machine", "machinery"]
