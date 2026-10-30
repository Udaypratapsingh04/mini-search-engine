from src.trie import Trie


class Autocomplete:
    """Frequency-ranked word suggestions from a {word: doc_freq} vocabulary."""

    def __init__(self, vocab: dict, cache_k: int = 5, min_prefix: int = 2):
        self.trie = Trie()
        for word, freq in vocab.items():
            self.trie.insert(word, freq)
        if cache_k:
            self.trie.build_cache(cache_k)
        self.min_prefix = min_prefix

    def suggest(self, prefix: str, limit: int = 5, use_cache: bool = True):
        prefix = prefix.strip().lower()
        if len(prefix) < self.min_prefix:
            return []
        return [w for w, _ in self.trie.autocomplete(prefix, limit, use_cache)]
