"""Trie for prefix search.

Prefix lookup is O(p) (p = prefix length), independent of dictionary size.
Each word's end node stores its document frequency. Optionally every node
caches its top-k words (space-for-time trade-off).
"""
import heapq


class TrieNode:
    __slots__ = ("children", "is_end", "freq", "top")

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.freq = 0
        self.top = None          # cached top-k [(word, freq), ...]


def _rank_key(item):
    # highest frequency first, then alphabetical (deterministic)
    return (-item[1], item[0])


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.size = 0            # number of distinct words
        self.cache_k = None      # k used by build_cache, None = no valid cache

    def insert(self, word: str, freq: int = 1) -> None:
        node = self.root
        for ch in word:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = node.children[ch] = TrieNode()
            node = nxt
        if not node.is_end:
            node.is_end = True
            self.size += 1
        node.freq += freq
        self.cache_k = None      # cache is stale after any insert

    def _find(self, prefix: str):
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def contains(self, word: str) -> bool:
        node = self._find(word)
        return bool(node and node.is_end)

    def words_with_prefix(self, prefix: str):
        """All (word, freq) under prefix, via iterative DFS."""
        start = self._find(prefix)
        if start is None:
            return []
        out = []
        stack = [(start, prefix)]
        while stack:
            node, word = stack.pop()
            if node.is_end:
                out.append((word, node.freq))
            for ch, child in node.children.items():
                stack.append((child, word + ch))
        return out

    def build_cache(self, k: int = 5) -> None:
        """Store the top-k words at every node (post-order)."""
        def build(node, prefix):
            cands = [(prefix, node.freq)] if node.is_end else []
            for ch, child in node.children.items():
                cands.extend(build(child, prefix + ch))
            node.top = heapq.nsmallest(k, cands, key=_rank_key)
            return node.top

        build(self.root, "")
        self.cache_k = k

    def autocomplete(self, prefix: str, limit: int = 5, use_cache: bool = True):
        """Top `limit` (word, freq) pairs for prefix, most frequent first."""
        node = self._find(prefix)
        if node is None:
            return []
        if use_cache and self.cache_k is not None and limit <= self.cache_k:
            return list(node.top[:limit])         # cache holds full words
        return heapq.nsmallest(limit, self.words_with_prefix(prefix), key=_rank_key)
