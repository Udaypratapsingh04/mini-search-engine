from functools import lru_cache
from nltk.stem import PorterStemmer
from src.tokenizer import tokenize

_stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def preprocess(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text)]
