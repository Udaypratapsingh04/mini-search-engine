"""Day 14: accuracy of SpellChecker on 30 handmade typos."""
from src.spellcheck import SpellChecker
from src.vocab import load_vocab

# (typo, intended word)
CASES = [
    ("comptuer", "computer"), ("graphcis", "graphics"), ("windwos", "windows"),
    ("sofware", "software"), ("hardwre", "hardware"), ("porgram", "program"),
    ("internt", "internet"), ("scence", "science"), ("goverment", "government"),
    ("religon", "religion"), ("hocky", "hockey"), ("basebal", "baseball"),
    ("encrption", "encryption"), ("presdent", "president"), ("questoin", "question"),
    ("problm", "problem"), ("systm", "system"), ("becuase", "because"),
    ("univeristy", "university"), ("footbal", "football"), ("medcal", "medical"),
    ("drvier", "driver"), ("netwrk", "network"), ("keybord", "keyboard"),
    ("monitr", "monitor"), ("memry", "memory"), ("shuttel", "shuttle"),
    ("moterbike", "motorbike"), ("diesase", "disease"), ("secuirty", "security"),
]

if __name__ == "__main__":
    vocab = load_vocab()
    checker = SpellChecker(vocab)
    right = 0
    for typo, want in CASES:
        got = checker.correct(typo)
        ok = got == want
        right += ok
        mark = "OK  " if ok else "MISS"
        note = "" if want in vocab else "   (intended word not in vocab)"
        print(f"{mark} {typo:12} -> {str(got):12} (wanted {want}){note}")
    acc = 100 * right / len(CASES)
    print(f"\nAccuracy: {right}/{len(CASES)} = {acc:.0f}%")
