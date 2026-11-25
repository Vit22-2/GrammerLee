from __future__ import annotations

from collections import defaultdict, Counter
import math
import random
from typing import Iterable, Sequence, Tuple

UNK = "<UNK>"
BOS = "<s>"
EOS = "</s>"

class TrigramLanguageModel:
    def __init__(self, smoothing: str = "kneser_ney", discount: float = 0.75) -> None:
        if smoothing not in ("kneser_ney", "laplace"):
            raise ValueError("smoothing must be 'kneser_ney' or 'laplace'")
        if not 0.0 < discount < 1.0:
            raise ValueError("discount must be strictly between 0 and 1")
        self.smoothing = smoothing
        self.discount = discount
        self.vocab: set = set()
        self._fitted = False
        self._prob = self._kn_prob if smoothing == "kneser_ney" else self._laplace_prob 

    def fit(self, sentences: Iterable[Sequence[str]]) -> "TrigramLanguageModel":
        d = self.discount
 
        # 1) raw counts: (w1, w2) -> Counter({w3: count}); vocabulary from the tokens
        tri = defaultdict(Counter)
        vocab = {UNK, BOS, EOS}
        for sent in sentences:
            vocab.update(sent)
            for i in range(len(sent) - 2):
                tri[(sent[i], sent[i + 1])][sent[i + 2]] += 1
        if not tri:
            raise ValueError("No trigrams found: need padded sentences with >= 3 tokens")
 
        self.vocab = vocab
        self._V = len(vocab)
 
        # 2) trigram level: per context -> (counts, C(w1,w2), interpolation weight)
        self._tri = {}
        for ctx, nxt in tri.items():
            total = sum(nxt.values())
            self._tri[ctx] = (dict(nxt), total, d * len(nxt) / total)
 
        # 3) bigram level (continuation counts): N1+(. w2 w3) = distinct w1 before (w2, w3)
        cont_bi = Counter()
        for (_w1, w2), nxt in tri.items():
            for w3 in nxt:
                cont_bi[(w2, w3)] += 1
 
        bi_total = Counter()   # w2 -> sum over w3 of cont_bi
        followers = Counter()  # w2 -> number of distinct w3 after it
        cont_uni = Counter()   # w3 -> number of distinct w2 before it
        for (w2, w3), c in cont_bi.items():
            bi_total[w2] += c
            followers[w2] += 1
            cont_uni[w3] += 1
 
        self._cont_bi = dict(cont_bi)
        self._bi = {w2: (t, d * followers[w2] / t) for w2, t in bi_total.items()}
 
        # 4) unigram level: continuation probability, with the leftover discount mass
        #    spread uniformly over the whole vocabulary so nothing is ever zero
        n_types = len(cont_bi)
        floor = (d * len(cont_uni) / n_types) / self._V
        self._p_uni = {w: max(cont_uni.get(w, 0) - d, 0.0) / n_types + floor for w in vocab}
 
        # words the generator may sample (a sentence never continues with <s>)
        self._gen_vocab = sorted(vocab - {BOS})
        self._fitted = True
        return self

    def _kn_prob(self, w1: str, w2: str, w3: str) -> float:
        d = self.discount
        p_uni = self._p_uni[w3]
 
        bi = self._bi.get(w2)
        if bi is None:                                   # w2 never had a follower
            p_bi = p_uni
        else:
            total, lam = bi
            p_bi = max(self._cont_bi.get((w2, w3), 0) - d, 0.0) / total + lam * p_uni
 
        tri = self._tri.get((w1, w2))
        if tri is None:                                  # unseen context -> back off
            return p_bi
        counts, total, lam = tri
        c = counts.get(w3, 0)
        if c:
            return (c - d) / total + lam * p_bi
        return lam * p_bi
 
    def _laplace_prob(self, w1: str, w2: str, w3: str) -> float:
        tri = self._tri.get((w1, w2))
        if tri is None:
            return 1.0 / self._V
        counts, total, _ = tri
        return (counts.get(w3, 0) + 1) / (total + self._V)

    def log_prob(self, context: Tuple[str, str], word: str) -> float:
        """Natural log of P(word | context). context = (w_{t-2}, w_{t-1}). Never -inf."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        vocab = self.vocab
        w1, w2 = context
        if w1 not in vocab:
            w1 = UNK
        if w2 not in vocab:
            w2 = UNK
        if word not in vocab:
            word = UNK
        return math.log(self._prob(w1, w2, word))

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        """Per-token perplexity over padded sentences (the two <s> are context only)."""
        total, n = 0.0, 0
        for sent in sentences:
            for i in range(2, len(sent)):
                total += self.log_prob((sent[i - 2], sent[i - 1]), sent[i])
                n += 1
        return math.exp(-total / n)

    def generate(self, prompt: str, max_new_tokens: int = 20, seed=None) -> str:
        """Continue a plain-text prompt (e.g. "the first"). Returns only the new words."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        rng = random.Random(seed)
        words = [w if w in self.vocab else UNK for w in prompt.lower().split()]
        history = [BOS, BOS] + words          # pads prompts shorter than 2 words
        w1, w2 = history[-2], history[-1]
 
        choices = self._gen_vocab
        out = []
        for _ in range(max_new_tokens):
            weights = [self._prob(w1, w2, w) for w in choices]
            nxt = rng.choices(choices, weights=weights, k=1)[0]
            if nxt == EOS:
                break
            out.append(nxt)
            w1, w2 = w2, nxt
        return " ".join(out)