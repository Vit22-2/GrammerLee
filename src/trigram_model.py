from collections import defaultdict, Counter
import random

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

    def _p_uni(self, w3):
        d = self.d
        total = self.total_bigram_types
        p = max(self.cont_uni.get(w3, 0) - d, 0) / total
        leftover = d * len(self.cont_uni) / total
        return p + leftover / self.vocab_size

    def _p_bi(self, w2, w3):
        d = self.d
        ctx = self.cont_bi_total.get(w2, 0)
        if ctx == 0:
            return self._p_uni(w3)
        p = max(self.cont_bi.get((w2, w3), 0) - d, 0) / ctx
        lam = d * self.followers[w2] / ctx
        return p + lam * self._p_uni(w3)

    def kneser_ney_prob(self, w1, w2, w3):
        d = self.d
        c_ctx = self.bigram_counts.get((w1, w2), 0)
        if c_ctx == 0:
            return self._p_bi(w2, w3)
        c3 = self.trigram_counts[(w1, w2)].get(w3, 0)
        lam = d * len(self.trigram_counts[(w1, w2)]) / c_ctx
        return max(c3 - d, 0) / c_ctx + lam * self._p_bi(w2, w3)

    def mle_prob(self, w1, w2, w3):
        context_count = self.bigram_counts.get((w1, w2), 0)
        if context_count == 0:
            return 0.0
        return self.trigram_counts.get((w1, w2), {}).get(w3, 0) / context_count

    def laplace_prob(self, w1, w2, w3):
        c3 = self.trigram_counts.get((w1, w2), {}).get(w3, 0)
        c2 = self.bigram_counts.get((w1, w2), 0)
        return (c3 + 1) / (c2 + self.vocab_size)

    def generate(self, prompt_tokens, max_len=20, smoothing="laplace"):
        w1, w2 = prompt_tokens[-2:]
        output = [w1, w2]

        prob_fn = {"mle": self.mle_prob, "laplace": self.laplace_prob,
                "kn": self.kneser_ney_prob}[smoothing]

        for _ in range(max_len):
            candidates = self.trigram_counts.get((w1, w2))
            if not candidates:
                break

            words, probs = zip(*[(w3, prob_fn(w1, w2, w3)) for w3 in candidates])
            next_word = random.choices(words, weights=probs, k=1)[0]

            if next_word == "</s>":
                break
            output.append(next_word)
            w1, w2 = w2, next_word

        return " ".join(w for w in output if w not in ("<s>", "</s>"))