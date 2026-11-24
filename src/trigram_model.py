from collections import defaultdict, Counter
import random

class TrigramLanguageModel:
    def __init__(self, vocab_size, discount=0.75):
        self.vocab_size = vocab_size
        self.d = discount
        self.trigram_counts = defaultdict(Counter)
        self.bigram_counts = Counter()       
        # Kneser-Ney "continuation" statistics
        self.cont_bi = Counter()
        self.cont_bi_total = Counter()
        self.followers = Counter()
        self.cont_uni = Counter()
        self.total_bigram_types = 0      

    def fit(self, sentences):
        for sent in sentences:
            for i in range(len(sent) - 2):
                w1, w2, w3 = sent[i], sent[i+1], sent[i+2]
                is_new_trigram = self.trigram_counts[(w1, w2)][w3] == 0
                self.trigram_counts[(w1, w2)][w3] += 1
                self.bigram_counts[(w1, w2)] += 1
                if is_new_trigram:
                    self.cont_bi[(w2, w3)] += 1
                    self.cont_bi_total[w2] += 1
                    if self.cont_bi[(w2, w3)] == 1:
                        self.followers[w2] += 1
                        self.cont_uni[w3] += 1
                        self.total_bigram_types += 1
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