import math

TRIGRAM_CONTEXT_SIZE = 2
BOS = "<s>"
EOS = "</s>"


def perplexity(total_log_prob, n_units):
    if n_units <= 0:
        raise ValueError("n_units must be positive")
    return math.exp(-total_log_prob / n_units)


def trigram_total_log_prob(model, sentences):
    padding = [BOS] * TRIGRAM_CONTEXT_SIZE
    total = 0.0
    count = 0
    for sent in sentences:
        if sent[:TRIGRAM_CONTEXT_SIZE] != padding or sent[-1] != EOS:
            raise ValueError("sentence must look like <s> <s> ... </s>")
        for i in range(TRIGRAM_CONTEXT_SIZE, len(sent)):
            context = tuple(sent[i - TRIGRAM_CONTEXT_SIZE : i])
            word = sent[i]
            total += model.log_prob(context, word)
            count += 1
    return total, count


if __name__ == "__main__":
    VOCAB_SIZE = 1000

    assert abs(perplexity(-5 * math.log(VOCAB_SIZE), 5) - VOCAB_SIZE) < 1e-9

    class Uniform:
        def log_prob(self, context, word):
            return -math.log(VOCAB_SIZE)

    sents = [
        ["<s>", "<s>", "the", "cat", "sat", "</s>"],
        ["<s>", "<s>", "hi", "</s>"],
    ]
    total, n = trigram_total_log_prob(Uniform(), sents)
    assert n == 6  # based on the above sents variable, (the, cat, sat, </s>, hi, </s>)
    assert abs(perplexity(total, n) - VOCAB_SIZE) < 1e-6
    print("ok")
