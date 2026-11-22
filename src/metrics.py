import math

TRIGRAM_CONTEXT_SIZE = 2
BOS = "<s>"
EOS = "</s>"


def perplexity(total_log_prob, n_units):
    if n_units <= 0:
        raise ValueError("n_units must be positive")
    return math.exp(-total_log_prob / n_units)


def count_scored_words(sentences):
    return sum(len(sent) - TRIGRAM_CONTEXT_SIZE for sent in sentences)


def trigram_total_log_prob(model, sentences):
    padding = [BOS] * TRIGRAM_CONTEXT_SIZE
    total = 0.0
    for sent in sentences:
        if sent[:TRIGRAM_CONTEXT_SIZE] != padding or sent[-1] != EOS:
            raise ValueError("sentence must look like <s> <s> ... </s>")
        for i in range(TRIGRAM_CONTEXT_SIZE, len(sent)):
            context = tuple(sent[i - TRIGRAM_CONTEXT_SIZE : i])
            word = sent[i]
            total += model.log_prob(context, word)
    return total


def run_benchmark(scorers, n_units):
    results = {}
    for name, score in scorers.items():
        results[name] = perplexity(score(), n_units)
    return results


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
    n = count_scored_words(sents)
    assert n == 6  # based on the above sents variable, (the, cat, sat, </s>, hi, </s>)

    total = trigram_total_log_prob(Uniform(), sents)
    assert abs(perplexity(total, n) - VOCAB_SIZE) < 1e-6

    scorers = {"uniform": lambda: trigram_total_log_prob(Uniform(), sents)}
    results = run_benchmark(scorers, n)
    assert abs(results["uniform"] - VOCAB_SIZE) < 1e-6
    print("ok")
