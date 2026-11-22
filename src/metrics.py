import math


def perplexity(total_log_prob, n_units):
    if n_units <= 0:
        raise ValueError("n_units must be positive")
    return math.exp(-total_log_prob / n_units)


if __name__ == "__main__":
    VOCAB_SIZE = 1000

    assert abs(perplexity(-5 * math.log(VOCAB_SIZE), 5) - VOCAB_SIZE) < 1e-9
    print("ok")
