import argparse
import io
import itertools
import sys
import threading
import time
from contextlib import contextmanager, redirect_stderr

from src.metrics import (
    count_scored_words,
    gpt2_total_log_prob,
    run_benchmark,
    trigram_total_log_prob,
)
from src.trigram_model import TrigramLanguageModel

MIN_FREQ = 3
GPT2_NAME = "gpt2"
N_TEST_PARAGRAPHS = None
PROMPTS = ("the history", "in the", "he was")
MAX_NEW_TOKENS = 30
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark a trigram model against GPT-2, or continue a prompt with GPT-2."
    )
    parser.add_argument(
        "--prompt", help="text for GPT-2 to continue (skips the benchmark)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="length limit for --prompt, in GPT-2 tokens (default: %(default)s)",
    )
    return parser.parse_args()


@contextmanager
def waiting(message):
    """Show 'message...' with repeating dots on stderr while the block runs."""
    out = sys.stderr
    done = threading.Event()

    def spin():
        if not out.isatty():
            return
        for dots in itertools.cycle(("", ".", "..", "...")):
            if done.is_set():
                break
            print(f"\r{message}{dots:<3}", end="", file=out, flush=True)
            time.sleep(0.4)

    thread = threading.Thread(target=spin)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()
        if out.isatty():
            print("\r" + " " * (len(message) + 3) + "\r", end="", file=out)


def trim_to_sentence(text, prompt):
    end = max(text.rfind(mark) for mark in ".!?")
    return text[: end + 1] if end >= len(prompt) else text


def run_prompt(prompt, max_new_tokens):
    with waiting("Loading GPT-2"), redirect_stderr(io.StringIO()):
        from src.gpt2_model import GPT2TextGenerator

        gpt2 = GPT2TextGenerator(GPT2_NAME)
    with waiting("Prompting"):
        text = gpt2.generate(prompt, max_new_tokens, seed=SEED)
    print(trim_to_sentence(text, prompt))


def run_full_benchmark():
    from src.data_loader import load_wikitext
    from src.gpt2_model import GPT2TextGenerator
    from src.preprocessing import NLTKPreprocessor

    data = load_wikitext()
    test_paragraphs = data["test"][:N_TEST_PARAGRAPHS]

    pre = NLTKPreprocessor(min_freq=MIN_FREQ).fit(data["train"])
    train_sents = pre.transform(data["train"])
    test_sents = pre.transform(test_paragraphs)
    test_text = "\n\n".join(test_paragraphs)
    n = count_scored_words(test_sents)

    trigram_kn = TrigramLanguageModel(smoothing="kneser_ney").fit(train_sents)
    trigram_laplace = TrigramLanguageModel(smoothing="laplace").fit(train_sents)
    gpt2 = GPT2TextGenerator(GPT2_NAME)

    scorers = {
        "Trigram (Laplace)": lambda: trigram_total_log_prob(
            trigram_laplace, test_sents
        ),
        "Trigram (Kneser-Ney)": lambda: trigram_total_log_prob(trigram_kn, test_sents),
        "GPT-2": lambda: gpt2_total_log_prob(gpt2.model, gpt2.tokenizer, test_text),
    }
    results = run_benchmark(scorers, n)

    for name, ppl in results.items():
        print(f"{name}: {ppl:,.2f}")

    generators = {"Trigram (Kneser-Ney)": trigram_kn, "GPT-2": gpt2}
    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt!r}")
        for name, generator in generators.items():
            print(f"  [{name}] {generator.generate(prompt, MAX_NEW_TOKENS, seed=SEED)}")


def main():
    args = parse_args()
    if args.prompt:
        run_prompt(args.prompt, args.max_new_tokens)
    else:
        run_full_benchmark()


if __name__ == "__main__":
    main()
