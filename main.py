from src.data_loader import load_wikitext
from src.gpt2_model import GPT2TextGenerator
from src.metrics import (
    count_scored_words,
    gpt2_total_log_prob,
    run_benchmark,
    trigram_total_log_prob,
)
from src.preprocessing import NLTKPreprocessor
from src.trigram_model import TrigramLanguageModel

MIN_FREQ = 3
GPT2_NAME = "gpt2"
N_TEST_PARAGRAPHS = None
PROMPTS = ("the history", "in the", "he was")
MAX_NEW_TOKENS = 30


def main():
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
        "Trigram (Laplace)": lambda: trigram_total_log_prob(trigram_laplace, test_sents),
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
            print(f"  [{name}] {generator.generate(prompt, MAX_NEW_TOKENS)}")


if __name__ == "__main__":
    main()
