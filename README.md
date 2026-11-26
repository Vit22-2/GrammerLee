# GrammarLee: Comparative Analysis of N-Gram and Autoregressive Transformer Models for Text Prediction

Compares a trigram language model (Laplace and Kneser-Ney smoothing) with
pretrained GPT-2 on WikiText-2, using test-set perplexity normalised per word so
the two models are comparable.

## Setup

Tested with Python 3.12 and [uv](https://docs.astral.sh/uv/).

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

### Benchmark

```
python main.py
```

The first run will download the WikiText-2 dataset (Hugging Face `datasets`),
the GPT-2 weights (about 500 MB), and the NLTK tokenizer data automatically.

It then prints the test-set perplexity of each model and sample completions for
a few prompts. For a quick test, set `N_TEST_PARAGRAPHS` at the top of `main.py`
to a small number such as 50 (the scores are then not comparable with the
full-test-set results below).

### Continue a prompt with GPT-2

```
python main.py --prompt "the history of" --max-new-tokens 40
```

Prints GPT-2's continuation of the prompt.

### Next-word suggestions

```
python main.py --suggest
```

Type some text and press Enter to see three suggested next words. Type `1`,
`2` or `3` to append a suggestion, or keep typing.

### All options

```
python main.py --help
```

## Results (full WikiText-2 test set)

| Model                      | Test perplexity (per word) |
| -------------------------- | -------------------------- |
| Trigram, Laplace smoothing | 9,599.75                   |
| Trigram, Kneser-Ney        | 218.72                     |
| GPT-2 (`gpt2`, zero-shot)  | 32.53                      |

## Project structure

- `main.py`: loads the data, fits the models, runs the benchmark, prints samples
- `src/data_loader.py`: loads and cleans WikiText-2
- `src/preprocessing.py`: NLTK tokenization, vocabulary, `<UNK>` handling,
  sentence padding
- `src/trigram_model.py`: trigram model with Laplace and Kneser-Ney smoothing,
  text generation
- `src/gpt2_model.py`: GPT-2 wrapper (Hugging Face) and text generation
- `src/metrics.py`: perplexity for the trigram models and for GPT-2 (sliding
  window), and the benchmark runner

## How the comparison works

The trigram predicts whole words and GPT-2 predicts subword (BPE) tokens, so a
per-token perplexity would not be comparable. Both models' total log-probability
on the test set is divided by the same count: the number of word-level (NLTK)
tokens, including punctuation and one end-of-sentence marker per sentence. GPT-2
is scored with a sliding window (1024 tokens, stride 512) so that each token has
as much context as possible.

> [!NOTE]
> GPT-2 is used as pretrained, without fine-tuning on WikiText-2, while the
> trigram is trained on it. Rare words (fewer than 3 occurrences in the training
> split) become `<UNK>` for the trigram, which makes its task slightly easier.

## License

This project is licensed under the [MIT License](LICENSE).
