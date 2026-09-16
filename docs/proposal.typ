// This proposal uses mousse-notes as a base template.
// https://github.com/dogeystamp/mousse-notes
// Install with `typst init @preview/mousse-notes dir`

#import "@preview/mousse-notes:2.0.0": *

#set document(
  title: [ITM 454],
  author: "Sakphea Seng, Someatra Pum, Techsean Srun, Vithourak Chiv",
)
#set page(paper: "a4")
#set text(size: 12pt)
#show: style

#title-page(
  subtitle: upper[ Final Project Proposal],
  primary: upper[
    Comparative Analysis of N-Gram \
    and Autoregressive Transformer \
    Models for Text Prediction
  ],
  secondary: upper[American University of Phnom Penh \ Professor Monyrath
    Buntoun \ Fall 2026],
)

#set heading(numbering: "1.1")

#let gap-after = 0.75em
#let gap-before = gap-after * 2
#let ratio = 1.15

#show heading.where(level: 1): it => {
  v(gap-before, weak: true)
  set text(size: 1em * calc.pow(ratio, 2), weight: "regular", style: "italic")

  block[
    #if it.numbering != none {
      counter(heading).display()
      h(0.5em)
    }
    #it.body
  ]
  v(gap-after, weak: true)
}

#show heading.where(level: 2): it => {
  set text(size: 1em * ratio, weight: "regular")
  v(gap-before, weak: true)

  block(sticky: true)[
    #if it.numbering != none {
      emph(text(size: 0.8em, counter(heading).display(it.numbering)))
      "."
      h(0.5em)
    }
    #emph(it.body)
    #box(width: 1fr)[#align(right)[#line(length: 100% - 0.8em, start: (0%, -0.225em), stroke: 0.75pt)]]
  ]

  v(gap-after, weak: true)
}

#show heading.where(level: 3): it => {
  v(gap-before, weak: true)
  set text(size: 1em, weight: "regular", style: "italic")

  block[
    #if it.numbering != none {
      counter(heading).display()
      h(0.5em)
    }
    #it.body
  ]
  v(gap-after, weak: true)
}

= Problem Statement

Text prediction (or Language Modeling) is the foundational task of estimating
the probability distribution of natural language text. It powers modern
autocomplete systems, writing assistants, and generative AI. The objective of
this project is to build and evaluate a self-supervised text prediction system
to determine the performance gap between traditional frequency-based statistical
models and modern context-aware neural network architectures.

= Proposed Approach and Methodology

== Text Preprocessing Pipeline

For the statistical baseline, we will implement a custom text-cleaning pipeline:
- *Tokenization:* Tokenization using `nltk.word_tokenize` to isolate words and
  punctuation.
- *Normalization:* Lowercase conversion. Unlike text classification, standard
  English stopwords will *not* be removed, as structural and relational words
  are strictly required for generating coherent sequences.
- *Vocabulary Handling:* Handling Out-Of-Vocabulary (OOV) words by replacing
  low-frequency terms with a special `<UNK>` token to maintain a closed
  vocabulary.

== Machine Learning Implementation

=== Statistical Baseline: Trigram Language Model

We will implement a Trigram Language Model. To avoid calculating the probability
of a word based on an infinitely long history, the model relies on the Markov
assumption, reducing the context to only the two preceding words. The transition
probabilities will be calculated using Maximum Likelihood Estimation (MLE) based
on frequency counts in the training corpus. To handle unseen trigrams, we will
apply Laplace smoothing or Kneser-Ney smoothing.

=== Neural Network Model: GPT-2

We will implement a pre-trained autoregressive, decoder-only transformer
(*GPT-2*) using the Hugging Face `transformers` library. Preprocessing will
bypass our manual pipeline and utilize the model's dedicated Byte-Pair Encoding
(BPE) tokenizer, which splits unknown words into subwords rather than relying on
`<UNK>` tokens, allowing it to predict text with an open vocabulary.

#pagebreak()
= Dataset Description

We will utilize the *WikiText-2* dataset (or a subset of Project Gutenberg).
This dataset contains millions of tokens of clean, continuous English text,
making it ideal for self-supervised training without requiring manual labels.
The text will be split into training, validation, and testing partitions.

= Expected Outcomes and Evaluation Metrics

Because this is a generative task, standard classification metrics like accuracy
or F1-score are not applicable. The models will be evaluated using *Perplexity*,
which measures how well a probability model predicts a sample. A lower
perplexity score indicates that the model is less "surprised" by the test data
and is more accurate at predicting the next sequence of words. We expect the
autoregressive transformer to achieve a significantly lower perplexity score
than the Trigram model due to its ability to retain long-range context beyond
two words.
