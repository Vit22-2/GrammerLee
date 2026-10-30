# NLTK-based text preprocessing for Trigram model

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import nltk

UNK = "<UNK>"
BOS = "<s>"
EOS = "</s>"
SPECIAL_TOKENS = (UNK, BOS, EOS)

# Download the sentence tokenizer data if missing
def _ensure_nltk() -> None:
  for resource, path in (("punkt_tab", "tokenizers/punkt_tab"), ("punkt", "tokenizers/punkt")):
    try:
      nltk.data.find(path)
      return
    except LookupError:
      try:
        if nltk.download(resource, quiet=True):
          nltk.data.find(path)
          return
      except Exception:
        continue
  raise RuntimeError(
      "Could not obtain NLTK 'punkt_tab'/'punkt'. Run: python -m nltk.downloader punkt_tab"
  )

# Lowercasing, tokenization and vocabulary mapping for the n-gram model.
class NLTKPreprocessor:
  def __init__(
      self,
      min_freq: int = 2,
      max_vocab: Optional[int] = None,
      lowercase: bool = True,
      n_bos: int = 2,
  ) -> None:
      
    self.min_freq = min_freq
    self.max_vocab = max_vocab
    self.lowercase = lowercase
    self.n_bos = n_bos
    self.word2idx: Dict[str, int] = {}
    self.idx2word: Dict[int, str] = {}
    self.word_counts: Counter = Counter()
    _ensure_nltk()

    # Tokenizing raw text to list of sentences
  def tokenize(self, text: str) -> List[List[str]]:
    if self.lowercase:
        text = text.lower()
    return [nltk.word_tokenize(s) for s in nltk.sent_tokenize(text)]

  # Build vocabulary from training text
  def fit(self, paragraphs: Iterable[str]) -> "NLTKPreprocessor":
    counts: Counter = Counter()
    for para in paragraphs:
      for sent in self.tokenize(para):
        counts.update(sent)
    self.word_counts = counts

    kept = [(w, c) for w, c in counts.most_common() if c >= self.min_freq]
    if self.max_vocab is not None:
      kept = kept[: max(0, self.max_vocab - len(SPECIAL_TOKENS))]

    words = list(SPECIAL_TOKENS) + [w for w, _ in kept if w not in SPECIAL_TOKENS]
    self.word2idx = {w: i for i, w in enumerate(words)}
    self.idx2word = {i: w for w, i in self.word2idx.items()}
    return self

  @property
  def vocab_size(self) -> int:
    return len(self.word2idx)

  def _require_fit(self) -> None:
    if not self.word2idx:
      raise RuntimeError("Call fit() (or load_vocab()) before transform().")

  # Replace out-of-vocabulary tokens with <UNK>
  def map_unk(self, tokens: Sequence[str]) -> List[str]:
    self._require_fit()
    return [t if t in self.word2idx else UNK for t in tokens]

  # Transform paragraphs to list of sentences
  def transform(self, paragraphs: Iterable[str], add_boundaries: bool = True) -> List[List[str]]:
    self._require_fit()
    out: List[List[str]] = []
    for para in paragraphs:
      for sent in self.tokenize(para):
        if not sent:
          continue
        
        toks = self.map_unk(sent)

        if add_boundaries:
          toks = [BOS] * self.n_bos + toks + [EOS]
        out.append(toks)
    return out
  
  # Token strings to ids
  def encode(self, sentences: Iterable[Sequence[str]]) -> List[List[int]]:
    self._require_fit()
    unk = self.word2idx[UNK]
    return [[self.word2idx.get(t, unk) for t in s] for s in sentences]

  def decode(self, ids: Iterable[int]) -> List[str]:
    return [self.idx2word.get(i, UNK) for i in ids]

  @staticmethod
  def flatten(sentences: Iterable[Sequence]) -> list:
    """One continuous token stream (handy for perplexity over a whole split)."""
    return [t for s in sentences for t in s]

  # Turn what the user has typed into trigram context tokens.
  def prepare_prompt(self, text: str) -> List[str]:
    self._require_fit()
    sents = self.tokenize(text)
    toks = self.map_unk(sents[-1] if sents else [])
    return [BOS] * self.n_bos + toks

  # Stats
  def oov_rate(self, paragraphs: Iterable[str]) -> float:
    self._require_fit()
    total = unk = 0
    for para in paragraphs:
      for sent in self.tokenize(para):
        total += len(sent)
        unk += sum(1 for t in sent if t not in self.word2idx)
    return unk / total if total else 0.0

  # Persistence
  def save_vocab(self, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
      "config": {
        "min_freq": self.min_freq,
        "max_vocab": self.max_vocab,
        "lowercase": self.lowercase,
        "n_bos": self.n_bos,
      },
      "word2idx": self.word2idx,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

  @classmethod
  def load_vocab(cls, path: str) -> "NLTKPreprocessor":
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pre = cls(**payload["config"])
    pre.word2idx = {w: int(i) for w, i in payload["word2idx"].items()}
    pre.idx2word = {i: w for w, i in pre.word2idx.items()}
    return pre

  # Save processed sentences as JSON
  @staticmethod
  def export_tokens(sentences: Sequence[Sequence], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sentences, ensure_ascii=False), encoding="utf-8")

  @staticmethod
  def load_tokens(path: str) -> List[List]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
  from src.data_loader import load_wikitext

  data = load_wikitext()
  pre = NLTKPreprocessor(min_freq=3).fit(data["train"])
  print("vocab size:", pre.vocab_size)
  print("test OOV rate:", f"{pre.oov_rate(data['test']):.2%}")
  sents = pre.transform(data["test"][:3])
  print(sents[0])
