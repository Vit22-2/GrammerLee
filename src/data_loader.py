# Dataset loading for the text-prediction project

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SPLITS = ("train", "validation", "test")

# Lines such as " = Valkyria Chronicles III = " or " = = Gameplay = = "
_HEADING_RE = re.compile(r"^\s*=+ .* =+\s*$")

_CONTRACTIONS = ("'s", "n't", "'re", "'ve", "'ll", "'d", "'m")

# Undo WikiText's token-level formatting
def clean_wikitext(text: str) -> str:
  text = text.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
  for c in _CONTRACTIONS:
      text = text.replace(" " + c, c)
  text = re.sub(r" ([.,;:!?%)\]}])", r"\1", text)   # no space before closing punct
  text = re.sub(r"([(\[{$]) ", r"\1", text)          # no space after opening punct
  text = re.sub(r"\s{2,}", " ", text)
  return text.strip()


def _paragraphs_from_lines(lines: Sequence[str], drop_headings: bool, clean: bool) -> List[str]:
  out: List[str] = []
  for line in lines:
      line = line.strip()
      if not line:
          continue
      if drop_headings and _HEADING_RE.match(line):
          continue
      out.append(clean_wikitext(line) if clean else line)
  return out

# Download/load WikiText via Hugging Face `datasets` and return paragraphs.
def load_wikitext(
    config: str = "wikitext-2-raw-v1",
    drop_headings: bool = True,
    clean: bool = True,
    cache_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    
  from datasets import load_dataset
  try:
    ds = load_dataset("Salesforce/wikitext", config, cache_dir=cache_dir)
  except Exception:  # older hub name
    ds = load_dataset("wikitext", config, cache_dir=cache_dir)

  return {
    split: _paragraphs_from_lines(ds[split]["text"], drop_headings, clean)
    for split in SPLITS
  }

# Fallback loader for a local plain-text corpus
def load_text_corpus(
    path: str,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    min_chars: int = 40,
) -> Dict[str, List[str]]:

  if abs(sum(ratios) - 1.0) > 1e-6 or len(ratios) != 3:
    raise ValueError("ratios must be three numbers summing to 1")

  raw = Path(path).read_text(encoding="utf-8", errors="ignore")

  # Strip Project Gutenberg licence boilerplate if present.
  start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw, re.S)
  end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", raw)
  if start:
    raw = raw[start.end():]
  if end:
    raw = raw[: raw.find(end.group(0))] if end.group(0) in raw else raw

  paras = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw)]
  paras = [p for p in paras if len(p) >= min_chars]

  n = len(paras)
  a = int(n * ratios[0])
  b = a + int(n * ratios[1])
  return {"train": paras[:a], "validation": paras[a:b], "test": paras[b:]}

# Short readable summary
def describe(splits: Dict[str, List[str]]) -> str:
  rows = []
  for name in SPLITS:
      paras = splits.get(name, [])
      words = sum(len(p.split()) for p in paras)
      rows.append(f"{name:<11} {len(paras):>7,} paragraphs  ~{words:>10,} words")
  return "\n".join(rows)


if __name__ == "__main__":
  data = load_wikitext()
  print(describe(data))
  print("\nSample:", data["train"][0][:300])