import torch

from src.metrics import GPT2_CONTEXT_SIZE


def suggest_next_words(model, tokenizer, text, k=3):
    if not text.strip():
        return []

    device = next(model.parameters()).device
    ids = tokenizer(text.rstrip(), return_tensors="pt").input_ids
    ids = ids[:, -GPT2_CONTEXT_SIZE:].to(device)

    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    probs = torch.softmax(logits, dim=-1)

    words = []
    seen = set()
    for token_id in torch.topk(probs, 50).indices.tolist():
        piece = tokenizer.decode([token_id])
        word = piece.strip()
        if piece.startswith(" ") and word.isalpha() and word.lower() not in seen:
            seen.add(word.lower())
            words.append(word)
        if len(words) == k:
            break
    return words
