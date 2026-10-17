"""
core.py - the pieces shared by training, evaluation, conversion and the app.

Keeping them in ONE place means the text is prepared exactly the same way
when the model learns and when it translates, so the two can never drift apart.

How this translator works (one model, many directions):
  * The base model is Google's mT5 (a multilingual encoder-decoder Transformer).
  * We add one special token per language: <en>, <jp>, <zh>.
  * To translate into Japanese we feed the model:  "<jp>" + source sentence.
    The token tells the model WHICH language to write; the model works out the
    source language by itself. This is how a single model covers every
    direction (en->ja, ja->zh, zh->en, ...).
"""

import json
import random

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# NOTE: the ORDER here matters. The tokens get ids 250100, 250101, 250102 when
# added to mT5's tokenizer, and a model trained earlier relies on that order.
LANG_TOKENS = {"en": "<en>", "ja": "<jp>", "zh": "<zh>"}
LANG_NAMES = {"en": "English", "ja": "Japanese", "zh": "Chinese"}

DEFAULT_BASE_MODEL = "google/mt5-base"
MAX_LEN = 128  # max tokens per sentence (the first version used 20, which cut sentences off)


# --------------------------------------------------------------------------
# Model + tokenizer
# --------------------------------------------------------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(path_or_name):
    """Load the tokenizer and make sure the three language tokens exist.

    use_fast=False keeps mT5's original SentencePiece behaviour, which turns
    rare characters (common in Japanese/Chinese) into byte tokens instead of
    <unk>. Safe to call on an already fine-tuned folder: tokens that exist
    are simply reused.
    """
    tokenizer = AutoTokenizer.from_pretrained(path_or_name, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": list(LANG_TOKENS.values())})
    return tokenizer


def load_model_and_tokenizer(path_or_name, device=None):
    """Load tokenizer + model from a Hugging Face id (google/mt5-base) or a local folder."""
    device = device or get_device()
    tokenizer = load_tokenizer(path_or_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(path_or_name)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        # the 3 new language tokens need rows in the embedding table
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return tokenizer, model


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_translation_splits(train_file=None, validation_file=None, test_file=None):
    """Return {"train", "validation", "test"} datasets, each row has a
    "translation" dict like {"en": "...", "ja": "...", "zh": "..."}.

    Default: the ALT (Asian Language Treebank) parallel corpus from the Hugging
    Face Hub. You can instead pass local .jsonl files (one JSON object per line
    with a "translation" dict) - handy for offline tests or your own data.
    """
    from datasets import load_dataset

    if train_file:
        files = {"train": train_file}
        if validation_file:
            files["validation"] = validation_file
        if test_file:
            files["test"] = test_file
        return load_dataset("json", data_files=files)

    errors = []
    for args in (("mutiyama/alt", "alt-parallel"), ("alt", "alt-parallel"), ("alt",)):
        try:
            return load_dataset(*args)
        except Exception as exc:  # try the next known name
            errors.append(f"{args}: {exc}")
    raise RuntimeError("Could not load the ALT dataset. Tried:\n" + "\n".join(errors))


def sample_training_pairs(translations, langs, rng):
    """Turn a list of ALT 'translation' dicts into (source_text, target_text) pairs.

    For each sentence we pick two different languages at random: one is the
    source, the other the target, and the target's language token is put in
    front of the source. Over many epochs every direction gets practised.
    """
    pairs = []
    for row in translations:
        src, tgt = rng.sample(langs, 2)
        src_text, tgt_text = row.get(src), row.get(tgt)
        if not src_text or not tgt_text:  # some rows miss a language
            continue
        pairs.append((LANG_TOKENS[tgt] + src_text, tgt_text))
    return pairs


def make_batch(pairs, tokenizer, max_len, device):
    """Tokenize (source, target) pairs into tensors the model can train on.

    * Sentences are padded only to the longest one in THIS batch (fast).
    * attention_mask tells the model which positions are real words, not padding.
    * Padding in the labels is set to -100 so it does not count in the loss.
    """
    sources = [s for s, _ in pairs]
    targets = [t for _, t in pairs]
    enc = tokenizer(sources, max_length=max_len, truncation=True, padding=True, return_tensors="pt")
    lab = tokenizer(text_target=targets, max_length=max_len, truncation=True, padding=True, return_tensors="pt")
    labels = lab["input_ids"].masked_fill(lab["attention_mask"] == 0, -100)
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
        "labels": labels.to(device),
    }


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------
@torch.no_grad()
def translate(texts, target_lang, tokenizer, model, num_beams=4, max_new_tokens=MAX_LEN, batch_size=16):
    """Translate one string or a list of strings into target_lang ("en", "ja" or "zh")."""
    single = isinstance(texts, str)
    texts = [texts] if single else list(texts)
    model.eval()
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = [LANG_TOKENS[target_lang] + (t or "").strip() for t in texts[i:i + batch_size]]
        enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        enc = enc.to(model.device)
        out = model.generate(**enc, num_beams=num_beams, max_new_tokens=max_new_tokens)
        results.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return results[0] if single else results


def seeded_rng(seed):
    return random.Random(seed)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
