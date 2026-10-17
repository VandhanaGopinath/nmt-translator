"""
convert_legacy_checkpoint.py - reuse a model you ALREADY trained in the original notebook.

The original notebook saved only the weights (mt5_translation.pt in Google Drive).
This turns that file into a normal model folder that evaluate.py and app.py can load,
so you can evaluate and demo it without training again.

    python convert_legacy_checkpoint.py --state_dict mt5_translation.pt --output_dir outputs/mt5-legacy

Tip: that model was trained with 20-token sentences, so retraining with train.py
(128 tokens) is expected to give better translations of longer sentences.
"""

import argparse

import torch
from transformers import AutoModelForSeq2SeqLM

from core import DEFAULT_BASE_MODEL, load_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state_dict", required=True, help="path to mt5_translation.pt")
    p.add_argument("--output_dir", default="outputs/mt5-legacy")
    p.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    args = p.parse_args()

    tokenizer = load_tokenizer(args.base_model)  # adds <en>, <jp>, <zh> in the original order
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tokenizer))  # 250103 rows, same as the original notebook

    state = torch.load(args.state_dict, map_location="cpu")
    model.load_state_dict(state, strict=True)  # raises a clear error if shapes/keys differ

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Converted. Model folder saved to {args.output_dir}")


if __name__ == "__main__":
    main()
