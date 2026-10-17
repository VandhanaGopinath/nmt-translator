"""
train.py - fine-tune mT5 to translate between English, Japanese and Chinese.

    python train.py                       # defaults: mt5-base, 2 epochs, ALT dataset
    python train.py --epochs 1 --max_train_samples 2000    # quick trial run

Needs a GPU for the real run (a free Colab T4 works - see colab_train.ipynb).
The best checkpoint (lowest validation loss) is saved to --output_dir.

Why fp32 and not fp16: mT5 is known to produce NaN losses in half precision,
so we train in full precision.
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from core import (
    DEFAULT_BASE_MODEL,
    LANG_TOKENS,
    MAX_LEN,
    get_device,
    load_model_and_tokenizer,
    load_translation_splits,
    make_batch,
    sample_training_pairs,
    seeded_rng,
    write_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default=DEFAULT_BASE_MODEL)
    p.add_argument("--output_dir", default="outputs/mt5-translator")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8, help="lower to 4 if the GPU runs out of memory")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--max_len", type=int, default=MAX_LEN)
    p.add_argument("--eval_every", type=int, default=500, help="validate every N steps")
    p.add_argument("--val_samples", type=int, default=300, help="validation sentences used for checkpoint selection")
    p.add_argument("--print_every", type=int, default=50)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_file", default=None, help="optional local .jsonl instead of the ALT dataset")
    p.add_argument("--validation_file", default=None)
    p.add_argument("--test_file", default=None)
    return p.parse_args()


@torch.no_grad()
def validation_loss(model, tokenizer, translations, langs, batch_size, max_len, device, seed):
    """Average loss on a FIXED set of validation pairs (same pairs every time, so numbers are comparable)."""
    model.eval()
    pairs = sample_training_pairs(translations, langs, seeded_rng(seed))
    losses = []
    for i in range(0, len(pairs), batch_size):
        batch = make_batch(pairs[i:i + batch_size], tokenizer, max_len, device)
        losses.append(model(**batch).loss.item())
    model.train()
    return float(np.mean(losses))


def save_loss_curve(log_rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in log_rows]
    losses = [r["train_loss"] for r in log_rows]
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses)
    plt.xlabel("training step")
    plt.ylabel("average training loss")
    plt.title("Training loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = seeded_rng(args.seed)
    device = get_device()
    langs = list(LANG_TOKENS.keys())
    print(f"Device: {device}")

    tokenizer, model = load_model_and_tokenizer(args.model_name_or_path, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    splits = load_translation_splits(args.train_file, args.validation_file, args.test_file)
    train_rows = splits["train"]["translation"]
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    val_rows = splits["validation"]["translation"][: args.val_samples] if "validation" in splits else None
    print(f"Training sentences: {len(train_rows)} | validation sentences: {len(val_rows) if val_rows else 0}")

    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)

    model.train()
    step, best_val, best_step = 0, float("inf"), None
    recent, log_rows = [], []
    start = time.time()

    def maybe_validate_and_save():
        nonlocal best_val, best_step
        if not val_rows:
            return
        val = validation_loss(model, tokenizer, val_rows, langs, args.batch_size, args.max_len, device, args.seed)
        flag = ""
        if val < best_val:
            best_val, best_step = val, step
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            flag = "  <- best so far, checkpoint saved"
        print(f"[validation] step {step}: loss {val:.3f}{flag}")

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        for i in range(0, len(order), args.batch_size):
            rows = [train_rows[j] for j in order[i:i + args.batch_size]]
            pairs = sample_training_pairs(rows, langs, rng)
            if not pairs:
                continue
            batch = make_batch(pairs, tokenizer, args.max_len, device)

            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            step += 1
            recent.append(loss.item())
            if step % args.print_every == 0:
                avg = float(np.mean(recent[-args.print_every:]))
                log_rows.append({"step": step, "epoch": epoch, "train_loss": avg, "lr": scheduler.get_last_lr()[0]})
                print(f"epoch {epoch} | step {step}/{total_steps} | train loss {avg:.3f} | "
                      f"{(time.time() - start) / 60:.1f} min elapsed")
            if step % args.eval_every == 0:
                maybe_validate_and_save()

        maybe_validate_and_save()  # end of every epoch

    if best_step is None:  # no validation data: just save the final model
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "training_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "epoch", "train_loss", "lr"])
        writer.writeheader()
        writer.writerows(log_rows)
    if log_rows:
        save_loss_curve(log_rows, os.path.join(args.output_dir, "loss_curve.png"))
    write_json(os.path.join(args.output_dir, "training_summary.json"), {
        **vars(args), "total_steps": total_steps, "best_validation_loss": None if best_step is None else best_val,
        "best_step": best_step, "minutes": round((time.time() - start) / 60, 1),
    })
    print(f"Done. Best model saved in {args.output_dir} (validation loss {best_val:.3f} at step {best_step}).")


if __name__ == "__main__":
    main()
