"""
Offline smoke test: runs the WHOLE pipeline (train -> evaluate -> convert -> app) on a
tiny made-up model, tokenizer and dataset, in about a minute on a laptop CPU.

It does not measure translation quality (the model is tiny and the data is a toy).
It checks the plumbing: tokens, batching, label masking, saving/loading, scoring, the app.

    python tests/smoke_test.py
"""

import json
import os
import subprocess
import sys
import tempfile

import sentencepiece as spm
import torch
import transformers
from transformers import AutoModelForSeq2SeqLM, MT5Config, MT5ForConditionalGeneration, T5Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from core import LANG_TOKENS, load_model_and_tokenizer, load_tokenizer, make_batch, translate  # noqa: E402

PY = sys.executable
ANIMALS = [("cat", "猫", "猫"), ("dog", "犬", "狗"), ("bird", "鳥", "鸟"), ("fish", "魚", "鱼"), ("horse", "馬", "马")]
VERBS = [("runs", "走る", "跑"), ("sleeps", "寝る", "睡"), ("eats", "食べる", "吃")]
ADVERBS = [("fast", "速く", "快"), ("slowly", "ゆっくり", "慢慢地"), ("often", "よく", "经常")]


def check(condition, message):
    print(("PASS  " if condition else "FAIL  ") + message)
    if not condition:
        sys.exit(1)


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        print(result.stdout[-3000:], result.stderr[-3000:])
        sys.exit(f"command failed: {' '.join(cmd)}")
    return result.stdout


def build_toy_world(tmp):
    rows = []
    for a in ANIMALS:
        for v in VERBS:
            for adv in ADVERBS:
                rows.append({"translation": {
                    "en": f"the {a[0]} {v[0]} {adv[0]}",
                    "ja": f"{a[1]}が{adv[1]}{v[1]}",
                    "zh": f"{a[2]}{adv[2]}{v[2]}",
                }})
    train = rows * 4  # 180 rows
    paths = {}
    for name, data in (("train", train), ("validation", rows[:30]), ("test", rows[:30])):
        paths[name] = os.path.join(tmp, f"{name}.jsonl")
        with open(paths[name], "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    corpus = os.path.join(tmp, "corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for r in rows:
            f.write("\n".join(r["translation"].values()) + "\n")
    spm.SentencePieceTrainer.train(
        input=corpus, model_prefix=os.path.join(tmp, "toy_spm"), vocab_size=90, model_type="unigram",
        character_coverage=1.0, pad_id=0, eos_id=1, unk_id=2, bos_id=-1, hard_vocab_limit=False,
    )
    spm_path = os.path.join(tmp, "toy_spm.model")
    if int(transformers.__version__.split(".")[0]) >= 5:  # v5: tokenizer built from a (piece, score) list
        sp = spm.SentencePieceProcessor(model_file=spm_path)
        vocab = [(sp.id_to_piece(i), sp.get_score(i)) for i in range(sp.get_piece_size())]
        tok = T5Tokenizer(vocab=vocab, extra_ids=0)
    else:  # v4: tokenizer built from the sentencepiece file
        tok = T5Tokenizer(vocab_file=spm_path, extra_ids=0)
    assert len(tok) > 30, f"toy tokenizer looks broken (vocab size {len(tok)})"
    base_dir = os.path.join(tmp, "toy_base")
    tok.save_pretrained(base_dir)
    cfg = MT5Config(vocab_size=len(tok), d_model=48, d_kv=12, d_ff=96, num_layers=2, num_decoder_layers=2,
                    num_heads=4, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    MT5ForConditionalGeneration(cfg).save_pretrained(base_dir)
    return paths, base_dir, len(tok)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        paths, base_dir, base_vocab = build_toy_world(tmp)
        data_args = ["--train_file", paths["train"], "--validation_file", paths["validation"], "--test_file", paths["test"]]

        # ---- 1. tokens and batching ------------------------------------------------
        tok = load_tokenizer(base_dir)
        ids = tok.convert_tokens_to_ids(list(LANG_TOKENS.values()))
        check(ids == [base_vocab, base_vocab + 1, base_vocab + 2], f"language tokens get consecutive new ids {ids}")
        batch = make_batch([("<jp>the cat runs fast", "猫が速く走る"), ("<zh>the dog sleeps", "狗睡")],
                           tok, 32, "cpu")
        labels = batch["labels"]
        check((labels == -100).any().item(), "padding in labels is masked with -100")
        check(not (labels == tok.pad_token_id).any().item(), "no real <pad> ids remain in labels")
        check(batch["attention_mask"].sum().item() < batch["attention_mask"].numel(), "attention mask marks padding")
        t2, m2 = load_model_and_tokenizer(base_dir, device="cpu")
        check(m2.get_input_embeddings().num_embeddings == len(t2) == base_vocab + 3,
              "embedding table was resized for the 3 language tokens")

        # ---- 2. training -------------------------------------------------------------
        out_dir = os.path.join(tmp, "model")
        stdout = run([PY, "train.py", "--model_name_or_path", base_dir, "--output_dir", out_dir, "--epochs", "30",
                      "--batch_size", "16", "--lr", "1e-3", "--max_len", "32", "--eval_every", "50",
                      "--print_every", "20", "--val_samples", "30", *data_args])
        for name in ("config.json", "training_log.csv", "loss_curve.png", "training_summary.json"):
            check(os.path.exists(os.path.join(out_dir, name)), f"train.py wrote {name}")
        check(any(f.startswith("model") for f in os.listdir(out_dir)), "train.py saved model weights")
        losses = [float(l.split("train loss")[1].split("|")[0]) for l in stdout.splitlines() if "train loss" in l]
        check(losses[-1] < losses[0] * 0.5, f"training loss went down ({losses[0]:.2f} -> {losses[-1]:.2f})")
        summary = json.load(open(os.path.join(out_dir, "training_summary.json")))
        check(summary["best_validation_loss"] is not None, f"best validation loss recorded ({summary['best_validation_loss']:.3f})")

        # ---- 3. translating with the trained model ---------------------------------
        t3, m3 = load_model_and_tokenizer(out_dir, device="cpu")
        out = translate("the cat runs fast", "ja", t3, m3, num_beams=2, max_new_tokens=32)
        check(isinstance(out, str) and len(out) > 0, f"translate() returns text: {out!r}")
        outs = translate(["the dog sleeps slowly", "the fish eats often"], "zh", t3, m3, num_beams=1, max_new_tokens=32)
        check(len(outs) == 2, f"translate() handles a list: {outs}")

        # ---- 4. evaluation -----------------------------------------------------------
        eval_dir = os.path.join(tmp, "eval")
        run([PY, "evaluate.py", "--model_dir", out_dir, "--out_dir", eval_dir, "--batch_size", "8", *data_args])
        results = json.load(open(os.path.join(eval_dir, "results.json")))
        check(len(results) == 6, f"evaluate.py scored all 6 directions: {sorted(results)}")
        avg_chrf = sum(v["chrf"] for v in results.values()) / 6
        check(avg_chrf > 20, f"toy model learned something (average chrF {avg_chrf:.1f}; bleu {[v['bleu'] for v in results.values()]})")
        check(os.path.exists(os.path.join(eval_dir, "results.md")) and os.path.exists(os.path.join(eval_dir, "samples.tsv")),
              "results.md and samples.tsv written")

        # ---- 5. converting a legacy (weights-only) checkpoint ----------------------
        legacy_model = AutoModelForSeq2SeqLM.from_pretrained(out_dir)
        legacy_path = os.path.join(tmp, "legacy.pt")
        torch.save(legacy_model.state_dict(), legacy_path)
        conv_dir = os.path.join(tmp, "converted")
        run([PY, "convert_legacy_checkpoint.py", "--state_dict", legacy_path, "--output_dir", conv_dir,
             "--base_model", base_dir])
        t4, m4 = load_model_and_tokenizer(conv_dir, device="cpu")
        a = translate("the bird eats fast", "zh", t3, m3, num_beams=2, max_new_tokens=32)
        b = translate("the bird eats fast", "zh", t4, m4, num_beams=2, max_new_tokens=32)
        check(a == b, f"converted legacy checkpoint gives identical output ({b!r})")

        # ---- 6. the Streamlit app ----------------------------------------------------
        os.environ["NMT_MODEL"] = out_dir
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
        check(not at.exception, "app starts without errors")
        at.text_area[0].set_value("the cat runs fast").run()
        at.button[0].click().run()
        check(not at.exception and len(at.success) == 1 and at.success[0].value, f"app shows a translation: {at.success[0].value!r}")

        # app should fail politely (no crash) when the model folder is missing
        os.environ["NMT_MODEL"] = os.path.join(tmp, "does-not-exist")
        at2 = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=120).run()
        check(not at2.exception and len(at2.error) == 1, "app shows a friendly error if no model is found")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
