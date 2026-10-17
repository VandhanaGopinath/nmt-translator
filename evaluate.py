"""
evaluate.py - measure translation quality on the held-out TEST split.

    python evaluate.py --model_dir outputs/mt5-translator
    python evaluate.py --model_dir outputs/mt5-translator --max_samples 200   # quick check

For each of the 6 directions (en->ja, en->zh, ja->en, ja->zh, zh->en, zh->ja) it
translates the test sentences and scores them with:
  * BLEU  - word/character overlap with the reference translation (0-100)
  * chrF  - character n-gram overlap; more forgiving for Japanese/Chinese

Writes results.json, results.md and samples.tsv into --out_dir.
Japanese BLEU is computed per character and Chinese with sacreBLEU's Chinese
tokenizer, so those numbers are NOT comparable with English BLEU.
"""

import argparse
import os

import sacrebleu

from core import LANG_TOKENS, load_model_and_tokenizer, load_translation_splits, translate, write_json

BLEU_TOKENIZER = {"en": "13a", "zh": "zh", "ja": "char"}  # keyed by the TARGET language


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="outputs/mt5-translator")
    p.add_argument("--out_dir", default=None, help="default: <model_dir>/eval")
    p.add_argument("--split", default="test")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--train_file", default=None)
    p.add_argument("--validation_file", default=None)
    p.add_argument("--test_file", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.model_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(args.model_dir)
    splits = load_translation_splits(args.train_file, args.validation_file, args.test_file)
    rows = splits[args.split]["translation"]
    if args.max_samples:
        rows = rows[: args.max_samples]
    langs = list(LANG_TOKENS.keys())
    print(f"Evaluating on {len(rows)} {args.split} sentences, {len(langs) * (len(langs) - 1)} directions")

    results, sample_lines = {}, ["direction\tsource\treference\tmodel_output"]
    for src in langs:
        for tgt in langs:
            if src == tgt:
                continue
            pairs = [(r[src], r[tgt]) for r in rows if r.get(src) and r.get(tgt)]
            sources = [s for s, _ in pairs]
            refs = [t for _, t in pairs]
            hyps = translate(sources, tgt, tokenizer, model, num_beams=args.num_beams, batch_size=args.batch_size)

            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=BLEU_TOKENIZER[tgt]).score
            chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
            key = f"{src}->{tgt}"
            results[key] = {"bleu": round(bleu, 2), "chrf": round(chrf, 2), "sentences": len(pairs)}
            print(f"{key}: BLEU {bleu:.1f} | chrF {chrf:.1f} | {len(pairs)} sentences")
            for s, r, h in list(zip(sources, refs, hyps))[:20]:
                sample_lines.append(f"{key}\t{s}\t{r}\t{h}")

    write_json(os.path.join(out_dir, "results.json"), results)
    with open(os.path.join(out_dir, "samples.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(sample_lines) + "\n")
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("| Direction | BLEU | chrF | Sentences |\n|---|---|---|---|\n")
        for key, v in results.items():
            f.write(f"| {key} | {v['bleu']} | {v['chrf']} | {v['sentences']} |\n")
    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
