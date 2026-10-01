"""Check that every row of a training mixture builds a training target, before a long run.

Training draws rows at random, so a bad row fails at whatever step it comes up: v6 died at
step ~55 on a CommonsenseQA question listing one option twice. This pushes every row through
the augmentation and prompt building training uses (several draws each, scales reversed or
not), with a stand-in tokenizer in which letters and yes / no are single tokens.

    python scripts/check_mix.py data/train_4b_v2.jsonl        # exit 1 on any failure
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from training.augment import DistractorPool, augment  # noqa: E402
from training.train_lora import build_prompt_for_training  # noqa: E402


class WordTokenizer:
    """Words and single non-word characters are tokens."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [hash(t) % 1_000_003 for t in re.findall(r"\w+|\W", text)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mix")
    ap.add_argument("--draws", type=int, default=4, help="augmentations per row")
    ap.add_argument("--prompt-format", default="qwen3")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.mix).open(encoding="utf-8") if l.strip()]
    pool, tok = DistractorPool.build(rows), WordTokenizer()
    failures, example = Counter(), {}
    for i, r in enumerate(rows):
        for d in range(args.draws):
            try:
                aug = augment(r, pool, random.Random(i * args.draws + d), reverse_prob=0.5)
                build_prompt_for_training(aug, tok, args.prompt_format)
            except Exception as e:  # noqa: BLE001 - every failure is reported
                key = (r.get("source"), type(e).__name__, str(e)[:100])
                failures[key] += 1
                example.setdefault(key, i)
    print(f"{args.mix}: {len(rows)} rows x {args.draws} draws, {sum(failures.values())} failures")
    for (src, kind, msg), n in failures.most_common(10):
        print(f"  {n:5d}  {src}  {kind}: {msg}  (line {example[(src, kind, msg)] + 1})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
