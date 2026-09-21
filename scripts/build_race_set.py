"""Build a labelled race set from the held-out evaluation datasets.

Draws items from the same public test splits the model card reports on, plus
BoolQ for the Noul primitive, and writes them with their ground truth so the
race can score both engines for real instead of comparing them to each other.

    banking77 (choice, 77 classes) -> mteb/banking77 test      (3076 rows)
    sst5      (score, 5 levels)    -> SetFit/sst5 test         (2210 rows)
    boolq     (noul, binary)       -> google/boolq validation  (3270 rows)

Banking77 carries 77 classes while direct Choice evaluation caps at 26, so the
option list is deterministically sub-sampled per item, always keeping the gold
option. Both engines receive the identical list.

    python scripts/build_race_set.py --total 27 --out data/race_set.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CHOICE_PROMPT = "What is the customer's intent?"
SCORE_PROMPT = "Rate the sentiment of this review:"
SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def _rng_for(item_key: str) -> random.Random:
    """Deterministic per-item RNG so the same race set is reproducible."""
    seed = int(hashlib.md5(item_key.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)


def build_choice(n: int, n_options: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("mteb/banking77", split="test")
    all_labels = sorted({row["label_text"] for row in ds})
    picks = _rng_for("banking77-order").sample(range(len(ds)), n)

    items = []
    for i in picks:
        row = ds[i]
        gold = row["label_text"]
        rng = _rng_for(f"banking77:{i}")
        distractors = [l for l in all_labels if l != gold]
        options = rng.sample(distractors, n_options - 1) + [gold]
        rng.shuffle(options)
        items.append({
            "id": f"banking77:{i}",
            "source": "banking77",
            "kind": "choice",
            "state": row["text"],
            "prompt": CHOICE_PROMPT,
            "options": options,
            "gold": gold,
        })
    return items


def build_score(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("SetFit/sst5", split="test")
    picks = _rng_for("sst5-order").sample(range(len(ds)), n)
    return [{
        "id": f"sst5:{i}",
        "source": "sst5",
        "kind": "score",
        "state": ds[i]["text"],
        "prompt": SCORE_PROMPT,
        "levels": SST5_LEVELS,
        "gold": SST5_LEVELS[int(ds[i]["label"])],
    } for i in picks]


def build_noul(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")
    picks = _rng_for("boolq-order").sample(range(len(ds)), n)
    items = []
    for i in picks:
        row = ds[i]
        statement = row["question"].strip()
        statement = statement[0].upper() + statement[1:] + "?"
        items.append({
            "id": f"boolq:{i}",
            "source": "boolq",
            "kind": "noul",
            # BoolQ passages run long; the race panel shows one line per item.
            "state": row["passage"][:900],
            "statement": statement,
            "gold": bool(row["answer"]),
        })
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=27,
                    help="total items, split across the three primitives")
    ap.add_argument("--options", type=int, default=8,
                    help="options shown per Choice item (banking77 has 77 classes)")
    ap.add_argument("--out", default="data/race_set.json")
    args = ap.parse_args()

    n_choice = args.total // 3
    n_score = args.total // 3
    n_noul = args.total - n_choice - n_score

    items = build_choice(n_choice, args.options)
    items += build_score(n_score)
    items += build_noul(n_noul)
    # Interleave so the race panel does not show three homogeneous blocks.
    _rng_for(f"race-order-{args.total}").shuffle(items)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"items": items}, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    counts: dict[str, int] = {}
    for it in items:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1
    print(f"[race] wrote {len(items)} items to {out}: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
