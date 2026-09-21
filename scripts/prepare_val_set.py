"""Prepare disjoint validation split (data/val.jsonl) for post-hoc temperature calibration.

Selects 1,000 augmented instances from training data sources (500 Choice, 250 Score, 250 Noul)
to fit temperature parameters T_choice, T_score, T_noul without data leakage into the held-out evaluation split.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from training.augment import augment, DistractorPool


def main():
    """Build calibration validation split from training set sources."""
    train_path = Path("data/train.jsonl")
    val_out_path = Path("data/val.jsonl")

    print(f"[val] Reading {train_path}...")
    all_raw = []
    by_kind = defaultdict(list)
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            all_raw.append(ex)
            by_kind[ex["kind"]].append(ex)

    print(f"[val] Building DistractorPool over {len(all_raw)} instances...")
    pool = DistractorPool.build(all_raw)

    rng = random.Random(42)
    val_examples = []

    # 500 augmented Choice instances (ensuring len(options) >= 2)
    choices = by_kind["choice"]
    rng.shuffle(choices)
    count = 0
    for ex in choices:
        try:
            aug = augment(ex, pool, rng)
            if aug.kind == "choice" and len(aug.options) >= 2:
                val_examples.append({
                    "state": aug.state[:1500],
                    "kind": "choice",
                    "prompt": aug.prompt,
                    "options": aug.options,
                    "label": aug.label,
                    "include_other": aug.include_other,
                    "source": aug.source,
                })
                count += 1
                if count >= 500:
                    break
        except Exception:
            continue

    # 250 Score instances
    scores = by_kind["score"]
    rng.shuffle(scores)
    count = 0
    for ex in scores:
        try:
            aug = augment(ex, pool, rng)
            val_examples.append({
                "state": aug.state[:1500],
                "kind": "score",
                "prompt": aug.prompt,
                "levels": aug.levels,
                "label": aug.label,
                "source": aug.source,
            })
            count += 1
            if count >= 250:
                break
        except Exception:
            continue

    # 250 Noul instances
    nouls = by_kind["noul"]
    rng.shuffle(nouls)
    count = 0
    for ex in nouls:
        try:
            aug = augment(ex, pool, rng)
            val_examples.append({
                "state": aug.state[:1500],
                "kind": "noul",
                "prompt": aug.prompt,
                "statement": aug.statement,
                "label": aug.label,
                "source": aug.source,
            })
            count += 1
            if count >= 250:
                break
        except Exception:
            continue

    rng.shuffle(val_examples)
    print(f"[val] Writing {len(val_examples)} validation instances to {val_out_path}...")
    with open(val_out_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("[val] Validation set successfully prepared.")


if __name__ == "__main__":
    main()
