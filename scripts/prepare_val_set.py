"""Carve a disjoint calibration split (data/val.jsonl) out of the training set.

Selects raw examples from data/train.jsonl (500 Choice, 250 Score, 250 Noul),
REMOVES them from train.jsonl, then writes their augmented form to val.jsonl for
fitting temperature parameters T_choice, T_score, T_noul.

The removal is the point: the first released calibration set was sampled from
train.jsonl without being taken out of it, so all 1,000 calibration instances were
also trained on. Temperatures fitted on memorized examples come out too low and the
deployed model runs systematically overconfident. Run this AFTER build_dataset.py
and BEFORE train_lora.py, exactly once per generated train.jsonl (running it again
carves a second, different split out of the already-reduced file).
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
import sys

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.augment import augment, DistractorPool

QUOTAS = {"choice": 500, "score": 250, "noul": 250}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--val-out", default="data/val.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_path = Path(args.train)
    val_out_path = Path(args.val_out)

    print(f"[val] Reading {train_path}...")
    all_raw: list[dict] = []
    by_kind: dict[str, list[int]] = defaultdict(list)  # kind -> indices into all_raw
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            by_kind[ex["kind"]].append(len(all_raw))
            all_raw.append(ex)

    print(f"[val] Building DistractorPool over {len(all_raw)} instances...")
    pool = DistractorPool.build(all_raw)

    rng = random.Random(args.seed)

    # Select raw indices per primitive, so the chosen examples can be removed from
    # the training file — calibration must never see trained-on instances.
    selected: list[tuple[str, int]] = []
    for kind, quota in QUOTAS.items():
        idxs = list(by_kind.get(kind, []))
        if len(idxs) < quota:
            raise RuntimeError(
                f"train.jsonl carries only {len(idxs)} {kind} examples; "
                f"cannot carve a {quota}-example calibration split."
            )
        rng.shuffle(idxs)
        selected.extend((kind, i) for i in idxs[:quota])

    selected_idx = {i for _, i in selected}

    # Augment the selected examples into calibration instances.
    val_examples: list[dict] = []
    dropped = 0
    for kind, i in selected:
        ex = all_raw[i]
        try:
            aug = augment(ex, pool, rng)
        except Exception:
            dropped += 1
            continue
        rec: dict = {
            "state": aug.state[:1500],
            "kind": kind,
            "prompt": aug.prompt,
            "label": aug.label,
            "source": aug.source,
        }
        if kind == "choice":
            if len(aug.options) < 2:
                dropped += 1
                continue
            rec["options"] = aug.options
            rec["include_other"] = aug.include_other
        elif kind == "score":
            rec["levels"] = aug.levels
        else:
            rec["statement"] = aug.statement
        val_examples.append(rec)

    rng.shuffle(val_examples)

    print(f"[val] Writing {len(val_examples)} calibration instances to {val_out_path} "
          f"({dropped} dropped during augmentation)...")
    with open(val_out_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Remove the selected raw examples from the training file. Every selected example
    # is removed, including the few whose augmentation failed: they were drawn for
    # calibration and must not silently re-enter training.
    kept = [ex for i, ex in enumerate(all_raw) if i not in selected_idx]
    print(f"[val] Rewriting {train_path}: {len(all_raw)} -> {len(kept)} instances "
          f"({len(selected_idx)} withheld for calibration)...")
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("[val] Calibration split prepared; train and calibration sets are disjoint.")


if __name__ == "__main__":
    main()
