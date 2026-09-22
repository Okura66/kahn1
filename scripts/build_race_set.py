"""Build a labelled race set by drawing from the reserved evaluation split.

Items are sampled from data/eval.jsonl, the same held-out file the benchmark reports
on, so the race measures the engine on exactly the distribution the numbers describe.
Every eval source is reserved: none of them appears in the training mixture (the
isolation is enforced in training/build_dataset.py).

    choice -> banking77 (77 classes), massive (60 classes)
    score  -> sst5 (5 ordinal levels)
    noul   -> rte, scitail (binary entailment)

Nine items per primitive by default, drawn with a fixed seed so the set is
reproducible. Choice sources carry far more classes than direct Choice evaluation
supports, so the option list is deterministically sub-sampled per item, always
keeping the gold option. Both engines receive the identical list.

    python scripts/build_race_set.py --per-kind 9 --out data/race_set.json
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

KINDS = ("choice", "score", "noul")
# BoolQ-style passages run long; the race panel shows one line per item.
MAX_STATE_CHARS = 900


def _rng_for(item_key: str) -> random.Random:
    """Deterministic per-item RNG so the same race set is reproducible."""
    seed = int(hashlib.md5(item_key.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)


def load_eval(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"No evaluation set at {path}. Build one first: "
            f"python training/build_dataset.py --eval-only --eval-out {path}"
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def to_choice(ex: dict, idx: int, n_options: int) -> dict:
    """Sub-sample the option list around the gold option, then permute it."""
    options = ex["options"]
    gold = options[ex["label"]]
    rng = _rng_for(f"{ex['source']}:{idx}")
    distractors = [o for o in options if o != gold]
    keep = min(max(n_options, 2), len(options)) - 1
    shown = rng.sample(distractors, keep) + [gold]
    rng.shuffle(shown)
    return {
        "id": f"{ex['source']}:{idx}",
        "source": ex["source"],
        "kind": "choice",
        "state": ex["state"][:MAX_STATE_CHARS],
        "prompt": ex["prompt"],
        "options": shown,
        "gold": gold,
    }


def to_score(ex: dict, idx: int) -> dict:
    return {
        "id": f"{ex['source']}:{idx}",
        "source": ex["source"],
        "kind": "score",
        "state": ex["state"][:MAX_STATE_CHARS],
        "prompt": ex["prompt"],
        "levels": ex["levels"],
        "gold": ex["levels"][ex["label"]],
    }


def to_noul(ex: dict, idx: int) -> dict:
    statement = ex["statement"].strip()
    if statement:
        statement = statement[0].upper() + statement[1:]
    return {
        "id": f"{ex['source']}:{idx}",
        "source": ex["source"],
        "kind": "noul",
        "state": ex["state"][:MAX_STATE_CHARS],
        "statement": statement,
        # Dataset label 1 == yes / entailment.
        "gold": bool(ex["label"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=9,
                    help="items drawn per primitive (choice, score, noul)")
    ap.add_argument("--options", type=int, default=8,
                    help="options shown per Choice item (banking77 carries 77 classes)")
    ap.add_argument("--eval", default="data/eval.jsonl",
                    help="reserved evaluation split to draw from")
    ap.add_argument("--sources", default=None,
                    help="comma-separated whitelist of eval sources to draw from "
                         "(e.g. banking77,massive,sst5_eval,rte_eval). All sources are "
                         "held-out; this only chooses which held-out sources the demo "
                         "shows. Omit to draw from every source.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/race_set.json")
    args = ap.parse_args()

    allowed = None
    if args.sources:
        allowed = {s.strip() for s in args.sources.split(",") if s.strip()}

    rows = load_eval(Path(args.eval))
    by_kind: dict[str, list[tuple[int, dict]]] = {k: [] for k in KINDS}
    for i, ex in enumerate(rows):
        if ex.get("kind") in by_kind:
            if allowed is not None and ex.get("source") not in allowed:
                continue
            by_kind[ex["kind"]].append((i, ex))

    empty = [k for k in KINDS if not by_kind[k]]
    if empty:
        raise RuntimeError(
            f"{args.eval} carries no {', '.join(empty)} example, so the race cannot "
            f"score that primitive. Rebuild it: python training/build_dataset.py --eval-only"
        )

    items: list[dict] = []
    for kind in KINDS:
        pool = by_kind[kind]
        n = min(args.per_kind, len(pool))
        if n < args.per_kind:
            print(f"[race] only {n} {kind} items available (asked for {args.per_kind})")
        picks = random.Random(f"{args.seed}:{kind}").sample(pool, n)
        for idx, ex in picks:
            if kind == "choice":
                items.append(to_choice(ex, idx, args.options))
            elif kind == "score":
                items.append(to_score(ex, idx))
            else:
                items.append(to_noul(ex, idx))

    # Interleave so the race panel does not show three homogeneous blocks.
    random.Random(args.seed).shuffle(items)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"items": items}, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    for it in items:
        counts[it["kind"]] = counts.get(it["kind"], 0) + 1
        sources[it["source"]] = sources.get(it["source"], 0) + 1
    print(f"[race] wrote {len(items)} items to {out}: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"[race] sources: " + ", ".join(f"{k}={v}" for k, v in sorted(sources.items())))


if __name__ == "__main__":
    main()
