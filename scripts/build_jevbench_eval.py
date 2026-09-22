"""Convert the public JevBench tasks into our evaluation JSONL schema.

JevBench (github.com/fstandhartinger/jevbench, MIT) is an independent, external
benchmark for Jev-class typed-decision models, with published Jev scores. Running
Kahn1 through it gives a real head-to-head against Jev on identical items — far
stronger than our 27-item demo race.

Mapping to our schema (data/eval.jsonl format consumed by eval.eval_full):
  choice -> options fold each candidate as "label: description" so the rich
            criteria text (the signal a cross-encoder scorer reads directly) is
            visible in the prompt; gold label = index of `expected`.
  noul   -> statement = the task instructions; gold = 1 if expected == "yes".
  score  -> levels = the ordered criteria descriptions; gold = the integer
            `expected` level index.

source is set to "jevbench-<tier>" so the per-source eval breakdown reproduces
JevBench's original / easy / hard tiers.

    python scripts/build_jevbench_eval.py --out data/jevbench_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TIERS = ("original", "easy", "hard")


def convert_task(task: dict, tier: str) -> dict | None:
    q = task["question"]
    kind = q["type"]
    state = task["state"]
    # ~15% of tasks carry a structured state (contacts, policies, tables); serialize
    # it to readable JSON so the model sees the full context as text.
    if not isinstance(state, str):
        state = json.dumps(state, indent=2, ensure_ascii=False)
    instructions = q.get("instructions", "")
    criteria = q.get("criteria")
    labels = task["labels"]
    expected = task["expected"]
    source = f"jevbench-{tier}"
    base = {"state": state, "source": source, "id": task["id"]}

    if kind == "choice":
        if not isinstance(criteria, dict):
            return None
        # Fold each candidate's description into its option text, in label order
        # so the gold index lines up. Falls back to the bare label if a
        # description is missing.
        options = [f"{lbl}: {criteria[lbl]}" if criteria.get(lbl) else str(lbl)
                   for lbl in labels]
        if expected not in labels:
            return None
        return {**base, "kind": "choice", "prompt": instructions,
                "options": options, "label": labels.index(expected)}

    if kind == "noul":
        # labels are always ["no", "yes"]; expected is "yes"/"no".
        gold = 1 if str(expected).lower() in ("yes", "true", "1") else 0
        return {**base, "kind": "noul", "prompt": instructions,
                "statement": instructions, "label": gold}

    if kind == "score":
        # criteria is the ordered list of level descriptions; expected is the
        # integer level index.
        levels = [str(c) for c in criteria] if isinstance(criteria, list) else [str(l) for l in labels]
        idx = int(expected)
        if not (0 <= idx < len(levels)):
            return None
        return {**base, "kind": "score", "prompt": instructions,
                "levels": levels, "label": idx}

    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/jevbench",
                    help="directory holding original.jsonl / easy.jsonl / hard.jsonl")
    ap.add_argument("--out", default="data/jevbench_eval.jsonl")
    args = ap.parse_args()

    src = Path(args.src)
    out_rows: list[dict] = []
    dropped = 0
    per = {}
    for tier in TIERS:
        path = src / f"{tier}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; download JevBench public tiers first.")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = convert_task(json.loads(line), tier)
            if rec is None:
                dropped += 1
                continue
            out_rows.append(rec)
            per[(tier, rec["kind"])] = per.get((tier, rec["kind"]), 0) + 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kinds = {}
    for r in out_rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"[jevbench] wrote {len(out_rows)} items to {args.out} ({dropped} dropped)")
    print(f"[jevbench] kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    print(f"[jevbench] per tier/kind: " +
          ", ".join(f"{t}/{k}={n}" for (t, k), n in sorted(per.items())))


if __name__ == "__main__":
    main()
