"""Kahn1 vs Jev 1.13.0 on the 231 public JevBench items, item by item.

Jev's outcomes are not re-measured here: they are the per-task results JevBench
itself publishes (results/v1.2/jevbench-v1.2-per-task.json, revision v1.3.0), so
the Jev side is an independent measurement. Kahn1's outcomes are the saved v3
predictions on the same items (reports/jevbench_v3_preds.json, in the order of
data/jevbench_eval.jsonl, built by scripts/build_jevbench_eval.py).

    python scripts/jevbench_vs_jev.py            # writes reports/JEVBENCH_VS_JEV.md + .json
"""

from __future__ import annotations

import json
import urllib.request
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
PER_TASK_URL = ("https://raw.githubusercontent.com/fstandhartinger/jevbench/main/"
                "results/v1.2/jevbench-v1.2-per-task.json")
ITEMS = _ROOT / "data" / "jevbench_eval.jsonl"
KAHN1 = _ROOT / "reports" / "jevbench_v3_preds.json"
OUT_MD = _ROOT / "reports" / "JEVBENCH_VS_JEV.md"
OUT_JSON = _ROOT / "reports" / "jevbench_vs_jev.json"
JEV_ID = "jev-1.13.0"
# Our tier names (source "jevbench-<tier>") -> JevBench's.
TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}


def main() -> None:
    with urllib.request.urlopen(PER_TASK_URL, timeout=60) as r:
        per_task = json.load(r)
    jev = per_task["systems"][JEV_ID]
    items = [json.loads(line) for line in ITEMS.open(encoding="utf-8")]
    k1 = json.loads(KAHN1.read_text(encoding="utf-8"))["corrects"]
    assert len(k1) == len(items)

    tally = defaultdict(lambda: {"n": 0, "kahn1": 0, "jev": 0})
    for it, ok in zip(items, k1):
        tier = it["source"].removeprefix("jevbench-")
        outcome = jev["public_tasks"][it["id"]][0]  # c correct, w wrong, f failed (scored wrong)
        for key in (tier, "all"):
            t = tally[key]
            t["n"] += 1
            t["kahn1"] += int(ok)
            t["jev"] += int(outcome == "c")

    groups = {k: {**v, "kahn1_acc": v["kahn1"] / v["n"], "jev_acc": v["jev"] / v["n"]}
              for k, v in tally.items()}
    full = {TIERS.get(k, k): jev["by_tier"][TIERS[k]]["accuracy"] for k in TIERS}
    OUT_JSON.write_text(json.dumps({
        "jevbench_revision": per_task["revision"], "jev": jev["display"],
        "public_items": groups, "jev_full_tier_accuracy": full,
    }, indent=1), encoding="utf-8")

    rows = [f"| {k} | {v['n']} | **{100 * v['kahn1_acc']:.1f} %** | **{100 * v['jev_acc']:.1f} %** |"
            for k, v in sorted(groups.items(), key=lambda kv: ["easy", "original", "hard", "all"].index(kv[0]))]
    OUT_MD.write_text(f"""# Kahn1 vs Jev on JevBench (public items)

- JevBench revision {per_task['revision']}; Jev = {jev['display']}, per-task outcomes published by JevBench.
- Kahn1: v3 checkpoint, k = 3, calibrated, predictions in `reports/jevbench_v3_preds.json`.
- Same 231 public items on both sides. "original" is JevBench's "standard" tier.

| Tier | Items | Kahn1 | Jev 1.13.0 |
|---|---:|---:|---:|
{chr(10).join(rows)}

Jev on the full tiers (public + held-out, as JevBench reports): easy {100 * full['easy']:.1f} %,
standard {100 * full['standard']:.1f} %, hard {100 * full['hard']:.1f} %. Kahn1 was not run on the held-out items,
which JevBench does not publish.
""", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
