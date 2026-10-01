"""Kahn1 v4 (Qwen3.5-4B) against v3, JEV and JevK5, every comparison paired and like for like.

- Held-out 14,663 (data/eval.jsonl): v4 and v3 are both scored by eval/eval_full.py (Choice
  over the 8 options eval/baselines.py:_prepare_choice_options picks, k = 3, calibrated). JEV
  gets the same 8 options for Choice (data/jev_eval_preds_choice8.jsonl) and, for Noul, the
  "supported" wording, the fairer one for it (data/jev_eval_preds_noul_supported.jsonl).
- Choice over every intent (77 / 60): the 1,184 items v3 answered through evaluate_two_stage,
  v4 on the same items, JEV's run with every intent.
- JevBench, 231 public items: v4 and v3 predictions, Jev's per-task outcomes published by
  JevBench, JevK5's own public run.

    python scripts/v4_report.py      # reports/V4_REPORT.md + .json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

R, D = _ROOT / "reports", _ROOT / "data"


def jl(path: Path) -> dict[int, dict]:
    return {json.loads(l)["i"]: json.loads(l) for l in path.open(encoding="utf-8") if "answer" in json.loads(l)}


def main() -> None:
    from eval.metrics import ece
    from scripts.choice_fairness import eight
    from scripts.jev_holdout import jev_outcome

    items = [json.loads(l) for l in (D / "eval.jsonl").open(encoding="utf-8")]
    v3 = json.loads((R / "eval_v3_temponly_preds.json").read_text(encoding="utf-8"))
    v4 = json.loads((R / "eval_v4_preds.json").read_text(encoding="utf-8"))
    jev_full, jev8, jev_sup = jl(D / "jev_eval_preds.jsonl"), jl(D / "jev_eval_preds_choice8.jsonl"), \
        jl(D / "jev_eval_preds_noul_supported.jsonl")

    def jev_like(i: int):
        it = items[i]
        if it["kind"] == "choice":
            return jev_outcome(eight(it), jev8[i]["answer"])[:2]
        if it["kind"] == "noul":
            return jev_outcome(it, jev_sup[i]["answer"])[:2]
        return jev_outcome(it, jev_full[i]["answer"])[:2]

    groups = defaultdict(list)
    for i, it in enumerate(items):
        for g in (it["source"], "kind:" + it["kind"], "all"):
            groups[g].append(i)
    held = {}
    for g, ids in groups.items():
        j = [jev_like(i) for i in ids]
        held[g] = {"n": len(ids),
                   "v4": sum(v4["corrects"][i] for i in ids) / len(ids),
                   "v3": sum(v3["corrects"][i] for i in ids) / len(ids),
                   "jev": sum(x[0] for x in j) / len(ids),
                   "v4_ece": ece([v4["confidences"][i] for i in ids], [v4["corrects"][i] for i in ids]),
                   "v3_ece": ece([v3["confidences"][i] for i in ids], [v3["corrects"][i] for i in ids]),
                   "jev_ece": ece([x[1] for x in j], [x[0] for x in j])}

    # Choice over every intent, same items.
    f3, f4 = jl(D / "kahn1_v3_choice_full.jsonl"), jl(D / "kahn1_v4_choice_full.jsonl") \
        if (D / "kahn1_v4_choice_full.jsonl").exists() else {}
    both = [i for i in f3 if i in f4 and i in jev_full]
    gold = lambda i: items[i]["options"][items[i]["label"]]
    full = {"n": len(both)}
    if both:
        full.update(v3=sum(f3[i]["answer"]["choice"] == gold(i) for i in both) / len(both),
                    v4=sum(f4[i]["answer"]["choice"] == gold(i) for i in both) / len(both),
                    jev=sum(jev_outcome(items[i], jev_full[i]["answer"])[0] for i in both) / len(both))

    # JevBench public, per item.
    jb = [json.loads(l) for l in (D / "jevbench_eval.jsonl").open(encoding="utf-8")]
    jb3 = json.loads((R / "jevbench_v3_preds.json").read_text(encoding="utf-8"))["corrects"]
    jb4 = json.loads((R / "jevbench_v4_preds.json").read_text(encoding="utf-8"))["corrects"]
    import urllib.request
    per_task = json.loads(urllib.request.urlopen(
        "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/results/v1.2/jevbench-v1.2-per-task.json",
        timeout=120).read())["systems"]["jev-1.13.0"]["public_tasks"]
    k5 = {json.loads(l)["task_id"]: bool(json.loads(l)["correct"])
          for l in (D / "jevk5" / "jevk5-v0.2.jsonl").open(encoding="utf-8")}
    tiers = defaultdict(lambda: defaultdict(int))
    for it, a, b in zip(jb, jb3, jb4):
        for t in (it["source"].removeprefix("jevbench-"), "all"):
            c = tiers[t]
            c["n"] += 1; c["v3"] += a; c["v4"] += b
            c["jev"] += per_task[it["id"]][0] == "c"; c["jevk5"] += k5[it["id"]]
    jbt = {t: {k: (v / c["n"] if k != "n" else v) for k, v in c.items()} for t, c in tiers.items()}

    out = {"heldout": held, "choice_every_intent": full, "jevbench": jbt}
    (R / "v4_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    P = lambda x: f"{100 * x:.1f} %"
    order = ["banking77", "massive", "rte_eval", "scitail_eval", "sst5_eval", "app_reviews_eval",
             "kind:choice", "kind:score", "kind:noul", "all"]
    def B(row: dict, key: str, keys: tuple) -> str:
        """The value, in bold when it is the best of the row."""
        return f"**{P(row[key])}**" if row[key] == max(row[k] for k in keys) else P(row[key])

    K3, K4 = ("v4", "v3", "jev"), ("v4", "v3", "jevk5", "jev")
    rows = "\n".join(f"| {g.replace('kind:', 'all ').replace('_eval', '')} | {held[g]['n']} | {B(held[g], 'v4', K3)} | "
                     f"{B(held[g], 'v3', K3)} | {B(held[g], 'jev', K3)} | {held[g]['v4_ece']:.3f} | {held[g]['v3_ece']:.3f} | "
                     f"{held[g]['jev_ece']:.3f} |" for g in order if g in held)
    jrows = "\n".join(f"| {t} | {c['n']} | {B(c, 'v4', K4)} | {B(c, 'v3', K4)} | {B(c, 'jevk5', K4)} | {B(c, 'jev', K4)} |"
                      for t, c in sorted(jbt.items(), key=lambda kv: ["easy", "original", "hard", "all"].index(kv[0])))
    fl = (f"| every intent (77 / 60), {full['n']} items | {B(full, 'v4', K3)} | {B(full, 'v3', K3)} | {B(full, 'jev', K3)} |"
          if both else "| every intent | not run | | |")
    (R / "V4_REPORT.md").write_text(f"""# Kahn1 v4 (Qwen3.5-4B + LoRA) — local evaluation

v4: Qwen3.5-4B, LoRA r=16 on attention and linear-attention projections, native chat
template, trained on data/train_v4.jsonl (the v3 mixture subsampled + DocNLI, ShARC, WANLI,
QuALITY), checkpoint selected on data/dev_v4.jsonl. v3: Qwen2.5-3B. Local, not published.

## Held-out 14,663 items, like for like

Choice over the same 8 options for every system; JEV's Noul asked whether the text supports
the statement. k = 3 and temperature calibration for v4 and v3.

| Dataset | Items | v4 | v3 | JEV 1.13.0 | v4 ECE | v3 ECE | JEV ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Choice over every intent

| Condition | v4 | v3 | JEV 1.13.0 |
|---|---:|---:|---:|
{fl}

## JevBench, 231 public items

| Tier | Items | v4 | v3 | JevK5 v0.2 | Jev 1.13.0 |
|---|---:|---:|---:|---:|---:|
{jrows}
""", encoding="utf-8")
    print((R / "V4_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
