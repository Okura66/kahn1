"""Kahn1 4B (Qwen3.5-4B + LoRA) against Kahn1 3B, JEV and JevK5, every comparison paired.

- Held-out 14,663 (data/eval.jsonl): both Kahn1 models scored by eval/eval_full.py (Choice over
  the 8 options eval/baselines.py:_prepare_choice_options picks, k = 3, calibrated). JEV gets
  the same 8 options for Choice and the "supported" wording for Noul.
- Choice over every intent (77 / 60): the 1,184 items the 3B answered through
  evaluate_two_stage, the 4B on the same items, JEV's run with every intent.
- JevBench, 231 public items: Kahn1 predictions (k = 3, calibrated), Jev's per-task outcomes
  published by JevBench, JevK5's own public run; exact McNemar test against JevK5.
- Hard teacher dev split (data/dev_teacher.jsonl, k = 1): base Qwen3.5-4B and the 4B.

    python scripts/kahn1_4b_report.py      # reports/KAHN1_4B_REPORT.md + .json
"""

from __future__ import annotations

import json
import sys
import urllib.request
from collections import defaultdict
from math import comb
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

R, D = _ROOT / "reports", _ROOT / "data"
JEV_PER_TASK = ("https://raw.githubusercontent.com/fstandhartinger/jevbench/main/results/v1.2/"
                "jevbench-v1.2-per-task.json")


def jl(path: Path) -> dict[int, dict]:
    rows = (json.loads(l) for l in path.open(encoding="utf-8"))
    return {r["i"]: r for r in rows if "answer" in r}


def mcnemar(a: list[int], b: list[int]) -> tuple[int, int, float]:
    """Discordant pairs (a right only, b right only) and the exact two-sided p-value."""
    x = sum(1 for i, j in zip(a, b) if i and not j)
    y = sum(1 for i, j in zip(a, b) if j and not i)
    n, k = x + y, min(x, y)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return x, y, p


def main() -> None:
    from eval.metrics import ece
    from scripts.choice_fairness import eight
    from scripts.jev_holdout import jev_outcome

    items = [json.loads(l) for l in (D / "eval.jsonl").open(encoding="utf-8")]
    k3 = json.loads((R / "eval_v3_temponly_preds.json").read_text(encoding="utf-8"))
    k4 = json.loads((R / "eval_v5_preds.json").read_text(encoding="utf-8"))
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
                   "k4b": sum(k4["corrects"][i] for i in ids) / len(ids),
                   "k3b": sum(k3["corrects"][i] for i in ids) / len(ids),
                   "jev": sum(x[0] for x in j) / len(ids),
                   "k4b_ece": ece([k4["confidences"][i] for i in ids], [k4["corrects"][i] for i in ids]),
                   "k3b_ece": ece([k3["confidences"][i] for i in ids], [k3["corrects"][i] for i in ids]),
                   "jev_ece": ece([x[1] for x in j], [x[0] for x in j])}

    f3 = jl(D / "kahn1_v3_choice_full.jsonl")
    f4 = jl(D / "kahn1_v5_choice_full.jsonl") if (D / "kahn1_v5_choice_full.jsonl").exists() else {}
    both = [i for i in f3 if i in f4 and i in jev_full]
    gold = lambda i: items[i]["options"][items[i]["label"]]
    full = {"n": len(both)}
    if both:
        full.update(k3b=sum(f3[i]["answer"]["choice"] == gold(i) for i in both) / len(both),
                    k4b=sum(f4[i]["answer"]["choice"] == gold(i) for i in both) / len(both),
                    jev=sum(jev_outcome(items[i], jev_full[i]["answer"])[0] for i in both) / len(both))

    jb = [json.loads(l) for l in (D / "jevbench_eval.jsonl").open(encoding="utf-8")]
    jb3 = json.loads((R / "jevbench_v3_preds.json").read_text(encoding="utf-8"))["corrects"]
    jb4 = json.loads((R / "jevbench_v5_preds.json").read_text(encoding="utf-8"))["corrects"]
    per_task = json.loads(urllib.request.urlopen(JEV_PER_TASK, timeout=120).read())["systems"]["jev-1.13.0"][
        "public_tasks"]
    k5 = {r["task_id"]: bool(r["correct"]) for r in map(json.loads, (D / "jevk5" / "jevk5-v0.2.jsonl").open(
        encoding="utf-8"))}
    tiers = defaultdict(lambda: defaultdict(int))
    for it, a, b in zip(jb, jb3, jb4):
        for t in (it["source"].removeprefix("jevbench-"), "all"):
            c = tiers[t]
            c["n"] += 1; c["k3b"] += a; c["k4b"] += b
            c["jev"] += per_task[it["id"]][0] == "c"; c["jevk5"] += k5[it["id"]]
    jbt = {t: {k: (v / c["n"] if k != "n" else v) for k, v in c.items()} for t, c in tiers.items()}
    x, y, p = mcnemar(jb4, [k5[it["id"]] for it in jb])

    dev = {}
    for name, f in (("base", R / "v4" / "devteacher_base.json"), ("k4b", R / "v5" / "k1_final.json")):
        if f.exists():
            h = json.loads(f.read_text(encoding="utf-8"))["heldout_sample"]
            dev[name] = {k: v["acc"] for k, v in h.items() if isinstance(v, dict)} | {"balanced": h["balanced_acc"]}

    out = {"heldout": held, "choice_every_intent": full, "jevbench": jbt,
           "jevbench_vs_jevk5": {"k4b_only": x, "jevk5_only": y, "p": p}, "dev_teacher": dev}
    (R / "kahn1_4b_report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    P = lambda v: f"{100 * v:.1f} %"
    def B(row: dict, key: str, keys: tuple) -> str:
        return f"**{P(row[key])}**" if row[key] == max(row[k] for k in keys) else P(row[key])

    order = ["banking77", "massive", "rte_eval", "scitail_eval", "sst5_eval", "app_reviews_eval",
             "kind:choice", "kind:score", "kind:noul", "all"]
    K3, K4 = ("k4b", "k3b", "jev"), ("k4b", "k3b", "jevk5", "jev")
    rows = "\n".join(f"| {g.replace('kind:', 'all ').replace('_eval', '')} | {held[g]['n']:,} | {B(held[g], 'k4b', K3)} | "
                     f"{B(held[g], 'k3b', K3)} | {B(held[g], 'jev', K3)} | {held[g]['k4b_ece']:.3f} | "
                     f"{held[g]['k3b_ece']:.3f} | {held[g]['jev_ece']:.3f} |" for g in order if g in held)
    jrows = "\n".join(f"| {t} | {c['n']} | {B(c, 'k4b', K4)} | {B(c, 'k3b', K4)} | {B(c, 'jevk5', K4)} | {B(c, 'jev', K4)} |"
                      for t, c in sorted(jbt.items(), key=lambda kv: ["easy", "original", "hard", "all"].index(kv[0])))
    fl = (f"| every intent (77 / 60), {full['n']:,} items | {B(full, 'k4b', K3)} | {B(full, 'k3b', K3)} | "
          f"{B(full, 'jev', K3)} |" if both else "| every intent | not run | | |")
    drows = "\n".join(f"| {k.replace('kind:', '').replace('lang:', 'lang ')} | {P(dev['base'][k])} | {P(dev['k4b'][k])} |"
                      for k in ("kind:choice", "kind:score", "kind:noul", "lang:en", "lang:fr", "all", "balanced")
                      if "base" in dev and "k4b" in dev and k in dev["base"] and k in dev["k4b"])
    (R / "KAHN1_4B_REPORT.md").write_text(f"""# Kahn1 4B (Qwen3.5-4B + LoRA): evaluation

Kahn1 4B: Qwen3.5-4B, LoRA r = 16 on the attention and linear-attention projections, native chat
template (thinking off), trained on data/train_v5.jsonl: a subsample of the 3B mixture with less
short classification, long-document decisions (DocNLI, ShARC, WANLI, QuALITY) and 402 hard decision
questions written by Claude Opus, English and French, each repeated 4 times with permuted options.
Checkpoint chosen on two dev splits, never on a benchmark. Kahn1 3B: Qwen2.5-3B (v3).

## Held-out 14,663 items, like for like

Choice over the same 8 options for every system (the gold one and 7 seeded distractors); JEV's
Noul asked whether the text supports the statement. k = 3 and temperature calibration for Kahn1.

| Dataset | Items | Kahn1 4B | Kahn1 3B | JEV 1.13.0 | 4B ECE | 3B ECE | JEV ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Choice over every intent

| Condition | Kahn1 4B | Kahn1 3B | JEV 1.13.0 |
|---|---:|---:|---:|
{fl}

## JevBench, 231 public items

Kahn1: k = 3, calibrated. Jev: JevBench's own per-task outcomes. JevK5: its public v0.2 run.

| Tier | Items | Kahn1 4B | Kahn1 3B | JevK5 v0.2 | Jev 1.13.0 |
|---|---:|---:|---:|---:|---:|
{jrows}

Kahn1 4B vs JevK5, paired: {x} items only Kahn1 4B gets right, {y} only JevK5; exact McNemar p = {p:.2f}.

## Hard decision dev split (317 items, k = 1)

Questions written by Claude Opus and checked by two blind solvers; never trained on.

| Group | Qwen3.5-4B base | Kahn1 4B |
|---|---:|---:|
{drows}
""", encoding="utf-8")
    print((R / "KAHN1_4B_REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
