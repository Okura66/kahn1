"""Kahn1 against JevK5 (github.com/allebee/jevk5), with JEV as the reference.

Two comparisons, each paired item by item:

- JevBench, 231 public items: JevK5's per-item outcomes from its own run through
  JevBench's runner (results/public231/jevk5-v0.2.jsonl in its repository), Jev's
  from JevBench's published per-task file, Kahn1's saved v3 predictions. JevBench's
  own v1.4 re-measurement of JevK5 (public and sealed accuracy) is reported next to it.
- BANKING77 test split: JevK5 run with its own script, bench/many_options.py
  (knockout, ceil(77/16)+1 passes), on items built by `build` below in the request
  shape that script uses (empty state, the text in the instructions, every intent as
  option_<i> in PolyAI's label order). Kahn1 and JEV are the held-out runs already on
  disk (data/eval.jsonl order). Items are paired by normalized text.

    python scripts/jevk5_vs_kahn1.py build     # data/jevk5/banking77_test.jsonl (JevK5's input)
    # then, with JevK5 installed (pip install "jevk5 @ git+https://github.com/allebee/jevk5@v0.2.2"):
    #   python bench/many_options.py run --items data/jevk5/banking77_test.jsonl \
    #       --methods knockout --out data/jevk5/banking77_test.results.jsonl
    python scripts/jevk5_vs_kahn1.py report    # reports/JEVK5_VS_KAHN1.md + .json

JevK5 v0.2 trained on 1,500 BANKING77 train texts (its README); the test split is
disjoint from them. Kahn1 never trained on BANKING77.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DATA = _ROOT / "data" / "jevk5"
ITEMS = DATA / "banking77_test.jsonl"
RESULTS = DATA / "banking77_test.results.jsonl"
PUBLIC231 = DATA / "jevk5-v0.2.jsonl"
POLYAI = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
JEVK5_PUBLIC_URL = "https://raw.githubusercontent.com/allebee/jevk5/main/results/public231/jevk5-v0.2.jsonl"
JB_PER_TASK = "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/results/v1.2/jevbench-v1.2-per-task.json"
JB_V14 = "https://raw.githubusercontent.com/fstandhartinger/jevbench/main/results/v1.4/jevbench-v1.4-results.json"
OUT_MD = _ROOT / "reports" / "JEVK5_VS_KAHN1.md"
OUT_JSON = _ROOT / "reports" / "jevk5_vs_kahn1.json"


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def norm(text: str) -> str:
    return " ".join(text.lower().split())


def build(_args) -> None:
    labels = json.loads(fetch(POLYAI + "categories.json"))
    assert len(labels) == 77
    rows = list(csv.DictReader(io.StringIO(fetch(POLYAI + "test.csv").decode())))
    DATA.mkdir(parents=True, exist_ok=True)
    with ITEMS.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "id": f"BANKING77:test:{i}", "dataset": "BANKING77", "state": {},
                "question": {"type": "choice",
                             "instructions": "Classify the banking intent of this user request:\n" + r["text"],
                             "criteria": {f"option_{k}": d for k, d in enumerate(labels)}},
                "expected": f"option_{labels.index(r['category'])}", "text": r["text"],
            }, ensure_ascii=False) + "\n")
    print(f"{len(rows)} BANKING77 test items -> {ITEMS}")


def banking(ece) -> dict:
    from scripts.jev_holdout import jev_outcome

    k5_items = {json.loads(l)["id"]: json.loads(l) for l in ITEMS.open(encoding="utf-8")}
    k5 = {}
    for line in RESULTS.open(encoding="utf-8"):
        r = json.loads(line)
        probs = r["designs"]["knockout"]["probs"]
        gold = int(r["expected"].split("_")[1])
        pick = max(range(len(probs)), key=lambda i: probs[i])
        k5.setdefault(norm(k5_items[r["id"]]["text"]), []).append((int(pick == gold), max(probs) / sum(probs),
                                                                     r["designs"]["knockout"]["seconds"]))
    eval_items = [json.loads(l) for l in (_ROOT / "data" / "eval.jsonl").open(encoding="utf-8")]
    kahn1 = json.loads((_ROOT / "reports" / "eval_v3_temponly_preds.json").read_text(encoding="utf-8"))
    jev = {}
    for line in (_ROOT / "data" / "jev_eval_preds.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        if "answer" in rec:
            jev[rec["i"]] = rec["answer"]
    rows = []
    for i, it in enumerate(eval_items):
        if it["source"] != "banking77" or i not in jev:
            continue
        cands = k5.get(norm(it["state"]))
        if not cands:
            continue
        kc, kp, ks = cands[0]
        jc, jp, _ = jev_outcome(it, jev[i])
        rows.append((kahn1["corrects"][i], kahn1["confidences"][i], jc, jp, kc, kp, ks))
    n = len(rows)
    col = lambda j: [r[j] for r in rows]
    # A partial run covers only the first intents: PolyAI's test file is ordered by intent.
    done_ids = {json.loads(l)["id"] for l in RESULTS.open(encoding="utf-8")}
    intents = {k5_items[i]["expected"] for i in done_ids}
    secs = sorted(col(6))
    return {
        "items_kahn1_eval": sum(1 for it in eval_items if it["source"] == "banking77"),
        "items_jevk5": sum(len(v) for v in k5.values()), "paired": n,
        "kahn1_acc": sum(col(0)) / n, "jev_acc": sum(col(2)) / n, "jevk5_acc": sum(col(4)) / n,
        "kahn1_ece": ece(col(1), col(0)), "jev_ece": ece(col(3), col(2)), "jevk5_ece": ece(col(5), col(4)),
        "jevk5_all_acc": sum(c for v in k5.values() for c, _, _ in v) / sum(len(v) for v in k5.values()),
        "jevk5_p50_s": secs[len(secs) // 2],
        "intents_covered": len(intents), "items_run": len(done_ids), "items_total": len(k5_items),
    }


def jevbench() -> dict:
    if not PUBLIC231.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        PUBLIC231.write_bytes(fetch(JEVK5_PUBLIC_URL))
    k5 = {json.loads(l)["task_id"]: bool(json.loads(l)["correct"]) for l in PUBLIC231.open(encoding="utf-8")}
    jev = json.loads(fetch(JB_PER_TASK))["systems"]["jev-1.13.0"]["public_tasks"]
    items = [json.loads(l) for l in (_ROOT / "data" / "jevbench_eval.jsonl").open(encoding="utf-8")]
    k1 = json.loads((_ROOT / "reports" / "jevbench_v3_preds.json").read_text(encoding="utf-8"))["corrects"]
    t = defaultdict(lambda: [0, 0, 0, 0])
    for it, ok in zip(items, k1):
        for key in (it["source"].removeprefix("jevbench-"), "all"):
            t[key][0] += 1
            t[key][1] += int(ok)
            t[key][2] += int(k5[it["id"]])
            t[key][3] += int(jev[it["id"]][0] == "c")
    v14 = json.loads(fetch(JB_V14))
    official = {s["key"]: {"public": s.get("public_accuracy"), "sealed": s.get("sealed_accuracy"),
                           "tiers": s.get("tiers")}
                for s in v14["systems"] if s["key"] in ("jevk5-v02", "jev-1.13.0")}
    return {"tiers": {k: {"n": n, "kahn1": a / n, "jevk5": b / n, "jev": c / n} for k, (n, a, b, c) in t.items()},
            "jevbench_v14": official, "jevbench_v14_revision": v14["revision"]}


def report(_args) -> None:
    from eval.metrics import ece

    jb = jevbench()
    bk = banking(ece)
    OUT_JSON.write_text(json.dumps({"jevbench": jb, "banking77": bk}, indent=1), encoding="utf-8")
    P = lambda x: f"{100 * x:.1f} %"
    def best(v, key):
        top = max(v["kahn1"], v["jevk5"], v["jev"])
        return f"**{P(v[key])}**" if v[key] == top else P(v[key])

    tiers = "\n".join(f"| {k} | {v['n']} | {best(v, 'kahn1')} | {best(v, 'jevk5')} | {best(v, 'jev')} |"
                      for k, v in sorted(jb["tiers"].items(), key=lambda kv: ["easy", "original", "hard", "all"].index(kv[0])))
    o = jb["jevbench_v14"]
    OUT_MD.write_text(f"""# Kahn1 vs JevK5 (and JEV)

JevK5 v0.2 (github.com/allebee/jevk5): Qwen3.5-4B + a LoRA distilled from Qwen3.6-27B,
Apache-2.0. Kahn1: v3 checkpoint, k = 3, temperature calibration.

## JevBench, 231 public items

JevK5: its own run through JevBench's runner (per-item file in its repository). Jev: the
per-task outcomes JevBench publishes. Kahn1: `reports/jevbench_v3_preds.json`.

| Tier | Items | Kahn1 | JevK5 v0.2 | Jev 1.13.0 |
|---|---:|---:|---:|---:|
{tiers}

JevBench {jb["jevbench_v14_revision"]} re-measured both on its own pods: public accuracy JevK5
{P(o["jevk5-v02"]["public"])}, Jev {P(o["jev-1.13.0"]["public"])}; on the 308 fresh sealed items
JevK5 {P(o["jevk5-v02"]["sealed"])}, Jev {P(o["jev-1.13.0"]["sealed"])}. Kahn1 cannot run the sealed items.

## BANKING77 test split (77 intents)

| | Kahn1 v3 | JevK5 v0.2 | JEV 1.13.0 |
|---|---:|---:|---:|
| Accuracy, {bk["paired"]} paired items | **{P(bk["kahn1_acc"])}** | {P(bk["jevk5_acc"])} | {P(bk["jev_acc"])} |
| ECE (15 bins, p_max) | {bk["kahn1_ece"]:.3f} | {bk["jevk5_ece"]:.3f} | {bk["jev_ece"]:.3f} |

- Coverage: {bk["items_run"]} of {bk["items_total"]} test items were run, which covers {bk["intents_covered"]} of 77
  intents (PolyAI's test file is ordered by intent). All three systems are scored on the same items.
- JevK5: its own `bench/many_options.py`, knockout over ceil(77/16)+1 = 6 passes, in its own
  request shape; {bk["items_jevk5"]} test items, {P(bk["jevk5_all_acc"])} on all of them; median
  {1000 * bk["jevk5_p50_s"]:.0f} ms per item on one RTX 5070 Ti without its optional
  flash-linear-attention kernels (JevK5 reports 116 ms on an H100 with them).
- Kahn1 and JEV: Kahn1's prompt and option order (the held-out run), JEV through its API.
- Paired by normalized text; {bk["items_kahn1_eval"]} BANKING77 items in Kahn1's eval set.
- JevK5 v0.2 trained on 1,500 BANKING77 train texts; Kahn1 never saw BANKING77.
""", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build").set_defaults(func=build)
    sub.add_parser("report").set_defaults(func=report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
