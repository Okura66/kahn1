"""Choice under the same option set on every side.

eval/baselines.py:_prepare_choice_options caps each Choice question at 8 options (the gold
one plus 7 distractors seeded from the state) before Kahn1 answers it, so Kahn1's held-out
banking77 / MASSIVE numbers are 8-option numbers. JEV (scripts/jev_holdout.py) and JevK5
(scripts/jevk5_vs_kahn1.py) were given every intent (77 / 60). This script measures both
conditions on the same items:

    python scripts/choice_fairness.py jev8          # JEV on the exact 8-option sets Kahn1 saw
    python scripts/choice_fairness.py kahn1-full    # Kahn1 on all intents (evaluate_two_stage), GPU
    python scripts/choice_fairness.py report        # reports/CHOICE_FAIRNESS.md + .json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EVAL = _ROOT / "data" / "eval.jsonl"
JEV8 = _ROOT / "data" / "jev_eval_preds_choice8.jsonl"
KAHN1_FULL = _ROOT / "data" / "kahn1_v3_choice_full.jsonl"
OUT_MD = _ROOT / "reports" / "CHOICE_FAIRNESS.md"
OUT_JSON = _ROOT / "reports" / "choice_fairness.json"
SOURCES = ("banking77", "massive")


def choice_items() -> list[tuple[int, dict]]:
    items = [json.loads(line) for line in EVAL.open(encoding="utf-8")]
    return [(i, it) for i, it in enumerate(items) if it["kind"] == "choice" and it["source"] in SOURCES]


def eight(it: dict) -> dict:
    """The item as Kahn1 saw it: the 8 options and the gold index among them."""
    from eval.baselines import _prepare_choice_options

    options, label, allow_other = _prepare_choice_options(it)
    return {**it, "options": options, "label": label, "allow_other": allow_other}


def done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["i"] for line in path.open(encoding="utf-8") if "answer" in json.loads(line)}


# ---------------------------------------------------------------------------

async def jev8(concurrency: int) -> None:
    from scripts.jev_holdout import payload
    from sysone import jev

    todo = [(i, eight(it)) for i, it in choice_items() if i not in done(JEV8)]
    print(f"{len(todo)} Choice items to send to JEV with 8 options", flush=True)
    sem, lock, n = asyncio.Semaphore(concurrency), asyncio.Lock(), 0
    async with jev.make_client(timeout=60.0) as client:
        with JEV8.open("a", encoding="utf-8") as out:
            async def one(i: int, it: dict) -> None:
                nonlocal n
                rec = {"i": i, "source": it["source"], "options": it["options"], "label": it["label"]}
                async with sem:
                    for attempt in range(5):
                        try:
                            r = await jev.evaluate(it["state"], {"answer": payload(it)}, client=client)
                            rec.update(answer=r.answers.get("answer", {}), latency_ms=round(r.latency_ms, 1))
                            break
                        except jev.JevError as exc:
                            rec["error"] = str(exc)[:300]
                            if not any(s in str(exc) for s in ("429", "HTTP 5", "request failed")):
                                break
                            await asyncio.sleep(2 ** attempt)
                async with lock:
                    if "answer" in rec:
                        rec.pop("error", None)
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    n += 1
                    if n % 500 == 0 or n == len(todo):
                        print(f"{n}/{len(todo)}", flush=True)

            await asyncio.gather(*(one(i, it) for i, it in todo))


def kahn1_full(model: str, max_items: int | None, fmt: str = "tags", out_path: Path = KAHN1_FULL,
               same_as: Path | None = None) -> None:
    """Kahn1 on every intent: evaluate_two_stage (one Noul per option, top 10, then a Choice).

    same_as restricts the run to the items already answered in another run's file, so two
    checkpoints are compared on the same items.
    """
    from sysone.engine import Engine, EngineConfig
    from sysone.types import ChoiceQuestion, Query

    engine = Engine(EngineConfig(model=model, max_model_len=2048, prompt_format=fmt))
    keep = done(same_as) if same_as else None
    todo = [(i, it) for i, it in choice_items()
            if i not in done(out_path) and (keep is None or i in keep)][:max_items]
    print(f"{len(todo)} Choice items for {model} with every intent", flush=True)
    t0 = time.perf_counter()
    with out_path.open("a", encoding="utf-8") as out:
        for n, (i, it) in enumerate(todo, 1):
            # model_construct: ChoiceQuestion validates at most 64 options, banking77 has 77;
            # the two-stage route itself has no such limit.
            q = Query.model_construct(state=it["state"], questions=[ChoiceQuestion.model_construct(
                kind="choice", key="answer", prompt=it["prompt"], options=it["options"], allow_other=False)])
            a = engine.evaluate_two_stage(q).answers["answer"]
            out.write(json.dumps({"i": i, "source": it["source"], "answer": {
                "choice": a.choice, "confidence": a.confidence,
                "p_max": max(a.probabilities.values())}}, ensure_ascii=False) + "\n")
            if n % 250 == 0 or n == len(todo):
                print(f"{n}/{len(todo)}  {n / (time.perf_counter() - t0):.1f} items/s", flush=True)
                out.flush()


def report() -> None:
    from eval.metrics import ece
    from scripts.jev_holdout import jev_outcome

    items = dict(choice_items())
    kahn1 = json.loads((_ROOT / "reports" / "eval_v3_temponly_preds.json").read_text(encoding="utf-8"))
    jev_full = {}
    for line in (_ROOT / "data" / "jev_eval_preds.jsonl").open(encoding="utf-8"):
        rec = json.loads(line)
        if "answer" in rec and rec["i"] in items:
            jev_full[rec["i"]] = rec["answer"]
    load = lambda p: {json.loads(l)["i"]: json.loads(l) for l in p.open(encoding="utf-8")
                      if "answer" in json.loads(l)} if p.exists() else {}
    j8, kf = load(JEV8), load(KAHN1_FULL)

    table = {}
    for src in SOURCES + ("all",):
        ids = [i for i, it in items.items() if src in ("all", it["source"])]
        row = {"n": len(ids)}
        # 8 options: Kahn1 held-out run vs JEV on the same 8-option sets.
        p8 = [i for i in ids if i in j8]
        if p8:
            k = [kahn1["corrects"][i] for i in p8]
            j = [jev_outcome(eight(items[i]), j8[i]["answer"]) for i in p8]
            row["opt8"] = {"paired": len(p8), "kahn1": sum(k) / len(p8), "jev": sum(x[0] for x in j) / len(p8),
                           "kahn1_ece": ece([kahn1["confidences"][i] for i in p8], k),
                           "jev_ece": ece([x[1] for x in j], [x[0] for x in j])}
        # Every intent: Kahn1 two-stage vs JEV's full-set run.
        pf = [i for i in ids if i in kf and i in jev_full]
        if pf:
            k = [int(kf[i]["answer"]["choice"] == items[i]["options"][items[i]["label"]]) for i in pf]
            j = [jev_outcome(items[i], jev_full[i]) for i in pf]
            row["full"] = {"paired": len(pf), "kahn1": sum(k) / len(pf), "jev": sum(x[0] for x in j) / len(pf),
                           "kahn1_ece": ece([kf[i]["answer"]["p_max"] for i in pf], k),
                           "jev_ece": ece([x[1] for x in j], [x[0] for x in j])}
        table[src] = row
    OUT_JSON.write_text(json.dumps(table, indent=1), encoding="utf-8")
    P = lambda x: f"{100 * x:.2f} %"
    lines = []
    for src, r in table.items():
        for cond, label in (("opt8", "8 options (gold + 7 seeded distractors)"), ("full", "every intent")):
            if cond in r:
                c = r[cond]
                lines.append(f"| {src} | {label} | {c['paired']} | {P(c['kahn1'])} | {P(c['jev'])} | "
                             f"{c['kahn1_ece']:.3f} | {c['jev_ece']:.3f} |")
    OUT_MD.write_text(f"""# Choice: the same option set on every side

Kahn1's held-out evaluation (eval/baselines.py:_prepare_choice_options) answers each Choice
question over 8 options: the gold one and 7 distractors seeded from the state. The first JEV
comparison gave JEV every intent. Both conditions, paired:

| Source | Options | Items | Kahn1 v3 | JEV 1.13.0 | Kahn1 ECE | JEV ECE |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

- 8 options: Kahn1 = the held-out run (k = 3, temperature calibration); JEV = the same 8 options.
- Every intent: Kahn1 = evaluate_two_stage (one Noul per option, top 10, then a Choice; k = 1,
  uncalibrated, which does not change its picks); JEV = its run with every intent.
""", encoding="utf-8")
    print(OUT_MD.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jev8")
    j.add_argument("--concurrency", type=int, default=6)
    k = sub.add_parser("kahn1-full")
    k.add_argument("--model", default="checkpoints/qwen_merged")
    k.add_argument("--max-items", type=int, default=None)
    k.add_argument("--prompt-format", default="tags", choices=["tags", "chatml", "qwen3"])
    k.add_argument("--out", type=Path, default=KAHN1_FULL)
    k.add_argument("--same-as", type=Path, default=None, help="only the items answered in this file")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "jev8":
        asyncio.run(jev8(args.concurrency))
    elif args.cmd == "kahn1-full":
        kahn1_full(args.model, args.max_items, args.prompt_format, args.out, args.same_as)
    else:
        report()


if __name__ == "__main__":
    main()
