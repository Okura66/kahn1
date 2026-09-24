"""Run the TypeSafe JEV API over the Kahn1 held-out set and compare, item by item.

Both systems answer the same 14,663 items of data/eval.jsonl, with the same state,
the same prompt and the same options, and are scored against the same ground truth.
Kahn1's side is not re-run: its per-item outcomes are the saved predictions of the
v3 evaluation (reports/eval_v3_temponly_preds.json), in eval-file order.

    python scripts/jev_holdout.py run                 # resumable, writes data/jev_eval_preds.jsonl
    python scripts/jev_holdout.py report              # writes reports/JEV_VS_KAHN1.md + .json

`run` sends one request per item (one question per request, as Kahn1 is scored),
N requests in flight at once, and appends each answer as it lands, so an
interrupted run picks up where it stopped. Needs JEV_API_KEY (or TYPESAFE_API_KEY).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sysone import jev  # noqa: E402

EVAL = _ROOT / "data" / "eval.jsonl"
JEV_PREDS = _ROOT / "data" / "jev_eval_preds.jsonl"
KAHN1_PREDS = _ROOT / "reports" / "eval_v3_temponly_preds.json"
REPORT_MD = _ROOT / "reports" / "JEV_VS_KAHN1.md"
REPORT_JSON = _ROOT / "reports" / "jev_vs_kahn1.json"


def load_items() -> list[dict]:
    return [json.loads(line) for line in EVAL.open(encoding="utf-8")]


# Noul variant: JEV's native form is the bare statement as instructions, which JEV
# tends to read as "is this true in general?". SUPPORTED says what Kahn1's own
# prompt says ("true for this state"), for a second, fairer JEV run.
NOUL_TEMPLATES = {"bare": "{statement}", "supported": "The text supports this statement: {statement}"}
JEV_PREDS_SUPPORTED = _ROOT / "data" / "jev_eval_preds_noul_supported.jsonl"


def payload(item: dict, noul_template: str = "bare") -> dict:
    statement = item.get("statement")
    if item["kind"] == "noul":
        statement = NOUL_TEMPLATES[noul_template].format(statement=statement)
    return jev.question_payload({
        "kind": item["kind"], "prompt": item.get("prompt"), "options": item.get("options"),
        "levels": item.get("levels"), "statement": statement,
    })


def preds_path(noul_template: str) -> Path:
    return JEV_PREDS if noul_template == "bare" else JEV_PREDS_SUPPORTED


def done_indices(path: Path = JEV_PREDS) -> set[int]:
    if not path.exists():
        return set()
    out = set()
    for line in path.open(encoding="utf-8"):
        rec = json.loads(line)
        if "answer" in rec:
            out.add(rec["i"])
    return out


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

async def run(concurrency: int, limit: int | None, model: str, noul_template: str = "bare") -> None:
    items = load_items()
    path = preds_path(noul_template)
    done = done_indices(path)
    # The "supported" variant only changes Noul prompts, so only Noul items are re-run.
    todo = [i for i in range(len(items)) if i not in done
            and (noul_template == "bare" or items[i]["kind"] == "noul")]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(items)} items, {len(done)} already answered, {len(todo)} to go", flush=True)
    if not todo:
        return
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    t0 = time.perf_counter()
    n_ok = n_err = 0

    async with jev.make_client(timeout=60.0) as client:
        with path.open("a", encoding="utf-8") as out:

            async def one(i: int) -> None:
                nonlocal n_ok, n_err
                item = items[i]
                rec = {"i": i, "source": item["source"]}
                async with sem:
                    for attempt in range(5):
                        try:
                            r = await jev.evaluate(item["state"], {"answer": payload(item, noul_template)},
                                                   model=model, client=client)
                            rec.update(answer=r.answers.get("answer", {}), latency_ms=round(r.latency_ms, 1),
                                       model=r.model)
                            break
                        except jev.JevError as exc:
                            rec["error"] = str(exc)[:300]
                            # 429 / 5xx / network: back off and retry; other 4xx will not improve.
                            retry = any(s in str(exc) for s in ("429", "HTTP 5", "request failed"))
                            if not retry:
                                break
                            await asyncio.sleep(2 ** attempt)
                async with lock:
                    if "answer" in rec:
                        rec.pop("error", None)
                        n_ok += 1
                    else:
                        n_err += 1
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    done = n_ok + n_err
                    if done % 250 == 0 or done == len(todo):
                        rate = done / (time.perf_counter() - t0)
                        print(f"{done}/{len(todo)}  ok {n_ok}  errors {n_err}  {rate:.1f} items/s", flush=True)

            await asyncio.gather(*(one(i) for i in todo))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def jev_outcome(item: dict, a: dict) -> tuple[int, float, list[float]]:
    """(correct, p_max, distribution over the item's options) for one JEV answer."""
    kind = item["kind"]
    if kind == "choice":
        probs = a.get("probabilities") or {}
        dist = [float(probs.get(o, 0.0)) for o in item["options"]]
        correct = int(a.get("choice") == item["options"][item["label"]])
    elif kind == "score":
        probs = a.get("probabilities") or {}
        dist = [float(probs.get(str(k), 0.0)) for k in range(len(item["levels"]))]
        pick = max(range(len(dist)), key=lambda k: dist[k]) if any(dist) else round(float(a.get("score", 0)))
        correct = int(pick == item["label"])
    else:
        p_yes = float(a.get("noul", 0.0))
        dist = [p_yes, 1.0 - p_yes]
        correct = int((p_yes > 0.5) == (item["label"] == 1))
    s = sum(dist)
    p_max = max(dist) / s if s > 0 else 0.0
    return correct, p_max, dist


def pct(x: float) -> str:
    return f"{100 * x:.2f} %"


def report() -> None:
    from eval.metrics import ece

    items = load_items()
    k1 = json.loads(KAHN1_PREDS.read_text(encoding="utf-8"))
    assert len(k1["corrects"]) == len(items), "Kahn1 predictions do not match data/eval.jsonl"

    def load(path: Path) -> dict[int, dict]:
        out = {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                rec = json.loads(line)
                if "answer" in rec:
                    out[rec["i"]] = rec
        return out

    jev_recs = load(JEV_PREDS)
    # Same run with the Noul items re-asked in the "supported" wording.
    jev_sup = {**jev_recs, **load(JEV_PREDS_SUPPORTED)}
    models = sorted({r.get("model", "") for r in jev_recs.values()})

    groups: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        groups[it["source"]].append(i)
        groups["kind:" + it["kind"]].append(i)
        groups["all"].append(i)

    def stats(idxs: list[int], recs: dict[int, dict] = jev_recs) -> dict:
        both = [i for i in idxs if i in recs]
        j = [jev_outcome(items[i], recs[i]["answer"]) for i in both]
        jc = [x[0] for x in j]
        kc = [k1["corrects"][i] for i in both]
        wins = sum(1 for a, b in zip(kc, jc) if a and not b)
        losses = sum(1 for a, b in zip(kc, jc) if b and not a)
        return {
            "n": len(idxs), "paired": len(both),
            "kahn1_acc": sum(kc) / len(both) if both else 0.0,
            "jev_acc": sum(jc) / len(both) if both else 0.0,
            "kahn1_ece": ece([k1["confidences"][i] for i in both], kc) if both else 0.0,
            "jev_ece": ece([x[1] for x in j], jc) if both else 0.0,
            "kahn1_only": wins, "jev_only": losses,
        }

    order = ["banking77", "massive", "rte_eval", "scitail_eval", "sst5_eval", "app_reviews_eval",
             "kind:choice", "kind:noul", "kind:score", "all"]
    table = {g: stats(groups[g]) for g in order if g in groups}
    lat = sorted(r["latency_ms"] for r in jev_recs.values())
    summary = {
        "jev_models": models,
        "kahn1": "v3 checkpoint, k = 3, temperature calibration (reports/QWEN_FULL_EVAL_v3_temponly.md)",
        "items": len(items), "jev_answered": len(jev_recs),
        "jev_latency_ms": {"p50": lat[len(lat) // 2], "p95": lat[int(0.95 * (len(lat) - 1))]} if lat else None,
        "groups": table,
        "noul_supported_template": NOUL_TEMPLATES["supported"],
        "groups_noul_supported": {g: stats(groups[g], jev_sup) for g in ("rte_eval", "scitail_eval", "kind:noul", "all")},
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    rows = []
    for g, s in table.items():
        name = g.replace("kind:", "all ").replace("_eval", "")
        rows.append(f"| {name} | {s['paired']} | **{pct(s['kahn1_acc'])}** | **{pct(s['jev_acc'])}** | "
                    f"{s['kahn1_ece']:.4f} | {s['jev_ece']:.4f} | {s['kahn1_only']} | {s['jev_only']} |")
    sup_rows = []
    for g, s in summary["groups_noul_supported"].items():
        base = table[g]["jev_acc"]
        sup_rows.append(f"| {g.replace('kind:', 'all ').replace('_eval', '')} | {s['paired']} | **{pct(s['kahn1_acc'])}** | "
                        f"{pct(base)} | **{pct(s['jev_acc'])}** | {s['jev_ece']:.4f} |")
    lat_line = (f"JEV round trip over the internet: p50 {summary['jev_latency_ms']['p50']:.0f} ms, "
                f"p95 {summary['jev_latency_ms']['p95']:.0f} ms." if lat else "")
    REPORT_MD.write_text(f"""# Kahn1 vs JEV — paired, on the Kahn1 held-out set

- Items: {len(items)} held-out items of `data/eval.jsonl`; JEV answered {len(jev_recs)}.
- JEV: TypeSafe API, model {", ".join(models) or "?"}, one question per request, run with `scripts/jev_holdout.py`.
- Kahn1: {summary['kahn1']}.
- Same state, prompt and options on both sides, scored against the same labels. ECE: 15 bins on p_max.
- {lat_line}

| Dataset | Paired items | Kahn1 | JEV | Kahn1 ECE | JEV ECE | Only Kahn1 right | Only JEV right |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Noul wording

JEV's native Noul form is the bare statement as `instructions` (also what JevBench sends). On SciTail,
whose negatives are hypotheses the text does not support, JEV then answers yes to most items: it seems to
judge whether the statement is true in general. Kahn1's prompt asks whether it is "true for this state".
The Noul items were re-run with `{NOUL_TEMPLATES["supported"]}` to give JEV the same question:

| Dataset | Items | Kahn1 | JEV, bare statement | JEV, "supported" wording | JEV ECE (supported) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(sup_rows)}

Kahn1's latency (p50 36.4 ms, k = 3) is local vLLM on one RTX 5070 Ti; JEV's is a hosted
API measured from the client, network included. They are not the same measurement.
""", encoding="utf-8")
    print(REPORT_MD.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--concurrency", type=int, default=6)
    r.add_argument("--limit", type=int, default=None, help="answer at most this many new items")
    r.add_argument("--model", default=jev.DEFAULT_MODEL)
    r.add_argument("--noul-template", choices=sorted(NOUL_TEMPLATES), default="bare",
                   help="how the Noul statement is worded for JEV (supported: re-runs Noul items only)")
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "run":
        asyncio.run(run(args.concurrency, args.limit, args.model, args.noul_template))
    else:
        report()


if __name__ == "__main__":
    main()
