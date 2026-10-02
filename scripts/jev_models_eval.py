"""Run open Jev-format decision models (Laya, Clef-flash) on the Kahn1 held-out set and JevBench.

Both take Jev's question fields (type, instructions, criteria) and return Jev's answer shape, so
each item is asked exactly as JEV was asked (scripts/jev_holdout.py), one question per request:

- held-out (data/eval.jsonl, 14,663 items): Choice over the same 8 options Kahn1 and JEV saw
  (scripts/choice_fairness.py:eight), Score over the same levels, Noul either as the bare
  statement (JEV's native form) or as "The text supports this statement: ..." (--noul supported,
  re-runs the Noul items only);
- JevBench (231 public items): the native JevBench question and state (data/jevbench/*.jsonl),
  criteria descriptions included, as JevBench's runner sends them to Jev.

Kahn1 is not re-run: its per-item outcomes are the saved k = 3 calibrated predictions
(reports/eval_v7_preds.json, reports/jevbench_v7_preds.json).

    python scripts/jev_models_eval.py run --system laya --set heldout            # resumable
    python scripts/jev_models_eval.py run --system laya --set heldout --noul supported
    python scripts/jev_models_eval.py run --system clef-flash --set jevbench --quant int8
    python scripts/jev_models_eval.py report    # reports/OPEN_DECISION_MODELS.md + .json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

D = _ROOT / "data"
R = _ROOT / "reports"
OUT = D / "compare"
REPORT_MD = R / "OPEN_DECISION_MODELS.md"
REPORT_JSON = R / "open_decision_models.json"
SYSTEMS = ("laya", "clef-flash")
# Laya on its multilingual checkpoint read up to 8,192 tokens: JevBench states longer than the
# English checkpoint's 512 tokens are not truncated.
EXTRA_SYSTEMS = ("laya-long",)
CLEF_PATH = Path.home() / "compare" / "clef-flash"
NOUL_TEMPLATES = {"bare": "{statement}", "supported": "The text supports this statement: {statement}"}


def preds_path(system: str, which: str, noul: str = "bare") -> Path:
    suffix = "" if which != "heldout" or noul == "bare" else f"_noul_{noul}"
    return OUT / f"{system}_{which}{suffix}.jsonl"


def heldout_requests(noul: str, path: Path = D / "eval.jsonl") -> list[tuple[int, object, dict]]:
    from scripts.choice_fairness import eight

    out = []
    for i, line in enumerate(path.open(encoding="utf-8")):
        it = json.loads(line)
        if noul != "bare" and it["kind"] != "noul":
            continue
        if it["kind"] == "choice":
            it = eight(it)
            q = {"type": "choice", "instructions": it["prompt"], "criteria": {o: o for o in it["options"]}}
        elif it["kind"] == "score":
            q = {"type": "score", "instructions": it["prompt"], "criteria": list(it["levels"])}
        else:
            q = {"type": "noul", "instructions": NOUL_TEMPLATES[noul].format(statement=it["statement"])}
        out.append((i, it["state"], q))
    return out


def jevbench_tasks() -> dict[str, dict]:
    tasks = {}
    for tier in ("original", "easy", "hard"):
        for line in (D / "jevbench" / f"{tier}.jsonl").open(encoding="utf-8"):
            if line.strip():
                t = json.loads(line)
                tasks[t["id"]] = t
    return tasks


def jevbench_requests() -> list[tuple[int, object, dict]]:
    tasks = jevbench_tasks()
    items = [json.loads(l) for l in (D / "jevbench_eval.jsonl").open(encoding="utf-8")]
    return [(i, tasks[it["id"]]["state"], tasks[it["id"]]["question"]) for i, it in enumerate(items)]


# ---------------------------------------------------------------------------
# runners: each answers a list of (state, question) with Jev answer dicts
# ---------------------------------------------------------------------------

class LayaRunner:
    """Laya through its recommended Router (picks the English or the multilingual checkpoint)."""

    name = "laya"

    def __init__(self, args, long: bool = False):
        from laya import Router
        self.router = Router(preload=True)
        self.batch = 1
        self.kw = {"model": "multilingual", "max_len": 8192} if long else {}

    def answer(self, reqs):
        out = []
        for state, q in reqs:
            r = self.router.predict(state, {"answer": q}, **self.kw)
            out.append({**r["answers"]["answer"], "_routing": r["routing"]["model"],
                        "_truncated": r["usage"].get("truncated", False)})
        return out


class ClefRunner:
    """Clef-flash through its release code (joint_schema_model.py), batched by length."""

    name = "clef-flash"

    def __init__(self, args):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        sys.path.insert(0, str(CLEF_PATH))
        from joint_schema_model import ClefModel, JointSchemaHead

        kw = {}
        if args.quant == "int8":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif args.quant == "nf4":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.bfloat16)
        # load_release_model's steps, minus the image/video processor (needs torchvision): text only.
        backbone = Qwen3_5ForConditionalGeneration.from_pretrained(CLEF_PATH, dtype=torch.bfloat16,
                                                                   device_map={"": "cuda"}, **kw)
        backbone.config.use_cache = False
        head = JointSchemaHead(**json.loads((CLEF_PATH / "joint_head_config.json").read_text()))
        head.load_state_dict(load_file(CLEF_PATH / "joint_head.safetensors"), strict=True)
        self.model = ClefModel(backbone, head.to(device="cuda", dtype=torch.bfloat16)).eval()
        self.tok = AutoTokenizer.from_pretrained(CLEF_PATH)
        self.batch = args.batch
        self.torch = torch

    def answer(self, reqs):
        from joint_schema_model import collate_records, encode_record, systemone_answer

        tok = self.tok
        enc = [encode_record(tok, {"state": s, "questions": {"answer": q}}, max_length=8192) for s, q in reqs]
        with self.torch.inference_mode():
            logits = self.model(collate_records(enc, tok.pad_token_id, self.torch.device("cuda")))
        out = []
        for (s, q), e, lg in zip(reqs, enc, logits):
            probs = dict(zip(e.questions[0].option_ids, lg[0].float().softmax(-1).tolist()))
            out.append({**systemone_answer(q, probs), "_tokens": len(e.input_ids)})
        return out


def run(args) -> None:
    runner_cls = {"laya": LayaRunner, "clef-flash": ClefRunner,
                  "laya-long": lambda a: LayaRunner(a, long=True)}[args.system]
    reqs = (heldout_requests(args.noul) if args.set == "heldout" else jevbench_requests() if args.set == "jevbench"
            else heldout_requests(args.noul, D / "contam_probe.jsonl"))
    path = preds_path(args.system, args.set, args.noul)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {json.loads(l)["i"] for l in path.open(encoding="utf-8") if l.strip()}
    todo = [r for r in reqs if r[0] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{args.system} {args.set} noul={args.noul}: {len(reqs)} requests, {len(done)} done, "
          f"{len(todo)} to go", flush=True)
    if not todo:
        return
    # Similar lengths together, so a batch pads little; results are keyed by item index.
    todo.sort(key=lambda r: len(json.dumps(r[1], ensure_ascii=False)) if not isinstance(r[1], str) else len(r[1]))
    runner = runner_cls(args)
    t0 = time.perf_counter()
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), runner.batch):
            chunk = todo[b:b + runner.batch]
            tb = time.perf_counter()
            answers = runner.answer([(s, q) for _, s, q in chunk])
            ms = 1000 * (time.perf_counter() - tb) / len(chunk)
            for (i, _, _), a in zip(chunk, answers):
                f.write(json.dumps({"i": i, "answer": a, "ms_per_item": round(ms, 1)}, ensure_ascii=False) + "\n")
            f.flush()
            n += len(chunk)
            if n % 500 < runner.batch or n == len(todo):
                print(f"{n}/{len(todo)}  {n / (time.perf_counter() - t0):.1f} items/s", flush=True)
    print("RUN DONE", flush=True)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def load_preds(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {r["i"]: r["answer"] for r in map(json.loads, path.open(encoding="utf-8"))}


def jevbench_outcome(task: dict, a: dict) -> tuple[int, float]:
    """(correct, p_max) for a Jev answer to a native JevBench task."""
    kind, exp = task["question"]["type"], task["expected"]
    if kind == "noul":
        p = float(a.get("noul", 0.0))
        gold = str(exp).lower() in ("yes", "true", "1")
        return int((p > 0.5) == gold), max(p, 1 - p)
    probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
    pick = max(probs, key=probs.get)
    s = sum(probs.values()) or 1.0
    return int(pick == str(exp)), probs[pick] / s


def report() -> None:
    from eval.metrics import ece
    from scripts.choice_fairness import eight
    from scripts.jev_holdout import jev_outcome
    from scripts.kahn1_4b_report import JEV_PER_TASK, mcnemar

    def jl(p: Path) -> dict[int, dict]:
        return {r["i"]: r for r in map(json.loads, p.open(encoding="utf-8"))} if p.exists() else {}

    items = [json.loads(l) for l in (D / "eval.jsonl").open(encoding="utf-8")]
    k1 = json.loads((R / "eval_v7_preds.json").read_text(encoding="utf-8"))
    jev_full, jev8, jev_sup = jl(D / "jev_eval_preds.jsonl"), jl(D / "jev_eval_preds_choice8.jsonl"), \
        jl(D / "jev_eval_preds_noul_supported.jsonl")

    def jev_like(i: int):
        it = items[i]
        if it["kind"] == "choice":
            return jev_outcome(eight(it), jev8[i]["answer"])[:2]
        if it["kind"] == "noul":
            return jev_outcome(it, jev_sup[i]["answer"])[:2]
        return jev_outcome(it, jev_full[i]["answer"])[:2]

    # Per system: item index -> (correct, p_max). Noul from the "supported" run when present.
    held: dict[str, dict[int, tuple[int, float]]] = {
        "Kahn1 4B": {i: (k1["corrects"][i], k1["confidences"][i]) for i in range(len(items))},
        "JEV 1.13.0": {i: jev_like(i) for i in range(len(items))},
    }
    noul_bare: dict[str, dict[int, tuple[int, float]]] = {
        "JEV 1.13.0": {i: jev_outcome(items[i], jev_full[i]["answer"])[:2]
                       for i, it in enumerate(items) if it["kind"] == "noul"}}
    meta: dict[str, dict] = {}
    for s in SYSTEMS:
        base, sup = load_preds(preds_path(s, "heldout")), load_preds(preds_path(s, "heldout", "supported"))
        if not base:
            continue
        res = {}
        for i, a in base.items():
            it = items[i]
            res[i] = jev_outcome(eight(it) if it["kind"] == "choice" else it, a)[:2]
        noul_bare[s] = {i: res[i] for i in res if items[i]["kind"] == "noul"}
        for i, a in sup.items():
            res[i] = jev_outcome(items[i], a)[:2]
        held[s] = res
        meta[s] = {"heldout_items": len(base), "noul_supported": len(sup),
                   "ms_per_item": sorted(r.get("ms_per_item", 0) for r in
                                         map(json.loads, preds_path(s, "heldout").open(encoding="utf-8")))[len(base) // 2],
                   "truncated": sum(1 for a in base.values() if a.get("_truncated"))}

    groups: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        for g in ("all", "kind:" + it["kind"], it["source"]):
            groups[g].append(i)
    names = list(held)
    complete = [n for n in names if len(held[n]) == len(items)]

    def cell(n, ids):
        rs = [held[n][i] for i in ids if i in held[n]]
        return {"n": len(rs), "acc": sum(r[0] for r in rs) / len(rs) if rs else None,
                "ece": ece([r[1] for r in rs], [r[0] for r in rs]) if rs else None}

    held_tab = {g: {n: cell(n, ids) for n in names} for g, ids in groups.items()}
    held_sig = {}
    for n in names[1:]:
        both = [i for i in range(len(items)) if i in held[n]]
        held_sig[n] = mcnemar([held["Kahn1 4B"][i][0] for i in both], [held[n][i][0] for i in both]) + (len(both),)
    bare_tab = {n: cell_ for n, cell_ in (
        (n, {"n": len(v), "acc": sum(x[0] for x in v.values()) / len(v)}) for n, v in noul_bare.items() if v)}

    # JevBench: Kahn1 (ours), Jev (JevBench's run), JevK5 (its authors' run), the rest ours.
    jb = [json.loads(l) for l in (D / "jevbench_eval.jsonl").open(encoding="utf-8")]
    tasks = jevbench_tasks()
    jb_k1 = json.loads((R / "jevbench_v7_preds.json").read_text(encoding="utf-8"))["corrects"]
    per_task = json.loads(urllib.request.urlopen(JEV_PER_TASK, timeout=120).read())["systems"]["jev-1.13.0"][
        "public_tasks"]
    k5 = {r["task_id"]: int(bool(r["correct"])) for r in map(json.loads, (D / "jevk5" / "jevk5-v0.2.jsonl").open(
        encoding="utf-8"))}
    jbc: dict[str, list] = {"Kahn1 4B": list(jb_k1), "Jev 1.13.0": [int(per_task[it["id"]][0] == "c") for it in jb],
                            "JevK5 v0.2": [k5[it["id"]] for it in jb]}
    jb_conf: dict[str, list] = {}
    for s in SYSTEMS + EXTRA_SYSTEMS:
        p = load_preds(preds_path(s, "jevbench"))
        if len(p) == len(jb):
            o = [jevbench_outcome(tasks[it["id"]], p[i]) for i, it in enumerate(jb)]
            jbc[s] = [x[0] for x in o]
            jb_conf[s] = [x[1] for x in o]
    tiers = {"all": list(range(len(jb)))}
    for i, it in enumerate(jb):
        tiers.setdefault(it["source"].removeprefix("jevbench-"), []).append(i)
    jb_tab = {t: {n: sum(c[i] for i in ids) / len(ids) for n, c in jbc.items()} | {"n": len(ids)}
              for t, ids in tiers.items()}
    jb_sig = {n: mcnemar(jbc["Kahn1 4B"], c) for n, c in jbc.items() if n != "Kahn1 4B"}
    jb_ece = {n: ece(jb_conf[n], jbc[n]) for n in jb_conf}

    REPORT_JSON.write_text(json.dumps({"heldout": held_tab, "heldout_mcnemar": held_sig, "noul_bare": bare_tab,
                                       "jevbench": jb_tab, "jevbench_mcnemar": jb_sig, "jevbench_ece": jb_ece,
                                       "meta": meta}, indent=1), encoding="utf-8")

    pct = lambda x: "" if x is None else f"{100 * x:.1f} %"
    disp = {"laya": "Laya", "clef-flash": "Clef-flash", "laya-long": "Laya, multilingual 8k"}.get
    L = ["# Open decision models on the Kahn1 held-out set and JevBench", ""]
    L += ["Same items, same options for every system. Kahn1 4B: k = 3, temperature calibration (saved",
          "predictions). JEV 1.13.0: its saved API answers (Choice over the same 8 options, Noul in the",
          "\"supported\" wording). Laya: its Router, shipped calibration. Clef-flash: its release code,",
          "raw softmax" + (", weights in int8 (bitsandbytes): 9B in bf16 does not fit in 16 GB." if
                           "clef-flash" in held else "."), ""]
    L += ["## Held-out set", "", "| Group | Items | " + " | ".join(disp(n, n) for n in names) + " |",
          "|---|---:|" + "---:|" * len(names)]
    order = ["all", "kind:choice", "kind:score", "kind:noul"] + sorted(g for g in groups if ":" not in g and g != "all")
    for g in order:
        row = held_tab[g]
        L.append(f"| {g.removeprefix('kind:')} | {len(groups[g])} | " +
                 " | ".join(pct(row[n]["acc"]) for n in names) + " |")
    L += ["", "ECE (lower is better): " + ", ".join(f"{n} {held_tab['all'][n]['ece']:.3f}" for n in names
                                                  if held_tab['all'][n]['ece'] is not None) + ".", ""]
    L += ["Noul as the bare statement (JEV's native form): " +
          ", ".join(f"{n} {pct(v['acc'])}" for n, v in bare_tab.items()) + ".", ""]
    L += ["Paired with Kahn1 4B on every item (exact McNemar): " + "; ".join(
        f"{n}: Kahn1 alone right {x}, {n} alone right {y}, p = {p:.2g}" for n, (x, y, p, _) in held_sig.items()) + ".",
          ""]
    L += ["## JevBench, 231 public items", "",
          "Kahn1 4B: our run (k = 3, calibrated). Jev: JevBench's own run. JevK5: its authors' run.",
          "Laya and Clef-flash: our run, native JevBench questions and states.", "",
          "| Tier | Items | " + " | ".join(disp(n, n) for n in jbc) + " |", "|---|---:|" + "---:|" * len(jbc)]
    for t in ("all", "original", "easy", "hard"):
        L.append(f"| {t} | {jb_tab[t]['n']} | " + " | ".join(pct(jb_tab[t][n]) for n in jbc) + " |")
    L += ["", "Paired with Kahn1 4B (exact McNemar): " + "; ".join(
        f"{n}: {x} / {y}, p = {p:.2g}" for n, (x, y, p) in jb_sig.items()) + ".", ""]
    if jb_ece:
        L += ["JevBench ECE: " + ", ".join(f"{n} {v:.3f}" for n, v in jb_ece.items()) + ".", ""]
    L += ["## Caveats", "",
          "- Laya was trained on BANKING77, MASSIVE, SST-5, SciTail, RTE and app_reviews (its model card),",
          "  the six sources of this held-out set: its held-out figures are in-distribution, not zero-shot.",
          "- Laya's English checkpoint reads 512 tokens; longer states are truncated "
          f"({meta.get('laya', {}).get('truncated', 0)} held-out items, "
          f"{sum(1 for a in load_preds(preds_path('laya', 'jevbench')).values() if a.get('_truncated'))} "
          "of the 231 JevBench items). Its multilingual checkpoint read up to 8,192 tokens (laya-long, nothing",
          "  truncated) scores lower on JevBench, so truncation does not explain Laya's JevBench score.",
          "- Clef-flash was run in int8 on one RTX 5070 Ti, not in bf16 as released. Its training data is not published;",
          "  a train-split probe and a gain-over-base check are in CONTAMINATION_PROBE.md.",
          "- Kahn1 4B is k = 3 (three option orders averaged) with temperature calibration, its published",
          "  setting; Laya and Clef-flash answer in one pass, each in its released setting. ECE is therefore",
          "  each system as shipped, not after a common recalibration.", ""]
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", choices=SYSTEMS + EXTRA_SYSTEMS, required=True)
    r.add_argument("--set", choices=("heldout", "jevbench", "probe"), required=True,
                   help="probe: train-split items of the held-out sources (scripts/contamination_probe.py)")
    r.add_argument("--noul", choices=tuple(NOUL_TEMPLATES), default="bare")
    r.add_argument("--quant", choices=("bf16", "int8", "nf4"), default="int8", help="Clef-flash weights")
    r.add_argument("--batch", type=int, default=8, help="Clef-flash items per forward pass")
    r.add_argument("--limit", type=int, default=None)
    sub.add_parser("report")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else report()


if __name__ == "__main__":
    main()
