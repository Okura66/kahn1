"""Quick, batched evaluation of a backbone or checkpoint, for comparing bases and prompt formats.

One forward pass per question (k = 1), no calibration, every prompt in one vLLM call. Scores
two sets:

- the stratified held-out sample checkpoint selection uses (scripts/select_checkpoint.py:
  400 items per primitive from data/eval.jsonl, seed 0), Choice capped at the 8 options the
  held-out evaluation uses (eval/baselines.py:_prepare_choice_options);
- the 231 public JevBench items (data/jevbench_eval.jsonl), all of their options.

It also reports the probability mass the model puts on the answer tokens before
renormalisation (vLLM returns full-vocabulary log-probs): how well an untrained model follows
the answer format.

    python scripts/eval_base.py --model Qwen/Qwen3.5-4B --format qwen3 --multimodal \
        --out reports/base_eval/qwen35_4b_qwen3.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval.baselines import _prepare_choice_options  # noqa: E402
from scripts.select_checkpoint import stratified_sample  # noqa: E402
from sysone.types import ChoiceQuestion, NoulQuestion, Query, ScoreQuestion  # noqa: E402


def question(ex: dict, cap: bool):
    if ex["kind"] == "choice":
        options, label, allow_other = _prepare_choice_options(ex) if cap else (ex["options"], ex["label"], False)
        return ChoiceQuestion(key="q", prompt=ex["prompt"], options=options, allow_other=allow_other), label
    if ex["kind"] == "score":
        return ScoreQuestion(key="q", prompt=ex["prompt"], levels=ex["levels"]), ex["label"]
    return NoulQuestion(key="q", statement=ex["statement"], prompt=ex.get("prompt", "")), ex["label"]


def score(engine, examples: list[dict], cap: bool) -> list[dict]:
    """One entry per example: correct, p_max, answer-token mass."""
    entries, meta = [], []
    for ex in examples:
        q, label = question(ex, cap)
        for e in engine._build_batch(Query(state=ex["state"], questions=[q]), n_permutations=1):
            entries.append(e)
            meta.append((ex, label))
    prompts = [e["spec"].full_text for e in entries]
    params = [engine._make_params(e["resolved"].token_ids) for e in entries]
    t0 = time.perf_counter()
    outputs = engine._llm.generate(prompts, params, use_tqdm=False)
    secs = time.perf_counter() - t0
    rows = []
    for out, e, (ex, label) in zip(outputs, entries, meta):
        lr = engine._extract(out, e["spec"], e["resolved"])
        mass = sum(math.exp(x) for x in lr.raw_logits if x != float("-inf"))
        probs = lr.probs
        if ex["kind"] == "noul":
            pred = 1 if probs[0] > 0.5 else 0          # NOUL_LABELS: index 0 = yes
            gold = 1 if label == 1 else 0
        else:
            pred = max(range(len(probs)), key=lambda i: probs[i])
            gold = label if label >= 0 else len(probs) - 1
        rows.append({"source": ex["source"], "kind": ex["kind"], "lang": ex.get("lang"), "correct": int(pred == gold),
                     "p_max": max(probs), "mass": mass})
    return rows, secs


def summarize(rows: list[dict]) -> dict:
    from eval.metrics import ece

    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups["kind:" + r["kind"]].append(r)
        groups[r["source"]].append(r)
        groups["all"].append(r)
        if r.get("lang"):
            groups["lang:" + r["lang"]].append(r)
    for g, rs in sorted(groups.items()):
        out[g] = {"n": len(rs), "acc": sum(r["correct"] for r in rs) / len(rs),
                  "ece": ece([r["p_max"] for r in rs], [r["correct"] for r in rs]),
                  "mass": sum(r["mass"] for r in rs) / len(rs)}
    kinds = [v["acc"] for k, v in out.items() if k.startswith("kind:")]
    out["balanced_acc"] = sum(kinds) / len(kinds)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--format", default="tags", choices=["tags", "chatml", "qwen3"])
    ap.add_argument("--multimodal", action="store_true", help="checkpoint has a vision tower: load text only")
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--per-kind", type=int, default=400)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    ap.add_argument("--sample-file", default=None,
                    help="score every item of this JSONL instead of the stratified held-out sample")
    ap.add_argument("--no-jevbench", action="store_true", help="skip JevBench (e.g. for checkpoint selection)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from sysone.engine import Engine, EngineConfig

    extra = {"language_model_only": True} if args.multimodal else {}
    engine = Engine(EngineConfig(model=args.model, prompt_format=args.format, max_model_len=args.max_model_len,
                                 gpu_memory_utilization=args.gpu_memory_utilization, extra_llm_kwargs=extra))
    engine._ensure_loaded()
    if args.sample_file:
        sample = [json.loads(l) for l in Path(args.sample_file).open(encoding="utf-8")]
    else:
        sample = stratified_sample(_ROOT / "data" / "eval.jsonl", args.per_kind)
    held, t1 = score(engine, sample, cap=True)
    res = {"model": args.model, "format": args.format, "k": 1, "calibration": None,
           "sample": args.sample_file or "stratified data/eval.jsonl", "heldout_sample": summarize(held),
           "seconds": {"heldout": round(t1, 1)}}
    if not args.no_jevbench:
        jb = [json.loads(l) for l in (_ROOT / "data" / "jevbench_eval.jsonl").open(encoding="utf-8")]
        jbr, t2 = score(engine, jb, cap=False)
        res["jevbench_public"] = summarize(jbr)
        res["seconds"]["jevbench"] = round(t2, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    h = res["heldout_sample"]
    line = (f"{args.model} [{args.format}]  sample balanced {100 * h['balanced_acc']:.1f}  "
            f"(choice {100 * h['kind:choice']['acc']:.1f} score {100 * h['kind:score']['acc']:.1f} "
            f"noul {100 * h['kind:noul']['acc']:.1f}, mass {h['all']['mass']:.2f})")
    if "jevbench_public" in res:
        j = res["jevbench_public"]
        line += (f"  |  JevBench {100 * j['all']['acc']:.1f} (easy {100 * j['jevbench-easy']['acc']:.1f} std "
                 f"{100 * j['jevbench-original']['acc']:.1f} hard {100 * j['jevbench-hard']['acc']:.1f}, mass "
                 f"{j['all']['mass']:.2f})")
    print(line + f"  {res['seconds']}", flush=True)


if __name__ == "__main__":
    main()
