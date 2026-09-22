"""Pick the best LoRA checkpoint by real held-out accuracy, not validation NLL.

The training loop selects `best` on a 150-example stratified validation NLL, which
is noisy: a 0.58-vs-0.60 gap is within sampling noise, and a late checkpoint on a
decayed LR often generalizes as well or better. This script settles it empirically:
merge each candidate, score it on a larger stratified sample of the full eval set,
and select on balanced accuracy (mean of per-primitive accuracy, so Choice does not
dominate), tie-broken by NLL. The winner is merged into checkpoints/qwen_merged.

Raw engine, no calibration — checkpoints are compared on their own probabilities.

    python scripts/select_checkpoint.py \
        --candidates checkpoints/qwen_lora_v3/best \
                     checkpoints/qwen_lora_v3/step_500 \
                     checkpoints/qwen_lora_v3/final \
        --per-kind 400
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TMP = Path("checkpoints/_select_tmp")


def stratified_sample(eval_path: Path, per_kind: int, seed: int = 0) -> list[dict]:
    by_kind: dict[str, list[dict]] = {}
    for line in eval_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ex = json.loads(line)
            by_kind.setdefault(ex["kind"], []).append(ex)
    out = []
    for k, rows in sorted(by_kind.items()):
        rng = random.Random(f"{seed}:{k}")
        rng.shuffle(rows)
        out.extend(rows[:per_kind])
    return out


def score_checkpoint(merged_dir: str, sample: list[dict], max_model_len: int) -> dict:
    from sysone.engine import Engine, EngineConfig
    from eval.baselines import run_full_system
    from eval.metrics import compute_all

    engine = Engine(EngineConfig(model=merged_dir, max_model_len=max_model_len))
    probs, labels, conf, corr, _ = run_full_system(engine, sample, n_permutations=1)

    # Per-kind accuracy, then a balanced mean so no primitive dominates.
    by_kind: dict[str, list[int]] = {}
    for ex, c in zip(sample, corr):
        by_kind.setdefault(ex["kind"], []).append(c)
    per_kind = {k: sum(v) / len(v) for k, v in by_kind.items()}
    balanced = sum(per_kind.values()) / len(per_kind)
    m = compute_all(probs, labels, conf, corr)
    try:
        engine.close()
    except Exception:
        pass
    return {"balanced_acc": balanced, "per_kind": per_kind,
            "global_acc": m.accuracy, "nll": m.nll, "ece": m.ece}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="LoRA adapter dirs to compare")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--eval", default="data/eval.jsonl")
    ap.add_argument("--per-kind", type=int, default=400)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--out", default="checkpoints/qwen_merged",
                    help="where the winning merged checkpoint is written")
    args = ap.parse_args()

    from scripts.merge_qwen_lora import merge_qwen

    sample = stratified_sample(Path(args.eval), args.per_kind)
    print(f"[select] scoring {len(args.candidates)} candidates on {len(sample)} items "
          f"({args.per_kind}/kind)\n")

    results = {}
    for cand in args.candidates:
        print(f"[select] === {cand} ===")
        if TMP.exists():
            shutil.rmtree(TMP)
        merge_qwen(base_model_name=args.base, adapter_path=cand, output_dir=str(TMP))
        res = score_checkpoint(str(TMP), sample, args.max_model_len)
        results[cand] = res
        pk = "  ".join(f"{k}={v*100:.1f}%" for k, v in sorted(res["per_kind"].items()))
        print(f"[select] {cand}: balanced={res['balanced_acc']*100:.2f}%  "
              f"global={res['global_acc']*100:.2f}%  NLL={res['nll']:.4f}  ECE={res['ece']:.4f}")
        print(f"[select]   per-kind: {pk}\n")

    winner = max(results, key=lambda c: (results[c]["balanced_acc"], -results[c]["nll"]))
    print("=" * 60)
    print(f"[select] WINNER: {winner} "
          f"(balanced {results[winner]['balanced_acc']*100:.2f}%)")
    print("=" * 60)

    # Merge the winner into the publish location.
    if TMP.exists():
        shutil.rmtree(TMP)
    merge_qwen(base_model_name=args.base, adapter_path=winner, output_dir=args.out)
    print(f"[select] winner merged into {args.out}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/checkpoint_selection.json").write_text(
        json.dumps({"winner": winner, "results": results,
                    "n_items": len(sample), "per_kind": args.per_kind}, indent=2),
        encoding="utf-8")
    print("[select] wrote reports/checkpoint_selection.json")


if __name__ == "__main__":
    main()
