"""sysone CLI interface.

Commands:
  sysone serve          — launches the FastAPI inference server
  sysone calibrate      — fits temperature scaling parameters on a JSONL dataset
  sysone evaluate FILE  — evaluates a JSON query (from file or stdin)
  sysone eval-report    — executes complete benchmark evaluation and outputs reports/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_serve(args):
    import os
    import uvicorn
    os.environ.setdefault("SYSONE_EAGER", "1")
    if args.model:
        os.environ["SYSONE_MODEL"] = args.model
    uvicorn.run(
        "sysone.server:app", host=args.host, port=args.port,
        reload=False, workers=1,
    )


def cmd_calibrate(args):
    """Calibrates temperature parameter T on a JSONL dataset (one example per line).

    Expected format per line:
      - Either pre-extracted logits: {"kind": "choice", "logits": [..], "label": 2}
      - Or raw data: {"state": "..", "kind": "choice", "prompt": "..", "options": [..], "label": 2}
    Writes the resulting TemperatureConfig to --out (default: calibration.json).
    """
    from .calibrate import fit_all, TemperatureConfig, collect_logits_from_dataset
    lines = [l.strip() for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        print("[calibrate] Empty dataset file.")
        return

    first_ex = json.loads(lines[0])
    if "logits" in first_ex:
        by_kind: dict[str, tuple[list, list]] = {"choice": ([], []), "score": ([], []), "noul": ([], [])}
        for line in lines:
            ex = json.loads(line)
            kind = ex["kind"]
            by_kind[kind][0].append(ex["logits"])
            by_kind[kind][1].append(ex["label"])
    else:
        from .engine import Engine, EngineConfig
        default_model = "checkpoints/qwen_merged" if Path("checkpoints/qwen_merged").exists() else "Qwen/Qwen2.5-3B-Instruct"
        model_name = getattr(args, "model", None) or default_model
        print(f"[calibrate] Extracting logits using model {model_name}...")
        engine = Engine(EngineConfig(model=model_name))
        by_kind = collect_logits_from_dataset(engine, args.data)

    cfg = fit_all(by_kind)
    out = Path(args.out)
    cfg.save(out)
    print(f"[calibrate] {out} → choice T={cfg.choice:.3f}, "
          f"score T={cfg.score:.3f}, noul T={cfg.noul:.3f}")



def cmd_evaluate(args):
    """Evaluates a JSON query payload read from a file (--file) or stdin."""
    from .engine import Engine, EngineConfig
    from .types import Query
    raw = Path(args.file).read_text(encoding="utf-8") if args.file else sys.stdin.read()
    req = json.loads(raw)
    eng = Engine(EngineConfig(model=args.model))
    query = Query(state=req["state"], questions=req["questions"])
    resp = eng.evaluate(query, n_permutations=req.get("n_permutations", 3))
    print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))


def cmd_eval_report(args):
    from eval.run_eval import run_full_report
    run_full_report(model=args.model, out=args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sysone")
    ap.add_argument("--model", default=None, help="backbone model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="launch FastAPI evaluation server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    p_cal = sub.add_parser("calibrate", help="fit temperature scaling on a JSONL dataset")
    p_cal.add_argument("--data", required=True, help="input JSONL dataset file")
    p_cal.add_argument("--out", default="calibration.json")
    p_cal.add_argument("--model", default="checkpoints/qwen_merged", help="model checkpoint path or name")
    p_cal.set_defaults(func=cmd_calibrate)

    p_eval = sub.add_parser("evaluate", help="evaluate a JSON query payload")
    p_eval.add_argument("--file", default=None, help="JSON input file path (or stdin)")
    p_eval.set_defaults(func=cmd_evaluate)

    p_rep = sub.add_parser("eval-report", help="generate reports/REPORT.md")
    p_rep.add_argument("--out", default="reports/REPORT.md")
    p_rep.set_defaults(func=cmd_eval_report)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
