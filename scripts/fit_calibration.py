"""Fit per-primitive temperatures for a checkpoint, in a given prompt format.

`sysone calibrate` builds its engine with the default config (Qwen2.5 prompt layout,
2048 tokens); this fits the same TemperatureConfig for any format and length, on the
training-disjoint data/val.jsonl, never on the benchmark.

    python scripts/fit_calibration.py --model ~/k1merged/v4 --format qwen3 --out calibration_v4.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--format", default="tags", choices=["tags", "chatml", "qwen3"])
    ap.add_argument("--data", default="data/val.jsonl")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from sysone.calibrate import collect_logits_from_dataset, fit_all
    from sysone.engine import Engine, EngineConfig

    engine = Engine(EngineConfig(model=args.model, prompt_format=args.format, max_model_len=args.max_model_len))
    engine._ensure_loaded()
    by_kind = collect_logits_from_dataset(engine, args.data)
    cfg = fit_all(by_kind)
    cfg.save(args.out)
    print(f"[calibrate] {args.out}: choice T={cfg.choice:.3f} score T={cfg.score:.3f} noul T={cfg.noul:.3f}")


if __name__ == "__main__":
    main()
