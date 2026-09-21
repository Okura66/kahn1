"""Run the three question primitives on CPU, with calibration and debiasing.

Usage:
    python scripts/run_cpu_demo.py --model Okura66/Kahn1-Qwen2.5-3B
    python scripts/run_cpu_demo.py --model ./Kahn1-Qwen2.5-3B --bench
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sysone.calibrate import CalibratedEngine, TemperatureConfig
from sysone.cpu import CPUEngine
from sysone.types import ChoiceQuestion, NoulQuestion, Query, ScoreQuestion

DEMO = Query(
    state=(
        "Customer message: 'I was charged twice for my subscription this month "
        "and I need an immediate refund.'"
    ),
    questions=[
        ChoiceQuestion(
            key="intent",
            prompt="What is the customer's intent?",
            options=["dispute_charge", "cancel_subscription", "update_payment",
                     "general_inquiry"],
            # The fallback is only meaningful when none of the options can apply.
            allow_other=False,
        ),
        ScoreQuestion(
            key="urgency",
            prompt="Rate the urgency level:",
            levels=["low", "medium", "high", "critical"],
        ),
        NoulQuestion(
            key="asks_refund",
            statement="The customer is asking for a refund.",
        ),
    ],
)


def show(resp) -> None:
    for key, ans in resp.answers.items():
        name = type(ans).__name__
        if name == "ChoiceAnswer":
            print(f"  {key:12s} -> {ans.choice}  (confidence {ans.confidence:.3f})")
        elif name == "ScoreAnswer":
            print(f"  {key:12s} -> {ans.level}  E[score]={ans.expected_score:.3f} "
                  f"(confidence {ans.confidence:.3f})")
        else:
            print(f"  {key:12s} -> p(yes)={ans.noul:.3f}")
            continue
        for label, p in sorted(ans.probabilities.items(), key=lambda kv: -kv[1]):
            print(f"       {p:6.2%}  {label}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Okura66/Kahn1-Qwen2.5-3B")
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--permutations", type=int, default=3)
    ap.add_argument("--calibration", default="calibration.json")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    engine = CPUEngine(
        model=args.model,
        dtype=args.dtype,
        batch_size=args.batch_size,
        num_threads=args.threads or None,
    )

    cal_path = Path(args.calibration)
    if cal_path.exists():
        cfg = TemperatureConfig.load(cal_path)
        print(f"[cpu] calibration T_choice={cfg.choice:.3f} "
              f"T_score={cfg.score:.3f} T_noul={cfg.noul:.3f}")
        runner = CalibratedEngine(engine, cfg)
    else:
        print(f"[cpu] no calibration file at {cal_path}; using raw probabilities")
        runner = engine

    for i in range(args.runs):
        t0 = time.perf_counter()
        resp = runner.evaluate(DEMO, n_permutations=args.permutations)
        elapsed = (time.perf_counter() - t0) * 1000.0
        print(f"\n=== run {i + 1}/{args.runs} | {elapsed:.0f} ms ===")
        show(resp)


if __name__ == "__main__":
    main()
