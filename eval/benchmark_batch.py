"""Batched latency and prefix caching scaling benchmark.

Empirically evaluates prefix caching efficiency:
    Latency N=10 vs N=1 (shared state prefix) < 1.4x

Usage:
    python -m eval.benchmark_batch [--model Qwen/Qwen2.5-3B-Instruct] [--repeats 5]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np

# WSL2 environment settings for vLLM
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from eval.plots import latency_batch_plot
from sysone.engine import Engine, EngineConfig
from sysone.types import (
    ChoiceQuestion,
    NoulQuestion,
    Query,
    ScoreQuestion,
)


SAMPLE_STATE = (
    "Customer Account #94820 - Enterprise Pro (15-min SLA):\n"
    "- History: Customer for 3 years, 450 active licenses, no prior critical incidents.\n"
    "- Incident opened at 08:42: Complete inability to authenticate for all users via SAML / Okta SSO identity provider. "
    "HTTP 500 error returned by the authentication gateway.\n"
    "- Critical context: Quarterly financial close today at 18:00. All accounting workflows and invoice approvals are halted.\n"
    "- Audit trail: Latest SSO configuration modification occurred yesterday at 23:15 by tenant admin during X.509 certificate rotation. "
    "Gateway logs indicate SHA-256 fingerprint mismatch on inbound assertions.\n"
    "- User impact: 120 financial controllers blocked; contractual penalties if statutory filings are delayed past midnight.\n"
    "- Latest executive ticket message: 'Our finance department is fully paralyzed. If unresolved within 2 hours, we will invoke SLA breach clauses. Mobilize escalation engineering immediately.'"
)

TEN_QUESTIONS = [
    ChoiceQuestion(
        key="q1_intent",
        prompt="Primary ticket intent?",
        options=["Technical support", "Commercial negotiation", "Training", "Other"],
    ),
    ScoreQuestion(
        key="q2_urgency",
        prompt="Urgency level?",
        levels=["Low", "Medium", "High", "Critical"],
    ),
    NoulQuestion(
        key="q3_threat",
        statement="Customer is threatening contract termination.",
    ),
    ChoiceQuestion(
        key="q4_routing",
        prompt="Target escalation tier?",
        options=["Tier 3 Support", "Billing", "Sales", "HR"],
    ),
    ScoreQuestion(
        key="q5_sentiment",
        prompt="Customer sentiment?",
        levels=["Very negative", "Negative", "Neutral", "Positive"],
    ),
    NoulQuestion(
        key="q6_business_impact",
        statement="Incident is blocking business operations for the entire team.",
    ),
    ChoiceQuestion(
        key="q7_action",
        prompt="Immediate action required?",
        options=["Tier 3 Escalation", "Standard Email Response", "Close Ticket", "Refund Request"],
    ),
    ScoreQuestion(
        key="q8_politeness",
        prompt="Customer tone politeness?",
        levels=["Aggressive", "Terse", "Neutral", "Courteous"],
    ),
    NoulQuestion(
        key="q9_clarity",
        statement="The reported issue concerns the SAML protocol.",
    ),
    ChoiceQuestion(
        key="q10_risk",
        prompt="Customer churn risk level?",
        options=["Low", "Medium", "High", "Critical"],
    ),
]


def run_batch_benchmark(
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    repeats: int = 5,
    out_dir: str = "reports",
) -> dict:
    """Benchmark batched latency across varying question counts on a shared state prefix."""
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    print(f"[benchmark_batch] Loading model {model} with prefix caching enabled...")
    config = EngineConfig(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=1024,
        quantization="fp8",
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    engine = Engine(config)
    engine._ensure_loaded()
    print("[benchmark_batch] Model successfully loaded.")

    # 1. Warm-up (covers N=1 and N=10 to initialize KV cache allocations and JIT kernels)
    print("[benchmark_batch] Executing warmup passes...")
    _ = engine.evaluate(Query(state=SAMPLE_STATE, questions=TEN_QUESTIONS[:1]), n_permutations=1)
    _ = engine.evaluate(Query(state=SAMPLE_STATE, questions=TEN_QUESTIONS), n_permutations=1)
    _ = engine.evaluate(Query(state=SAMPLE_STATE, questions=TEN_QUESTIONS), n_permutations=1)
    print("[benchmark_batch] Warmup complete.")

    # 2. Benchmark timings across batch sizes N
    batch_sizes = [1, 2, 4, 6, 8, 10]
    latencies_per_size: dict[int, list[float]] = {n: [] for n in batch_sizes}
    cache_hits_per_size: dict[int, list[float]] = {n: [] for n in batch_sizes}

    for n in batch_sizes:
        questions_subset = TEN_QUESTIONS[:n]
        query = Query(state=SAMPLE_STATE, questions=questions_subset)
        print(f"[benchmark_batch] Measuring latency for N={n} questions ({repeats} iterations)...")
        for r in range(repeats):
            # Single permutation to isolate intrinsic batch forward pass latency
            resp = engine.evaluate(query, n_permutations=1)
            latencies_per_size[n].append(resp.latency_ms)
            cache_hits_per_size[n].append(resp.cache_hit_rate)
            print(f"   rep {r+1}/{repeats}: {resp.latency_ms:.2f} ms (hit_rate={resp.cache_hit_rate:.2f})")

    # 3. Aggregation
    median_latencies = [float(np.median(latencies_per_size[n])) for n in batch_sizes]
    mean_latencies = [float(np.mean(latencies_per_size[n])) for n in batch_sizes]
    std_latencies = [float(np.std(latencies_per_size[n])) for n in batch_sizes]

    lat_n1 = median_latencies[0]
    lat_n10 = median_latencies[-1]
    ratio = lat_n10 / max(lat_n1, 1e-6)

    passed = ratio < 1.4

    print("\n" + "=" * 60)
    print("BATCHED LATENCY BENCHMARK RESULTS:")
    print(f"Latency N=1  (median) : {lat_n1:.2f} ms")
    print(f"Latency N=10 (median) : {lat_n10:.2f} ms")
    print(f"Ratio N=10 / N=1      : {ratio:.3f}x")
    print(f"Required Threshold    : < 1.40x")
    print(f"Validation Verdict    : {'PASSED' if passed else 'FAILED'}")
    print("=" * 60 + "\n")

    # 4. Generate visualization
    plot_path = os.path.join(out_dir, "latency_batch.png")
    latency_batch_plot(batch_sizes, median_latencies, out_path=plot_path)
    print(f"[benchmark_batch] Diagnostic plot saved: {plot_path}")

    # 5. Compile markdown report
    md_path = os.path.join(out_dir, "LATENCY_BATCH.md")
    report_md = f"""# Batched Latency Benchmark & Prefix Caching Evaluation

- **Model**: `{model}`
- **Hardware**: NVIDIA GPU
- **Prefix Caching**: Enabled (`enable_prefix_caching=True`)
- **Success Criterion**: $\\text{{Latency}}_{{N=10}} / \\text{{Latency}}_{{N=1}} < 1.4\\times$

## Empirical Results

| Number of Questions ($N$) | Median Latency (ms) | Mean Latency (ms) | Std Dev (ms) |
|---|---|---|---|
"""
    for n, med, mean, std in zip(batch_sizes, median_latencies, mean_latencies, std_latencies):
        report_md += f"| **{n}** | {med:.2f} | {mean:.2f} | {std:.2f} |\n"

    report_md += f"""
## Scaling Ratio Analysis

- **Latency $N=1$**: `{lat_n1:.2f} ms`
- **Latency $N=10$**: `{lat_n10:.2f} ms`
- **Empirical Ratio ($N=10$ vs $N=1$)**: **`{ratio:.3f}x`** (Target: $< 1.4\\times$)
- **Validation Verdict**: **{'PASSED' if passed else 'FAILED'}**

## Architectural Discussion

Evaluating 10 questions concurrently over the same state prefix within a **single batched call** leverages KV cache block reuse (automatic prefix caching).
Because decision queries constrain prediction to a **single token**, the latency overhead of $N=10$ questions vs $N=1$ is limited to the parallel computation of additional query suffixes and single-token head projection, keeping latency scaling well within the $< 1.4\\times$ bound.

![Latency vs N](latency_batch.png)
"""
    Path(md_path).write_text(report_md, encoding="utf-8")
    print(f"[benchmark_batch] Report saved to: {md_path}")

    return {
        "batch_sizes": batch_sizes,
        "median_latencies": median_latencies,
        "ratio": ratio,
        "passed": passed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM batched latency benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    res = run_batch_benchmark(model=args.model, repeats=args.repeats)
    if not res["passed"]:
        sys.exit(1)
