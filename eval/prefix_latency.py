"""Prefix-caching latency: N questions (1..10) about one shared state, short and long.

The short state is eval/benchmark_batch.py's sample state, the long one the same text 5 times.
Qwen3.5 is hybrid (Gated DeltaNet + attention), and vLLM caches its prefix in blocks of 528 tokens,
so a state shorter than a block is never reused. Engine defaults (bf16, prefix caching, eager mode),
k = 1, median of 10 requests per point after 3 warm-up requests. Writes nothing: prints the table.

    python -m eval.prefix_latency Okura66/Kahn1-Qwen3.5-4B      # reports/LATENCY_PREFIX.md
"""
import os
import sys

import numpy as np

os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
sys.path[:0] = [".", "src"]
from eval.benchmark_batch import SAMPLE_STATE, TEN_QUESTIONS  # noqa: E402
from sysone.engine import Engine, EngineConfig  # noqa: E402
from sysone.types import Query  # noqa: E402


def main():
    model = sys.argv[1]
    engine = Engine(EngineConfig(model=model, dtype="bfloat16", gpu_memory_utilization=0.90, max_model_len=4096,
                                 enable_prefix_caching=True))
    engine._ensure_loaded()
    tok = engine._tokenizer if hasattr(engine, "_tokenizer") else None
    for label, state in (("short", SAMPLE_STATE), ("long", "\n\n".join([SAMPLE_STATE] * 5))):
        n_tok = len(tok.encode(state)) if tok is not None else -1
        for _ in range(3):
            engine.evaluate(Query(state=state, questions=TEN_QUESTIONS), n_permutations=1)
        print(f"== {label} state, {n_tok} tokens")
        for n in (1, 2, 4, 6, 8, 10):
            lat, hit = [], []
            for _ in range(10):
                r = engine.evaluate(Query(state=state, questions=TEN_QUESTIONS[:n]), n_permutations=1)
                lat.append(r.latency_ms)
                hit.append(r.cache_hit_rate)
            print(f"N={n:2d}  median {np.median(lat):7.1f} ms  hit_rate {np.mean(hit):.2f}")


if __name__ == "__main__":
    main()
