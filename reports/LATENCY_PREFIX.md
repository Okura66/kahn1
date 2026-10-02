# Kahn1 4B: latency and prefix caching

Median latency of one request carrying N questions about the same state, for a short and a long
state. Qwen3.5 is a hybrid model (Gated DeltaNet and attention layers), and vLLM caches its prefix in
blocks of 528 tokens ("Mamba cache mode align"): a state shorter than one block is never reused.

| Questions per request | 212-token state | 1,064-token state |
|---:|---:|---:|
| 1 | 45.5 ms | 49.3 ms |
| 2 | 83.8 ms | 92.8 ms |
| 4 | 136.4 ms | 108.7 ms |
| 6 | 189.4 ms | 104.1 ms |
| 8 | 249.1 ms | 121.7 ms |
| 10 | 318.8 ms | 142.7 ms |
| Prompt tokens served from the cache | 0 % | 93 % |

- On the long state, 10 questions take about 3 times one question, about 10 ms per extra question.
- On the short state no block fills, every question recomputes the state, and latency grows almost
  linearly: 10 questions take 7 times one.

## How it was measured

- Model: the published Kahn1 4B (`Okura66/Kahn1-Qwen3.5-4B`; the LoRA adapter merged into
  Qwen3.5-4B), bf16.
- Engine: `sysone.engine.Engine` with its defaults (vLLM 0.29.0, prefix caching on, eager mode, which
  is also what `sysone serve` runs), `max_model_len` 4096, one RTX 5070 Ti 16 GB under WSL2.
- Requests: k = 1 (`n_permutations=1`), the first N of the ten sample questions of
  `eval/benchmark_batch.py`. The short state is its sample state (212 tokens), the long one the same
  text five times (1,064 tokens).
- Each point is the median of 10 requests, after 3 warm-up requests of 10 questions on the same state.
  The cache share is the engine's `cache_hit_rate`, averaged over the 10 requests.
- Script: `python -m eval.prefix_latency Okura66/Kahn1-Qwen3.5-4B` (`eval/prefix_latency.py`).

These figures are separate from the k = 3 latency of the held-out run (p50 88.7 ms, p95 279.3 ms,
[KAHN1_4B_HELDOUT.md](KAHN1_4B_HELDOUT.md)), which mixes states of every length.
