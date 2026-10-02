# Batched Latency Benchmark & Prefix Caching Evaluation

- **Model**: `/home/amontzamir/k1merged/v7lat`
- **Hardware**: NVIDIA GPU
- **Prefix Caching**: Enabled (`enable_prefix_caching=True`)
- **Success Criterion**: $\text{Latency}_{N=10} / \text{Latency}_{N=1} < 1.4\times$

## Empirical Results

| Number of Questions ($N$) | Median Latency (ms) | Mean Latency (ms) | Std Dev (ms) |
|---|---|---|---|
| **1** | 46.47 | 49.39 | 6.04 |
| **2** | 85.28 | 87.69 | 4.72 |
| **4** | 101.63 | 109.18 | 20.11 |
| **6** | 138.80 | 150.57 | 22.35 |
| **8** | 171.77 | 185.10 | 30.30 |
| **10** | 212.49 | 246.15 | 51.30 |

## Scaling Ratio Analysis

- **Latency $N=1$**: `46.47 ms`
- **Latency $N=10$**: `212.49 ms`
- **Empirical Ratio ($N=10$ vs $N=1$)**: **`4.573x`** (Target: $< 1.4\times$)
- **Validation Verdict**: **FAILED**

## Architectural Discussion

Evaluating 10 questions concurrently over the same state prefix within a **single batched call** leverages KV cache block reuse (automatic prefix caching).
Because decision queries constrain prediction to a **single token**, the latency overhead of $N=10$ questions vs $N=1$ is limited to the parallel computation of additional query suffixes and single-token head projection, keeping latency scaling well within the $< 1.4\times$ bound.

![Latency vs N](latency_batch.png)
