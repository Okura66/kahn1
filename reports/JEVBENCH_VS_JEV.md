# Kahn1 vs Jev on JevBench (public items)

- JevBench revision v1.3.0; Jev = Jev 1.13.0 (TypeSafe AI), per-task outcomes published by JevBench.
- Kahn1: v3 checkpoint, k = 3, calibrated, predictions in `reports/jevbench_v3_preds.json`.
- Same 231 public items on both sides. "original" is JevBench's "standard" tier.

| Tier | Items | Kahn1 | Jev 1.13.0 |
|---|---:|---:|---:|
| easy | 48 | **100.0 %** | **100.0 %** |
| original | 72 | **84.7 %** | **98.6 %** |
| hard | 111 | **42.3 %** | **73.0 %** |
| all | 231 | **67.5 %** | **86.6 %** |

Jev on the full tiers (public + held-out, as JevBench reports): easy 100.0 %,
standard 99.0 %, hard 74.1 %. Kahn1 was not run on the held-out items,
which JevBench does not publish.
