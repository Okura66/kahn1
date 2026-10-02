# Kahn1 4B (Qwen3.5-4B + LoRA): evaluation

Kahn1 4B: Qwen3.5-4B, LoRA r = 16 on the attention and linear-attention
projections, native chat template (thinking off), trained for two epochs (2,010 steps) on
a 32,170-row mix: public NLI, topic, intent and sentiment sources (ShARC, WANLI
and DocNLI whole), QuALITY, BoolQ, CLINC150, a replay of ARC, CommonsenseQA and the MMLU-Pro
validation questions (never MMLU-Pro test), and 2,836 distinct hard decision questions written by
Claude Opus, English and French (402 x 3, 600 x 3, 1,834 x 2; 20.7 % of the rows). Loss:
cross-entropy restricted to the candidate tokens + squared EMD on Score questions. Checkpoint chosen
on dev accuracy on the dev splits, never on a benchmark. Kahn1 3B: Qwen2.5-3B-Instruct + LoRA.

## Held-out 14,663 items, like for like

Choice over the same 8 options for every system (the gold one and 7 seeded distractors); JEV's
Noul asked whether the text supports the statement. k = 3 and temperature calibration for Kahn1.

| Dataset | Items | Kahn1 4B | Kahn1 3B | JEV 1.13.0 | 4B ECE | 3B ECE | JEV ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 3,076 | 91.8 % | 91.1 % | **94.8 %** | 0.018 | 0.026 | 0.024 |
| massive | 2,974 | 92.9 % | 90.8 % | **94.1 %** | 0.013 | 0.011 | 0.017 |
| rte | 277 | 85.9 % | 82.7 % | **89.9 %** | 0.048 | 0.105 | 0.034 |
| scitail | 2,126 | **74.1 %** | 65.0 % | 72.4 % | 0.141 | 0.248 | 0.077 |
| sst5 | 2,210 | 55.0 % | 52.4 % | **57.7 %** | 0.052 | 0.098 | 0.178 |
| app_reviews | 4,000 | 50.2 % | **50.9 %** | 48.9 % | 0.146 | 0.096 | 0.252 |
| all choice | 6,050 | 92.3 % | 90.9 % | **94.5 %** | 0.015 | 0.017 | 0.019 |
| all score | 6,210 | 51.9 % | 51.4 % | **52.0 %** | 0.112 | 0.097 | 0.226 |
| all noul | 2,403 | **75.5 %** | 67.0 % | 74.4 % | 0.127 | 0.232 | 0.071 |
| all | 14,663 | 72.4 % | 70.3 % | **73.2 %** | 0.072 | 0.082 | 0.113 |

## Choice over every intent

| Condition | Kahn1 4B | Kahn1 3B | JEV 1.13.0 |
|---|---:|---:|---:|
| every intent (77 / 60), 1,184 items | 71.6 % | 68.4 % | **79.1 %** |

## JevBench, 231 public items

Kahn1: k = 3, calibrated (our run). Jev: JevBench's own per-task outcomes. JevK5: its authors' own
public v0.2 run; JevBench's own run of JevK5 v0.2 scores 85.3 % on the same 231 items
(results/v1.4/measurement-aggregates.json). Three different runners on the same items.

| Tier | Items | Kahn1 4B | Kahn1 3B | JevK5 v0.2 | Jev 1.13.0 |
|---|---:|---:|---:|---:|---:|
| easy | 48 | **100.0 %** | **100.0 %** | **100.0 %** | **100.0 %** |
| standard | 72 | 97.2 % | 84.7 % | 95.8 % | **98.6 %** |
| hard | 111 | **75.7 %** | 42.3 % | 73.9 % | 73.0 % |
| all | 231 | **87.4 %** | 67.5 % | 86.1 % | 86.6 % |

Kahn1 4B vs JevK5's own run, paired: 13 items only Kahn1 4B gets right, 10 only JevK5; exact McNemar p = 0.68.
Kahn1 4B vs Jev's published per-task outcomes, paired: 13 items only Kahn1 4B gets right, 11 only Jev;
exact McNemar p = 0.84.

Hard tier by family (the family is the third field of the item id, hard-<author>-<family>-<nn>;
Kahn1 4B: k = 3, calibrated). Items right:

| Family | Items | Kahn1 4B | JevK5 v0.2 | Jev 1.13.0 |
|---|---:|---:|---:|---:|
| adversarial | 6 | 6 | 5 | 6 |
| ambiguous | 7 | 5 | 6 | 6 |
| judge_hard | 17 | 12 | 13 | 13 |
| long_policy | 19 | 13 | 11 | 12 |
| multi_hop | 18 | 14 | 14 | 15 |
| probability | 10 | 10 | 7 | 7 |
| routing_hard | 5 | 5 | 5 | 5 |
| temporal_numeric | 15 | 5 | 7 | 4 |
| tradeoff | 6 | 6 | 6 | 5 |
| trap | 8 | 8 | 8 | 8 |

## Hard decision dev split (317 items, k = 1)

Questions written by Claude Opus and checked by two blind solvers; never trained on, but used to
choose the checkpoint: a dev score, not a benchmark.

| Group | Qwen3.5-4B base | Kahn1 4B |
|---|---:|---:|
| choice | 46.6 % | 67.1 % |
| score | 42.7 % | 57.3 % |
| noul | 57.3 % | 76.0 % |
| lang en | 48.9 % | 67.0 % |
| lang fr | 48.8 % | 69.0 % |
| all | 48.9 % | 67.5 % |
| balanced | 48.8 % | 66.8 % |
