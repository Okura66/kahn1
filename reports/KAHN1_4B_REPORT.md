# Kahn1 4B (Qwen3.5-4B + LoRA): evaluation

Kahn1 4B (version 2, internal v6): Qwen3.5-4B, LoRA r = 16 on the attention and linear-attention
projections, native chat template (thinking off), trained for two epochs on data/train_4b_v2.jsonl
(22,406 rows): the 3B mixture's NLI, topic, intent and sentiment sources (sentiment scales whole),
DocNLI, ShARC and WANLI (halved), QuALITY, BoolQ, more CLINC150, a replay of ARC, CommonsenseQA and
the MMLU-Pro validation questions (never MMLU-Pro test), and hard decision questions written by
Claude Opus, English and French (402 round-1 questions x 5, 600 round-2 questions x 4; 19.7 % of the
rows). Loss: cross-entropy restricted to the candidate tokens + squared EMD on Score questions.
Checkpoint chosen on dev accuracy on two dev splits, never on a benchmark. Kahn1 4B v1 (internal
v5): the first 4B, trained on data/train_v5.jsonl. Kahn1 3B: Qwen2.5-3B (v3).

## Held-out 14,663 items, like for like

Choice over the same 8 options for every system (the gold one and 7 seeded distractors); JEV's
Noul asked whether the text supports the statement. k = 3 and temperature calibration for Kahn1.

| Dataset | Items | Kahn1 4B | Kahn1 4B v1 | Kahn1 3B | JEV 1.13.0 | 4B ECE | 3B ECE | JEV ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| banking77 | 3,076 | 91.5 % | 91.5 % | 91.1 % | **94.8 %** | 0.022 | 0.026 | 0.024 |
| massive | 2,974 | 93.1 % | 92.4 % | 90.8 % | **94.1 %** | 0.013 | 0.011 | 0.017 |
| rte | 277 | 84.1 % | 86.6 % | 82.7 % | **89.9 %** | 0.055 | 0.105 | 0.034 |
| scitail | 2,126 | 68.2 % | 70.4 % | 65.0 % | **72.4 %** | 0.166 | 0.248 | 0.077 |
| sst5 | 2,210 | 50.2 % | 49.2 % | 52.4 % | **57.7 %** | 0.141 | 0.098 | 0.178 |
| app_reviews | 4,000 | 50.9 % | 48.2 % | **50.9 %** | 48.9 % | 0.106 | 0.096 | 0.252 |
| all choice | 6,050 | 92.3 % | 91.9 % | 90.9 % | **94.5 %** | 0.010 | 0.017 | 0.019 |
| all score | 6,210 | 50.6 % | 48.6 % | 51.4 % | **52.0 %** | 0.118 | 0.097 | 0.226 |
| all noul | 2,403 | 70.0 % | 72.2 % | 67.0 % | **74.4 %** | 0.152 | 0.232 | 0.071 |
| all | 14,663 | 71.0 % | 70.3 % | 70.3 % | **73.2 %** | 0.077 | 0.082 | 0.113 |

## Choice over every intent

| Condition | Kahn1 4B | Kahn1 3B | JEV 1.13.0 |
|---|---:|---:|---:|
| every intent (77 / 60), 1,184 items | 70.4 % | 68.4 % | **79.1 %** |

## JevBench, 231 public items

Kahn1: k = 3, calibrated. Jev: JevBench's own per-task outcomes. JevK5: its public v0.2 run.

| Tier | Items | Kahn1 4B | Kahn1 4B v1 | Kahn1 3B | JevK5 v0.2 | Jev 1.13.0 |
|---|---:|---:|---:|---:|---:|---:|
| easy | 48 | **100.0 %** | **100.0 %** | **100.0 %** | **100.0 %** | **100.0 %** |
| original | 72 | **98.6 %** | 91.7 % | 84.7 % | 95.8 % | **98.6 %** |
| hard | 111 | 72.1 % | 70.3 % | 42.3 % | **73.9 %** | 73.0 % |
| all | 231 | 86.1 % | 83.1 % | 67.5 % | 86.1 % | **86.6 %** |

Kahn1 4B vs JevK5, paired: 12 items only Kahn1 4B gets right, 12 only JevK5; exact McNemar p = 1.00.

## Hard decision dev split (317 items, k = 1)

Questions written by Claude Opus and checked by two blind solvers; never trained on, but used to
choose the checkpoint: a dev score, not a benchmark.

| Group | Qwen3.5-4B base | Kahn1 4B v1 | Kahn1 4B |
|---|---:|---:|---:|
| choice | 46.6 % | 59.6 % | 69.2 % |
| score | 42.7 % | 57.3 % | 56.0 % |
| noul | 57.3 % | 69.8 % | 69.8 % |
| lang en | 48.9 % | 60.5 % | 66.1 % |
| lang fr | 48.8 % | 66.7 % | 66.7 % |
| all | 48.9 % | 62.1 % | 66.2 % |
| balanced | 48.8 % | 62.2 % | 65.0 % |
