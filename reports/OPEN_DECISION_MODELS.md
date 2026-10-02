# Open decision models on the Kahn1 held-out set and JevBench

Same items, same options for every system. Kahn1 4B: k = 3, temperature calibration (saved
predictions). JEV 1.13.0: its saved API answers (Choice over the same 8 options, Noul in the
"supported" wording). Laya: its Router, shipped calibration. Clef-flash: its release code,
raw softmax, weights in int8 (bitsandbytes): 9B in bf16 does not fit in 16 GB.

## Held-out set

| Group | Items | Kahn1 4B | JEV 1.13.0 | Laya | Clef-flash |
|---|---:|---:|---:|---:|---:|
| all | 14663 | 72.4 % | 73.2 % | 59.1 % | 74.8 % |
| choice | 6050 | 92.3 % | 94.5 % | 83.2 % | 98.6 % |
| score | 6210 | 51.9 % | 52.0 % | 29.5 % | 53.7 % |
| noul | 2403 | 75.5 % | 74.4 % | 74.8 % | 69.7 % |
| app_reviews_eval | 4000 | 50.2 % | 48.9 % | 23.5 % | 51.2 % |
| banking77 | 3076 | 91.8 % | 94.8 % | 86.1 % | 99.3 % |
| massive | 2974 | 92.9 % | 94.1 % | 80.1 % | 97.8 % |
| rte_eval | 277 | 85.9 % | 89.9 % | 77.3 % | 85.9 % |
| scitail_eval | 2126 | 74.1 % | 72.4 % | 74.5 % | 67.6 % |
| sst5_eval | 2210 | 55.0 % | 57.7 % | 40.4 % | 58.2 % |

ECE (lower is better): Kahn1 4B 0.072, JEV 1.13.0 0.113, laya 0.184, clef-flash 0.101.

Noul as the bare statement (JEV's native form): JEV 1.13.0 48.7 %, laya 69.5 %, clef-flash 53.4 %.

Paired with Kahn1 4B on every item (exact McNemar): JEV 1.13.0: Kahn1 alone right 819, JEV 1.13.0 alone right 930, p = 0.0085; laya: Kahn1 alone right 2969, laya alone right 1010, p = 6.3e-221; clef-flash: Kahn1 alone right 836, clef-flash alone right 1187, p = 6.1e-15.

## JevBench, 231 public items

Kahn1 4B: our run (k = 3, calibrated). Jev: JevBench's own run. JevK5: its authors' run.
Laya and Clef-flash: our run, native JevBench questions and states.

| Tier | Items | Kahn1 4B | Jev 1.13.0 | JevK5 v0.2 | Laya | Clef-flash | Laya, multilingual 8k |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 231 | 87.4 % | 86.6 % | 86.1 % | 57.6 % | 83.5 % | 46.8 % |
| original | 72 | 97.2 % | 98.6 % | 95.8 % | 70.8 % | 97.2 % | 40.3 % |
| easy | 48 | 100.0 % | 100.0 % | 100.0 % | 95.8 % | 100.0 % | 89.6 % |
| hard | 111 | 75.7 % | 73.0 % | 73.9 % | 32.4 % | 67.6 % | 32.4 % |

Paired with Kahn1 4B (exact McNemar): Jev 1.13.0: 13 / 11, p = 0.84; JevK5 v0.2: 13 / 10, p = 0.68; laya: 77 / 8, p = 2.8e-15; clef-flash: 21 / 12, p = 0.16; laya-long: 103 / 9, p = 2.3e-21.

JevBench ECE: laya 0.122, clef-flash 0.070, laya-long 0.289.

## Caveats

- Laya was trained on BANKING77, MASSIVE, SST-5, SciTail, RTE and app_reviews (its model card),
  the six sources of this held-out set: its held-out figures are in-distribution, not zero-shot.
- Laya's English checkpoint reads 512 tokens; longer states are truncated (0 held-out items, 57 of the 231 JevBench items). Its multilingual checkpoint read up to 8,192 tokens (laya-long, nothing
  truncated) scores lower on JevBench, so truncation does not explain Laya's JevBench score.
- Clef-flash was run in int8 on one RTX 5070 Ti, not in bf16 as released. Its training data is not published;
  a train-split probe and a gain-over-base check are in CONTAMINATION_PROBE.md.
- Kahn1 4B is k = 3 (three option orders averaged) with temperature calibration, its published
  setting; Laya and Clef-flash answer in one pass, each in its released setting. ECE is therefore
  each system as shipped, not after a common recalibration.
