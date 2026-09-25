# Choice: the same option set on every side

Kahn1's held-out evaluation (eval/baselines.py:_prepare_choice_options) answers each Choice
question over 8 options: the gold one and 7 distractors seeded from the state. The first JEV
comparison gave JEV every intent. Both conditions, paired:

| Source | Options | Items | Kahn1 v3 | JEV 1.13.0 | Kahn1 ECE | JEV ECE |
|---|---|---:|---:|---:|---:|---:|
| banking77 | 8 options (gold + 7 seeded distractors) | 3076 | 91.09 % | 94.77 % | 0.026 | 0.024 |
| banking77 | every intent | 604 | 67.38 % | 79.80 % | 0.162 | 0.084 |
| massive | 8 options (gold + 7 seeded distractors) | 2974 | 90.79 % | 94.15 % | 0.011 | 0.017 |
| massive | every intent | 580 | 69.48 % | 78.28 % | 0.128 | 0.091 |
| all | 8 options (gold + 7 seeded distractors) | 6050 | 90.94 % | 94.46 % | 0.017 | 0.019 |
| all | every intent | 1184 | 68.41 % | 79.05 % | 0.141 | 0.086 |

- 8 options: Kahn1 = the held-out run (k = 3, temperature calibration); JEV = the same 8 options.
- Every intent: Kahn1 = evaluate_two_stage (one Noul per option, top 10, then a Choice; k = 1,
  uncalibrated, which does not change its picks); JEV = its run with every intent.
