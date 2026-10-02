# Train-split probe

Accuracy and mean log-probability of the gold answer on test items (the held-out set) and on
up to 2000 train items per source, same prompts, same 8 options. A gap well above Kahn1's (which
never saw these sources) means the system saw the train split.

| Source | Kahn1 4B test | Kahn1 4B train | gap | laya test | laya train | gap | clef-flash test | clef-flash train | gap | tev1 test | tev1 train | gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **acc** | | | | | | | | | | | | |
| banking77 | 91.8 % | 91.5 % | -0.3 | 86.1 % | 84.7 % | -1.4 | 99.3 % | 99.2 % | -0.1 | 94.3 % | 94.8 % | +0.5 |
| massive | 92.9 % | 93.4 % | +0.5 | 80.1 % | 81.3 % | +1.2 | 97.8 % | 98.7 % | +0.9 | 91.6 % | 92.3 % | +0.7 |
| sst5_eval | 55.0 % | 55.4 % | +0.4 | 40.4 % | 36.6 % | -3.8 | 58.2 % | 60.5 % | +2.3 | 57.0 % | 57.9 % | +0.9 |
| scitail_eval | 74.1 % | 72.0 % | -2.2 | 68.3 % | 66.8 % | -1.6 | 50.3 % | 48.2 % | -2.1 | 60.3 % | 58.1 % | -2.3 |
| rte_eval | 85.9 % | 87.3 % | +1.4 | 78.0 % | 73.7 % | -4.3 | 77.3 % | 77.1 % | -0.2 | 81.6 % | 81.9 % | +0.3 |
| **mean_logp_gold** | | | | | | | | | | | | |
| banking77 | -0.254 | -0.258 | -0.004 | -0.595 | -0.666 | -0.071 | -0.036 | -0.037 | -0.001 | -0.183 | -0.163 | +0.020 |
| massive | -0.217 | -0.202 | +0.016 | -0.689 | -0.678 | +0.011 | -0.082 | -0.052 | +0.030 | -0.243 | -0.237 | +0.005 |
| sst5_eval | -1.097 | -1.134 | -0.037 | -1.384 | -1.459 | -0.075 | -1.029 | -1.000 | +0.029 | -0.979 | -0.975 | +0.005 |
| scitail_eval | -0.553 | -0.610 | -0.057 | -0.602 | -0.597 | +0.005 | -1.263 | -1.316 | -0.053 | -0.760 | -0.802 | -0.042 |
| rte_eval | -0.307 | -0.289 | +0.018 | -0.455 | -0.561 | -0.106 | -0.563 | -0.577 | -0.014 | -0.395 | -0.391 | +0.003 |

Item counts (test / train): banking77 3076 / 2000, massive 2974 / 2000, sst5_eval 2210 / 2000, scitail_eval 2126 / 2000, rte_eval 277 / 2000.

## Gain over the base model

Full held-out set, k = 1, no calibration. Base models are read with Kahn1's prompt format (qwen3 template),
the 9B in fp8 (vLLM); Clef-flash is its held-out run above (int8, its own head, Noul in the supported wording).

| Group | Qwen3.5-4B (base) | Kahn1 4B, k = 1 | Qwen3.5-9B (base, fp8) | Clef-flash |
|---|---:|---:|---:|---:|
| all | 62.2 % | 72.0 % | 66.3 % | 74.8 % |
| kind:choice | 88.3 % | 92.1 % | 92.0 % | 98.6 % |
| kind:score | 39.3 % | 51.2 % | 45.2 % | 53.7 % |
| kind:noul | 55.7 % | 75.2 % | 55.9 % | 69.7 % |
| banking77 | 87.9 % | 91.2 % | 91.8 % | 99.3 % |
| massive | 88.7 % | 93.1 % | 92.3 % | 97.8 % |

banking77: Clef-flash removes 92 % of Qwen3.5-9B's errors; Kahn1 4B removes 27 % of Qwen3.5-4B's (Kahn1 never saw banking77).

massive: Clef-flash removes 71 % of Qwen3.5-9B's errors; Kahn1 4B removes 39 % of Qwen3.5-4B's (Kahn1 never saw massive).

A gain that large on intents is what training on intent data gives, these datasets or close ones; the probe
and this check cannot tell which.
