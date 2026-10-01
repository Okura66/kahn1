#!/bin/bash
cd /mnt/c/dev/OpenJEV
export PYTHONPATH=.:src VLLM_WSL2_ENABLE_PIN_MEMORY=1 PYTHONUNBUFFERED=1
TR=~/.venvs/k1train/bin/python; EV=~/.venvs/sysone/bin/python; W=/home/amontzamir/k1merged/v7
L=/mnt/c/dev/OpenJEV/.v7_final.log
step() { echo "=== $1 $(date +%H:%M)" >> $L; shift; "$@" > /tmp/v7_step.log 2>&1 && { grep -aE 'Global Accuracy|items/s|Total Instances|sample balanced|calibrate' /tmp/v7_step.log | tail -3 >> $L; echo "OK" >> $L; } || { echo "FAILED" >> $L; tail -10 /tmp/v7_step.log >> $L; }; }
: > $L
step "merge final" $TR scripts/merge_qwen_lora.py --base Qwen/Qwen3.5-4B --adapter checkpoints/qwen35_lora_v7/final --output $W --device cuda
step "JevBench k=1 + dev_teacher" $EV scripts/eval_base.py --model $W --format qwen3 --max-model-len 8192 --sample-file data/dev_teacher.jsonl --out reports/v7/k1_final.json
step "calibration (val.jsonl)" $EV scripts/fit_calibration.py --model $W --format qwen3 --out calibration_v7.json
step "JevBench 231, k=3, calibrated" $EV -m eval.eval_full --model $W --prompt-format qwen3 --eval data/jevbench_eval.jsonl --calibration calibration_v7.json --n-permutations 3 --max-model-len 8192 --out reports/JEVBENCH_v7.md --preds-out reports/jevbench_v7_preds.json
step "held-out 14,663, k=3, calibrated" $EV -m eval.eval_full --model $W --prompt-format qwen3 --eval data/eval.jsonl --calibration calibration_v7.json --n-permutations 3 --max-model-len 4096 --out reports/QWEN35_V7_FULL_EVAL.md --preds-out reports/eval_v7_preds.json
step "Choice, every intent, same 1,184 items" $EV scripts/choice_fairness.py kahn1-full --model $W --prompt-format qwen3 --out data/kahn1_v7_choice_full.jsonl --same-as data/kahn1_v3_choice_full.jsonl
echo "FINAL DONE $(date +%H:%M)" >> $L
