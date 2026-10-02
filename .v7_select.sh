#!/bin/bash
# Merge each v7 checkpoint on the GPU, score it on the two dev splits (never on the benchmark
# or JevBench), then delete the merged copy to keep the WSL disk small.
cd /mnt/c/dev/OpenJEV
export PYTHONPATH=.:src VLLM_WSL2_ENABLE_PIN_MEMORY=1 PYTHONUNBUFFERED=1
TR=~/.venvs/k1train/bin/python; EV=~/.venvs/sysone/bin/python; M=~/k1merged
L=/mnt/c/dev/OpenJEV/.v7_select.log
mkdir -p $M reports/v7
for c in "$@"; do
  test -d checkpoints/qwen35_lora_v7/$c || { echo "skip $c" >> $L; continue; }
  $TR scripts/merge_qwen_lora.py --base Qwen/Qwen3.5-4B --adapter checkpoints/qwen35_lora_v7/$c --output $M/v7_$c --device cuda > /tmp/merge_$c.log 2>&1 || { echo "MERGE FAILED $c" >> $L; tail -5 /tmp/merge_$c.log >> $L; continue; }
  for d in dev_teacher dev_teacher_t3 dev_v4; do
    $EV scripts/eval_base.py --model $M/v7_$c --format qwen3 --max-model-len 8192 --sample-file data/$d.jsonl --no-jevbench --out reports/v7/${d}_$c.json > /tmp/${d}_$c.log 2>&1 && echo "$d $c $(grep -a 'sample balanced' /tmp/${d}_$c.log | cut -c1-200)" >> $L || { echo "EVAL FAILED $d $c" >> $L; tail -8 /tmp/${d}_$c.log >> $L; }
  done
  rm -rf $M/v7_$c
done
echo "SELECT DONE" >> $L
