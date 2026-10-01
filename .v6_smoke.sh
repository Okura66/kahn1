#!/bin/bash
# A few steps of the v6 recipe on real data, to check the loss before a 10-hour run.
cd /mnt/c/dev/OpenJEV
export PYTHONUNBUFFERED=1 PYTHONPATH=.:src
~/.venvs/k1train/bin/python training/train_lora.py --model Qwen/Qwen3.5-4B \
  --train data/train_4b_v2.jsonl --eval data/dev_teacher.jsonl --output-dir /tmp/v6_smoke \
  --epochs 1 --lr 3e-5 --micro-batch 1 --grad-accum 4 --max-steps 3 --val-every 3 --save-every 1000 --eval-samples 24 \
  --rank 16 --alpha 32 --targets qwen35-attention --max-len 6144 --prompt-format qwen3 \
  --loss options --ordinal-weight 1.0 > /mnt/c/dev/OpenJEV/.v6_smoke.log 2>&1
echo "SMOKE EXIT $?" >> /mnt/c/dev/OpenJEV/.v6_smoke.log
