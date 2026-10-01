#!/bin/bash
# Kahn1 4B v2 (internal v6): round-2 teacher mix + replay, 2 epochs, candidate-restricted CE
# + ordinal EMD on Score. Each change can be turned off: --loss vocab, --ordinal-weight 0,
# and the mix is rebuilt with --replay none / without --round2.
cd /mnt/c/dev/OpenJEV
export PYTHONUNBUFFERED=1 PYTHONPATH=.:src
LOSS=${LOSS:-options}; ORD=${ORD:-1.0}; EPOCHS=${EPOCHS:-2}; TRAIN=${TRAIN:-data/train_4b_v2.jsonl}
~/.venvs/k1train/bin/python training/train_lora.py --model Qwen/Qwen3.5-4B \
  --train $TRAIN --eval data/dev_teacher.jsonl --output-dir checkpoints/qwen35_lora_v6 \
  --epochs $EPOCHS --lr 3e-5 --micro-batch 1 --grad-accum 32 --val-every 50 --save-every 200 --eval-samples 317 \
  --rank 16 --alpha 32 --targets qwen35-attention --max-len 6144 --prompt-format qwen3 \
  --loss $LOSS --ordinal-weight $ORD > /mnt/c/dev/OpenJEV/.v6_train.log 2>&1
echo "TRAIN EXIT $?" >> /mnt/c/dev/OpenJEV/.v6_train.log
