#!/bin/bash
# Kahn1 4B, next run (internal v7). Same toggles as v6 (LOSS, ORD, EPOCHS, TRAIN), plus:
# - 'best' picked on dev accuracy (SELECT_ON=acc): in v6 the val NLL bottomed at step 250 of
#   1400 while dev accuracy kept rising to the last step; SELECT_ON=nll is the old behaviour;
# - an optional early stop on that score (PATIENCE validations without progress, 0 = off);
# - the mix checked row by row first, so a bad row fails here and not at step 55.
cd /mnt/c/dev/OpenJEV
export PYTHONUNBUFFERED=1 PYTHONPATH=.:src
LOSS=${LOSS:-options}; ORD=${ORD:-1.0}; EPOCHS=${EPOCHS:-2}; TRAIN=${TRAIN:-data/train_4b_v2.jsonl}
SELECT_ON=${SELECT_ON:-acc}; PATIENCE=${PATIENCE:-0}; OUT=${OUT:-checkpoints/qwen35_lora_v7}
LOG=/mnt/c/dev/OpenJEV/.v7_train.log
~/.venvs/k1train/bin/python scripts/check_mix.py $TRAIN > $LOG 2>&1 || { echo "TRAIN EXIT mix check failed" >> $LOG; exit 1; }
~/.venvs/k1train/bin/python training/train_lora.py --model Qwen/Qwen3.5-4B \
  --train $TRAIN --eval data/dev_teacher.jsonl --output-dir $OUT \
  --epochs $EPOCHS --lr 3e-5 --micro-batch 1 --grad-accum 32 --val-every 50 --save-every 100 --eval-samples 317 \
  --rank 16 --alpha 32 --targets qwen35-attention --max-len 6144 --prompt-format qwen3 \
  --loss $LOSS --ordinal-weight $ORD --select-on $SELECT_ON --early-stop-patience $PATIENCE >> $LOG 2>&1
echo "TRAIN EXIT $?" >> $LOG
