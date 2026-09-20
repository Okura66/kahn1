#!/usr/bin/env bash
set -e

export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=.

cd /mnt/c/dev/OpenJEV

echo "=== Lancement Évaluation Exhaustive 8 260 exemples pour Qwen Merged ==="
~/.venvs/sysone/bin/python -m eval.eval_full \
  --model checkpoints/qwen_merged \
  --eval data/eval.jsonl \
  --calibration calibration_qwen.json \
  --n-permutations 3 \
  --out reports/QWEN_FULL_EVAL_8260.md \
  --preds-out reports/eval_qwen_8260_preds.json

echo "=== Évaluation exhaustive terminée avec succès ! ==="
