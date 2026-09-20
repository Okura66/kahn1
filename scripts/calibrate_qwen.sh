#!/usr/bin/env bash
set -e

export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=.

cd /mnt/c/dev/OpenJEV

echo "=== Calibration post-hoc de température pour Qwen Merged ==="
~/.venvs/sysone/bin/python -c "
import sys
from sysone.cli import main
sys.argv = ['sysone', 'calibrate', '--data', 'data/val.jsonl', '--model', 'checkpoints/qwen_merged', '--out', 'calibration_qwen.json']
main()
"
echo "=== Calibration terminée ! ==="
