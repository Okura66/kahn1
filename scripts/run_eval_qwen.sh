#!/usr/bin/env bash
set -e

# Configuration environnement WSL2 / GPU
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=.

cd /mnt/c/dev/OpenJEV

echo "=== Lancement Benchmark Qwen Merged (checkpoints/qwen_merged) ==="
~/.venvs/sysone/bin/python -m eval.benchmark_qwen --model checkpoints/qwen_merged --out reports/QWEN_BENCHMARK.md "$@"
echo "=== Benchmark terminé avec succès ! ==="
