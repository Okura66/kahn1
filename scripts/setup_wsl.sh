#!/usr/bin/env bash
# Configure Linux virtual memory settings for large model weights loading in WSL2
sudo sysctl -w vm.overcommit_memory=1 >/dev/null 2>&1 || sysctl -w vm.overcommit_memory=1 >/dev/null 2>&1 || true
sudo swapon /swapfile >/dev/null 2>&1 || swapon /swapfile >/dev/null 2>&1 || true
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
export VLLM_USE_FLASHINFER_SAMPLER=0
