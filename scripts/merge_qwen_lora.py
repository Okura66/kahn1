"""Merge Qwen2.5-3B LoRA adapter weights into a standalone checkpoint.

Produces a standard standalone Hugging Face checkpoint in checkpoints/qwen_merged,
directly consumable by vLLM with prefix caching and zero LoRA runtime overhead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_qwen(
    base_model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    adapter_path: str = "checkpoints/qwen_lora/best",
    output_dir: str = "checkpoints/qwen_merged",
) -> None:
    """Load base model and LoRA adapter, merge weights, and save standalone checkpoint."""
    print(f"[merge_qwen] Loading base model {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    print(f"[merge_qwen] Loading LoRA adapter weights from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    print("[merge_qwen] Merging LoRA weights into base parameters (merge_and_unload)...")
    merged_model = model.merge_and_unload()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    print(f"[merge_qwen] Saving standalone merged checkpoint to {output_dir}...")
    merged_model.save_pretrained(str(out_p))

    print(f"[merge_qwen] Saving tokenizer to {output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(str(out_p))

    print(f"[merge_qwen] Standalone merged checkpoint saved to {output_dir}.")


def main():
    ap = argparse.ArgumentParser(description="Merge Qwen2.5-3B LoRA adapter weights")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/qwen_lora/best")
    ap.add_argument("--output", default="checkpoints/qwen_merged")
    args = ap.parse_args()

    merge_qwen(base_model_name=args.base, adapter_path=args.adapter, output_dir=args.output)


if __name__ == "__main__":
    main()
