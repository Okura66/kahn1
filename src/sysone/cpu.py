"""CPU inference backend (transformers instead of vLLM).

vLLM is GPU-only, which makes the engine unusable for local development, CI and
laptop debugging. This module swaps only the backend: prompt construction,
single-token option resolution, permutation debiasing, calibration and answer
composition all come from the regular `Engine`.

The seam is narrow on purpose. `Engine` reaches its backend exclusively through
`self._llm.generate(prompts, params_list)` and reads vLLM-shaped outputs, so a
small adapter exposing that surface is enough for every code path — including
`CalibratedEngine` and `evaluate_two_stage`.

    from sysone.cpu import CPUEngine
    from sysone.calibrate import CalibratedEngine, TemperatureConfig

    engine = CPUEngine("Okura66/Kahn1-Qwen2.5-3B", dtype="float32")
    engine = CalibratedEngine(engine, TemperatureConfig.load("calibration.json"))
    response = engine.evaluate(query, n_permutations=3)

Expect seconds per prompt rather than the tens of milliseconds vLLM reaches on a
GPU: there is no paged KV cache and no prefix caching here. Prefer `float32`.
`bfloat16` halves memory but runs ~7x slower on CPUs without AMX, because the
matmuls fall back to emulation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .engine import Engine, EngineConfig


# ---------------------------------------------------------------------------
# vLLM-shaped stubs
# ---------------------------------------------------------------------------

@dataclass
class CPUSamplingParams:
    """The subset of vllm.SamplingParams that Engine._make_params constructs.

    `allowed_token_ids` is what Engine._detect_restrict_param looks for; keeping
    the name identical means the detection logic needs no special case.
    """

    max_tokens: int = 1
    temperature: float = 0.0
    logprobs: int = 1
    allowed_token_ids: list[int] = field(default_factory=list)


@dataclass
class _Logprob:
    logprob: float


@dataclass
class _Completion:
    logprobs: list[dict[int, _Logprob]]


@dataclass
class _RequestOutput:
    outputs: list[_Completion]
    prompt_token_ids: list[int]
    num_cached_tokens: int = 0


# ---------------------------------------------------------------------------
# Torch backend
# ---------------------------------------------------------------------------

class TorchCPUBackend:
    """Scores one continuation token per prompt with a single batched forward pass."""

    def __init__(self, model, tokenizer, batch_size: int = 8):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def generate(self, prompts: list[str], params_list: list[CPUSamplingParams]):
        import torch

        outputs: list[_RequestOutput] = []
        with torch.inference_mode():
            for start in range(0, len(prompts), self.batch_size):
                chunk = prompts[start:start + self.batch_size]
                chunk_params = params_list[start:start + self.batch_size]
                enc = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                mask = enc["attention_mask"]
                # A raw forward() does not derive position_ids from the mask (only
                # generate() does). Without this, left padding shifts RoPE positions
                # and silently corrupts every prompt shorter than the batch maximum.
                position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
                logits = self.model(
                    input_ids=enc["input_ids"],
                    attention_mask=mask,
                    position_ids=position_ids,
                ).logits[:, -1, :].float()

                for row, params in enumerate(chunk_params):
                    ids = list(params.allowed_token_ids)
                    # Restrict then renormalize: this reproduces vLLM's
                    # allowed_token_ids masking followed by log_softmax, so the
                    # values Engine._extract reads carry the same meaning.
                    logprobs = torch.log_softmax(logits[row, ids], dim=-1)
                    lp = {tid: _Logprob(float(lg)) for tid, lg in zip(ids, logprobs)}
                    n_real = int(mask[row].sum())
                    outputs.append(_RequestOutput(
                        outputs=[_Completion(logprobs=[lp])],
                        prompt_token_ids=list(range(n_real)),
                    ))
        return outputs


class CPUEngine(Engine):
    """Engine backed by transformers on CPU rather than vLLM on GPU."""

    def __init__(
        self,
        model: str | None = None,
        dtype: str = "float32",
        batch_size: int = 8,
        num_threads: int | None = None,
        config: EngineConfig | None = None,
        tokenizer=None,
    ):
        cfg = config or EngineConfig()
        if model is not None:
            cfg.model = model
        cfg.dtype = dtype
        super().__init__(cfg, tokenizer=tokenizer)
        self.batch_size = batch_size
        self.num_threads = num_threads

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.num_threads:
            torch.set_num_threads(self.num_threads)

        self._SamplingParams = CPUSamplingParams
        self._sampling_fields = {
            "max_tokens", "temperature", "logprobs", "allowed_token_ids",
        }
        self._restrict_param = "allowed_token_ids"

        t0 = time.perf_counter()
        tok = self._tokenizer or AutoTokenizer.from_pretrained(self.config.model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self._tokenizer = tok

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            dtype=getattr(torch, self.config.dtype),
            low_cpu_mem_usage=True,
        )
        model.eval()
        self._llm = TorchCPUBackend(model, tok, batch_size=self.batch_size)
        print(
            f"[cpu] loaded {self.config.model} dtype={self.config.dtype} "
            f"threads={torch.get_num_threads()} in {time.perf_counter() - t0:.1f}s"
        )

    def close(self) -> None:
        """Drop the model and reclaim host memory."""
        if self._llm is not None:
            self._llm = None
            import gc

            gc.collect()
