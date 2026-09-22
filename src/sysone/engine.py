"""Inference engine: vLLM wrapper, logit extraction, and unified batching.

Architecture Principles:
  - Unified batch execution: all questions × permutations are executed in a
    single llm.generate() invocation, avoiding sequential per-question calls.
  - allowed_token_ids sets unselected token logits to -inf prior to log_softmax,
    ensuring the returned distribution is constrained and normalized.
    Empirically verified to sum to 1 within numerical tolerance (1e-3).
  - Common state prefix is byte-identical across questions, enabling KV cache reuse.

Internal API:
  evaluate(query, n_permutations=3) -> EvaluateResponse

The engine exposes raw logits in addition to probabilities to support temperature
scaling calibration applied prior to softmax normalization.

vLLM SamplingParams parameter signature is inspected dynamically at runtime
(see _detect_restrict_param).
"""

from __future__ import annotations

import inspect
import math
import os
import time

# Enable pinned memory / UVA support required by vLLM on WSL2 and disable flashinfer sampler
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from dataclasses import dataclass, field
from typing import Any

from .debias import (
    Permutation,
    generate_permutations,
    generate_bidirectional_permutations,
    remap_distribution,
    average_distributions,
    geometric_mean_distributions,
)
from .prompt import (
    PromptSpec,
    build_prompt_spec,
    OTHER_LABEL_TEXT,
)
from .tokens import (
    ResolvedTokens,
    resolve_choice_tokens,
    resolve_noul_tokens,
)
from .types import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluateResponse,
    NoulAnswer,
    NoulQuestion,
    Query,
    ScoreAnswer,
    ScoreQuestion,
    confidence_from_probs,
)


# ---------------------------------------------------------------------------
# Token Restriction Parameter Detection (Dynamic vLLM API compatibility)
# ---------------------------------------------------------------------------

def _detect_restrict_param(SamplingParams) -> str:
    """Inspects the runtime signature of SamplingParams to identify the parameter
    responsible for restricting token vocabulary.

    Supported candidate arguments: allowed_token_ids, allowed_tokens, logit_bias.
    Raises RuntimeError if no known restriction parameter is detected.
    """
    field_names: set[str] = set()
    try:
        sig = inspect.signature(SamplingParams)
        field_names |= set(sig.parameters.keys())
    except (ValueError, TypeError):
        pass
    try:
        sp0 = SamplingParams()
        field_names |= set(sp0.__dict__.keys())
        # attrs-based (vLLM commonly leverages attrs/dataclasses)
        if hasattr(sp0, "__attrs_attrs__"):
            for a in sp0.__attrs_attrs__:
                field_names.add(a.name)
    except Exception:
        pass
    for cand in ("allowed_token_ids", "allowed_tokens"):
        if cand in field_names:
            return cand
    if "logit_bias" in field_names:
        return "logit_bias"
    raise RuntimeError(
        f"No token-restriction parameter found on SamplingParams "
        f"(known fields: {sorted(field_names)}). Unrecognized vLLM version."
    )


# ---------------------------------------------------------------------------
# Logit Evaluation Result
# ---------------------------------------------------------------------------

@dataclass
class LogitResult:
    """Evaluation pass output for a single prompt: raw logits and restricted
    probabilities over target option tokens."""
    spec: PromptSpec
    resolved: ResolvedTokens
    # Raw logits restricted to option tokens (before vocabulary masking,
    # sufficient for post-hoc temperature scaling calibration prior to softmax).
    raw_logits: list[float]
    # Restricted and normalized probability distribution (sums to ~1.0)
    probs: list[float]
    # Argmax predicted token id
    pred_token_id: int


# ---------------------------------------------------------------------------
# Engine Configuration & Runtime Wrapper
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    # Default backbone: Qwen2.5-3B-Instruct (high-performance SLM, ~1.3 GB VRAM).
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 2048
    quantization: str | None = None

    enforce_eager: bool = True
    enable_prefix_caching: bool = True
    # Numerical tolerance threshold for probability summation checks
    sum_tolerance: float = 1e-3
    # Cardinality threshold above which two-stage evaluation routing is required
    two_stage_threshold: int = 26
    two_stage_top_k: int = 10
    # Maximum supported cardinality ceiling
    max_cardinality: int = 255


class Engine:
    """vLLM engine wrapper providing unified batch inference and logit extraction.

    The model is loaded lazily upon first call to allow testing and mock
    instantiation in non-GPU environments.
    """

    def __init__(self, config: EngineConfig | None = None, tokenizer=None):
        self.config = config or EngineConfig()
        self._tokenizer = tokenizer
        self._llm: Any = None
        self._SamplingParams = None
        self._restrict_param: str | None = None
        self._sampling_fields: set[str] = set()
        self._resolved_cache: dict[tuple, Any] = {}
        self._params_cache: dict[tuple, Any] = {}

    # -- Lazy model initialization --
    def _ensure_loaded(self):
        if self._llm is not None:
            return
        from vllm import LLM, SamplingParams  # lazy import
        from transformers import AutoTokenizer
        self._SamplingParams = SamplingParams
        field_names: set[str] = set()
        try:
            sig = inspect.signature(SamplingParams)
            field_names |= set(sig.parameters.keys())
        except (ValueError, TypeError):
            pass
        try:
            sp0 = SamplingParams()
            field_names |= set(sp0.__dict__.keys())
            if hasattr(sp0, "__attrs_attrs__"):
                for a in sp0.__attrs_attrs__:
                    field_names.add(a.name)
        except Exception:
            pass
        self._sampling_fields = field_names
        self._restrict_param = _detect_restrict_param(SamplingParams)
        self._tokenizer = self._tokenizer or AutoTokenizer.from_pretrained(
            self.config.model
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        llm_kwargs: dict[str, Any] = dict(
            model=self.config.model,
            enable_prefix_caching=self.config.enable_prefix_caching,
            dtype=self.config.dtype,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            enforce_eager=self.config.enforce_eager,
            # Direct Choice evaluation requests one logprob per candidate, up to 26
            # letters plus the fallback. vLLM's default cap is 20, which rejected any
            # query with more than 20 candidates (VLLMValidationError on `logprobs`).
            max_logprobs=self.config.two_stage_threshold + 2,
        )
        if self.config.quantization is not None:
            llm_kwargs["quantization"] = self.config.quantization
        self._llm = LLM(**llm_kwargs)

    def close(self) -> None:
        """Explicitly releases GPU resources and the underlying vLLM EngineCore process."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    # -- SamplingParams construction --
    def _make_params(self, option_ids: list[int]):
        key = tuple(option_ids)
        if key in self._params_cache:
            return self._params_cache[key]
        sp = self._SamplingParams
        kwargs: dict[str, Any] = dict(
            max_tokens=1,
            temperature=0.0,
            logprobs=max(len(option_ids), 1),
        )
        if "logprob_token_ids" in self._sampling_fields:
            kwargs["logprob_token_ids"] = list(option_ids)
        if self._restrict_param == "logit_bias":
            bias: dict[int, float] = {tid: 0.0 for tid in option_ids}
            kwargs["logit_bias"] = bias
        else:
            kwargs[self._restrict_param] = list(option_ids)
        res = sp(**kwargs)
        self._params_cache[key] = res
        return res

    # -- Logits & probability extraction --
    def _extract(
        self,
        output,
        spec: PromptSpec,
        resolved: ResolvedTokens,
    ) -> LogitResult:
        """Extracts raw logits and constrained probabilities from vLLM RequestOutput."""
        out = output.outputs[0]
        lp = out.logprobs[0]  # dict[int, Logprob] for the 1st generated token
        raw_logits: list[float] = []
        probs: list[float] = []
        for tid in resolved.token_ids:
            if tid in lp:
                raw_logits.append(float(lp[tid].logprob))  # logprob already constrained log-softmax
                probs.append(float(math.exp(lp[tid].logprob)))
            else:
                # Token absent from returned top-k logprobs -> probability ~0
                raw_logits.append(float("-inf"))
                probs.append(0.0)
        total = sum(probs)
        if abs(total - 1.0) > self.config.sum_tolerance and total > 0:
            # Renormalize defensively if numerical drift exceeds tolerance
            probs = [p / total for p in probs]
        # Predicted token = argmax of probabilities
        pred_idx = max(range(len(probs)), key=lambda i: probs[i])
        pred_tid = resolved.token_ids[pred_idx]
        return LogitResult(
            spec=spec, resolved=resolved,
            raw_logits=raw_logits, probs=probs, pred_token_id=pred_tid,
        )

    # -- Token resolution --
    def _resolve(self, spec: PromptSpec) -> ResolvedTokens:
        key = (spec.kind, spec.suffix, spec.n_options)
        if key in self._resolved_cache:
            return self._resolved_cache[key]
        if spec.kind == "noul":
            res = resolve_noul_tokens(self._tokenizer, spec.suffix)
        else:
            res = resolve_choice_tokens(self._tokenizer, spec.suffix, spec.n_options)
        self._resolved_cache[key] = res
        return res

    # -- Batch prompt generation (questions × permutations) --
    def _build_batch(
        self,
        query: Query,
        n_permutations: int,
        seed: int = 0,
    ) -> list[dict]:
        """Constructs the list of prompt specifications to score in a single unified batch.

        Each batch entry contains:
          - prompt full text (via PromptSpec)
          - PromptSpec
          - ResolvedTokens
          - Permutation mapping (identity for score/noul questions)
          - question index and Question instance
        """
        entries: list[dict] = []
        for qi, question in enumerate(query.questions):
            if isinstance(question, ChoiceQuestion):
                base_options = list(question.options)
                n_opts = len(base_options)
                # Cardinality > threshold requires two-stage evaluation. Handled in
                # evaluate_two_stage; here only direct Choice evaluation is supported.
                if n_opts > self.config.two_stage_threshold:
                    raise NotImplementedError(
                        f"Choice with {n_opts} options exceeds threshold "
                        f"{self.config.two_stage_threshold}. "
                        "Use evaluate_two_stage or raise the threshold."
                    )
                # Effective options scored: base options + 'other' if allow_other
                eff_options = list(base_options)
                if question.allow_other:
                    eff_options.append(OTHER_LABEL_TEXT)
                n_eff = len(eff_options)
                perms = generate_permutations(n_eff, n_permutations, seed=seed + qi)
                for perm in perms:
                    permuted = [eff_options[i] for i in perm.order]
                    # include_other=False because OTHER is already incorporated into eff_options
                    spec = build_prompt_spec(
                        query.state, question,
                        options=permuted, include_other=False,
                    )
                    resolved = self._resolve(spec)
                    entries.append(dict(
                        qi=qi, question=question, perm=perm,
                        spec=spec, resolved=resolved,
                        eff_options=eff_options, permuted=permuted,
                    ))
            elif isinstance(question, ScoreQuestion):
                # Score questions: if n_permutations >= 2, apply dual-pass reversal debiasing
                # (Pass 1 ascending, Pass 2 descending). If n_permutations == 1, single identity pass.
                levels = question.levels
                n_lvls = len(levels)
                if n_permutations >= 2 and n_lvls >= 2:
                    perms = generate_bidirectional_permutations(n_lvls)
                    for perm in perms:
                        permuted_levels = [levels[i] for i in perm.order]
                        spec = build_prompt_spec(query.state, question, options=permuted_levels)
                        resolved = self._resolve(spec)
                        entries.append(dict(
                            qi=qi, question=question, perm=perm,
                            spec=spec, resolved=resolved,
                        ))
                else:
                    spec = build_prompt_spec(query.state, question)
                    resolved = self._resolve(spec)
                    perm = Permutation(order=list(range(n_lvls)))
                    entries.append(dict(
                        qi=qi, question=question, perm=perm,
                        spec=spec, resolved=resolved,
                    ))
            elif isinstance(question, NoulQuestion):
                spec = build_prompt_spec(query.state, question)
                resolved = self._resolve(spec)
                perm = Permutation(order=[0, 1])
                entries.append(dict(
                    qi=qi, question=question, perm=perm,
                    spec=spec, resolved=resolved,
                ))
            else:
                raise TypeError(f"Unsupported question type: {type(question)}")
        return entries

    # -- API principale --
    # -- Main evaluation API --
    def evaluate(self, query: Query, n_permutations: int = 3) -> EvaluateResponse:
        """Evaluates all questions in a SINGLE batch sharing prefix state KV caching.

        Returns an EvaluateResponse containing answers, latency_ms, and cache_hit_rate.
        n_permutations: k permutations for Choice question debiasing (k=1 fast mode,
        k=3 default). Score and Noul questions are not subject to permutation shuffling.
        """
        self._ensure_loaded()
        t0 = time.perf_counter()
        entries = self._build_batch(query, n_permutations)
        prompts = [e["spec"].full_text for e in entries]
        # A single llm.generate call processes the entire batch with per-prompt SamplingParams.
        # vLLM accepts a list of SamplingParams corresponding 1:1 with prompts,
        # applying exact allowed_token_ids and logprob_token_ids per prompt.
        params_list = [self._make_params(e["resolved"].token_ids) for e in entries]
        outputs = self._llm.generate(prompts, params_list)
        # Compute prefix cache hit rate from vLLM output statistics
        cache_hit_rate = self._cache_hit_rate(outputs)

        # Logit and probability extraction
        results_per_entry = [
            self._extract(out, e["spec"], e["resolved"])
            for out, e in zip(outputs, entries)
        ]

        # Group results by question index
        answers: dict[str, Answer] = {}
        by_qi: dict[int, list[dict]] = {}
        for e, lr in zip(entries, results_per_entry):
            by_qi.setdefault(e["qi"], []).append(dict(entry=e, result=lr))

        for qi, items in by_qi.items():
            question = query.questions[qi]
            if isinstance(question, ChoiceQuestion):
                answers[question.key] = self._compose_choice(question, items)
            elif isinstance(question, ScoreQuestion):
                answers[question.key] = self._compose_score(question, items)
            elif isinstance(question, NoulQuestion):
                answers[question.key] = self._compose_noul(question, items)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return EvaluateResponse(
            answers=answers, latency_ms=latency_ms,
            cache_hit_rate=cache_hit_rate,
        )

    def _make_params_with_logprobs(self, max_opts: int):
        sp = self._SamplingParams
        kwargs: dict[str, Any] = dict(
            max_tokens=1, temperature=0.0, logprobs=max(max_opts, 1),
        )
        return sp(**kwargs)

    def _cache_hit_rate(self, outputs) -> float:
        """Computes prefix cache hit rate from vLLM execution metrics."""
        try:
            hits = 0
            total = 0
            for out in outputs:
                # 1. vLLM V1 exposes num_cached_tokens directly on RequestOutput
                cached = getattr(out, "num_cached_tokens", None)
                p_tokens = getattr(out, "prompt_token_ids", None)
                n_p = len(p_tokens) if p_tokens is not None else 0
                if cached is not None and n_p > 0:
                    hits += cached
                    total += n_p
                    continue

                # 2. Fallback via RequestMetrics
                m = getattr(out, "metrics", None)
                if m is not None:
                    ch = getattr(m, "num_computed_prefix_tokens", 0)
                    t = getattr(m, "num_total_tokens", n_p)
                    if t > 0:
                        hits += ch
                        total += t
            if total > 0:
                return hits / total
        except Exception:
            pass
        return 0.0

    # -- Answer composition --
    def _compose_choice(self, question: ChoiceQuestion, items: list[dict]) -> ChoiceAnswer:
        """Averages remapped probabilities across permutations and constructs ChoiceAnswer."""
        eff_options = items[0]["entry"]["eff_options"]
        # Scored option labels: A, B, ... (cardinality = len(eff_options))
        # 'other' corresponds to the final option if allow_other is True.
        perm_results = []
        for it in items:
            perm: Permutation = it["entry"]["perm"]
            probs = it["result"].probs  # indexed by presentation position
            remapped = remap_distribution(perm, probs)  # indexed by canonical option
            perm_results.append((perm, remapped))
        avg = average_distributions([r for _, r in perm_results])
        # Map to option string labels
        prob_map = {eff_options[i]: avg[i] for i in range(len(eff_options))}
        # Renormalize defensively
        s = sum(prob_map.values())
        if s > 0:
            prob_map = {k: v / s for k, v in prob_map.items()}
        choice = max(prob_map, key=prob_map.get)
        conf = confidence_from_probs(list(prob_map.values()))
        return ChoiceAnswer(
            choice=choice, probabilities=prob_map, confidence=conf,
        )

    def _compose_score(self, question: ScoreQuestion, items: list[dict]) -> ScoreAnswer:
        """Computes expected values: score Σ p_i × i (0-based) and expected_score Σ p_i × (i+1) (1-based).

        For dual-pass reversal debiasing, remaps to natural canonical order and aggregates
        via normalized geometric mean to neutralize positional asymmetries and middle-bias.
        """
        levels = question.levels
        n_lvls = len(levels)
        if len(items) > 1:
            remapped_list = []
            for it in items:
                perm: Permutation = it["entry"]["perm"]
                pos_probs = it["result"].probs
                remapped = remap_distribution(perm, pos_probs)
                remapped_list.append(remapped)
            probs = geometric_mean_distributions(remapped_list)
        else:
            probs = items[0]["result"].probs

        # Renormalize defensively
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
        prob_map = {levels[i]: probs[i] for i in range(n_lvls)}
        score = sum(probs[i] * i for i in range(n_lvls))
        expected_score = sum(probs[i] * (i + 1) for i in range(n_lvls))
        level = levels[max(range(n_lvls), key=lambda i: probs[i])]
        conf = confidence_from_probs(probs)
        return ScoreAnswer(
            score=score,
            expected_score=expected_score,
            level=level,
            probabilities=prob_map,
            confidence=conf,
        )

    def _compose_noul(self, question: NoulQuestion, items: list[dict]) -> NoulAnswer:
        """Noul: probability of affirmative 'yes' token (p(yes) clamped to [0, 1])."""
        probs = items[0]["result"].probs  # [p_yes, p_no]
        s = sum(probs)
        if s > 0:
            probs = [p / s for p in probs]
        p_yes = probs[0] if len(probs) >= 1 else 0.0
        return NoulAnswer(noul=max(0.0, min(1.0, p_yes)))

    # -- Two-stage pipeline for high-cardinality questions --
    def evaluate_two_stage(self, query: Query) -> EvaluateResponse:
        """Two-stage retrieval and reranking for high cardinality (> 26 options):
          1. Independently score each option using binary Noul ("Does this option match?").
          2. Select top-k candidate options (k=10).
          3. Final Choice question over top-k candidates.
        """
        self._ensure_loaded()
        t0 = time.perf_counter()
        # For each Choice question exceeding cardinality threshold:
        # Stage 1: independent Noul per candidate option.
        # Stage 2: Choice over selected top-k candidates.
        noul_entries: list[dict] = []
        direct_entries: list[dict] = []
        choice_to_options: dict[int, list[str]] = {}
        for qi, question in enumerate(query.questions):
            if isinstance(question, ChoiceQuestion) and len(question.options) > self.config.two_stage_threshold:
                # Stage 1: one Noul prompt per candidate option
                choice_to_options[qi] = list(question.options)
                for oi, opt in enumerate(question.options):
                    noul_q = NoulQuestion(
                        key=f"{question.key}__noul_{oi}",
                        statement=f"The option “{opt}” fits this state.",
                    )
                    spec = build_prompt_spec(query.state, noul_q)
                    resolved = self._resolve(spec)
                    noul_entries.append(dict(
                        qi=qi, oi=oi, option=opt,
                        question=question, spec=spec, resolved=resolved,
                    ))
            else:
                direct_entries.extend(self._build_batch(
                    type(query)(state=query.state, questions=[question]),
                    n_permutations=1,
                ))

        # Stage 1: batch execution of all Noul prompts
        answers: dict[str, Answer] = {}
        if noul_entries:
            prompts = [e["spec"].full_text for e in noul_entries]
            params_list = [self._make_params(e["resolved"].token_ids) for e in noul_entries]
            outputs = self._llm.generate(prompts, params_list)
            # Aggregate score per option
            option_scores: dict[int, list[tuple[str, float]]] = {}
            for e, out in zip(noul_entries, outputs):
                lr = self._extract(out, e["spec"], e["resolved"])
                p_yes = lr.probs[0] if lr.probs else 0.0
                option_scores.setdefault(e["qi"], []).append((e["option"], p_yes))
            # Stage 2: top-k selection per question -> Choice
            stage2_questions = []
            mapping = []  # (qi, top_options)
            for qi, scored in option_scores.items():
                scored.sort(key=lambda x: -x[1])
                top_k = min(self.config.two_stage_top_k, len(scored))
                top = [opt for opt, _ in scored[:top_k]]
                orig_q = query.questions[qi]
                stage2_q = ChoiceQuestion(
                    key=orig_q.key, prompt=orig_q.prompt,
                    options=top, allow_other=False,  # top-k is already filtered
                )
                stage2_questions.append(stage2_q)
                mapping.append((qi, top))
            # Batch execution of stage 2
            s2_query = type(query)(state=query.state, questions=stage2_questions)
            s2_entries = self._build_batch(s2_query, n_permutations=1)
            s2_prompts = [e["spec"].full_text for e in s2_entries]
            s2_params_list = [self._make_params(e["resolved"].token_ids) for e in s2_entries]
            s2_outputs = self._llm.generate(s2_prompts, s2_params_list)
            s2_results = [self._extract(o, e["spec"], e["resolved"])
                          for o, e in zip(s2_outputs, s2_entries)]
            for (qi, top), entry, lr in zip(mapping, s2_entries, s2_results):
                q = query.questions[qi]
                prob_map = {top[i]: lr.probs[i] for i in range(len(top))}
                s = sum(prob_map.values())
                if s > 0:
                    prob_map = {k: v / s for k, v in prob_map.items()}
                choice = max(prob_map, key=prob_map.get)
                conf = confidence_from_probs(list(prob_map.values()))
                answers[q.key] = ChoiceAnswer(
                    choice=choice, probabilities=prob_map, confidence=conf,
                )

        # Direct evaluation entries (questions below threshold)
        if direct_entries:
            prompts = [e["spec"].full_text for e in direct_entries]
            params_list = [self._make_params(e["resolved"].token_ids) for e in direct_entries]
            outputs = self._llm.generate(prompts, params_list)
            results = [self._extract(o, e["spec"], e["resolved"])
                       for o, e in zip(outputs, direct_entries)]
            by_qi: dict[int, list[dict]] = {}
            for e, lr in zip(direct_entries, results):
                by_qi.setdefault(e["qi"], []).append(dict(entry=e, result=lr))
            for qi, items in by_qi.items():
                q = query.questions[qi]
                if isinstance(q, ChoiceQuestion):
                    answers[q.key] = self._compose_choice(q, items)
                elif isinstance(q, ScoreQuestion):
                    answers[q.key] = self._compose_score(q, items)
                elif isinstance(q, NoulQuestion):
                    answers[q.key] = self._compose_noul(q, items)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return EvaluateResponse(
            answers=answers, latency_ms=latency_ms,
            cache_hit_rate=0.0,
        )
