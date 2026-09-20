"""Baseline evaluation pipeline.

Compares 4 configurations on held-out evaluation datasets:
  1. Raw un-tuned backbone, logit extraction, WITHOUT calibration or debiasing.
  2. Same raw backbone generating autoregressive JSON, parsed
     (explicitly measuring JSON parse failure rate and token generation latency overhead).
  3. Fine-tuned model WITHOUT temperature scaling or debiasing.
  4. Full production system (fine-tuned + temperature calibration + cyclic debiasing).

Baseline #2 (autoregressive JSON parsing) represents the anti-pattern sysone eliminates:
generating arbitrary syntax to extract structured labels introduces parse failures and non-calibrated logits.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from eval.metrics import (
    MetricsBundle, compute_all, latency_stats, LatencyStats,
)


# ---------------------------------------------------------------------------
# Baseline Configuration
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    name: str
    model: str
    debias: bool = False
    calibrate: bool = False
    finetuned: bool = False
    generate_json: bool = False


# ---------------------------------------------------------------------------
# Baseline 1: Raw Backbone (Single-pass Logit Extraction)
# ---------------------------------------------------------------------------

def _prepare_choice_options(ex: dict, max_opts: int = 8) -> tuple[list[str], int, bool]:
    """Prepare a deterministic subset of options when total candidate options exceed max_opts.

    Guarantees the target ground truth option is retained and sampling is fully deterministic.
    """
    raw_options = ex["options"]
    label = ex.get("label", -1)
    if len(raw_options) <= max_opts:
        return list(raw_options), label, (label == -1)

    import hashlib
    import random
    h = int(hashlib.md5(ex["state"].encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(h)

    if label >= 0 and label < len(raw_options):
        target = raw_options[label]
        distractors = [opt for i, opt in enumerate(raw_options) if i != label]
        rng.shuffle(distractors)
        chosen = distractors[:max_opts - 1]
        chosen.append(target)
        rng.shuffle(chosen)
        new_label = chosen.index(target)
        return chosen, new_label, False
    else:
        distractors = list(raw_options)
        rng.shuffle(distractors)
        chosen = distractors[:max_opts]
        return chosen, -1, True


def run_baseline_raw(
    engine,
    examples: list[dict],
    n_permutations: int = 1,
) -> tuple[list, list, list, list, LatencyStats]:
    """Evaluate raw un-tuned backbone with n_permutations=1 (no debiasing) and no calibration.

    Returns (all_probs, all_labels, confidences, corrects, latency_stats).
    """
    from sysone.types import (
        Query, ChoiceQuestion, ScoreQuestion, NoulQuestion,
    )
    all_probs = []
    all_labels = []
    confidences = []
    corrects = []
    latencies = []
    for ex in examples:
        if ex["kind"] == "choice":
            options, label, allow_other = _prepare_choice_options(ex)
            q = ChoiceQuestion(key="q", prompt=ex["prompt"],
                                options=options,
                                allow_other=allow_other)
        elif ex["kind"] == "score":
            q = ScoreQuestion(key="q", prompt=ex["prompt"], levels=ex["levels"])
            label = ex["label"]
        elif ex["kind"] == "noul":
            q = NoulQuestion(key="q", statement=ex["statement"])
            label = ex["label"]
        else:
            continue
        query = Query(state=ex["state"], questions=[q])
        t0 = time.perf_counter()
        resp = engine.evaluate(query, n_permutations=n_permutations)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ans = resp.answers["q"]
        if ex["kind"] == "choice":
            probs_dict = ans.probabilities
            all_opts = options + (["__other__"] if allow_other else [])
            probs = [probs_dict.get(o, 0.0) for o in all_opts]
            all_probs.append(probs)
            lbl = label if label >= 0 else len(all_opts) - 1
            all_labels.append(lbl)
            confidences.append(max(probs))
            pred = max(range(len(probs)), key=lambda i: probs[i])
            corrects.append(1 if pred == lbl else 0)
        elif ex["kind"] == "score":
            levels = ex["levels"]
            probs = [ans.probabilities.get(l, 0.0) for l in levels]
            all_probs.append(probs)
            all_labels.append(label)
            confidences.append(max(probs))
            pred = max(range(len(probs)), key=lambda i: probs[i])
            corrects.append(1 if pred == label else 0)
        elif ex["kind"] == "noul":
            p_yes = ans.noul
            all_probs.append([p_yes, 1.0 - p_yes])
            all_labels.append(label)
            confidences.append(max(p_yes, 1.0 - p_yes))
            pred = 1 if p_yes > 0.5 else 0
            corrects.append(1 if pred == label else 0)
    lat_stats = latency_stats(latencies)
    return all_probs, all_labels, confidences, corrects, lat_stats


# ---------------------------------------------------------------------------
# Baseline 2: Autoregressive JSON Generation Anti-Pattern
# ---------------------------------------------------------------------------

JSON_PROMPT_TEMPLATE = """<system>Respond strictly with a valid JSON object.</system>
<user>
## State
{state}

## Question
{question}

Respond with a JSON object of the form {{"answer": <value>}} where <value> is the 0-based integer index of the correct option.
</user>
<assistant>{{"answer":"""


def run_baseline_json(
    llm, SamplingParams,
    examples: list[dict],
    max_tokens: int = 16,
) -> tuple[list, list, list, list, LatencyStats, float]:
    """Generate autoregressive JSON and parse answers, recording syntax failures.

    Returns (all_probs, all_labels, confidences, corrects, lat_stats, parse_failure_rate).
    Probabilities are degenerate one-hot distributions placed on the extracted index,
    demonstrating the lack of soft calibrated confidence estimates.
    """
    all_probs = []
    all_labels = []
    confidences = []
    corrects = []
    latencies = []
    parse_failures = 0
    for ex in examples:
        if ex["kind"] == "choice":
            options, label, allow_other = _prepare_choice_options(ex)
            opt_lines = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)])
            if allow_other:
                opt_lines += f"\n{len(options)}. Other / None of the above"
            q_text = f"{ex['prompt']}\n\nOptions:\n{opt_lines}"
            n = len(options) + (1 if allow_other else 0)
            prompt = JSON_PROMPT_TEMPLATE.format(
                state=ex["state"], question=q_text,
            )
            lbl = label if label >= 0 else len(options)
        elif ex["kind"] == "score":
            q_text = ex["prompt"]
            n = len(ex["levels"])
            prompt = JSON_PROMPT_TEMPLATE.format(state=ex["state"], question=q_text)
            lbl = ex["label"]
        elif ex["kind"] == "noul":
            q_text = f"Is the proposition true? {ex['statement']}"
            n = 2
            prompt = JSON_PROMPT_TEMPLATE.format(state=ex["state"], question=q_text)
            lbl = ex["label"]
        else:
            continue
        params = SamplingParams(
            max_tokens=max_tokens, temperature=0.0,
        )
        t0 = time.perf_counter()
        out = llm.generate([prompt], params)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        text = out[0].outputs[0].text.strip()
        # Parse output for pattern "answer": <int>
        m = re.search(r'"answer"\s*:\s*(\d+)', text)
        if m is None:
            parse_failures += 1
            pred = -1
        else:
            pred = int(m.group(1))
        # Degenerate probabilities: 1.0 on prediction, 0 elsewhere
        probs = [0.0] * n
        if 0 <= pred < n:
            probs[pred] = 1.0
            corrects.append(1 if pred == lbl else 0)
        else:
            corrects.append(0)
        all_probs.append(probs)
        all_labels.append(lbl)
        confidences.append(1.0 if pred >= 0 else 0.0)
    lat_stats = latency_stats(latencies)
    parse_failure_rate = parse_failures / max(len(examples), 1)
    return all_probs, all_labels, confidences, corrects, lat_stats, parse_failure_rate


# ---------------------------------------------------------------------------
# Baselines 3 & 4: Fine-tuned Model Without / With Calibration and Debiasing
# ---------------------------------------------------------------------------

def run_baseline_finetuned(
    engine,
    examples: list[dict],
    n_permutations: int = 1,
) -> tuple:
    """Evaluate fine-tuned model without calibration or debiasing."""
    return run_baseline_raw(engine, examples, n_permutations=n_permutations)


def run_full_system(
    calibrated_engine,
    examples: list[dict],
    n_permutations: int = 3,
) -> tuple:
    """Evaluate full pipeline: fine-tuned weights + temperature scaling + debiasing (k=3)."""
    from sysone.types import (
        Query, ChoiceQuestion, ScoreQuestion, NoulQuestion,
    )
    all_probs = []
    all_labels = []
    confidences = []
    corrects = []
    latencies = []
    for ex in examples:
        if ex["kind"] == "choice":
            options, label, allow_other = _prepare_choice_options(ex)
            q = ChoiceQuestion(key="q", prompt=ex["prompt"],
                                options=options,
                                allow_other=allow_other)
        elif ex["kind"] == "score":
            q = ScoreQuestion(key="q", prompt=ex["prompt"], levels=ex["levels"])
            label = ex["label"]
        elif ex["kind"] == "noul":
            q = NoulQuestion(key="q", statement=ex["statement"])
            label = ex["label"]
        else:
            continue
        query = Query(state=ex["state"], questions=[q])
        t0 = time.perf_counter()
        resp = calibrated_engine.evaluate(query, n_permutations=n_permutations)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        ans = resp.answers["q"]
        if ex["kind"] == "choice":
            all_opts = options + (["__other__"] if allow_other else [])
            probs = [ans.probabilities.get(o, 0.0) for o in all_opts]
            all_probs.append(probs)
            lbl = label if label >= 0 else len(all_opts) - 1
            all_labels.append(lbl)
            confidences.append(max(probs))
            pred = max(range(len(probs)), key=lambda i: probs[i])
            corrects.append(1 if pred == lbl else 0)
        elif ex["kind"] == "score":
            levels = ex["levels"]
            probs = [ans.probabilities.get(l, 0.0) for l in levels]
            all_probs.append(probs)
            all_labels.append(label)
            confidences.append(max(probs))
            pred = max(range(len(probs)), key=lambda i: probs[i])
            corrects.append(1 if pred == label else 0)
        elif ex["kind"] == "noul":
            p_yes = ans.noul
            all_probs.append([p_yes, 1.0 - p_yes])
            all_labels.append(label)
            confidences.append(max(p_yes, 1.0 - p_yes))
            pred = 1 if p_yes > 0.5 else 0
            corrects.append(1 if pred == label else 0)
    lat_stats = latency_stats(latencies)
    return all_probs, all_labels, confidences, corrects, lat_stats
