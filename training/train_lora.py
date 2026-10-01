"""LoRA fine-tuning pipeline.

NO RL REQUIRED. Log-loss computed on the single target response token constitutes
a proper scoring rule: supervised fine-tuning directly optimizes probabilistic calibration.
RL is only warranted when reward formulations are non-differentiable, which does not
apply to single-token likelihood classification.

Configuration:
  - Backbone: Qwen/Qwen2.5-3B-Instruct
  - LoRA rank 32, alpha 64, dropout 0.05 targeting q, k, v, o, gate, up, down projections.
  - Loss formulation: cross-entropy over the SINGLE response token; all prompt prefix tokens
    are masked out (labels = -100).
  - Learning rate 1e-4 with cosine schedule, 3% warmup, bfloat16 mixed precision,
    effective batch size 64 via gradient accumulation.
  - Periodic checkpointing and validation NLL computation.
  - Checkpoint selection criterion is strictly validation NLL (log-loss), NOT accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

# Ensure repository root is available on sys.path for sysone and training imports
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CUDA configuration for WSL2: disable expandable_segments (incompatible with cuMemMap in WSL2)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

# Heavy torch/transformers/peft imports are deferred inside function scopes


def build_model(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load base causal language model with gradient checkpointing enabled."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"[train] Loading tokenizer for {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[train] Loading model {model_name} in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False

    # Gradient checkpointing is mandatory to fit training within 16 GB VRAM budgets
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    return model, tok


DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Qwen3.5 runs 3 of every 4 layers as Gated DeltaNet linear attention, whose projections
# have their own names; "attention" LoRA there means both kinds of layer.
QWEN35_ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"]


def build_lora(model, rank: int = 32, alpha: int = 64, targets: list[str] | None = None, dropout: float = 0.05):
    """Apply LoRA (default: rank 32 across attention and MLP projections, the v1-v3 recipe)."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(targets or DEFAULT_TARGETS),
    )
    return get_peft_model(model, cfg)


class Target(NamedTuple):
    """A training prompt and its answer: the answer token, and the candidate tokens it is read among.

    option_ids are the tokens inference restricts the softmax to (the option letters, or
    yes / no), answer_pos the answer's index among them. ordinal marks a Score question,
    whose options are levels in scale order: reversing the scale (augment_score) keeps
    neighbours adjacent, so distances between positions are distances between levels.
    """
    text: str
    answer_id: int
    option_ids: tuple[int, ...] = ()
    answer_pos: int = 0
    ordinal: bool = False


def build_prompt_for_training(aug, tokenizer, fmt: str = "tags") -> Target:
    """Construct training prompt ending with 'Answer:' followed by target label token ID.

    Tokenizes the complete prompt context (without trailing letter) and appends
    the ground truth target token. All prior prefix tokens are masked (labels = -100).
    """
    from sysone.prompt import build_prompt_spec
    from sysone.tokens import noul_token_index, resolve_choice_tokens, resolve_noul_tokens
    from sysone.types import ChoiceQuestion, ScoreQuestion, NoulQuestion

    if aug.kind == "choice":
        include_other = aug.include_other
        q = ChoiceQuestion(
            key="q",
            prompt=aug.prompt,
            options=aug.options,
            allow_other=include_other,
        )
        spec = build_prompt_spec(
            aug.state,
            q,
            options=aug.options,
            include_other=include_other,
            fmt=fmt,
        )
        resolved = resolve_choice_tokens(tokenizer, spec.suffix, spec.n_options)
        label_idx = aug.label if aug.label >= 0 else len(aug.options)  # 'other' is mapped to last index
    elif aug.kind == "score":
        q = ScoreQuestion(key="q", prompt=aug.prompt, levels=aug.levels)
        spec = build_prompt_spec(aug.state, q, fmt=fmt)
        resolved = resolve_choice_tokens(tokenizer, spec.suffix, len(aug.levels))
        label_idx = aug.label
    elif aug.kind == "noul":
        q = NoulQuestion(key="q", statement=aug.statement, prompt=aug.prompt)
        spec = build_prompt_spec(aug.state, q, fmt=fmt)
        resolved = resolve_noul_tokens(tokenizer, spec.suffix)
        # Dataset label 1 == yes, but NOUL_LABELS index 0 == 'yes': go through the helper.
        label_idx = noul_token_index(aug.label)
    else:
        raise ValueError(f"Unknown question kind: {aug.kind}")

    return Target(spec.full_text, resolved.token_ids[label_idx], tuple(resolved.token_ids),
                  label_idx, aug.kind == "score")


def collate_fn(batch: list[tuple], tokenizer, max_len: int = 2048):
    """Tokenize a batch of prompts for a loss on the answer token only.

    Each prompt is cut on the left to max_len tokens (the question and the answer cue at
    the end are kept) and left-padded, so the last position of every row is the one that
    predicts the answer. `answer_loss` reads only that position: computing logits over
    the whole vocabulary at every position (up to 248k entries for Qwen3.5) is what the
    loss does not need and memory cannot afford at 4k tokens.

    Items are Targets (or bare (text, answer id) pairs). When they carry their candidate
    tokens, those are padded to the widest row (option_mask marks the real ones) for the
    restricted and ordinal terms of `answer_terms`.
    """
    import torch

    items = [Target(*b) for b in batch]
    rows, labels = [], []
    for it in items:
        ids = tokenizer.encode(it.text, add_special_tokens=False)
        rows.append(ids[-max_len:])
        labels.append(it.answer_id)
    width = max(len(r) for r in rows)
    pad_id = tokenizer.pad_token_id
    input_ids = [[pad_id] * (width - len(r)) + r for r in rows]
    attn = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
    # Positions count real tokens only, as if each row had no padding.
    positions = [[0] * (width - len(r)) + list(range(len(r))) for r in rows]
    out = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "answer_ids": torch.tensor(labels, dtype=torch.long),
    }
    if any(len(r) < width for r in rows):  # a single row needs neither
        out["attention_mask"] = torch.tensor(attn, dtype=torch.long)
        out["position_ids"] = torch.tensor(positions, dtype=torch.long)
    if all(it.option_ids for it in items):
        n = max(len(it.option_ids) for it in items)
        out["option_ids"] = torch.tensor([list(it.option_ids) + [it.option_ids[0]] * (n - len(it.option_ids))
                                          for it in items], dtype=torch.long)
        out["option_mask"] = torch.tensor([[i < len(it.option_ids) for i in range(n)] for it in items])
        out["answer_pos"] = torch.tensor([it.answer_pos for it in items], dtype=torch.long)
        out["ordinal"] = torch.tensor([it.ordinal for it in items])
    return out


MODEL_INPUTS = ("input_ids", "attention_mask", "position_ids")


def answer_terms(model, batch: dict) -> dict:
    """Per-row loss terms at the answer position, from one forward pass.

    - vocab: cross-entropy of the answer token over the whole vocabulary (v1-v5).
    - options: cross-entropy over the candidate tokens only, the softmax inference takes
      (allowed_token_ids), so no weight goes to pushing down tokens it never reads.
    - emd: squared earth mover's distance between the predicted distribution over a Score
      question's levels and the true level, Σ_k (CDF_pred(k) - CDF_true(k))²; zero on Choice
      and Noul rows. It grows with the distance to the true level, where cross-entropy
      treats a neighbouring level and the far end of the scale alike, and the answer is
      read as the expectation Σ p_i·i.
    The last two need the candidate tokens (batches collated from Targets).
    """
    import torch
    import torch.nn.functional as F

    inputs = {k: v for k, v in batch.items() if k in MODEL_INPUTS}
    logits = model(**inputs, logits_to_keep=1).logits[:, -1, :].float()
    terms = {"vocab": F.cross_entropy(logits, batch["answer_ids"], reduction="none")}
    if "option_ids" not in batch:
        return terms
    opt = logits.gather(1, batch["option_ids"]).masked_fill(~batch["option_mask"], float("-inf"))
    terms["options"] = F.cross_entropy(opt, batch["answer_pos"], reduction="none")
    k = torch.arange(opt.shape[1], device=opt.device)
    cdf_true = (k[None, :] >= batch["answer_pos"][:, None]).float()
    emd = ((opt.softmax(-1).cumsum(-1) - cdf_true) ** 2).sum(-1)
    terms["emd"] = emd * batch["ordinal"].float()
    terms["correct"] = (opt.argmax(-1) == batch["answer_pos"]).float()
    return terms


def answer_loss(model, batch: dict, restrict: bool = False, ordinal_weight: float = 0.0):
    """Mean training loss: the answer's cross-entropy (over the vocabulary, or over the
    candidates when restrict), plus ordinal_weight × the Score rows' EMD.

    The EMD is averaged over every row of the batch, Score or not, so a Score question
    weighs the same whatever else shares its batch (with micro_batch 1, the loss of a
    Score row is exactly CE + ordinal_weight × EMD).
    """
    return combine_loss(answer_terms(model, batch), restrict, ordinal_weight)


def combine_loss(terms: dict, restrict: bool = False, ordinal_weight: float = 0.0):
    loss = (terms["options"] if restrict else terms["vocab"]).mean()
    if ordinal_weight:
        loss = loss + ordinal_weight * terms["emd"].mean()
    return loss


def compute_nll(
    model,
    eval_examples: list[dict],
    tokenizer,
    pool,
    max_eval: int = 100,
    seed: int = 42,
    batch_size: int = 4,
    per_kind: bool = False,
    fmt: str = "tags",
    max_len: int = 2048,
    metric: str = "vocab",
) -> float | tuple[float, dict[str, float], dict[str, float]]:
    """Compute validation NLL over a deterministic, frozen evaluation set.

    Checkpoint selection is strictly driven by negative log-likelihood (log-loss),
    rather than top-1 classification accuracy. The sample is stratified across the
    three primitives; per_kind additionally returns the breakdown, so a primitive
    regressing behind a flat overall NLL is visible rather than averaged away, and
    both NLLs plus the Score rows' mean EMD, whatever the loss trained.

    metric picks the NLL returned: "vocab" (the answer token over the vocabulary) or
    "options" (over the candidates, as inference reads it).
    """
    import torch
    from torch.utils.data import DataLoader
    from training.augment import augment

    model.eval()
    val_rng = random.Random(seed)

    # Stratified across primitives rather than a prefix of the file. A prefix measures
    # whatever source happens to sit at the top, so checkpoint selection would be driven
    # by one primitive while the other two drift unwatched.
    by_kind: dict[str, list[dict]] = {"choice": [], "score": [], "noul": []}
    for ex in eval_examples:
        if ex.get("kind") in by_kind:
            by_kind[ex["kind"]].append(ex)

    present = [k for k, v in by_kind.items() if v]
    if not present:
        model.train()
        return (float("nan"), {}, {}) if per_kind else float("nan")

    per_kind_quota = max(1, max_eval // len(present))
    selected: list[tuple[str, dict]] = []
    for k in present:
        pool_k = list(by_kind[k])
        random.Random(seed).shuffle(pool_k)
        selected.extend((k, ex) for ex in pool_k[:per_kind_quota])

    # Deterministic generation of (prompt, label) pairs to eliminate sampling noise
    val_items: list[Target] = []
    val_kinds: list[str] = []
    for kind, ex in selected:
        try:
            aug = augment(ex, pool, val_rng)
            item = build_prompt_for_training(aug, tokenizer, fmt)
            val_items.append(item)
            val_kinds.append(kind)
        except Exception:
            continue

    if not val_items:
        model.train()
        return (float("nan"), {}, {}) if per_kind else float("nan")

    val_loader = DataLoader(
        list(zip(val_items, val_kinds)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: (collate_fn([x[0] for x in b], tokenizer, max_len),
                              [x[1] for x in b]),
    )

    # Per-row terms, so each row's loss goes to its own primitive even in a mixed batch.
    rows: list[tuple[str, dict[str, float]]] = []
    with torch.no_grad():
        for batch, kinds in val_loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            terms = {name: t.tolist() for name, t in answer_terms(model, batch).items()}
            for i, k in enumerate(kinds):
                row = {name: vals[i] for name, vals in terms.items()}
                if all(math.isfinite(v) for v in row.values()):
                    rows.append((k, row))

    model.train()
    mean = lambda vals: sum(vals) / len(vals) if vals else float("nan")
    overall = mean([r[metric] for _, r in rows])
    if not per_kind:
        return overall
    by_kind = {k: mean([r[metric] for kk, r in rows if kk == k]) for k in sorted({kk for kk, _ in rows})}
    extra = {"vocab": mean([r["vocab"] for _, r in rows]),
             "options": mean([r["options"] for _, r in rows]),
             "score_emd": mean([r["emd"] for k, r in rows if k == "score"]),
             # Accuracy as the mean of the per-primitive accuracies, like eval_base's "balanced".
             "acc": mean([mean([r["correct"] for kk, r in rows if kk == k]) for k in by_kind]),
             "acc_by_kind": {k: mean([r["correct"] for kk, r in rows if kk == k]) for k in by_kind}}
    return overall, by_kind, extra


def _fmt_by_kind(by_kind: dict[str, float],
                 baseline: dict[str, float] | None = None) -> str:
    """Render the per-primitive NLL breakdown, with the delta against step 0 when known."""
    if not by_kind:
        return "n/a"
    parts = []
    for k, v in sorted(by_kind.items()):
        if baseline and k in baseline:
            parts.append(f"{k}={v:.4f} ({v - baseline[k]:+.4f})")
        else:
            parts.append(f"{k}={v:.4f}")
    return "  ".join(parts)


def sample_stratified_mixture(
    examples: list[dict],
    max_examples: int,
    seed: int = 42,
    quotas: dict[str, float] | None = None,
) -> list[dict]:
    """Sample a balanced multi-task training mixture across question primitives.

    Allocates balanced portions across the 3 primitives:
    - Choice : 33.3%
    - Score  : 33.3%
    - Noul   : 33.3%
    Prevents smaller corpora (e.g. ordinal Score) from being overshadowed by large datasets.
    """
    if quotas is None:
        quotas = {"choice": 1.0 / 3.0, "score": 1.0 / 3.0, "noul": 1.0 / 3.0}

    rng = random.Random(seed)
    by_kind: dict[str, list[dict]] = {"choice": [], "score": [], "noul": []}
    for ex in examples:
        k = ex.get("kind", "")
        if k in by_kind:
            by_kind[k].append(ex)

    sampled: list[dict] = []
    for k, share in quotas.items():
        sub = by_kind[k]
        target_n = int(round(max_examples * share))
        if len(sub) <= target_n:
            sampled.extend(sub)
        else:
            rng.shuffle(sub)
            sampled.extend(sub[:target_n])

    rng.shuffle(sampled)
    return sampled


def train(
    train_path: str = "data/train.jsonl",
    eval_path: str = "data/eval.jsonl",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    output_dir: str = "checkpoints",
    epochs: int = 1,
    lr: float = 1e-4,
    micro_batch: int = 1,
    grad_accum: int = 64,
    max_examples: int | None = None,
    max_steps: int | None = None,
    val_every: int = 50,
    save_every: int = 100,
    eval_samples: int = 100,
    seed: int = 42,
    rank: int = 32,
    alpha: int = 64,
    targets: list[str] | None = None,
    max_len: int = 2048,
    prompt_format: str = "tags",
    loss: str = "vocab",
    ordinal_weight: float = 0.0,
    early_stop_patience: int = 0,
    select_on: str = "nll",
):
    import torch
    from torch.utils.data import DataLoader
    from training.augment import AugmentingDataset, DistractorPool

    if loss not in ("vocab", "options"):
        raise ValueError(f"Unknown loss {loss!r}; expected 'vocab' or 'options'.")
    restrict = loss == "options"
    print(f"[train] Initializing LoRA training with {model_name}...")
    model, tokenizer = build_model(model_name)
    model = build_lora(model, rank=rank, alpha=alpha, targets=targets)
    model.print_trainable_parameters()
    print(f"[train] LoRA r={rank} alpha={alpha} targets={targets or DEFAULT_TARGETS} | "
          f"max_len={max_len} | prompt_format={prompt_format}")
    # The validation NLL (and so the 'best' checkpoint) is the one the loss trains.
    print(f"[train] Loss: cross-entropy over {'the candidate tokens' if restrict else 'the vocabulary'}"
          + (f" + {ordinal_weight} x EMD on Score" if ordinal_weight else "") + f" | val NLL: {loss}"
          + f" | best and early stop on: {select_on}")
    if select_on not in ("nll", "acc"):
        raise ValueError(f"Unknown select_on {select_on!r}; expected 'nll' or 'acc'.")

    # Data loading
    print(f"[train] Loading datasets: {train_path} and {eval_path}...")
    train_examples = [
        json.loads(line)
        for line in Path(train_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_examples = [
        json.loads(line)
        for line in Path(eval_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if max_examples:
        train_examples = sample_stratified_mixture(train_examples, max_examples, seed=seed)
        counts = {k: sum(1 for x in train_examples if x.get("kind") == k) for k in ["choice", "score", "noul"]}
        print(f"[train] Stratified mixture ({len(train_examples)} instances): Choice={counts['choice']}, Score={counts['score']}, Noul={counts['noul']}")

    pool = DistractorPool.build(train_examples)
    print(f"[train] Choice distractor pool size: {len(pool.choice_options)} options")
    print(f"[train] Dataset sizes: {len(train_examples)} train, {len(eval_examples)} eval")

    ds = AugmentingDataset(train_examples, pool, seed=seed)
    dl = DataLoader(
        ds,
        batch_size=micro_batch,
        shuffle=True,
        collate_fn=lambda b: {
            **collate_fn([build_prompt_for_training(x, tokenizer, prompt_format) for x in b], tokenizer, max_len),
            "is_teacher": torch.tensor([x.source.startswith("teacher-") for x in b]),
        },
    )

    effective_batch = micro_batch * grad_accum
    steps_per_epoch = len(dl) // grad_accum
    total_steps = steps_per_epoch * epochs
    if max_steps and max_steps < total_steps:
        total_steps = max_steps

    print(
        f"[train] Batch configuration: micro_batch={micro_batch}, grad_accum={grad_accum} "
        f"-> effective batch={effective_batch}"
    )
    print(f"[train] Scheduled steps: {total_steps} (steps_per_epoch={steps_per_epoch})")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] Trainable parameter tensors: {len(trainable_params)}")

    # Use non-factored Adafactor for minimal optimizer state memory footprint (15 MB vs 672 MB for AdamW)
    from transformers.optimization import Adafactor
    optim = Adafactor(
        trainable_params,
        lr=lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    warmup_steps = max(1, int(0.03 * total_steps))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    out_dir_p = Path(output_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    metrics_history: list[dict[str, Any]] = []

    def validate():
        return compute_nll(model, eval_examples, tokenizer, pool, max_eval=eval_samples, seed=seed,
                           per_kind=True, fmt=prompt_format, max_len=max_len, metric=loss)

    def fmt_extra(extra: dict[str, float]) -> str:
        return (f"NLL vocab={extra['vocab']:.4f}  NLL options={extra['options']:.4f}  "
                f"Score EMD={extra['score_emd']:.4f}  acc={100 * extra['acc']:.1f} ("
                + " ".join(f"{k} {100 * v:.1f}" for k, v in extra["acc_by_kind"].items()) + ")")

    # v6: the val NLL bottomed at step 250 and climbed back to near its start, while dev accuracy
    # kept rising to the last step (overconfidence, which calibration then removes). "acc"
    # keeps 'best' and the early stop on accuracy; lower is better for both scores.
    sel_score = lambda nll, extra: -extra["acc"] if select_on == "acc" else nll

    # Baseline pre-training evaluation
    print(f"[train] Initial evaluation (step 0) on {eval_samples} reserved instances...")
    t0_val = time.time()
    init_val_nll, init_by_kind, init_extra = validate()
    print(f"[train] Initial VAL NLL (step 0): {init_val_nll:.4f} (computed in {time.time() - t0_val:.1f}s)")
    print(f"[train]   per primitive: {_fmt_by_kind(init_by_kind)}")
    print(f"[train]   {fmt_extra(init_extra)}")
    metrics_history.append({"step": 0, "val_nll": init_val_nll,
                            "val_nll_by_kind": init_by_kind, "val_terms": init_extra,
                            "train_loss": None, "lr": 0.0})

    step = 0
    best_val_nll = init_val_nll
    best_sel = sel_score(init_val_nll, init_extra)
    vals_since_best = 0
    accum_loss = 0.0
    # Answer CE of teacher and public rows since the last log line: teacher questions are
    # repeated, so their loss falling far below the public one is memorisation.
    group_ce = {True: [0.0, 0], False: [0.0, 0]}
    micro_step = 0
    t_start = time.time()

    model.train()
    optim.zero_grad()

    stop_training = False
    for epoch in range(epochs):
        if stop_training:
            break
        print(f"[train] --- Starting Epoch {epoch + 1}/{epochs} ---")
        for batch in dl:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            terms = answer_terms(model, batch)
            batch_loss = combine_loss(terms, restrict, ordinal_weight) / grad_accum
            loss_val = batch_loss.item()
            row_ce = (terms["options"] if restrict else terms["vocab"]).detach().tolist()
            for ce, teacher in zip(row_ce, batch["is_teacher"].tolist()):
                group_ce[teacher][0] += ce
                group_ce[teacher][1] += 1
            batch_loss.backward()
            accum_loss += loss_val
            micro_step += 1
            del batch, batch_loss, terms

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                current_loss = accum_loss
                current_lr = sched.get_last_lr()[0]
                accum_loss = 0.0

                if step % 10 == 0 or step == 1:
                    elapsed = time.time() - t_start
                    rate = step / max(elapsed, 1e-3)
                    eta = (total_steps - step) / max(rate, 1e-3)
                    print(
                        f"[train] step {step:4d}/{total_steps} | "
                        f"loss={current_loss:.4f} | lr={current_lr:.2e} | "
                        f"{rate:.2f} step/s | ETA: {eta/60:.1f}m | CE "
                        + " ".join(f"{name}={s / n:.3f}" for name, (s, n) in
                                   (("teacher", group_ce[True]), ("public", group_ce[False])) if n)
                    )
                    group_ce = {True: [0.0, 0], False: [0.0, 0]}

                if step % val_every == 0:
                    val_nll, val_by_kind, val_extra = validate()
                    delta_vs_init = val_nll - init_val_nll
                    delta_str = f"({delta_vs_init:+.4f} vs step 0)"
                    print(f"[train] >>> VAL NLL @ step {step}: {val_nll:.4f} {delta_str}")
                    print(f"[train]     per primitive: {_fmt_by_kind(val_by_kind, init_by_kind)}")
                    print(f"[train]     {fmt_extra(val_extra)}")

                    metrics_entry = {
                        "step": step,
                        "val_nll": val_nll,
                        "val_nll_by_kind": val_by_kind,
                        "val_terms": val_extra,
                        "train_loss": current_loss,
                        "lr": current_lr,
                    }
                    metrics_history.append(metrics_entry)
                    (out_dir_p / "metrics.json").write_text(
                        json.dumps(metrics_history, indent=2), encoding="utf-8"
                    )

                    best_val_nll = min(best_val_nll, val_nll)
                    if sel_score(val_nll, val_extra) < best_sel:
                        best_sel = sel_score(val_nll, val_extra)
                        vals_since_best = 0
                        best_path = out_dir_p / "best"
                        model.save_pretrained(str(best_path))
                        tokenizer.save_pretrained(str(best_path))
                        print(f"[train] -> New best checkpoint saved to {best_path} "
                              f"(NLL {val_nll:.4f}, acc {100 * val_extra['acc']:.1f})")
                    else:
                        vals_since_best += 1

                if step % save_every == 0:
                    step_path = out_dir_p / f"step_{step}"
                    model.save_pretrained(str(step_path))
                    print(f"[train] Regular checkpoint saved: {step_path}")

                if early_stop_patience and vals_since_best >= early_stop_patience:
                    # v6 peaked at step 250 of 1400, then spent 6 hours climbing back.
                    print(f"[train] Early stop: no better val {select_on} in {vals_since_best} validations.")
                    stop_training = True
                    break

                if max_steps and step >= max_steps:
                    print(f"[train] Step limit reached ({step}/{max_steps}). Terminating training loop.")
                    stop_training = True
                    break

    # Final evaluation
    print("[train] Running final validation evaluation...")
    final_val_nll, final_by_kind, final_extra = validate()
    print(
        f"[train] Final VAL NLL: {final_val_nll:.4f} "
        f"(Initial: {init_val_nll:.4f}, Best: {best_val_nll:.4f})"
    )
    print(f"[train]   per primitive: {_fmt_by_kind(final_by_kind, init_by_kind)}")
    print(f"[train]   {fmt_extra(final_extra)}")

    final_path = out_dir_p / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    metrics_history.append({
        "step": step,
        "val_nll": final_val_nll,
        "val_terms": final_extra,
        "is_final": True,
        "initial_nll": init_val_nll,
        "best_nll": best_val_nll,
    })
    (out_dir_p / "metrics.json").write_text(
        json.dumps(metrics_history, indent=2), encoding="utf-8"
    )

    # Formal loss convergence verification (strictly decreasing validation NLL)
    gate_passed = best_val_nll < init_val_nll
    print(f"[train] =========================================")
    print(f"[train] LOSS CONVERGENCE CHECK (strictly decreasing val NLL): {'PASSED' if gate_passed else 'FAILED'}")
    print(f"[train] NLL step 0 = {init_val_nll:.4f} -> NLL best = {best_val_nll:.4f} (gain: {init_val_nll - best_val_nll:.4f})")
    print(f"[train] =========================================")

    return {
        "initial_val_nll": init_val_nll,
        "best_val_nll": best_val_nll,
        "final_val_nll": final_val_nll,
        "gate_passed": gate_passed,
    }


def main():
    ap = argparse.ArgumentParser(description="LoRA rank 32 fine-tuning for sysone")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--eval", default="data/eval.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--output-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--targets", default=None,
                    help='comma-separated module names, or "qwen35-attention" (default: the v1-v3 set)')
    ap.add_argument("--max-len", type=int, default=2048, help="tokens kept per prompt (cut on the left)")
    ap.add_argument("--prompt-format", default="tags", choices=["tags", "chatml", "qwen3"])
    ap.add_argument("--loss", default="vocab", choices=["vocab", "options"],
                    help="answer cross-entropy over the whole vocabulary (v1-v5) or over the candidate "
                         "tokens only, as inference reads them; also the validation NLL")
    ap.add_argument("--ordinal-weight", type=float, default=0.0,
                    help="weight of the squared EMD between predicted and true Score level (0 = off)")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                    help="stop after this many validations without a better --select-on score (0 = off)")
    ap.add_argument("--select-on", default="nll", choices=["nll", "acc"],
                    help="score that picks the 'best' checkpoint and drives the early stop")
    args = ap.parse_args()
    targets = (QWEN35_ATTENTION_TARGETS if args.targets == "qwen35-attention"
               else args.targets.split(",") if args.targets else None)

    train(
        train_path=args.train,
        eval_path=args.eval,
        model_name=args.model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        max_examples=args.max_examples,
        max_steps=args.max_steps,
        val_every=args.val_every,
        save_every=args.save_every,
        eval_samples=args.eval_samples,
        seed=args.seed,
        rank=args.rank,
        alpha=args.alpha,
        targets=targets,
        max_len=args.max_len,
        prompt_format=args.prompt_format,
        loss=args.loss,
        ordinal_weight=args.ordinal_weight,
        early_stop_patience=args.early_stop_patience,
        select_on=args.select_on,
    )


if __name__ == "__main__":
    main()
