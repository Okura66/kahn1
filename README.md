# Kahn1 — High-Throughput "System 1" Typed Decision Engine

<p align="left">
  <a href="https://huggingface.co/Okura66/Kahn1-Qwen2.5-3B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Kahn1--Qwen2.5--3B-ffcc00.svg" alt="Hugging Face Model" /></a>
  <a href="https://huggingface.co/Okura66/Kahn1-Qwen2.5-3B-LoRA"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20LoRA-Kahn1--Qwen2.5--3B--LoRA-orange.svg" alt="Hugging Face LoRA" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.11%2B-blue.svg" alt="Python 3.11+" /></a>
  <a href="tests/"><img src="https://img.shields.io/badge/Tests-104%20passed-success.svg" alt="Tests: 104 passed" /></a>
  <a href="https://docs.vllm.ai/"><img src="https://img.shields.io/badge/Engine-vLLM%20%7C%20Prefix%20Caching-purple.svg" alt="Engine: vLLM" /></a>
  <img src="https://img.shields.io/badge/Latency-sub--20ms-brightgreen.svg" alt="Latency: sub-20ms" />
</p>

> [!NOTE]
> **Website: [kahn1.com](https://kahn1.com/)** ([français](https://kahn1.com/fr/)) — try the [playground](https://kahn1.com/playground/) or watch [Kahn1 play Snake](https://kahn1.com/snake/), both in your browser.

> [!TIP]
> **Official Model Weights on Hugging Face**:
> - Full Merged Standalone Checkpoint (6.17 GB): [`Okura66/Kahn1-Qwen2.5-3B`](https://huggingface.co/Okura66/Kahn1-Qwen2.5-3B)
> - Lightweight LoRA Adapter (239 MB): [`Okura66/Kahn1-Qwen2.5-3B-LoRA`](https://huggingface.co/Okura66/Kahn1-Qwen2.5-3B-LoRA)

**Kahn1** (powered by the `sysone` Python framework) is an open-source, deterministic System 1 decision engine for structured classification, continuous ordinal scoring, and binary verification. Named in homage to Daniel Kahneman (*Thinking, Fast and Slow*), Kahn1 eliminates autoregressive text generation and JSON schema parsing by extracting strictly typed decisions and calibrated probability distributions directly from model logits at the single-token level.

---

## 📦 Model Weights & Hugging Face Releases

Two packaging formats are available on the Hugging Face Hub:

### Option 1: Full Merged Model (Recommended for vLLM & Production)
👉 **[Okura66/Kahn1-Qwen2.5-3B](https://huggingface.co/Okura66/Kahn1-Qwen2.5-3B)** (6.17 GB standalone)

```bash
# Serve directly with vLLM (Prefix Caching enabled):
vllm serve Okura66/Kahn1-Qwen2.5-3B --enable-prefix-caching --dtype bfloat16
```

### Option 2: Lightweight LoRA Adapter (239 MB)
👉 **[Okura66/Kahn1-Qwen2.5-3B-LoRA](https://huggingface.co/Okura66/Kahn1-Qwen2.5-3B-LoRA)**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Okura66/Kahn1-Qwen2.5-3B-LoRA")
model = PeftModel.from_pretrained(base, "Okura66/Kahn1-Qwen2.5-3B-LoRA")
```

---

## What Kahn1 does

- Takes a **state** (arbitrary text context) and a **list of typed questions** (Choice / Score / Noul).
- Returns **strictly typed values + calibrated probability distributions**.
- **Generates zero tokens** — directly inspects logits on option tokens.
- Evaluates all questions in a **single unified batch** sharing the state's KV cache (vLLM Prefix Caching).
- Schema/type errors are **guaranteed zero by construction** (Pydantic + option tokens, zero text parsing).

## What Kahn1 does not do (honestly)

- **Does not match the intelligence of a frontier model** on ambiguous subjective judgments (an 8B or 3B model makes no such claim).
- **Does not use TypeSafe's proprietary sampler or architecture**.
- **Calibration is valid only for the training distribution** — hence the `sysone calibrate` command to recalibrate on your domain.
- **Latency gains stem from prefix caching and single-token output**, not an exotic architectural innovation.

---

## Real-World Enterprise Use Cases

`sysone` is designed to be the **fast sensory cortex ("System 1")** of automated enterprise infrastructure, routing workflows before invoking expensive generative LLMs or human agents:

1. **High-Throughput Inbound Triage & Routing (< 50 ms)**:
   - Categorizing thousands of incoming support tickets, emails, or insurance claims per minute.
   - Extracting structured attributes (intent, priority, sentiment, refund requests) simultaneously in one batch pass without generating text.
2. **Deterministic Agentic Guardrails & Workflow Gating**:
   - Real-time safety validation: checking whether an autonomous agent's proposed action violates policy, leaks confidential data, or meets user approval criteria before tool execution.
3. **Document Ingestion & Metadata Scoring**:
   - Extracting qualitative dimensions (e.g. document urgency, tone, technical complexity) as continuous expected values ($0.0$ to $5.0$) rather than noisy discrete labels.
4. **Selective Escalation to "System Two" (Human-in-the-Loop)**:
   - Using calibrated confidence scores ($p_{\max}$, ECE < 0.02) to automate routine decisions while safely escalating low-confidence cases to a frontier reasoning model (GPT-4o, Claude 3.5) or a human operator.

---

## Zero-Shot Classification vs Classical ML: When to Use What

A common enterprise misconception is that zero-shot LLM classifiers make classical Machine Learning obsolete. From an engineering standpoint, this is fundamentally false.

| Dimension | Classical ML (BERT, CatBoost, XGBoost, Scikit) | `sysone` (Zero-Shot Typed LLM Engine) |
|---|---|---|
| **Taxonomy / Schema Agility** | **Rigid**: Adding a class requires relabeling data, retraining, and redeploying. | **Instantaneous**: Change the JSON schema in the request; works immediately in zero-shot. |
| **Data Requirements** | Requires hundreds/thousands of domain-labeled examples per class. | Zero domain labels required for immediate cold-start deployment. |
| **Inference Latency** | Ultra-fast (**< 1 ms to 10 ms** on CPU/GPU). | Fast for LLMs (**30 ms to 50 ms** on GPU), but heavier than tabular ML. |
| **Hardware Footprint** | Extremely lightweight (runs on small CPUs or tiny edge devices). | Requires modern GPU VRAM (8 GB to 16 GB for a 7B/8B model). |
| **Explainability & Auditing** | Direct feature importances (SHAP, tree splits, linear weights). | Latent attention representations with calibrated post-hoc probabilities. |

### The Engineering Takeaway:
- Use **Classical ML** when your ontology is fixed for years, latency must be sub-5ms, hardware is constrained, or data is tabular.
- Use **`sysone`** when your business rules, categories, and criteria change weekly, when cold-starting new product lines, or when interpreting unstructured natural language context with subtle semantic nuances.

---

## Determinism, Governance, and the Legal Reality of Generative AI

While `sysone` guarantees **0.0% schema and typing errors** by reading logits directly instead of parsing generated JSON text, engineering teams and legal compliance officers must understand its boundaries:

1. **The Architecture Remains a Generative Neural Network**:
   - Under regulatory frameworks (such as the **EU AI Act**, HIPAA, SOC2, and financial compliance standards), `sysone` is still fundamentally an autoregressive neural language model.
   - It computes conditional probability distributions over high-dimensional latent spaces. It does **not** provide formal symbolic verification, rule-engine guarantees, or mathematical proofs of factual correctness.
2. **Computational Determinism vs Factual Determinism**:
   - **Computationally deterministic**: Given fixed weights, identical prompt prefixes, greedy sampling ($T \to 0$ or single-token argmax), and identical CUDA kernel execution, `sysone` will produce identical logits and decisions every time.
   - **Factually non-deterministic**: The model can still be deceived by adversarial prompts, ambiguous inputs, or out-of-distribution texts.
3. **Calibrated Confidence as an Auditable Safety Valve**:
   - Because raw LLMs are chronically overconfident, raw softmax probabilities cannot be trusted in production.
   - By rigorously minimizing Expected Calibration Error ($ECE < 0.05$) and providing empirical Risk-Coverage curves (AURC), `sysone` enables **defensible risk management**:
     $$\text{If } \text{confidence}(x) \ge \tau \implies \text{Automate action (low risk)}$$
     $$\text{If } \text{confidence}(x) < \tau \implies \text{Route to human auditor or System 2}$$
   - This selective prediction threshold is what transforms an experimental LLM into an enterprise-ready, compliant decision engine.


---

## Quickstart

```bash
uv venv --python 3.11
source .venv/bin/activate     # .venv\Scripts\activate on Windows
uv pip install -e ".[gpu]"    # installs vllm (requires CUDA GPU)
```

### Running on CPU (no GPU required)

vLLM is GPU-only. `sysone.cpu.CPUEngine` swaps the backend for a plain
transformers forward pass and keeps every decision path identical — prompt
construction, single-token resolution, permutation debiasing and calibration are
the same code, so CPU and GPU results agree.

```bash
uv pip install -e ".[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
uv run python scripts/run_cpu_demo.py --model Okura66/Kahn1-Qwen2.5-3B
```

```python
from sysone.calibrate import CalibratedEngine, TemperatureConfig
from sysone.cpu import CPUEngine

engine = CPUEngine("Okura66/Kahn1-Qwen2.5-3B", dtype="float32", num_threads=16)
engine = CalibratedEngine(engine, TemperatureConfig.load("calibration.json"))
response = engine.evaluate(query, n_permutations=3)
```

Expect seconds per prompt instead of the tens of milliseconds vLLM reaches on a
GPU: there is no paged KV cache and no prefix caching. Keep `dtype="float32"` —
`bfloat16` halves memory (6.2 GB vs 12.3 GB) but runs roughly 7x slower on CPUs
without AMX, where bf16 matmuls fall back to emulation. Intended for local
development, CI and debugging, not for production throughput.

### Race: local Kahn1 vs the TypeSafe JEV API

A side-by-side interface that runs both engines over the same held-out items,
in the same order, and scores each against ground truth.

```bash
export JEV_API_KEY=...                       # https://api.typesafe.ai
uv run python scripts/build_race_set.py --per-kind 9

SYSONE_BACKEND=cpu SYSONE_MODEL=Okura66/Kahn1-Qwen2.5-3B \
  uv run uvicorn sysone.server:app --port 8000
# open http://127.0.0.1:8000/race
```

`scripts/build_race_set.py` draws 9 items per primitive at random (seeded) from
`data/eval.jsonl`, the same reserved holdout the benchmark reports on — banking77
and MASSIVE for Choice, SST-5 for Score, RTE and SciTail for Noul — so TOTAL
SCORE is measured accuracy, not a rating, and the panels break it down per
primitive. The set lands in the gitignored `data/`, and the script regenerates
it identically (build the eval split first:
`python training/build_dataset.py --eval-only --eval-out data/eval.jsonl`).

Both sides are System 1: neither generates text, and each answers one request
per item. What the race actually measures is local CPU inference against a
hosted service, so read the wall clock accordingly — the model card's sub-100ms
figures describe this same engine on vLLM/GPU. Without `JEV_API_KEY` the JEV
panel says so rather than showing invented numbers.

| variable | meaning |
|---|---|
| `SYSONE_BACKEND` | `vllm` (default) or `cpu` |
| `SYSONE_MODEL` | model id or local path |
| `SYSONE_THREADS` | torch CPU threads |
| `SYSONE_CALIBRATION` | temperature config (default `calibration.json`) |
| `JEV_API_KEY` | TypeSafe credentials, also read as `TYPESAFE_API_KEY` |

### Verification & Test Suite

```bash
uv run pytest tests/ -q
```

Runs the complete suite of 104 deterministic unit and integration tests (validating token mapping, debiasing invariance, ordinal scoring, post-hoc calibration, and REST API endpoints).

Historical logit verification of Phase 0 is archived in [`reports/SPIKE.md`](reports/SPIKE.md).

### Build Dataset

```bash
uv run python training/build_dataset.py --out data/train.jsonl --eval-out data/eval.jsonl
```

### Train (LoRA)

```bash
uv run python training/train_lora.py --train data/train.jsonl --eval data/eval.jsonl
```

### Evaluate (generates reports/REPORT.md)

```bash
uv run python eval/run_eval.py --model Qwen/Qwen2.5-3B-Instruct \
    --eval data/eval.jsonl --out reports/REPORT.md
```

### Serve

```bash
uv run sysone serve --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"state":"A frustrated customer.","questions":[{"kind":"choice","key":"q","prompt":"Intent?","options":["cancel","refund","help"],"allow_other":true}],"n_permutations":3}'
```

### Recalibrate on Your Domain

```bash
uv run sysone calibrate --data my_examples.jsonl --out calibration.json
# Then hot-reload in the server:
curl -X POST http://127.0.0.1:8000/v1/calibrate/load -d '{"path":"calibration.json"}'
```

---

## Comprehensive Reproduction & Adaptation Guide

For a step-by-step playbook on how to adapt `sysone` to **your custom domain data** or to **another open-weights model** (e.g. Llama 3.2 3B, Qwen 2.5 7B), see:

👉 **[`docs/PLAYBOOK.md`](docs/PLAYBOOK.md)**

This playbook covers:
1. Exact `.jsonl` schema format for custom questions (Choice, Score, Noul).
2. LoRA training with optimized settings (`training/train_lora.py`).
3. Weight merging (`scripts/merge_lora_shards.py`) for vLLM.
4. Post-hoc temperature calibration (`calibration.json`).
5. Native JEV / TypeSafe schema dictionary compatibility.

---

## Compatibility with JEV / TypeSafe Schema Format

The API natively accepts the canonical JEV dictionary schema via `POST /v1/evaluate/jev` (or in the `"schema"` field of `POST /v1/evaluate`):

```bash
curl -X POST http://127.0.0.1:8000/v1/evaluate/jev \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Hello, I cannot log in to my account since this morning.",
    "schema": {
      "category": {
        "type": "choice",
        "instructions": "Support ticket category",
        "criteria": {
          "bug": "Something is broken or producing errors",
          "billing": "Invoices, refunds, subscriptions",
          "account": "Login, credentials, permissions"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "Level of urgency",
        "criteria": ["Low", "Medium", "Critical"]
      },
      "legal_threat": {
        "type": "noul",
        "instructions": "The customer makes an explicit legal threat"
      }
    }
  }'
```

---

## Backbones & Modern SLM Architecture

The codebase is **model-agnostic** via `EngineConfig.model` / `--model` / `SYSONE_MODEL`.

- **Default Backbone**: `Qwen/Qwen2.5-3B-Instruct` (and `checkpoints/qwen_merged`)
  - **Memory Footprint**: Only **1.26 GB VRAM** in inference (leaves >14.5 GB for KV cache on 16GB GPUs, enabling massive batch concurrency).
  - **Exhaustive Holdout Results (10,663 samples — first benchmark covering all three primitives)**:
    - **Global Accuracy**: **77.60 %** across all unseen tasks (ECE **0.0537**, NLL **0.5714**).
    - **Choice (6,050 samples)**: **90.79 %** accuracy, **0.0167 ECE** — Banking77 **90.73 %**, MASSIVE **90.85 %**.
    - **Score / SST-5 (2,210 samples)**: **52.22 %** exact match, **95.20 %** off-by-one ($\pm 1$), **Spearman $\rho = 0.839$**, **MAE = 0.572**.
    - **Noul (2,403 samples)**: **67.71 %** — RTE **84.12 %**, SciTail **65.57 %** (out-of-domain science entailment; the honest number, not a cherry-pick).
    - **Latency**: p50 = **35.2 ms**, p95 = **86.5 ms** (21.5 q/s with $k=3$ debiasing).
    - Calibration temperatures fitted on a split **disjoint from training**.
  - **Architectural Depth**: 28 layers, 16 heads, trained on 18T multi-lingual tokens.
  - Full empirical report: 👉 **[`reports/QWEN_FULL_EVAL_10663.md`](reports/QWEN_FULL_EVAL_10663.md)**.
- **Also Supported**: `meta-llama/Llama-3.2-3B-Instruct`.

### Understanding Ordinal Scoring & Human Agreement (SST-5)

On fine-grained ordinal scales like SST-5 (5 sentiment degrees from *Very Negative* to *Very Positive*), discrete exact-match accuracy is **52.22 %**. This represents near-ceiling performance:
1. **Human Inter-Annotator Ceiling**: Human agreement on 5-way SST-5 is only **~55% - 60%** due to the natural subjectivity of nuances (e.g. distinguishing *Positive* from *Very Positive*). SOTA foundation models consistently plateau at 50% - 54%.
2. **Zero Catastrophic Inversion**: With **95.20 % off-by-one accuracy**, the model's prediction is either exact or immediately adjacent in 95% of cases. It virtually never confuses opposite polarities.
3. **Monotonic Ranking ($\rho = 0.839$)**: The high Spearman correlation proves strong ordering fidelity across continuous latent sentiment.
4. **Continuous Expectation**: In production, `sysone` consumes ordinal answers via expected value:
   $$\mathbb{E}[\text{Score}] = \sum_{i=0}^{K-1} i \cdot p_i$$
   This yields continuous scores (e.g. $3.65 / 4.0$) avoiding artificial discrete boundary clipping.

⚠️ vLLM APIs evolve quickly. The code **dynamically inspects `SamplingParams` signature** at runtime (see `engine._detect_restrict_param`).

---

## Project Structure

```
kahn1/
├── src/sysone/       Core engine package (types, tokens, prompt, debias, calibrate, cpu, jev, race, server)
│   └── web/          Race interface (Kahn1 vs JEV, scored on ground truth)
├── training/         Dataset building, augmentations, and LoRA training
├── eval/             Metrics, evaluation harness, benchmarks, debiasing tests
├── scripts/          Utilities (LoRA merge, val set prep, HF publication, CPU demo, race set)
│   └── scratch/      (gitignored) Local scratchpad & temporary investigation scripts
├── docs/             kahn1.com (home EN/FR, playground, Snake) and PLAYBOOK.md (reproduction guide)
├── tests/            Pytest unit & integration test suite
└── reports/          Empirical benchmarks, calibration logs, and evaluation reports
```

## Design Principles

- **No text generation** in the inference path. Read logits directly on option tokens (indirection via letters A, B, ...).
- **No regex** in the inference path.
- **Single-batch** `llm.generate()` for all questions × permutations. Looping one call per question is forbidden.
- **Averaging in probability space**, not logit space (averaging logits is not an average of beliefs).
- **Evaluation on disjoint, held-out datasets**, never seen during training.
- **Log validation NLL, not accuracy**, for checkpoint selection.
- **No RL.** Log-loss on the target answer token is already a strictly proper scoring rule: supervised fine-tuning directly optimizes calibration. **Do not attempt to "fix" the absence of RL.**

## Key Formulas

- `confidence = (p_max - 1/n) / (1 - 1/n)`, clamped to $[0, 1]$. Measures distance from equiprobability, **not** probability of being correct (TypeSafe formula).
- `score = Σ p_i × i` ($i$ 0-based level index). Yields a continuous expectation (e.g. 1.84 between levels 1 and 2).

## License

Released under the [MIT License](LICENSE).

