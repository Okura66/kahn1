Set up Kahn1 on this machine and prove it works with one real request.

Kahn1 (https://github.com/Okura66/kahn1, MIT) is an open-source decision engine: a fine-tuned Qwen2.5-3B that answers typed questions (choice, score, noul) about a text by reading the probabilities of option tokens. It is served by a FastAPI app, `sysone`. Guide: https://kahn1.com/get-started/

Rules: work step by step and show each command before you run it. Ask me before using sudo, installing system packages, or changing anything outside the project folder. If a step fails, show the exact error, explain the likely cause and propose a fix before going on. Never invent output: if the model could not run, say so.

1. Inspect the machine and tell me what you find: OS (Linux, macOS, Windows, WSL2), Python version (3.11+ needed), free RAM and disk space, and whether an NVIDIA GPU is usable (`nvidia-smi`). Then pick the backend and say why:
   - GPU (vLLM): only on Linux or WSL2 with an NVIDIA GPU and CUDA, 8 GB of VRAM or more.
   - CPU (transformers) otherwise: about 13 GB of free RAM, a few seconds per question.
   The model download is about 6 GB either way.

2. Install uv if it is missing (https://docs.astral.sh/uv/). If this folder is not already a clone of the repository, clone it:
   git clone https://github.com/Okura66/kahn1 && cd kahn1
   uv venv --python 3.11
   GPU: uv pip install -e ".[gpu]"
   CPU: uv pip install -e ".[cpu]" --extra-index-url https://download.pytorch.org/whl/cpu
   The model's tokenizer needs transformers 5 or newer. Check with
   uv run python -c "import transformers; print(transformers.__version__)"
   and run `uv pip install -U transformers` if it prints 4.x.

3. Start the server in the background and keep its log:
   GPU: SYSONE_MODEL=Okura66/Kahn1-Qwen2.5-3B uv run sysone serve --port 8000
   CPU: SYSONE_BACKEND=cpu SYSONE_MODEL=Okura66/Kahn1-Qwen2.5-3B uv run sysone serve --port 8000
   On Windows PowerShell, set each variable first with $env:NAME = "value". Poll GET http://127.0.0.1:8000/health until it returns {"status": "ok"}.

4. POST this JSON to http://127.0.0.1:8000/v1/evaluate/jev (write it to a file and send it with curl -d @file, or use Python). The first call loads the model and can take minutes:
   {"state": "Hi, I was charged twice for order #48213. I want a refund, otherwise I will dispute the charge with my bank.",
    "schema": {
      "intent": {"type": "choice", "instructions": "What is the customer's main request?",
                 "criteria": {"refund": "the customer wants their money back", "delivery": "question about a delivery", "account": "cannot access their account"}},
      "urgency": {"type": "score", "instructions": "How urgent is this message?", "criteria": ["low", "medium", "high", "critical"]},
      "churn_risk": {"type": "noul", "instructions": "The customer threatens to leave the service or to escalate."}},
    "n_permutations": 1}
   Show me the JSON answer and check it: intent has a choice and probabilities that sum to 1, urgency has a level and a score, churn_risk is a number between 0 and 1. Report latency_ms, and send the same request a second time to show the warm latency.

5. Finish with a short summary: backend used, install path, how to stop and restart the server, and where to go next (https://kahn1.com/get-started/ for the API, the response format, tuning and calibration on my own data).
