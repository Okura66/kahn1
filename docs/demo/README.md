# Kahn1 playground

Static page (`index.html` + `samples.json`), publishable as-is on GitHub Pages
(Settings → Pages → branch `main`, folder `/docs` → `https://okura66.github.io/kahn1/demo/`).

Two backends, toggled from the page:

- **server** — the sysone FastAPI server. Default URL `http://localhost:8000`,
  editable in the page. CORS allows `localhost` and `*.github.io` origins
  (override with `SYSONE_CORS_ORIGIN_REGEX`).
- **webgpu** — model in the browser tab, no server. [wllama](https://github.com/ngxson/wllama)
  (llama.cpp → WASM + WebGPU) loads `mradermacher/Kahn1-Qwen2.5-3B-GGUF` from the Hub
  (Q4_K_M / IQ4_XS / Q3_K_M; Q2_K outputs noise). Weights are cached per site in the
  browser; the CLEAR button in the page deletes them. That GGUF is **v1**, whose Noul
  answers are inverted: `GGUF.noulInverted` flips them — set it to `false` for a v3 GGUF.

## Run the server backend (WSL2, vLLM)

```bash
cd /mnt/c/dev/OpenJEV
SYSONE_MODEL=checkpoints/qwen_merged PYTHONPATH=src \
  ~/.venvs/sysone/bin/python -m uvicorn sysone.server:app --host 0.0.0.0 --port 8000
```

For the webgpu backend alone, any static server works:
`python -m http.server 8000 --directory docs`.

Then open `http://localhost:8000/demo/` (served by the server itself) or the
GitHub Pages URL. The model loads on the first run.
