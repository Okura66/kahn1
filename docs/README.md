# Kahn1 playground

Static page (`index.html` + `samples.json`), published by GitHub Pages
(branch `main`, folder `/docs`) on https://demo.kahn1.com/. `demo/` only redirects
old links to the root.

Two backends, toggled from the page:

- **server** — the sysone FastAPI server. Default URL `http://localhost:8000`,
  editable in the page. CORS allows `localhost` and `*.github.io` origins
  (override with `SYSONE_CORS_ORIGIN_REGEX`).
- **webgpu** — model in the browser tab, no server. [wllama](https://github.com/ngxson/wllama)
  (llama.cpp → WASM + WebGPU) loads `mradermacher/Kahn1-Qwen2.5-3B-GGUF` from the Hub
  (Q4_K_M / IQ4_XS / Q3_K_M; Q2_K outputs noise). Weights are cached per site in the
  browser; the CLEAR button in the page deletes them. That GGUF is **v1**, whose Noul
  answers are inverted: `GGUF.noulInverted` flips them — set it to `false` for a v3 GGUF.
  Without a usable WebGPU adapter it runs on the WASM CPU backend; if WebGPU fails to
  load the model, the page retries with `n_gpu_layers: 0` and the chip reads `wasm cpu (fallback)`.

## Run the server backend (WSL2, vLLM)

```bash
cd /mnt/c/dev/OpenJEV
SYSONE_MODEL=checkpoints/qwen_merged PYTHONPATH=src \
  ~/.venvs/sysone/bin/python -m uvicorn sysone.server:app --host 0.0.0.0 --port 8000
```

For the webgpu backend alone, any static server works:
`python -m http.server 8000 --directory docs`.

Then open `http://localhost:8000/demo/` (the sysone server serves `docs/` there),
`http://localhost:8000/` (static server), or https://demo.kahn1.com/. The model loads on the first run.
