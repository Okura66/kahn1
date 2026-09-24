# kahn1.com

Static site published by GitHub Pages (branch `main`, folder `/docs`, custom domain
in `CNAME`) on https://kahn1.com/. No build step: every page is plain HTML.

| path | file | what it is |
|---|---|---|
| `/` | `index.html` | home page, English: the project, use cases, caveats, benchmark results |
| `/fr/` | `fr/index.html` | the same home page in French |
| `/benchmarks/` | `benchmarks/index.html` | every benchmark: Kahn1 vs JEV, held-out details, Snake, latency |
| `/fr/resultats/` | `fr/resultats/index.html` | the same in French |
| `/get-started/` | `get-started/index.html` | the guide: AI-agent setup prompt, install, serve, API, calibration |
| `/fr/demarrer/` | `fr/demarrer/index.html` | the same guide in French |
| `/playground/` | `playground/index.html` + `samples.json` | the playground |
| `/snake/` | `snake/index.html` | Kahn1 plays Snake |
| `/demo/` | `demo/index.html` | redirects old links to `/playground/` |
| 404 | `404.html` | Pages serves it at any missing path (absolute links only) |

The two home pages share `assets/site.css` and `assets/site.js` (consent banner, view
counter, the playground and Snake edge tabs). The playground and Snake keep their CSS
inline. `sitemap.xml`, `robots.txt`, `llms.txt` and the share images `assets/og.png` /
`assets/og-fr.png` go with the home pages. EN and FR carry the same facts: when a number
changes, change it in both pages, in their JSON-LD blocks, and in `llms.txt`.

`get-started/setup-prompt.md` and `fr/demarrer/setup-prompt.md` are the setup prompts the guide shows
and links to (Claude Code `claude-cli://open?q=`, Claude Desktop `claude://code/new?q=`,
Claude Code on the web `claude.ai/code?prompt=`); the links embed the prompt, so edit the file
and rebuild the page together. The Claude Code link caps `q` at 5,000 characters.

The animation on the home pages (the logit reader) replays recorded runs of the v3
checkpoint (CPU, temperature calibration, k = 1), not live calls; its data sits in the
page's `reader-data` block. The Kahn1 vs JEV numbers come from
`reports/jev_vs_kahn1.json` (`scripts/jev_holdout.py`) and `reports/jevbench_vs_jev.json`
(`scripts/jevbench_vs_jev.py`).

## Playground and Snake backends

Two backends, toggled from the page:

- **server** — the sysone FastAPI server. Default URL `http://localhost:8000`,
  editable in the page. CORS allows `localhost`, `127.0.0.1` and `https://kahn1.com`
  only, with no wildcard (override with `SYSONE_CORS_ORIGIN_REGEX`).
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

Then open `http://localhost:8000/demo/playground/` (the sysone server serves `docs/`
under `/demo`), `http://localhost:8000/playground/` (static server), or
https://kahn1.com/playground/. The model loads on the first run. Links between pages
are relative, so the site works under `/demo/` too.
