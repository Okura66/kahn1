"""FastAPI evaluation server exposing batched inference endpoints.

Endpoints:
    POST /v1/evaluate
    { "state": "...", "questions": [...], "n_permutations": 3 }
    -> { "answers": {...}, "latency_ms": 87, "cache_hit_rate": 0.94 }

Exposes prefix cache hit rate in the response payload to monitor KV cache sharing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .engine import Engine, EngineConfig
from .types import Answer, EvaluateResponse, Query


# -- Request Models --
class EvaluateRequest(BaseModel):
    state: str
    questions: list[dict[str, Any]] | None = None
    schema_jev: dict[str, Any] | None = Field(default=None, alias="schema")
    n_permutations: int = 3

    model_config = {"populate_by_name": True}


class JevEvaluateRequest(BaseModel):
    state: str
    schema_jev: dict[str, Any] = Field(..., alias="schema")
    n_permutations: int = 3

    model_config = {"populate_by_name": True}


# -- Global Engine Instance (initialized on startup) --
_engine: Engine | None = None
_calibrated: Any = None


def get_engine() -> Engine:
    """Return the shared engine, honouring SYSONE_BACKEND ('vllm' by default, or 'cpu').

    The CPU backend lets the server run without a GPU; see sysone.cpu.
    """
    global _engine
    if _engine is None:
        import os
        from pathlib import Path
        default_model = "checkpoints/qwen_merged" if Path("checkpoints/qwen_merged").exists() else "Qwen/Qwen2.5-3B-Instruct"
        model = os.environ.get("SYSONE_MODEL", default_model)
        if os.environ.get("SYSONE_BACKEND", "vllm").lower() == "cpu":
            from .cpu import CPUEngine
            _engine = CPUEngine(
                model=model,
                dtype=os.environ.get("SYSONE_DTYPE", "float32"),
                num_threads=int(os.environ.get("SYSONE_THREADS", "0")) or None,
            )
        else:
            _engine = Engine(EngineConfig(model=model))
    return _engine


def get_runner():
    """Engine wrapped in temperature calibration when SYSONE_CALIBRATION points at a file."""
    global _calibrated
    if _calibrated is not None:
        return _calibrated
    import os
    from pathlib import Path
    path = Path(os.environ.get("SYSONE_CALIBRATION", "calibration.json"))
    if path.exists():
        from .calibrate import CalibratedEngine, TemperatureConfig
        _calibrated = CalibratedEngine(get_engine(), TemperatureConfig.load(path))
        return _calibrated
    return get_engine()


app = FastAPI(title="sysone", version="0.1.0")


@app.on_event("startup")
async def _load():
    # Lazy model loading: model is eagerly loaded only if SYSONE_EAGER=1,
    # otherwise deferred until first invocation (useful in test mode).
    import os
    if os.environ.get("SYSONE_EAGER") == "1":
        get_engine()._ensure_loaded()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    global _calibrated
    # Direct support for JEV schema format or sysone questions format
    if req.schema_jev is not None:
        query = Query.from_jev(state=req.state, schema=req.schema_jev)
    elif req.questions is not None:
        query = Query(state=req.state, questions=req.questions)
    else:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=422,
            detail="Request must provide either 'questions' (sysone list) or 'schema' (JEV dict)."
        )

    try:
        engine = get_engine()
        if _calibrated is not None:
            return _calibrated.evaluate(query, n_permutations=req.n_permutations)
        return engine.evaluate(query, n_permutations=req.n_permutations)
    except ModuleNotFoundError as e:
        if "vllm" in str(e).lower():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="vLLM is unavailable in this Python environment (it requires Linux/WSL2 with CUDA). "
                       "Set SYSONE_BACKEND=cpu to run without a GPU."
            )
        raise


@app.post("/v1/evaluate/jev", response_model=EvaluateResponse)
async def evaluate_jev(req: JevEvaluateRequest) -> EvaluateResponse:
    """Dedicated endpoint directly accepting JEV / TypeSafe schema requests."""
    global _calibrated
    query = Query.from_jev(state=req.state, schema=req.schema_jev)
    engine = get_engine()
    if _calibrated is not None:
        return _calibrated.evaluate(query, n_permutations=req.n_permutations)
    return engine.evaluate(query, n_permutations=req.n_permutations)



@app.post("/v1/calibrate/load")
async def load_calibration(path: str):
    """Loads a temperature scaling configuration file (TemperatureConfig JSON)."""
    global _calibrated
    from .calibrate import TemperatureConfig, CalibratedEngine
    cfg = TemperatureConfig.load(path)
    _calibrated = CalibratedEngine(get_engine(), cfg)
    return {"status": "loaded", "config": cfg.__dict__}


# ---------------------------------------------------------------------------
# Race: local Kahn1 vs the TypeSafe JEV cloud API
# ---------------------------------------------------------------------------

def _web_page(name: str) -> HTMLResponse:
    page = Path(__file__).parent / "web" / name
    if not page.exists():
        return HTMLResponse(f"<h1>{name} not found</h1>", status_code=404)
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/race", response_class=HTMLResponse)
async def race_page():
    """Serve the side-by-side race interface."""
    return _web_page("race.html")


@app.get("/api/race/items")
async def race_items(path: str | None = None):
    """Describe the race set without running anything."""
    from . import jev, race as race_mod

    try:
        items = race_mod.load_items(path or race_mod.DEFAULT_RACE_SET)
    except FileNotFoundError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=str(exc))
    return {
        "n_items": len(items),
        "kinds": {k: sum(1 for i in items if i["kind"] == k)
                  for k in ("choice", "score", "noul")},
        "sources": sorted({i["source"] for i in items}),
        "jev_configured": jev.has_key(),
        "backend": os.environ.get("SYSONE_BACKEND", "vllm"),
        "model": os.environ.get("SYSONE_MODEL", ""),
    }


@app.get("/api/race/stream")
async def race_stream(path: str | None = None, n_permutations: int = 1,
                      limit: int | None = None, mode: str = "sequential",
                      concurrency: int = 8):
    """Stream race events as server-sent events, one per answered item.

    mode: "sequential" (per-item latency) or "batch" (throughput: Kahn1 one vLLM
    batch, JEV bounded-concurrency requests). concurrency bounds the JEV batch mode.
    """
    from fastapi.responses import StreamingResponse

    from . import race as race_mod

    try:
        items = race_mod.load_items(path or race_mod.DEFAULT_RACE_SET)
    except FileNotFoundError as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=str(exc))
    if limit:
        items = items[:limit]

    async def events():
        try:
            async for ev in race_mod.race(items, get_runner(),
                                          n_permutations=n_permutations,
                                          mode=mode, concurrency=concurrency):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as exc:  # surface failures in the page instead of hanging
            payload = {"type": "fatal", "message": f"{type(exc).__name__}: {exc}"}
            yield f"data: {json.dumps(payload)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------
# Demo: interactive playground
#
# The page itself is static (docs/, published on GitHub Pages) and talks to
# this server over CORS, or runs the model in the browser with WebGPU. /demo
# serves the same files locally so the page also works without Pages.
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parents[2] / "docs"

# Origins allowed to call the API from a browser: local pages, GitHub Pages and
# the playground's own domain. Override with SYSONE_CORS_ORIGIN_REGEX.
_CORS_REGEX = os.environ.get(
    "SYSONE_CORS_ORIGIN_REGEX",
    r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://[\w-]+\.github\.io|https://([\w-]+\.)?kahn1\.com",
)

from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(CORSMiddleware, allow_origin_regex=_CORS_REGEX,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def _private_network_access(request, call_next):
    """Chrome asks before a public https page (GitHub Pages) may reach localhost."""
    response = await call_next(request)
    if request.headers.get("access-control-request-private-network") == "true":
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    if request.url.path.startswith("/demo"):
        # Revalidate on every load so an edited page is never served stale.
        response.headers["Cache-Control"] = "no-cache"
    return response


class DemoRequest(BaseModel):
    state: str
    schema_jev: dict[str, Any] = Field(..., alias="schema")
    n_permutations: int = 1

    model_config = {"populate_by_name": True}


@app.get("/api/demo/info")
async def demo_info():
    """Which model the server backend runs, without loading it."""
    engine = get_engine()
    return {
        "backend": os.environ.get("SYSONE_BACKEND", "vllm"),
        "model": engine.config.model,
        "calibrated": Path(os.environ.get("SYSONE_CALIBRATION", "calibration.json")).exists(),
        "loaded": engine._llm is not None,
    }


@app.post("/api/demo/evaluate")
def demo_evaluate(req: DemoRequest):
    """Evaluate one playground query. Sync on purpose: FastAPI runs it in a thread,
    so a multi-second first model load does not block the event loop."""
    import time

    from fastapi import HTTPException

    try:
        query = Query.from_jev(state=req.state, schema=req.schema_jev)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    runner = get_runner()
    engine = getattr(runner, "engine", runner)  # unwrap CalibratedEngine
    t0 = time.perf_counter()
    try:
        engine._ensure_loaded()
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"{type(exc).__name__}: {exc}")
    load_ms = (time.perf_counter() - t0) * 1000.0
    try:
        resp = runner.evaluate(query, n_permutations=req.n_permutations)
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {**resp.model_dump(), "load_ms": round(load_ms, 1),
            "kinds": {q.key: q.kind for q in query.questions}}


if DEMO_DIR.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/demo", StaticFiles(directory=DEMO_DIR, html=True), name="demo")
