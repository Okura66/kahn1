"""FastAPI evaluation server exposing batched inference endpoints.

Endpoints:
    POST /v1/evaluate
    { "state": "...", "questions": [...], "n_permutations": 3 }
    -> { "answers": {...}, "latency_ms": 87, "cache_hit_rate": 0.94 }

Exposes prefix cache hit rate in the response payload to monitor KV cache sharing.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
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
    global _engine
    if _engine is None:
        import os
        from pathlib import Path
        default_model = "checkpoints/qwen_merged" if Path("checkpoints/qwen_merged").exists() else "Qwen/Qwen2.5-3B-Instruct"
        cfg = EngineConfig(
            model=os.environ.get("SYSONE_MODEL", default_model),
        )
        _engine = Engine(cfg)
    return _engine


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
            detail="La requête doit spécifier soit 'questions' (liste sysone) soit 'schema' (dictionnaire JEV)."
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
                detail="vLLM n'est pas disponible dans cet environnement Python (vLLM requiert Linux/WSL2 avec CUDA)."
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
# Battle Arena Routes: Kahn1 vs Gemini Flash
# ---------------------------------------------------------------------------

from fastapi.responses import HTMLResponse
from pathlib import Path
from .arena import SHOWCASE_PRESETS, SCHEMA_BATCH_PRESETS, get_dataset_index, run_battle, run_schema_battle


class ArenaBattleRequest(BaseModel):
    state: str
    prompt: str = "Classification :"
    options: list[str]
    preset_id: str | None = None
    sample_index: int | None = None
    ground_truth: str | None = None
    gemini_model: str = "gemini-3.5-flash-lite"
    include_jev: bool = True


class ArenaSchemaBattleRequest(BaseModel):
    state: str
    questions: dict[str, Any]
    preset_id: str | None = None
    multiplier: int = 1
    include_jev: bool = True
    model_name: str = "gemini-3.5-flash-lite"


@app.get("/", response_class=HTMLResponse)
@app.get("/arena", response_class=HTMLResponse)
async def serve_arena():
    """Serves the visual comparison web interface."""
    html_path = Path(__file__).parent / "web" / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Fichier arena introuvable</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/arena/presets")
async def get_presets():
    """Returns pre-configured demonstration scenarios."""
    return SHOWCASE_PRESETS


@app.get("/api/arena/sample")
async def get_sample(dataset: str | None = None):
    """Retrieves a random sample from holdout dataset examples."""
    idx = get_dataset_index()
    return idx.get_random_sample(dataset=dataset)


@app.get("/api/arena/batch/draw")
async def draw_batch(n: int = 100, dataset: str | None = None, kahn1_filter: str | None = None):
    """Draws n random items (default 100) for visual grid evaluation."""
    idx = get_dataset_index()
    return idx.draw_batch(n=n, dataset=dataset, kahn1_filter=kahn1_filter)


@app.post("/api/arena/batch/eval_gemini")
async def batch_eval_gemini(item: dict, model_name: str = "gemini-3.5-flash-lite"):
    """Evaluates an individual item using the Gemini model in the benchmark arena."""
    from .arena import eval_gemini_batch_item
    return await eval_gemini_batch_item(item, model_name=model_name)


@app.post("/api/arena/batch/eval_jev")
async def batch_eval_jev(item: dict):
    """Evaluates an individual item using the TypeSafe JEV model in the benchmark arena."""
    from .arena import eval_jev_batch_item
    return await eval_jev_batch_item(item)


@app.post("/api/arena/battle")
async def arena_battle(req: ArenaBattleRequest):
    """Executes a comparative benchmark run between Kahn1, Gemini Flash, and TypeSafe JEV."""
    result = await run_battle(
        state=req.state,
        prompt=req.prompt,
        options=req.options,
        preset_id=req.preset_id,
        sample_index=req.sample_index,
        ground_truth=req.ground_truth,
        gemini_model=req.gemini_model,
        include_jev=req.include_jev,
    )
    return result.__dict__


@app.get("/api/arena/schema_presets")
async def get_schema_presets():
    """Returns multi-question batch showcase presets."""
    return SCHEMA_BATCH_PRESETS


@app.get("/api/arena/schema_random")
async def get_random_schema():
    """Returns a random multi-question batch scenario across all domains."""
    from .arena import get_random_schema_preset
    return get_random_schema_preset()


@app.post("/api/arena/schema_battle")
async def arena_schema_battle(req: ArenaSchemaBattleRequest):
    """Executes a multi-question batch comparative evaluation."""
    result = await run_schema_battle(
        state=req.state,
        questions=req.questions,
        preset_id=req.preset_id,
        multiplier=req.multiplier,
        include_jev=req.include_jev,
        model_name=req.model_name,
    )
    return result.__dict__

