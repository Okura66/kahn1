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
