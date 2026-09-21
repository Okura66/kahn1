"""Head-to-head race: local Kahn1 against the TypeSafe JEV cloud API.

Both engines answer the same held-out items, in the same order, and are scored
against ground truth rather than against each other. Items come from the public
test splits (see scripts/build_race_set.py), so every answer has a known label
and TOTAL SCORE is measured accuracy, not a rating.

Neither engine generates text: Kahn1 reads option-token logits locally, JEV
returns typed answers from one cloud request. So this is System 1 vs System 1,
and the visible gap is local CPU inference against a hosted service — not
parallel against autoregressive. Kahn1 on CPU costs seconds per item; the same
engine on vLLM/GPU is what the model card's sub-100ms figures describe.

Each engine runs its items sequentially and emits an event as each lands, so
the two panels fill independently and the wall clock is directly comparable.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from . import jev
from .types import ChoiceQuestion, NoulQuestion, Query, ScoreQuestion

DEFAULT_RACE_SET = Path("data/race_set.json")
KEY_WIDTH = 46


# ---------------------------------------------------------------------------
# Race set
# ---------------------------------------------------------------------------

def load_items(path: str | Path = DEFAULT_RACE_SET) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"No race set at {p}. Build one first: "
            f"python scripts/build_race_set.py --total 27"
        )
    return json.loads(p.read_text(encoding="utf-8"))["items"]


def line_key(item: dict) -> str:
    """Label shown for an item: its state, elided to keep one line per item."""
    text = " ".join(item["state"].split())
    if len(text) > KEY_WIDTH:
        text = text[: KEY_WIDTH - 1] + "…"
    return text


def render_line(item: dict, payload: dict[str, Any], last: bool) -> str:
    """Format one JSON body line the way the panels display it."""
    body = ", ".join(f'"{k}": {json.dumps(v)}' for k, v in payload.items())
    return f'  "{line_key(item)}": {{{body}}}' + ("" if last else ",")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def is_correct(item: dict, pick: Any) -> bool:
    """Exact match against ground truth, per primitive."""
    if item["kind"] == "noul":
        return bool(pick) == bool(item["gold"])
    return pick == item["gold"]


@dataclass
class EngineRun:
    """Accumulated state for one side of the race."""

    name: str
    n_items: int
    correct: int = 0
    answered: int = 0
    latencies: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    started: float = field(default_factory=time.perf_counter)
    error: str | None = None
    # Per-primitive tally: a single accuracy figure hides which primitive fails.
    by_kind: dict[str, list[int]] = field(default_factory=dict)

    def record(self, item: dict, ok: bool, latency_ms: float) -> None:
        self.answered += 1
        self.correct += int(ok)
        self.latencies.append(latency_ms)
        tally = self.by_kind.setdefault(item["kind"], [0, 0])
        tally[0] += int(ok)
        tally[1] += 1

    @property
    def score(self) -> float:
        """Measured accuracy over answered items, on a 0-100 scale."""
        if not self.answered:
            return 0.0
        return 100.0 * self.correct / self.answered

    def summary(self, total_ms: float) -> dict[str, Any]:
        lat = sorted(self.latencies)
        p50 = lat[len(lat) // 2] if lat else 0.0
        return {
            "engine": self.name,
            "total_ms": round(total_ms, 1),
            "p50_ms": round(p50, 1),
            "answered": self.answered,
            "correct": self.correct,
            "n_items": self.n_items,
            "score": round(self.score, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "by_kind": {k: {"correct": v[0], "total": v[1]}
                        for k, v in sorted(self.by_kind.items())},
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Kahn1 side
# ---------------------------------------------------------------------------

def build_query(item: dict) -> Query:
    kind = item["kind"]
    if kind == "choice":
        q: Any = ChoiceQuestion(
            key="q", prompt=item["prompt"], options=item["options"],
            # The held-out benchmark scores Choice without the fallback option.
            allow_other=False,
        )
    elif kind == "score":
        q = ScoreQuestion(key="q", prompt=item["prompt"], levels=item["levels"])
    elif kind == "noul":
        q = NoulQuestion(key="q", statement=item["statement"])
    else:
        raise ValueError(f"Unsupported item kind: {item['kind']!r}")
    return Query(state=item["state"], questions=[q])


def normalize_kahn1(item: dict, answer: Any) -> dict[str, Any]:
    """Reduce a sysone Answer to {"pick", "confidence", "json"}."""
    kind = item["kind"]
    if kind == "choice":
        return {
            "pick": answer.choice,
            "confidence": answer.confidence,
            "json": {"choice": answer.choice,
                     "confidence": round(answer.confidence, 4)},
        }
    if kind == "score":
        return {
            "pick": answer.level,
            "confidence": answer.confidence,
            "json": {"score": round(answer.score, 4),
                     "confidence": round(answer.confidence, 4),
                     "level": answer.level},
        }
    if kind == "noul":
        return {
            "pick": answer.noul > 0.5,
            "confidence": max(answer.noul, 1.0 - answer.noul),
            "json": {"noul": round(answer.noul, 4)},
        }
    raise ValueError(f"Unsupported item kind: {kind!r}")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

async def _run_kahn1(items, engine, n_permutations, emit):
    loop = asyncio.get_running_loop()

    # Load weights before the clock starts. Folding a one-off multi-second model
    # load into the first item would misreport both its latency and the p50.
    backend = getattr(engine, "engine", engine)  # unwrap CalibratedEngine
    t_load = time.perf_counter()
    try:
        await loop.run_in_executor(None, backend._ensure_loaded)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        await emit({"type": "error", "engine": "kahn1", "message": msg})
        run = EngineRun(name="kahn1", n_items=len(items))
        run.error = msg
        await emit({"type": "done", **run.summary(0.0)})
        return
    await emit({
        "type": "ready", "engine": "kahn1",
        "load_ms": round((time.perf_counter() - t_load) * 1000.0, 1),
    })

    run = EngineRun(name="kahn1", n_items=len(items))
    for idx, item in enumerate(items):
        query = build_query(item)
        t0 = time.perf_counter()
        try:
            # Inference is synchronous and CPU-bound; keep the event loop free
            # so the JEV side keeps streaming while this blocks.
            resp = await loop.run_in_executor(
                None, lambda q=query: engine.evaluate(q, n_permutations=n_permutations)
            )
        except Exception as exc:
            run.error = f"{type(exc).__name__}: {exc}"
            await emit({"type": "error", "engine": "kahn1", "message": run.error})
            break
        elapsed = (time.perf_counter() - t0) * 1000.0
        norm = normalize_kahn1(item, resp.answers["q"])
        ok = is_correct(item, norm["pick"])
        run.record(item, ok, elapsed)
        await emit({
            "type": "line", "engine": "kahn1", "index": idx,
            "text": render_line(item, norm["json"], idx == len(items) - 1),
            "correct": ok, "expected": None if ok else item["gold"],
            "latency_ms": round(elapsed, 1), "score": round(run.score, 1),
        })
    total = (time.perf_counter() - run.started) * 1000.0
    await emit({"type": "done", **run.summary(total)})


async def _run_jev(items, emit, model):
    run = EngineRun(name="jev", n_items=len(items))
    if not jev.has_key():
        run.error = ("JEV_API_KEY is not set in the server environment; "
                     "the JEV side cannot run.")
        await emit({"type": "error", "engine": "jev", "message": run.error})
        await emit({"type": "done", **run.summary(0.0)})
        return

    for idx, item in enumerate(items):
        try:
            result = await jev.evaluate(
                item["state"], {"q": jev.question_payload(item)}, model=model
            )
            norm = jev.normalize(item, result.answers.get("q", {}))
        except Exception as exc:
            run.error = f"{type(exc).__name__}: {exc}"
            await emit({"type": "error", "engine": "jev", "message": run.error})
            break
        ok = is_correct(item, norm["pick"])
        run.record(item, ok, result.latency_ms)
        run.input_tokens += result.input_tokens
        run.output_tokens += result.output_tokens
        await emit({
            "type": "line", "engine": "jev", "index": idx,
            "text": render_line(item, norm["json"], idx == len(items) - 1),
            "correct": ok, "expected": None if ok else item["gold"],
            "latency_ms": round(result.latency_ms, 1), "score": round(run.score, 1),
        })
    total = (time.perf_counter() - run.started) * 1000.0
    await emit({"type": "done", **run.summary(total)})


async def race(
    items: list[dict],
    engine,
    n_permutations: int = 1,
    jev_model: str = jev.DEFAULT_MODEL,
) -> AsyncIterator[dict]:
    """Run both engines concurrently, yielding events as each item lands."""
    queue: asyncio.Queue = asyncio.Queue()

    async def emit(ev: dict) -> None:
        await queue.put(ev)

    yield {
        "type": "start",
        "n_items": len(items),
        "kinds": {k: sum(1 for i in items if i["kind"] == k)
                  for k in ("choice", "score", "noul")},
        "n_permutations": n_permutations,
        "jev_configured": jev.has_key(),
    }

    tasks = [
        asyncio.create_task(_run_kahn1(items, engine, n_permutations, emit)),
        asyncio.create_task(_run_jev(items, emit, jev_model)),
    ]
    finished = 0
    try:
        while finished < len(tasks):
            ev = await queue.get()
            if ev["type"] == "done":
                finished += 1
            yield ev
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
