"""Client for the TypeSafe JEV System 1 API (https://api.typesafe.ai/v1/systemone).

JEV is a cloud System 1 decision engine: one request returns typed answers for
every question about a state, with no autoregressive generation. Its response
shape mirrors sysone's own primitives, which makes a direct comparison possible.

    choice -> {"type": "choice", "choice": str, "confidence": float,
               "probabilities": {option: p}}
    score  -> {"type": "score", "score": float, "confidence": float,
               "legend": {"0": level, ...}, "probabilities": {"0": p, ...}}
    noul   -> {"type": "noul", "noul": float}

The key is read from JEV_API_KEY (or TYPESAFE_API_KEY). When it is missing this
module raises rather than fabricating a plausible-looking answer: a demo that
invents latencies and confidences is worse than one that admits it cannot run.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"


class JevError(RuntimeError):
    """Raised when JEV cannot be reached or answers with a non-200 status."""


class JevKeyMissing(JevError):
    """Raised when no API key is configured in the environment."""


def api_key() -> str:
    key = os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY") or ""
    if not key:
        raise JevKeyMissing(
            "No JEV credentials found. Set JEV_API_KEY (or TYPESAFE_API_KEY) "
            "in the environment before running the race."
        )
    return key


def has_key() -> bool:
    return bool(os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY"))


@dataclass
class JevResult:
    """One JEV round trip: the parsed answers plus what it cost to obtain them."""

    answers: dict[str, Any]
    latency_ms: float
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


def question_payload(item: dict) -> dict[str, Any]:
    """Translate one race item into a JEV question specification."""
    kind = item["kind"]
    if kind == "choice":
        return {
            "type": "choice",
            "instructions": item["prompt"],
            # JEV expects {name: description}; the option text is its own description.
            "criteria": {opt: opt for opt in item["options"]},
        }
    if kind == "score":
        return {
            "type": "score",
            "instructions": item["prompt"],
            "criteria": list(item["levels"]),
        }
    if kind == "noul":
        return {"type": "noul", "instructions": item["statement"]}
    raise ValueError(f"Unsupported item kind: {kind!r}")


async def evaluate(
    state: str,
    questions: dict[str, dict[str, Any]],
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
) -> JevResult:
    """POST one state and its questions, and time the full round trip."""
    import httpx

    payload = {"model": model, "state": state, "questions": questions}
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(API_URL, headers=headers, json=payload)
    except Exception as exc:  # network, DNS, timeout
        raise JevError(f"JEV request failed: {type(exc).__name__}: {exc}") from exc
    elapsed = (time.perf_counter() - t0) * 1000.0

    if resp.status_code != 200:
        raise JevError(f"JEV returned HTTP {resp.status_code}: {resp.text[:300]}")

    body = resp.json()
    usage = body.get("usage") or {}
    return JevResult(
        answers=body.get("answers", {}),
        latency_ms=elapsed,
        model=body.get("model", ""),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        raw=body,
    )


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------

def normalize(item: dict, answer: dict[str, Any]) -> dict[str, Any]:
    """Reduce a JEV answer to the shape the race scoreboard consumes.

    Returns {"pick": <comparable to item['gold']>, "confidence": float, "json": dict}.
    """
    kind = item["kind"]
    if kind == "choice":
        return {
            "pick": answer.get("choice"),
            "confidence": float(answer.get("confidence", 0.0)),
            "json": {
                "choice": answer.get("choice"),
                "confidence": round(float(answer.get("confidence", 0.0)), 4),
            },
        }
    if kind == "score":
        levels = item["levels"]
        probs = answer.get("probabilities") or {}
        if probs:
            # Keys are stringified level indices.
            best = max(probs, key=lambda k: probs[k])
            idx = int(best)
        else:
            idx = int(round(float(answer.get("score", 0.0))))
        idx = max(0, min(len(levels) - 1, idx))
        return {
            "pick": levels[idx],
            "confidence": float(answer.get("confidence", 0.0)),
            "json": {
                "score": round(float(answer.get("score", 0.0)), 4),
                "confidence": round(float(answer.get("confidence", 0.0)), 4),
                "level": levels[idx],
            },
        }
    if kind == "noul":
        p_yes = float(answer.get("noul", 0.0))
        return {
            "pick": p_yes > 0.5,
            "confidence": max(p_yes, 1.0 - p_yes),
            "json": {"noul": round(p_yes, 4)},
        }
    raise ValueError(f"Unsupported item kind: {kind!r}")
