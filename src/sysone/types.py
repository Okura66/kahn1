"""Pydantic schemas and response contracts for sysone.

Question primitives:
    - ChoiceQuestion: categorical selection over mutually exclusive discrete options.
    - ScoreQuestion: ordinal classification over discrete ascending levels.
    - NoulQuestion: binary evaluation of a factual or logical proposition.

Answer structures:
    - ChoiceAnswer: discrete choice, full probability distribution, and normalized confidence.
    - ScoreAnswer: categorical argmax level, continuous expectations (0-based and 1-based),
      distribution, and normalized confidence.
    - NoulAnswer: calibrated continuous scalar belief in [0, 1].

Guarantees:
    Type safety is guaranteed by construction: outputs are synthesized directly
    from constrained token logits over pre-resolved option vocabularies, eliminating
    arbitrary text generation, regex extraction, and schema parsing errors.

Mathematical formulas:
    - Normalized confidence:
        confidence = max(0.0, min(1.0, (p_max - 1/n) / (1 - 1/n)))
      Quantifies deviation from the uniform distribution, invariant to cardinality n.
      Zero indicates maximum entropy; one indicates degenerate probability mass.
    - Continuous score expectation:
        E[score] = sum(p_i * i)  (0-based expectation)
        E[expected_score] = sum(p_i * (i + 1))  (1-based expectation)
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Question Primitives
# ---------------------------------------------------------------------------

class ChoiceQuestion(BaseModel):
    """Categorical question with discrete candidate options."""

    kind: Literal["choice"] = "choice"
    key: str
    prompt: str
    options: list[str] = Field(..., min_length=2, max_length=64)
    allow_other: bool = True

    @field_validator("options")
    @classmethod
    def _unique_options(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("Les options doivent être uniques.")
        return v


class ScoreQuestion(BaseModel):
    """Ordinal question with discrete ranked levels."""

    kind: Literal["score"] = "score"
    key: str
    prompt: str
    levels: list[str] = Field(..., min_length=2, max_length=64)

    @field_validator("levels")
    @classmethod
    def _unique_levels(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("Les niveaux doivent être uniques.")
        return v


class NoulQuestion(BaseModel):
    """Binary assessment question (yes/no or true/false proposition)."""

    kind: Literal["noul"] = "noul"
    key: str
    statement: str


Question = Union[ChoiceQuestion, ScoreQuestion, NoulQuestion]


class Query(BaseModel):
    """Atomic evaluation batch containing a shared context state and target questions."""

    state: str = Field(..., min_length=1)
    questions: list[Question] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_keys(self) -> "Query":
        keys = [q.key for q in self.questions]
        if len(set(keys)) != len(keys):
            raise ValueError("Les clés de questions doivent être uniques.")
        return self

    @classmethod
    def from_jev(
        cls,
        state: str,
        schema: dict[str, Any],
    ) -> "Query":
        """Build a canonical Query instance from a JEV / TypeSafe dictionary schema.

        Supports the three canonical primitive types:
        - choice: expects 'criteria' (dict or list) or 'options'
        - score: expects 'criteria' (list) or 'levels'
        - noul: expects 'instructions' or 'statement'
        """
        questions: list[Question] = []
        for key, spec in schema.items():
            if not isinstance(spec, dict):
                raise ValueError(
                    f"Spécification invalide pour la clé {key!r} : attendu dict, reçu {type(spec).__name__}"
                )

            q_type = spec.get("type") or spec.get("kind")
            instructions = spec.get("instructions") or spec.get("prompt") or ""
            criteria = spec.get("criteria")

            if q_type == "choice":
                if isinstance(criteria, dict):
                    options = [f"{lbl}: {desc}" if desc else lbl for lbl, desc in criteria.items()]
                elif isinstance(criteria, list):
                    options = [str(opt) for opt in criteria]
                elif "options" in spec:
                    options = [str(opt) for opt in spec["options"]]
                else:
                    raise ValueError(f"La question choice {key!r} doit contenir 'criteria' ou 'options'.")

                allow_other = spec.get("allow_other", True)
                questions.append(
                    ChoiceQuestion(
                        key=key,
                        prompt=instructions,
                        options=options,
                        allow_other=allow_other,
                    )
                )

            elif q_type == "score":
                if isinstance(criteria, list):
                    levels = [str(lvl) for lvl in criteria]
                elif "levels" in spec:
                    levels = [str(lvl) for lvl in spec["levels"]]
                else:
                    raise ValueError(f"La question score {key!r} doit contenir 'criteria' ou 'levels'.")

                questions.append(
                    ScoreQuestion(
                        key=key,
                        prompt=instructions,
                        levels=levels,
                    )
                )

            elif q_type == "noul":
                statement = instructions or spec.get("statement", "")
                if not statement:
                    raise ValueError(f"La question noul {key!r} doit spécifier 'instructions' ou 'statement'.")
                questions.append(
                    NoulQuestion(
                        key=key,
                        statement=statement,
                    )
                )

            else:
                raise ValueError(
                    f"Type de question JEV non supporté pour {key!r} : {q_type!r}. Attendu: choice, score, noul."
                )

        return cls(state=state, questions=questions)


# ---------------------------------------------------------------------------
# Answer Primitives
# ---------------------------------------------------------------------------

def confidence_from_probs(probs: list[float]) -> float:
    """Compute normalized confidence invariant to option cardinality n.

    Formula:
        confidence = max(0.0, min(1.0, (p_max - 1/n) / (1 - 1/n)))

    Measures dispersion relative to the uniform distribution (equiprobability).
    A uniform distribution yields 0.0 confidence, while a degenerate point mass
    yields 1.0 confidence, regardless of prediction correctness.
    """
    n = len(probs)
    if n == 0:
        raise ValueError("La liste de probabilités est vide.")
    if n == 1:
        return 0.0
    p_max = max(probs)
    return max(0.0, min(1.0, (p_max - 1.0 / n) / (1.0 - 1.0 / n)))


class ChoiceAnswer(BaseModel):
    """Evaluation response for a ChoiceQuestion."""

    choice: str
    probabilities: dict[str, float]  # Sums to 1.0
    confidence: float

    @model_validator(mode="after")
    def _check_sum(self) -> "ChoiceAnswer":
        s = sum(self.probabilities.values())
        if abs(s - 1.0) > 1e-3:
            raise ValueError(
                f"Les probabilités doivent sommer à 1.0 (somme={s})."
            )
        if self.choice not in self.probabilities:
            raise ValueError(
                f"choice={self.choice!r} absent des probabilités "
                f"{list(self.probabilities)}."
            )
        return self


class ScoreAnswer(BaseModel):
    """Evaluation response for a ScoreQuestion."""

    score: float  # Zero-based continuous expectation: sum(p_i * i) in range [0, M-1]
    level: str    # Argmax categorical level for display
    probabilities: dict[str, float]
    confidence: float
    expected_score: float = 0.0  # One-based continuous expectation: sum(p_i * (i + 1)) in range [1, M]

    @model_validator(mode="after")
    def _check(self) -> "ScoreAnswer":
        s = sum(self.probabilities.values())
        if abs(s - 1.0) > 1e-3:
            raise ValueError(
                f"Les probabilités doivent sommer à 1.0 (somme={s})."
            )
        if self.level not in self.probabilities:
            raise ValueError(
                f"level={self.level!r} absent des probabilités."
            )
        # Compute 1-based continuous expectation if unpopulated: sum(p_i * (i + 1))
        if self.expected_score == 0.0:
            levels = list(self.probabilities.keys())
            self.expected_score = sum(
                self.probabilities[lvl] * (i + 1)
                for i, lvl in enumerate(levels)
            )
        return self


class NoulAnswer(BaseModel):
    """Evaluation response for a NoulQuestion representing belief in [0, 1]."""

    noul: float  # Calibrated probability in [0.0, 1.0]

    @field_validator("noul")
    @classmethod
    def _range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"noul doit être dans [0,1] (valeur={v}).")
        return v


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


# ---------------------------------------------------------------------------
# API Response Schema
# ---------------------------------------------------------------------------

class EvaluateResponse(BaseModel):
    """End-to-end evaluation payload containing answers and inference telemetry."""

    answers: dict[str, Answer]
    latency_ms: float
    cache_hit_rate: float
