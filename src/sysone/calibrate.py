"""Post-hoc probability calibration routines.

Temperature Scaling:
    Optimizes a positive scalar temperature T per question primitive ('choice', 'score', 'noul')
    by minimizing Negative Log-Likelihood (NLL) via L-BFGS-B on held-out validation logits.
    T is applied directly to unnormalized candidate logits prior to softmax:
        p_i = exp(z_i / T) / sum_j exp(z_j / T)
    T > 1 softens overconfident distributions; T < 1 sharpens underconfident distributions.

Isotonic Regression:
    Non-parametric monotonic piecewise-constant mapping fit on confidence vs. accuracy pairs,
    providing complementary 1D empirical calibration.

Serialization & Deployment:
    Calibrated temperature values are serialized into JSON configuration files and reloaded
    at inference time via CalibratedEngine or the CLI `sysone calibrate` command.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class TemperatureConfig:
    """Calibrated scaling parameters per primitive kind with optional isotonic parameters."""

    choice: float = 1.0
    score: float = 1.0
    noul: float = 1.0
    isotonic: dict[str, dict] = field(default_factory=dict)  # Optional bin thresholds

    def save(self, path: str | Path):
        """Serialize configuration parameters to disk as formatted JSON."""
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureConfig":
        """Load configuration parameters from a JSON file."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**d)

    def T_for(self, kind: str) -> float:
        """Return the scalar temperature parameter associated with a question primitive kind."""
        return {"choice": self.choice, "score": self.score, "noul": self.noul}[kind]


def apply_temperature(logits: list[float], T: float) -> list[float]:
    """Apply temperature scaling (1/T) to raw logits followed by numerically stable softmax."""
    if T <= 0:
        raise ValueError("T must be > 0.")
    scaled = [l / T for l in logits]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    s = sum(exps)
    return [e / s for e in exps]


# ---------------------------------------------------------------------------
# NLL Optimization Objective
# ---------------------------------------------------------------------------

def _nll(T: float, all_logits: list[list[float]], labels: list[int]) -> float:
    """Compute average Negative Log-Likelihood: -mean(log p_correct) under temperature T."""
    if T <= 0:
        return 1e9
    total = 0.0
    for logits, y in zip(all_logits, labels):
        probs = apply_temperature(logits, T)
        p = probs[y]
        if p <= 0:
            total += 50.0  # Numerical clipping to prevent -inf
        else:
            total += -math.log(p)
    return total / len(labels)


def fit_temperature(
    all_logits: list[list[float]],
    labels: list[int],
    kind: str = "choice",
) -> float:
    """Find the scalar temperature T that minimizes NLL via L-BFGS-B optimization.

    Args:
        all_logits: Sequence of raw unnormalized logit vectors over active candidates.
        labels: Zero-based target label indices.
        kind: Target primitive kind label for logging.

    Returns:
        Optimal temperature scalar T in range [1e-3, 100.0].
    """
    from scipy.optimize import minimize

    if not all_logits:
        return 1.0
    res = minimize(
        lambda T: _nll(float(T[0]), all_logits, labels),
        x0=[1.0],
        method="L-BFGS-B",
        bounds=[(1e-3, 100.0)],
    )
    return float(res.x[0])


def fit_all(
    by_kind: dict[str, tuple[list[list[float]], list[int]]],
) -> TemperatureConfig:
    """Fit temperature parameters across all available primitive groups.

    Args:
        by_kind: Mapping from primitive kind ('choice', 'score', 'noul')
                 to (logits_list, ground_truth_labels) pairs.
    """
    cfg = TemperatureConfig()
    for kind, (logits, labels) in by_kind.items():
        if logits:
            T = fit_temperature(logits, labels, kind)
            setattr(cfg, kind, T)
    return cfg


# ---------------------------------------------------------------------------
# Isotonic Regression Calibrator
# ---------------------------------------------------------------------------

def _new_isotonic():
    """Build a clipped isotonic regressor, importing sklearn only when one is needed."""
    from sklearn.isotonic import IsotonicRegression

    return IsotonicRegression(out_of_bounds="clip")


@dataclass
class IsotonicCalibrator:
    """One-dimensional isotonic calibrator mapping raw confidence (p_max) to empirical accuracy.

    Provides a non-parametric monotonic mapping used as a complementary calibration method.
    """

    iso: "IsotonicRegression" = field(default_factory=lambda: _new_isotonic())

    def fit(self, confidences: list[float], correct: list[int]):
        """Fit isotonic regression on confidence and binary correctness indicators."""
        self.iso.fit(np.array(confidences), np.array(correct))

    def predict(self, confidence: float) -> float:
        """Map raw confidence to empirical calibrated probability."""
        return float(self.iso.predict([confidence])[0])

    def to_dict(self) -> dict:
        """Serialize fitted thresholds to a dictionary."""
        xs = self.iso.X_thresholds_.tolist() if hasattr(self.iso, "X_thresholds_") else []
        ys = self.iso.y_thresholds_.tolist() if hasattr(self.iso, "y_thresholds_") else []
        return {"x": xs, "y": ys}

    @classmethod
    def from_dict(cls, d: dict) -> "IsotonicCalibrator":
        """Reconstruct calibrator from serialized threshold arrays."""
        c = cls()
        if d.get("x") and d.get("y"):
            c.iso.fit(np.array(d["x"]), np.array(d["y"]))
        return c


# ---------------------------------------------------------------------------
# Calibrated Engine Wrapper
# ---------------------------------------------------------------------------

class CalibratedEngine:
    """Wraps an underlying inference Engine to scale logits prior to answer synthesis.

    Intercepts raw candidate logits produced by the backbone, scales them by the
    per-primitive temperature T, recomputes probability distributions, and formats answers.
    """

    def __init__(self, engine, config: TemperatureConfig):
        self.engine = engine
        self.config = config

    def evaluate(self, query, n_permutations: int = 3):
        """Execute calibrated evaluation batch over a shared query context."""
        import time
        from .types import (
            ChoiceAnswer, ScoreAnswer, NoulAnswer, ChoiceQuestion, ScoreQuestion,
            NoulQuestion, confidence_from_probs, EvaluateResponse,
        )
        from .prompt import OTHER_LABEL_TEXT
        from .debias import remap_distribution, average_distributions
        eng = self.engine
        eng._ensure_loaded()
        t0 = time.perf_counter()
        entries = eng._build_batch(query, n_permutations)
        prompts = [e["spec"].full_text for e in entries]
        params_list = [eng._make_params(e["resolved"].token_ids) for e in entries]
        outputs = eng._llm.generate(prompts, params_list)
        results = [eng._extract(o, e["spec"], e["resolved"])
                   for o, e in zip(outputs, entries)]
        # Apply temperature scaling to raw logits before probability computation
        for e, lr in zip(entries, results):
            T = self.config.T_for(e["spec"].kind)
            lr.probs = apply_temperature(lr.raw_logits, T)
        # Re-aggregate and compose answers per question
        answers = {}
        by_qi = {}
        for e, lr in zip(entries, results):
            by_qi.setdefault(e["qi"], []).append(dict(entry=e, result=lr))
        for qi, items in by_qi.items():
            q = query.questions[qi]
            if isinstance(q, ChoiceQuestion):
                answers[q.key] = eng._compose_choice(q, items)
            elif isinstance(q, ScoreQuestion):
                answers[q.key] = eng._compose_score(q, items)
            elif isinstance(q, NoulQuestion):
                answers[q.key] = eng._compose_noul(q, items)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return EvaluateResponse(
            answers=answers, latency_ms=latency_ms,
            cache_hit_rate=eng._cache_hit_rate(outputs),
        )


def collect_logits_from_dataset(engine, dataset_path: str | Path) -> dict[str, tuple[list[list[float]], list[int]]]:
    """Evaluate dataset examples in a single unified batch to harvest unnormalized candidate logits."""
    from .tokens import noul_token_index
    from .types import Query, ChoiceQuestion, ScoreQuestion, NoulQuestion

    by_kind: dict[str, tuple[list, list]] = {"choice": ([], []), "score": ([], []), "noul": ([], [])}
    lines = Path(dataset_path).read_text(encoding="utf-8").splitlines()
    examples = [json.loads(l) for l in lines if l.strip()]

    engine._ensure_loaded()
    all_entries = []
    metadata = []  # (kind, target_idx)

    for ex in examples:
        kind = ex["kind"]
        label = ex["label"]
        if kind == "choice":
            q = ChoiceQuestion(
                key="q",
                prompt=ex["prompt"],
                options=ex["options"],
                allow_other=ex.get("include_other", label == -1),
            )
            target_idx = label if label >= 0 else len(ex["options"])
        elif kind == "score":
            q = ScoreQuestion(key="q", prompt=ex["prompt"], levels=ex["levels"])
            target_idx = label
        elif kind == "noul":
            q = NoulQuestion(key="q", statement=ex["statement"],
                             prompt=ex.get("prompt", ""))
            # Dataset label 1 == yes, NOUL_LABELS index 0 == 'yes'.
            target_idx = noul_token_index(label)
        else:
            continue

        query = Query(state=ex["state"], questions=[q])
        entries = engine._build_batch(query, n_permutations=1)
        all_entries.extend(entries)
        metadata.append((kind, target_idx))

    if not all_entries:
        return by_kind

    engine._ensure_loaded()
    prompts = [e["spec"].full_text for e in all_entries]
    params_list = [engine._make_params(e["resolved"].token_ids) for e in all_entries]
    outputs = engine._llm.generate(prompts, params_list)

    for out, entry, (kind, target_idx) in zip(outputs, all_entries, metadata):
        lr = engine._extract(out, entry["spec"], entry["resolved"])
        by_kind[kind][0].append(lr.raw_logits)
        by_kind[kind][1].append(target_idx)

    return by_kind


