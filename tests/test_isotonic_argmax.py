"""Isotonic recalibration must never change the predicted label.

Regression: a map that lowers p_max redistributed the mass to the other
classes and could lift the runner-up above the winner, leaving ScoreAnswer.level
pointing at a class that was no longer the argmax of its own probabilities.
"""

from sysone.calibrate import CalibratedEngine, TemperatureConfig
from sysone.types import ChoiceAnswer, ScoreAnswer

# Maps any p_max in [0.5, 0.52] down to 0.49.
LOWERING = {"x": [0.5, 0.52], "y": [0.49, 0.49]}


def _engine(kind):
    return CalibratedEngine(None, TemperatureConfig(isotonic={kind: LOWERING}))


def test_score_level_stays_argmax_when_map_lowers_confidence():
    a = ScoreAnswer(score=0.5, level="a", confidence=0.2,
                    probabilities={"a": 0.505, "b": 0.485, "c": 0.01})
    a = _engine("score")._recalibrate("score", a)
    assert max(a.probabilities, key=a.probabilities.get) == a.level == "a"
    assert abs(sum(a.probabilities.values()) - 1.0) < 1e-9


def test_choice_stays_argmax_when_map_lowers_confidence():
    a = ChoiceAnswer(choice="x", confidence=0.1,
                     probabilities={"x": 0.51, "y": 0.48, "z": 0.01})
    a = _engine("choice")._recalibrate("choice", a)
    assert max(a.probabilities, key=a.probabilities.get) == a.choice == "x"
