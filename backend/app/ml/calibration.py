from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


CALIBRATION_METHOD = "sigmoid_platt"
PROBABILITY_EPSILON = 1e-12


@dataclass
class PlattProbabilityCalibrator:
    """One-dimensional sigmoid calibration fitted on held-out predictions."""

    coefficient: float | None = None
    intercept: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self.coefficient is not None and self.intercept is not None

    @staticmethod
    def _logits(probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 1:
            raise ValueError("Probabilities must be a one-dimensional array.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Probabilities must be finite.")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("Probabilities must be between 0 and 1.")
        safe = np.clip(values, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
        return np.log(safe / (1.0 - safe))

    def fit(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
    ) -> "PlattProbabilityCalibrator":
        labels = np.asarray(target, dtype=int)
        logits = self._logits(probabilities)
        if labels.ndim != 1 or len(labels) != len(logits):
            raise ValueError("Calibration target must align with probabilities.")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("Calibration requires both recovery labels 0 and 1.")

        estimator = LogisticRegression(solver="lbfgs", random_state=0)
        estimator.fit(logits.reshape(-1, 1), labels)
        self.coefficient = float(estimator.coef_[0, 0])
        self.intercept = float(estimator.intercept_[0])
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Probability calibrator has not been fitted yet.")
        adjusted = self.coefficient * self._logits(probabilities) + self.intercept
        result = np.empty_like(adjusted, dtype=float)
        positive = adjusted >= 0.0
        result[positive] = 1.0 / (1.0 + np.exp(-adjusted[positive]))
        exp_values = np.exp(adjusted[~positive])
        result[~positive] = exp_values / (1.0 + exp_values)
        return result

    def to_dict(self) -> dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Probability calibrator has not been fitted yet.")
        return {
            "method": CALIBRATION_METHOD,
            "coefficient": self.coefficient,
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlattProbabilityCalibrator":
        if payload.get("method") != CALIBRATION_METHOD:
            raise ValueError("Unsupported probability calibration method.")
        return cls(
            coefficient=float(payload["coefficient"]),
            intercept=float(payload["intercept"]),
        )
