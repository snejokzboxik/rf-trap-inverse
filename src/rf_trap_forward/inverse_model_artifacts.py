"""Import-stable serializable wrappers for persisted inverse estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class ClippedInverseModel:
    """Prediction wrapper that preserves documented displacement bounds."""

    estimator: object
    lower_bound_m: float = -500.0e-6
    upper_bound_m: float = 500.0e-6

    def predict(self, X_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return finite base predictions clipped coordinate-wise in metres."""

        predictions = np.asarray(self.estimator.predict(X_m), dtype=float)
        if predictions.ndim != 2 or predictions.shape[1] != 8:
            raise ValueError("base estimator must predict shape (N, 8)")
        if not np.all(np.isfinite(predictions)):
            raise ValueError("base estimator returned nonfinite predictions")
        return np.clip(predictions, self.lower_bound_m, self.upper_bound_m)
