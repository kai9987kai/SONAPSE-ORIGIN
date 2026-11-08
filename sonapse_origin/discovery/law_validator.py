"""Candidate Law Falsification and Out-of-Distribution Validation."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class CandidateLawValidator:
    """Falsification harness testing whether discovered equations hold across unseen environments and causal ablations."""

    def __init__(self, r2_promotion_threshold: float = 0.65, max_relative_error: float = 0.35):
        self.r2_promotion_threshold = r2_promotion_threshold
        self.max_relative_error = max_relative_error

    def validate_power_law(
        self,
        candidate_model: Dict[str, Any],
        test_x: np.ndarray,
        test_y: np.ndarray,
    ) -> Dict[str, Any]:
        """Tests power law equation y = a * x^b on out-of-distribution validation data."""
        a = candidate_model.get("a", 1.0)
        b = candidate_model.get("exponent_b", 1.0)

        valid = (test_x > 1e-4) & (test_y > 1e-4)
        if np.sum(valid) < 3:
            return {"promoted": False, "reason": "insufficient_test_samples"}

        x_eval = test_x[valid]
        y_eval = test_y[valid]
        pred_y = a * (x_eval ** b)

        ss_res = np.sum((y_eval - pred_y) ** 2)
        ss_tot = np.sum((y_eval - np.mean(y_eval)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

        rel_errors = np.abs((y_eval - pred_y) / (y_eval + 1e-6))
        mean_rel_error = float(np.mean(rel_errors))

        promoted = bool(r2 >= self.r2_promotion_threshold and mean_rel_error <= self.max_relative_error)

        return {
            "test_r_squared": r2,
            "mean_relative_error": mean_rel_error,
            "promoted": promoted,
            "status": "CANDIDATE_DISCOVERY" if promoted else "FALSIFIED_OR_EMPIRICAL_ONLY",
            "epistemic_warning": "Empirical candidate regularity only; not verified as a universal law of sentience.",
        }
