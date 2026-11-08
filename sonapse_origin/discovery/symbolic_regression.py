"""Symbolic Regression and Architecture-Independent Law Discovery."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class SymbolicLawDiscoverer:
    """Searches for interpretable closed-form mathematical equations connecting macro-properties.
    
    e.g. Stability = f(Integration, Efficiency, Environmental Complexity).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def fit_power_law(
        self,
        x: np.ndarray,
        y: np.ndarray,
        var_x_name: str = "Integration",
        var_y_name: str = "Stability",
    ) -> Dict[str, Any]:
        """Fits y = a * x^b power law relationship using log-linear regression."""
        # Filter strictly positive values
        valid = (x > 1e-4) & (y > 1e-4)
        if np.sum(valid) < 5:
            return {"r_squared": 0.0, "equation": "insufficient_data"}

        log_x = np.log(x[valid])
        log_y = np.log(y[valid])

        A = np.vstack([log_x, np.ones(len(log_x))]).T
        b_coeff, log_a = np.linalg.lstsq(A, log_y, rcond=None)[0]
        a_coeff = np.exp(log_a)

        # R^2 calculation
        pred_y = a_coeff * (x[valid] ** b_coeff)
        ss_res = np.sum((y[valid] - pred_y) ** 2)
        ss_tot = np.sum((y[valid] - np.mean(y[valid])) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

        equation = f"{var_y_name} = {a_coeff:.3f} * ({var_x_name})^{b_coeff:.3f}"
        return {
            "a": float(a_coeff),
            "exponent_b": float(b_coeff),
            "r_squared": float(r2),
            "equation": equation,
            "sample_size": int(np.sum(valid)),
        }

    def fit_multivariate_linear(
        self,
        X_matrix: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        target_name: str = "Performance",
    ) -> Dict[str, Any]:
        """Fits sparse multivariate linear law: y = beta_0 + sum_j beta_j * X_j."""
        N, D = X_matrix.shape
        A = np.hstack([X_matrix, np.ones((N, 1))])
        coeffs = np.linalg.lstsq(A, y, rcond=None)[0]

        pred_y = A @ coeffs
        ss_res = np.sum((y - pred_y) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

        terms = [f"{coeffs[i]:+.3f}*{feature_names[i]}" for i in range(D) if abs(coeffs[i]) > 0.01]
        intercept = coeffs[-1]
        equation = f"{target_name} = {intercept:.3f} " + (" ".join(terms) if terms else "")

        return {
            "coefficients": coeffs[:-1].tolist(),
            "intercept": float(intercept),
            "r_squared": float(r2),
            "equation": equation,
        }
