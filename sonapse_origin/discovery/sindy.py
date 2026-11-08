"""Sparse Identification of Nonlinear Dynamics (SINDy) Discovery Engine."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class SINDyDiscoveryEngine:
    """Discovers sparse dynamical equations from time-series observations:
    
        dX/dt = Theta(X) * Xi
        
    using sequential thresholded least squares (STLSQ) on a library of candidate functions.
    """

    def __init__(
        self,
        poly_order: int = 2,
        include_trig: bool = True,
        threshold: float = 0.05,
        max_iter: int = 10,
    ):
        self.poly_order = poly_order
        self.include_trig = include_trig
        self.threshold = threshold
        self.max_iter = max_iter
        self.feature_names: List[str] = []

    def build_library(self, X: np.ndarray, var_names: Optional[List[str]] = None) -> np.ndarray:
        """Constructs candidate feature matrix Theta(X) of shape (N_samples, N_features)."""
        N, D = X.shape
        v_names = var_names or [f"x{i}" for i in range(D)]
        columns = [np.ones((N, 1))]
        self.feature_names = ["1"]

        # 1. Linear terms
        for i in range(D):
            columns.append(X[:, i : i + 1])
            self.feature_names.append(v_names[i])

        # 2. Quadratic terms
        if self.poly_order >= 2:
            for i in range(D):
                for j in range(i, D):
                    columns.append((X[:, i] * X[:, j])[:, np.newaxis])
                    self.feature_names.append(f"{v_names[i]}*{v_names[j]}")

        # 3. Trigonometric / non-linear terms
        if self.include_trig:
            for i in range(D):
                columns.append(np.sin(X[:, i : i + 1]))
                self.feature_names.append(f"sin({v_names[i]})")
                columns.append(np.tanh(X[:, i : i + 1]))
                self.feature_names.append(f"tanh({v_names[i]})")

        return np.hstack(columns)

    def fit(
        self,
        X: np.ndarray,
        dX_dt: np.ndarray,
        var_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Estimates sparse coefficient matrix Xi via Sequential Thresholded Least Squares (STLSQ).
        
        Returns:
            Xi: sparse coefficient matrix of shape (N_library_features, D)
            equations: string representations of the discovered candidate differential equations.
        """
        Theta = self.build_library(X, var_names=var_names)
        D = dX_dt.shape[1]
        Xi = np.linalg.lstsq(Theta, dX_dt, rcond=None)[0]

        # Iterative thresholding to enforce sparsity
        for _ in range(self.max_iter):
            small_indices = np.abs(Xi) < self.threshold
            Xi[small_indices] = 0.0
            for ind in range(D):
                big_indices = ~small_indices[:, ind]
                if np.any(big_indices):
                    Xi[big_indices, ind] = np.linalg.lstsq(
                        Theta[:, big_indices], dX_dt[:, ind], rcond=None
                    )[0]

        v_names = var_names or [f"x{i}" for i in range(D)]
        equations = []
        for ind in range(D):
            terms = []
            for f_idx, coeff in enumerate(Xi[:, ind]):
                if abs(coeff) > 1e-4:
                    terms.append(f"{coeff:+.3f}*{self.feature_names[f_idx]}")
            eq_str = f"d({v_names[ind]})/dt = " + (" ".join(terms) if terms else "0.0")
            equations.append(eq_str)

        return Xi, equations

    def predict_derivative(self, X: np.ndarray, Xi: np.ndarray) -> np.ndarray:
        """Evaluates Theta(X) * Xi on new unseen data."""
        Theta = self.build_library(X)
        return Theta @ Xi
