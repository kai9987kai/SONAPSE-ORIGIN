"""Predictive Processing (PP) Hierarchical Generative Experiments."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class HierarchicalPredictiveCoding:
    """Implements a two-level hierarchical predictive coding circuit:
    
        Level 1 (Sensory): x_0
        Level 2 (Latent):  x_1
        Top-down prediction: p_0 = g(W_pred @ x_1)
        Prediction error: e_0 = x_0 - p_0
        Variational free energy proxy: F = (1/2) * ||e_0||^2 + (1/2) * ||x_1||^2
    """

    def __init__(self, sensory_dim: int = 8, latent_dim: int = 4, seed: int = 42):
        self.sensory_dim = sensory_dim
        self.latent_dim = latent_dim
        self.rng = np.random.default_rng(seed)

        # Generative top-down weights: latent -> sensory prediction
        self.W_pred = self.rng.normal(0, 0.4, (sensory_dim, latent_dim))
        # Recognition bottom-up weights: prediction error -> latent update
        self.W_recog = self.W_pred.T.copy()
        self.x_1 = np.zeros(latent_dim, dtype=np.float64)

    def infer_and_minimize_error(
        self,
        sensory_input: np.ndarray,
        num_inference_steps: int = 10,
        lr: float = 0.05,
    ) -> Dict[str, Any]:
        """Iteratively updates latent states to minimize sensory prediction error."""
        free_energy_history = []

        for _ in range(num_inference_steps):
            # Top-down prediction
            p_0 = np.tanh(self.W_pred @ self.x_1)
            # Bottom-up prediction error
            e_0 = sensory_input - p_0
            # Free energy proxy
            F = 0.5 * np.sum(e_0 ** 2) + 0.05 * np.sum(self.x_1 ** 2)
            free_energy_history.append(float(F))

            # Latent update: gradient descent on free energy dF/dx_1 = - W_pred^T e_0 + 0.1 x_1
            dx_1 = self.W_recog @ e_0 - 0.1 * self.x_1
            self.x_1 += lr * dx_1

        final_prediction = np.tanh(self.W_pred @ self.x_1)
        final_error = sensory_input - final_prediction

        return {
            "initial_free_energy": free_energy_history[0],
            "final_free_energy": free_energy_history[-1],
            "free_energy_reduction": free_energy_history[0] - free_energy_history[-1],
            "mean_squared_prediction_error": float(np.mean(final_error ** 2)),
            "latent_state": self.x_1.copy(),
            "predicted_input": final_prediction,
        }
