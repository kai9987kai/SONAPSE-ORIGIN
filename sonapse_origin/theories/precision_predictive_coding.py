"""Hierarchical Precision-Weighted Predictive Coding Subsystem.

Implements Karl Friston's formulation of hierarchical predictive processing:
- Descending top-down generative predictions mu_l
- Ascending bottom-up prediction errors epsilon_l = Pi_l * (x_l - g(mu_{l+1}))
- Dynamic precision weighting Pi_l modulated by attentional / neuromodulatory state
- Minimization of Variational Free Energy across hierarchical tiers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PredictiveCodingState:
    mu: List[np.ndarray]  # Internal belief states per hierarchical level [L0, L1, L2]
    errors: List[np.ndarray]  # Weighted prediction errors per level
    precisions: List[np.ndarray]  # Diagonal precision weights per level
    free_energy: float


class HierarchicalPredictiveCoder:
    """3-tier hierarchical predictive coding network with dynamic precision gating."""

    def __init__(
        self,
        layer_sizes: Tuple[int, int, int] = (16, 8, 4),
        dt: float = 0.05,
        seed: int = 42,
    ):
        self.layer_sizes = layer_sizes
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # Generative projection matrices: L2 -> L1, and L1 -> L0
        # g_1(mu_2) = W_12 @ mu_2
        # g_0(mu_1) = W_01 @ mu_1
        self.W_01 = self.rng.normal(0.0, 0.5 / np.sqrt(layer_sizes[1]), (layer_sizes[0], layer_sizes[1]))
        self.W_12 = self.rng.normal(0.0, 0.5 / np.sqrt(layer_sizes[2]), (layer_sizes[1], layer_sizes[2]))

        # Initialize smooth harmonic generative basis for sensory patterns
        l0, l1 = layer_sizes[0], layer_sizes[1]
        t = np.linspace(0, np.pi, l0)
        for j in range(l1):
            mode = np.cos(j * t)
            self.W_01[:, j] = mode / (np.linalg.norm(mode) + 1e-8)

        # High-level prior mean and precision
        self.prior_mu2 = np.zeros(layer_sizes[2])
        self.prior_pi2 = np.ones(layer_sizes[2])

    def infer(
        self,
        sensory_input: np.ndarray,
        precisions: Optional[List[np.ndarray]] = None,
        adaptation_steps: int = 25,
        learning_rate: float = 0.1,
    ) -> PredictiveCodingState:
        """Runs variational message passing to infer optimal beliefs and minimize free energy."""
        l0, l1, l2 = self.layer_sizes

        # Default precisions if not supplied
        if precisions is None:
            precisions = [
                np.ones(l0),  # Sensory precision Pi_0
                np.ones(l1),  # Intermediate precision Pi_1
                np.ones(l2),  # Abstract prior precision Pi_2
            ]

        # Initialize beliefs
        mu_0 = np.copy(sensory_input)
        mu_1 = np.zeros(l1)
        mu_2 = np.zeros(l2)

        eps_0 = np.zeros(l0)
        eps_1 = np.zeros(l1)
        eps_2 = np.zeros(l2)

        # Gradient descent on free energy (belief updating)
        for _ in range(adaptation_steps):
            # Top-down generative predictions
            pred_0 = np.tanh(self.W_01 @ mu_1)
            pred_1 = np.tanh(self.W_12 @ mu_2)

            # Precision-weighted prediction errors
            eps_0 = precisions[0] * (sensory_input - pred_0)
            eps_1 = precisions[1] * (mu_1 - pred_1)
            eps_2 = precisions[2] * (mu_2 - self.prior_mu2)

            # Belief gradients via predictive coding backpropagation:
            delta_0 = eps_0 * (1.0 - pred_0**2)
            grad_mu_1 = -eps_1 + (self.W_01.T @ delta_0)

            delta_1 = eps_1 * (1.0 - pred_1**2)
            grad_mu_2 = -eps_2 + (self.W_12.T @ delta_1)

            # Gradient update
            mu_1 += self.dt * learning_rate * grad_mu_1
            mu_2 += self.dt * learning_rate * grad_mu_2

        # Compute total variational free energy: sum of precision-weighted squared errors
        fe = 0.5 * (
            np.sum(precisions[0] * (sensory_input - np.tanh(self.W_01 @ mu_1)) ** 2)
            + np.sum(precisions[1] * (mu_1 - np.tanh(self.W_12 @ mu_2)) ** 2)
            + np.sum(precisions[2] * (mu_2 - self.prior_mu2) ** 2)
            - np.sum(np.log(np.clip(precisions[0], 1e-6, 1e6)))
            - np.sum(np.log(np.clip(precisions[1], 1e-6, 1e6)))
            - np.sum(np.log(np.clip(precisions[2], 1e-6, 1e6)))
        )

        return PredictiveCodingState(
            mu=[mu_0, mu_1, mu_2],
            errors=[eps_0, eps_1, eps_2],
            precisions=precisions,
            free_energy=float(fe),
        )

    def evaluate_noise_robustness(
        self,
        clean_pattern: np.ndarray,
        noise_level: float = 0.5,
        dynamic_precision: bool = True,
    ) -> Dict[str, float]:
        """Compares inference under sensory corruption with adaptive vs static precision."""
        noisy_sensory = clean_pattern + self.rng.normal(0.0, noise_level, size=clean_pattern.shape)

        if dynamic_precision:
            # Attenuate sensory precision Pi_0 in noisy regime, trusting top-down priors
            pi_0 = np.ones(self.layer_sizes[0]) / (1.0 + 3.0 * noise_level)
            pi_1 = np.ones(self.layer_sizes[1]) * 1.5
            pi_2 = np.ones(self.layer_sizes[2]) * 2.0
            precisions = [pi_0, pi_1, pi_2]
        else:
            # Naive static precision ignores noise
            precisions = [np.ones(s) for s in self.layer_sizes]

        state = self.infer(noisy_sensory, precisions=precisions)
        
        # Reconstruction of sensory pattern from high-level inferred causes
        recon_pattern = np.tanh(self.W_01 @ state.mu[1])
        reconstruction_error = float(np.mean((clean_pattern - recon_pattern) ** 2))

        return {
            "reconstruction_error": reconstruction_error,
            "free_energy": state.free_energy,
            "top_level_activity_norm": float(np.linalg.norm(state.mu[2])),
        }
