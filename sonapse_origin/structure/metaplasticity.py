"""Continuous Metamorphic Plasticity and BCM Sliding Threshold Rule."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class BCMMetaplasticityEngine:
    """Implements Bienenstock-Cooper-Munro (BCM) sliding modification threshold with structural rewiring:
    
        dW_ij/dt = eta * x_j * x_i * (x_i - theta_i) - lambda * W_ij
        theta_i = E[x_i^2] / x_0
    """

    def __init__(
        self,
        num_units: int,
        learning_rate: float = 0.01,
        weight_decay: float = 0.001,
        target_activity: float = 0.5,
        tau_theta: float = 50.0,
        seed: int = 42,
    ):
        self.num_units = num_units
        self.eta = learning_rate
        self.decay = weight_decay
        self.x_0 = target_activity
        self.tau_theta = tau_theta

        self.theta = np.full(num_units, target_activity, dtype=np.float64)
        self.activity_history: List[np.ndarray] = []
        self.rng = np.random.default_rng(seed)

    def step(
        self,
        W: np.ndarray,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Updates synaptic weights and adapts modification thresholds."""
        # 1. Update sliding threshold: d theta_i / dt = (post_i^2 / x_0 - theta_i) / tau_theta
        d_theta = ((post_activity ** 2) / (self.x_0 + 1e-4) - self.theta) / self.tau_theta
        self.theta = np.clip(self.theta + d_theta, 0.05, 5.0)

        # 2. BCM update: dW_ij = eta * pre_j * post_i * (post_i - theta_i) - decay * W_ij
        # post_activity is shape (N,), pre_activity is shape (N,)
        bcm_factor = post_activity * (post_activity - self.theta)  # shape: (N,)
        delta_W = self.eta * np.outer(bcm_factor, pre_activity) - self.decay * W

        new_W = np.clip(W + delta_W, -3.0, 3.0)
        return new_W, self.theta.copy()

    def metamorphic_rewiring(
        self,
        W: np.ndarray,
        correlation_matrix: np.ndarray,
        growth_threshold: float = 0.4,
    ) -> Tuple[np.ndarray, int]:
        """Prunes connections driving excessive BCM depression and sprouts co-active pathways."""
        new_W = W.copy()
        num_nodes = W.shape[0]
        sprouted = 0

        # Sprout between correlated nodes
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and abs(new_W[i, j]) < 0.02:
                    if correlation_matrix[i, j] > growth_threshold:
                        new_W[i, j] = float(self.rng.normal(0.2, 0.05))
                        sprouted += 1

        return new_W, sprouted
