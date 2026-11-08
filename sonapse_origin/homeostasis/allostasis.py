"""Allostatic Active Inference Controller: Anticipatory Interoceptive Regulation.

Extends reactive homeostasis to predictive allostasis (Sterling, 2012; Friston et al., 2015):
Predicts prospective metabolic decline tau steps ahead and selects policies that
minimize Expected Free Energy (EFE) with respect to viability priors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class AllostaticActiveInferenceController:
    """Anticipatory interoceptive controller minimizing Expected Free Energy (EFE)."""

    def __init__(
        self,
        lookahead_steps: int = 5,
        target_setpoint: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.lookahead_steps = lookahead_steps
        self.target = target_setpoint if target_setpoint is not None else np.array([0.9, 1.0, 0.5, 0.1])
        self.rng = np.random.default_rng(seed)

    def predict_future_state(
        self,
        current_h: np.ndarray,
        action: int,
        steps: int,
    ) -> np.ndarray:
        """Forward generative transition model: predicts h_{t+steps} under candidate action."""
        h_pred = current_h.copy()
        for _ in range(steps):
            # Natural metabolic baseline decay
            h_pred[0] -= 0.025  # Energy drains
            h_pred[3] += 0.015  # Fatigue accumulates

            # Anticipated effect of action
            if action == 1:  # Consume resource
                h_pred[0] += 0.08
            elif action == 2:  # Thermoregulate / stabilize
                h_pred[2] = 0.5 * h_pred[2] + 0.5 * 0.5
            elif action == 0:  # Rest / recover
                h_pred[3] = max(0.0, h_pred[3] - 0.03)

        return np.clip(h_pred, 0.0, 1.0)

    def select_action(
        self,
        current_h: np.ndarray,
        num_actions: int = 3,
        ambient_hazard: float = 0.0,
    ) -> Tuple[int, Dict[str, float]]:
        """Selects policy minimizing Expected Free Energy: G(a) = ||h_{t+tau} - target||_Q^2 + Risk."""
        weights = np.array([3.0, 4.0, 1.5, 1.5])
        action_efe = {}

        for a in range(num_actions):
            predicted_h = self.predict_future_state(current_h[:4], a, self.lookahead_steps)
            diff = predicted_h - self.target
            # Pragmatic value: distance to preferred operating envelope
            pragmatic_cost = float(np.sum(weights * (diff ** 2)))
            # Epistemic / risk penalty under ambient volatility
            risk_penalty = ambient_hazard * (1.0 if a != 2 else 0.1)
            efe = pragmatic_cost + risk_penalty
            action_efe[a] = efe

        # Select action with minimal expected free energy
        best_action = int(min(action_efe, key=action_efe.get))
        return best_action, action_efe
