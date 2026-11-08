"""Viability Variables and Homeostatic Regulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ViabilityEnvelope:
    """Defines preferred operating bounds [min_val, max_val] for viability variables."""
    energy_range: Tuple[float, float] = (0.3, 1.0)
    integrity_range: Tuple[float, float] = (0.5, 1.0)
    temperature_range: Tuple[float, float] = (0.4, 0.6)
    fatigue_range: Tuple[float, float] = (0.0, 0.7)
    uncertainty_range: Tuple[float, float] = (0.0, 0.4)
    prediction_error_range: Tuple[float, float] = (0.0, 0.3)


class HomeostaticRegulator:
    """Calculates homeostatic costs, setpoint deviations, and regulatory drives."""

    def __init__(
        self,
        envelope: Optional[ViabilityEnvelope] = None,
        q_weights: Optional[np.ndarray] = None,
    ):
        self.envelope = envelope or ViabilityEnvelope()
        # Cost weights Q
        self.q = q_weights if q_weights is not None else np.array([2.5, 4.0, 1.0, 1.5, 0.8, 0.5, 0.5, 0.3, 1.2])
        self.target = np.array([0.9, 1.0, 0.5, 0.1, 0.8, 0.1, 0.05, 0.5, 0.95])

    def compute_cost(self, h_vec: np.ndarray) -> float:
        """J_H = (h - h*)^T Q (h - h*)."""
        diff = h_vec - self.target
        return float(np.sum(self.q * (diff ** 2)))

    def is_viable(self, h_vec: np.ndarray) -> bool:
        """Checks if key variables reside within the viability envelope."""
        energy, integrity, temp, fatigue = h_vec[0], h_vec[1], h_vec[2], h_vec[3]
        return bool(
            self.envelope.energy_range[0] <= energy <= self.envelope.energy_range[1]
            and self.envelope.integrity_range[0] <= integrity <= self.envelope.integrity_range[1]
            and self.envelope.temperature_range[0] <= temp <= self.envelope.temperature_range[1]
            and self.envelope.fatigue_range[0] <= fatigue <= self.envelope.fatigue_range[1]
        )

    def compute_regulatory_drives(self, h_vec: np.ndarray) -> np.ndarray:
        """Gradient of homeostatic cost: drives agent actions towards setpoints."""
        return 2.0 * self.q * (self.target - h_vec)
