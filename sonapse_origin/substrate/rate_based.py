"""Continuous Rate-Based Recurrent Neural Dynamics."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np


class ContinuousRateRNN:
    """Continuous-time rate-based recurrent network:
    
        tau_i * dx_i/dt = -x_i + sigma(sum_j W_ij x_j + b_i + u_i)
    """

    def __init__(
        self,
        num_units: int,
        tau: float = 10.0,
        dt: float = 1.0,
        activation: str = "tanh",
        bias: Optional[np.ndarray] = None,
    ):
        self.num_units = num_units
        self.tau = tau
        self.dt = dt
        self.activation_str = activation
        self.bias = bias if bias is not None else np.zeros(num_units, dtype=np.float64)
        self.x = np.zeros(num_units, dtype=np.float64)

    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation_str == "tanh":
            return np.tanh(z)
        elif self.activation_str == "sigmoid":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))
        elif self.activation_str == "relu":
            return np.maximum(0.0, z)
        return z

    def reset(self) -> None:
        self.x.fill(0.0)

    def step(
        self,
        W: np.ndarray,
        external_input: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Forward Euler integration step."""
        u = external_input if external_input is not None else np.zeros(self.num_units)
        net_in = W @ self.x + self.bias + u
        activated = self._activate(net_in)
        dx = (-self.x + activated) / self.tau
        self.x += dx * self.dt
        return self.x.copy()
