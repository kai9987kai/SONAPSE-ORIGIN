"""Coupled Kuramoto Oscillators: Synchronisation and Phase Locking."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np


class KuramotoOscillators:
    """Coupled Kuramoto oscillator dynamics:
    
        d theta_i / dt = omega_i + sum_j K_ij * sin(theta_j - theta_i)
        
    Order parameter:
        r * exp(i * psi) = (1/N) * sum_j exp(i * theta_j)
    """

    def __init__(
        self,
        num_oscillators: int,
        natural_frequencies: Optional[np.ndarray] = None,
        coupling_strength: float = 1.0,
        dt: float = 0.05,
        seed: int = 42,
    ):
        self.num_oscillators = num_oscillators
        self.coupling_strength = coupling_strength
        self.dt = dt
        rng = np.random.default_rng(seed)

        if natural_frequencies is not None:
            self.omega = natural_frequencies
        else:
            # Lorentzian or Gaussian frequency distribution
            self.omega = rng.normal(1.0, 0.2, num_oscillators)

        self.theta = rng.uniform(0.0, 2.0 * np.pi, num_oscillators)

    def reset(self, seed: Optional[int] = None) -> None:
        rng = np.random.default_rng(seed)
        self.theta = rng.uniform(0.0, 2.0 * np.pi, self.num_oscillators)

    def order_parameter(self) -> Tuple[float, float]:
        """Calculates the Kuramoto order parameter r (coherence in [0, 1]) and mean phase psi."""
        complex_order = np.mean(np.exp(1j * self.theta))
        r = float(np.abs(complex_order))
        psi = float(np.angle(complex_order))
        return r, psi

    def step(
        self,
        coupling_matrix: Optional[np.ndarray] = None,
        external_drive: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Integrates phase dynamics by dt. Returns (phases, coherence_r)."""
        K = coupling_matrix if coupling_matrix is not None else np.ones((self.num_oscillators, self.num_oscillators)) * (self.coupling_strength / self.num_oscillators)
        ext = external_drive if external_drive is not None else np.zeros(self.num_oscillators)

        # Pairwise phase differences: theta_j - theta_i
        # theta is shape (N,) -> diff[i, j] = theta[j] - theta[i]
        phase_diff = self.theta[np.newaxis, :] - self.theta[:, np.newaxis]
        phase_interaction = np.sum(K * np.sin(phase_diff), axis=1)

        dtheta = self.omega + phase_interaction + ext
        self.theta = (self.theta + dtheta * self.dt) % (2.0 * np.pi)

        r, _ = self.order_parameter()
        return self.theta.copy(), r
