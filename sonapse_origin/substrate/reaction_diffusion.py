"""Continuous Neural Fields and Reaction-Diffusion Substrates."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy.ndimage import convolve


class ContinuousNeuralField:
    """Amari-style continuous neural field:
    
        tau * du(x,t)/dt = -u(x,t) + integral_Omega w(x, y) * f(u(y, t)) dy + I(x, t)
    """

    def __init__(
        self,
        grid_size: int = 32,
        tau: float = 5.0,
        dt: float = 0.5,
        sigma_exc: float = 1.5,
        sigma_inh: float = 3.5,
        a_exc: float = 1.0,
        a_inh: float = 0.6,
    ):
        self.grid_size = grid_size
        self.tau = tau
        self.dt = dt

        # Generate Mexican-hat kernel
        k_radius = int(3 * sigma_inh)
        x = np.arange(-k_radius, k_radius + 1)
        y = np.arange(-k_radius, k_radius + 1)
        xx, yy = np.meshgrid(x, y)
        dist_sq = xx**2 + yy**2

        kernel_exc = a_exc * np.exp(-dist_sq / (2 * sigma_exc**2)) / (2 * np.pi * sigma_exc**2)
        kernel_inh = a_inh * np.exp(-dist_sq / (2 * sigma_inh**2)) / (2 * np.pi * sigma_inh**2)
        self.kernel = kernel_exc - kernel_inh

        self.u = np.zeros((grid_size, grid_size), dtype=np.float64)

    def reset(self) -> None:
        self.u.fill(0.0)

    def step(self, external_input: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward Euler step with spatial 2D convolution."""
        ext = external_input if external_input is not None else np.zeros_like(self.u)
        # Activation function: sigmoid
        firing = 1.0 / (1.0 + np.exp(-self.u))
        # Non-local synaptic integration via spatial convolution
        recurrent_input = convolve(firing, self.kernel, mode="wrap")

        du = (-self.u + recurrent_input + ext) / self.tau
        self.u += du * self.dt
        return self.u.copy()
