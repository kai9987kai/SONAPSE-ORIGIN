"""Multidimensional Artificial Affective Dynamics and Stability."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class AffectiveDynamicsEngine:
    """Simulates 9-dimensional affective control states:
    
        da/dt = -Lambda * a + W_a * a + B_h * h + B_s * s + eta
        
    Used as internal regulatory signals and prioritization weights.
    These are mathematical control states, not claims of felt emotion.
    """

    AFFECT_NAMES = [
        "valence",          # [-1, +1]
        "arousal",          # [0, 1]
        "dread",            # [0, 1]
        "calmness",         # [0, 1]
        "fear",             # [0, 1]
        "joy",              # [0, 1]
        "surprise",         # [0, 1]
        "curiosity",        # [0, 1]
        "restorative_drive" # [0, 1]
    ]

    def __init__(
        self,
        dt: float = 0.05,
        seed: int = 42,
    ):
        self.dt = dt
        self.dim = len(self.AFFECT_NAMES)
        self.rng = np.random.default_rng(seed)

        # Natural relaxation rates Lambda (diagonal)
        self.Lambda = np.diag([0.3, 0.4, 0.25, 0.3, 0.45, 0.35, 0.5, 0.3, 0.2])

        # Interaction matrix W_a: mutual inhibitions and synergies
        # e.g. fear suppresses calmness; joy enhances valence; fear increases arousal
        self.W_a = np.zeros((self.dim, self.dim), dtype=np.float64)
        # valence (0): joy (+) fear (-) dread (-)
        self.W_a[0, 5] = 0.2
        self.W_a[0, 4] = -0.2
        self.W_a[0, 2] = -0.15
        # arousal (1): fear (+), surprise (+), joy (+), calmness (-)
        self.W_a[1, 4] = 0.2
        self.W_a[1, 6] = 0.2
        self.W_a[1, 5] = 0.1
        self.W_a[1, 3] = -0.1
        # calmness (3): fear (-), dread (-), arousal (-)
        self.W_a[3, 4] = -0.2
        self.W_a[3, 2] = -0.15
        self.W_a[3, 1] = -0.1

        # Coupling matrix from homeostatic variables B_h: shape (9, 9)
        self.B_h = np.zeros((self.dim, 9), dtype=np.float64)
        # Low integrity (h[1]) triggers fear (a[4]) and dread (a[2])
        self.B_h[4, 1] = -0.8
        self.B_h[2, 1] = -0.5
        # High fatigue (h[3]) triggers restorative drive (a[8]) and reduces arousal (a[1])
        self.B_h[8, 3] = 0.7
        self.B_h[1, 3] = -0.3
        # High prediction error (h[6]) triggers surprise (a[6]) and curiosity (a[7])
        self.B_h[6, 6] = 0.8
        self.B_h[7, 6] = 0.4
        # High energy (h[0]) triggers joy (a[5])
        self.B_h[5, 0] = 0.5

        self.a = np.array([0.0, 0.2, 0.0, 0.8, 0.0, 0.2, 0.0, 0.5, 0.1], dtype=np.float64)

    def is_lyapunov_stable(self) -> Tuple[bool, float]:
        """Checks if linearized system matrix (-Lambda + W_a) has negative real eigenvalues."""
        A_lin = -self.Lambda + self.W_a
        eigvals = np.linalg.eigvals(A_lin)
        max_real = float(np.max(np.real(eigvals)))
        return max_real < 0.0, max_real

    def step(
        self,
        h_vec: np.ndarray,
        sensory_salience: Optional[np.ndarray] = None,
        noise_scale: float = 0.01,
    ) -> np.ndarray:
        """Forward Euler step with non-linear saturating bounding."""
        s = sensory_salience if sensory_salience is not None else np.zeros(self.dim)
        noise = self.rng.normal(0, noise_scale, size=self.dim)

        # Baseline offset for homeostatic variables (e.g. 1.0 - integrity)
        h_mod = h_vec.copy()
        h_mod[1] = 1.0 - h_mod[1]  # Damage trigger

        da = -self.Lambda @ self.a + self.W_a @ self.a + self.B_h @ h_mod + s + noise
        self.a += da * self.dt

        # Bounding / non-linear saturation
        self.a[0] = np.clip(self.a[0], -1.0, 1.0)  # Valence
        self.a[1:] = np.clip(self.a[1:], 0.0, 1.0) # Non-negative states

        return self.a.copy()
