"""Bioelectric Morphogenesis & Basal Cognition Substrate.

Inspired by Michael Levin's bioelectric patterning and morphological computation:
- Non-neural somatic cell lattice with membrane potentials V_mem
- Voltage-gated gap junctions (connexin-like conductance g_ij)
- Target morphological attractor field V_target
- Spontaneous anatomical pattern restoration / regeneration post severe tissue lesion.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class MorphogenesisState:
    voltages: np.ndarray  # 2D grid (H, W) of cell membrane potentials
    gap_junction_conductance: np.ndarray  # (H, W, 4) conductances to [N, S, E, W]
    pattern_divergence: float  # MSE against target anatomical attractor
    differentiated_types: np.ndarray  # [0: Undifferentiated, 1: Anterior/Head, 2: Trunk, 3: Posterior/Tail]


class BioelectricMorphogeneticTissue:
    """Simulates bioelectric anatomical patterning and regeneration in a cell collective."""

    def __init__(
        self,
        grid_shape: Tuple[int, int] = (10, 10),
        dt: float = 0.05,
        base_conductance: float = 0.4,
        seed: int = 42,
    ):
        self.height, self.width = grid_shape
        self.dt = dt
        self.base_conductance = base_conductance
        self.rng = np.random.default_rng(seed)

        # Membrane biophysics parameters
        self.c_mem = 1.0
        self.g_leak = 0.1
        self.v_rest = -0.5  # Hyperpolarized default ground state
        self.g_pump = 0.3

        # Target morphological bioelectric pattern: monotonic anterior-posterior gradient
        # Anterior (x ~ 0): depolarized (+0.8), Posterior (x ~ 1): hyperpolarized (-0.8)
        self.v_target = np.zeros(grid_shape, dtype=np.float64)
        for r in range(self.height):
            for c in range(self.width):
                frac_x = c / max(1, self.width - 1)
                self.v_target[r, c] = 0.8 - 1.6 * frac_x

        # Initial tissue state initialized close to resting potential with noise
        self.voltages = np.full(grid_shape, self.v_rest) + self.rng.normal(0, 0.05, grid_shape)

    def step(self, steps: int = 1) -> None:
        """Solves bioelectric differential equation: C_m * dV/dt = -I_ion + I_gap."""
        h, w = self.height, self.width

        for _ in range(steps):
            # Compute gap junction currents from 4 nearest neighbors (von Neumann neighborhood)
            i_gap = np.zeros((h, w), dtype=np.float64)

            # North neighbor (r-1)
            diff_n = np.roll(self.voltages, 1, axis=0) - self.voltages
            diff_n[0, :] = 0.0  # boundary condition
            i_gap += self.base_conductance * diff_n

            # South neighbor (r+1)
            diff_s = np.roll(self.voltages, -1, axis=0) - self.voltages
            diff_s[-1, :] = 0.0
            i_gap += self.base_conductance * diff_s

            # West neighbor (c-1)
            diff_w = np.roll(self.voltages, 1, axis=1) - self.voltages
            diff_w[:, 0] = 0.0
            i_gap += self.base_conductance * diff_w

            # East neighbor (c+1)
            diff_e = np.roll(self.voltages, -1, axis=1) - self.voltages
            diff_e[:, -1] = 0.0
            i_gap += self.base_conductance * diff_e

            # Biological morphogenetic organizer: only boundary poles (anterior c=0 and posterior c=-1)
            # possess active bioelectric anchor pumps. Interior tissue relies on gap-junction diffusion.
            pump_mask = np.zeros((h, w), dtype=np.float64)
            pump_mask[:, 0] = 1.0
            pump_mask[:, -1] = 1.0

            # Ion pump and leak current:
            i_ion = self.g_leak * (self.voltages - self.v_rest) - pump_mask * self.g_pump * np.tanh(self.v_target - self.voltages)

            # Voltage update
            dv_dt = (-i_ion + i_gap) / self.c_mem
            self.voltages += self.dt * dv_dt
            self.voltages = np.clip(self.voltages, -1.5, 1.5)

    def apply_lesion(self, fraction_cut: float = 0.4) -> None:
        """Surgically excises a fraction of tissue (sets potentials to zero / damage state)."""
        cut_cols = int(self.width * fraction_cut)
        # Damage posterior half
        self.voltages[:, self.width - cut_cols :] = 0.0

    def get_state(self) -> MorphogenesisState:
        """Returns current bioelectric morphology and differentiated tissue phenotypes."""
        h, w = self.height, self.width
        # Differentiate based on stable membrane voltage regimes:
        # > +0.3: Anterior (Head), -0.3 to +0.3: Trunk, < -0.3: Posterior (Tail)
        differentiated = np.zeros((h, w), dtype=np.int32)
        differentiated[self.voltages > 0.3] = 1
        differentiated[(self.voltages >= -0.3) & (self.voltages <= 0.3)] = 2
        differentiated[self.voltages < -0.3] = 3

        divergence = float(np.mean((self.voltages - self.v_target) ** 2))

        return MorphogenesisState(
            voltages=np.copy(self.voltages),
            gap_junction_conductance=np.full((h, w, 4), self.base_conductance),
            pattern_divergence=divergence,
            differentiated_types=differentiated,
        )
