"""Unified Dynamical Substrate Simulator."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .spiking import LIFPopulation
from .rate_based import ContinuousRateRNN
from .oscillators import KuramotoOscillators
from .reaction_diffusion import ContinuousNeuralField
from .wave_comm import SignalChannelComparator, SignalSubstrateMode


class UnifiedDynamicalSimulator:
    """Coordinates simulation across spiking, rate-based, oscillatory, and field substrates."""

    def __init__(
        self,
        substrate_type: str = "rate",
        num_units: int = 32,
        dt: float = 0.5,
        seed: int = 42,
    ):
        self.substrate_type = substrate_type
        self.num_units = num_units
        self.dt = dt
        self.seed = seed

        if substrate_type == "lif":
            self.lif = LIFPopulation(num_neurons=num_units, dt=dt)
        elif substrate_type == "kuramoto":
            self.osc = KuramotoOscillators(num_oscillators=num_units, dt=dt, seed=seed)
        elif substrate_type == "field":
            self.field = ContinuousNeuralField(grid_size=int(np.sqrt(num_units)) or 8, dt=dt)
        else:
            self.rate = ContinuousRateRNN(num_units=num_units, dt=dt)

        self.step_count = 0

    def step(
        self,
        W: np.ndarray,
        external_input: Optional[np.ndarray] = None,
        intervention: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs one forward simulation step with optional causal lesion / clamping."""
        u = intervention or {}
        W_eff = W.copy()

        # Handle lesion intervention: zero out rows/cols for lesioned units
        if "lesion_units" in u:
            for unit in u["lesion_units"]:
                W_eff[unit, :] = 0.0
                W_eff[:, unit] = 0.0

        ext = external_input.copy() if external_input is not None else np.zeros(self.num_units)
        if "clamp_units" in u:
            for unit, val in u["clamp_units"].items():
                ext[int(unit)] = val

        if self.substrate_type == "lif":
            spikes, v = self.lif.step(W_eff, external_current=ext)
            state_out = v
            aux = {"spikes": spikes.tolist(), "firing_rate": float(np.mean(spikes))}
        elif self.substrate_type == "kuramoto":
            phases, r = self.osc.step(coupling_matrix=W_eff, external_drive=ext)
            state_out = phases
            aux = {"coherence_r": r}
        elif self.substrate_type == "field":
            field_u = self.field.step(external_input=ext.reshape(self.field.u.shape) if ext.size == self.field.u.size else None)
            state_out = field_u.flatten()[:self.num_units]
            aux = {"mean_field": float(np.mean(field_u))}
        else:
            rates = self.rate.step(W_eff, external_input=ext)
            state_out = rates
            aux = {"mean_rate": float(np.mean(rates))}

        self.step_count += 1
        return {
            "step": self.step_count,
            "substrate": self.substrate_type,
            "state": state_out,
            "auxiliary": aux,
        }
