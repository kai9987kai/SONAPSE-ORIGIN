"""Leaky Integrate-and-Fire (LIF) Neural Dynamics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class LIFPopulation:
    """Leaky Integrate-and-Fire population with refractory periods and synaptic currents:
    
        tau_m * dV_i/dt = -(V_i - V_rest) + R * I_i(t)
    """

    def __init__(
        self,
        num_neurons: int,
        tau_m: float = 20.0,      # Membrane time constant (ms)
        v_rest: float = -65.0,    # Resting membrane potential (mV)
        v_reset: float = -70.0,   # Post-spike reset potential (mV)
        v_thresh: float = -50.0,  # Firing threshold (mV)
        tau_syn: float = 5.0,     # Synaptic decay constant (ms)
        refractory_ms: float = 2.0,# Refractory period duration (ms)
        dt: float = 0.5,          # Simulation step size (ms)
    ):
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.tau_syn = tau_syn
        self.refractory_ms = refractory_ms
        self.dt = dt

        self.v = np.full(num_neurons, v_rest, dtype=np.float64)
        self.i_syn = np.zeros(num_neurons, dtype=np.float64)
        self.refractory_timer = np.zeros(num_neurons, dtype=np.float64)
        self.spike_history: List[np.ndarray] = []

    def reset(self) -> None:
        self.v.fill(self.v_rest)
        self.i_syn.fill(0.0)
        self.refractory_timer.fill(0.0)
        self.spike_history.clear()

    def step(
        self,
        W: np.ndarray,
        external_current: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Integrates forward by dt. Returns (spikes, membrane_potentials)."""
        ext = external_current if external_current is not None else np.zeros(self.num_neurons)

        # 1. Update refractory timers
        non_refractory = self.refractory_timer <= 0.0
        self.refractory_timer = np.maximum(0.0, self.refractory_timer - self.dt)

        # 2. Membrane potential update for non-refractory neurons
        dv = (-(self.v - self.v_rest) + (self.i_syn + ext)) / self.tau_m
        self.v[non_refractory] += dv[non_refractory] * self.dt

        # 3. Detect spikes
        spikes = (self.v >= self.v_thresh) & non_refractory
        self.v[spikes] = self.v_reset
        self.refractory_timer[spikes] = self.refractory_ms
        self.spike_history.append(spikes.copy())

        # 4. Synaptic input update: decay current + add incoming spikes
        # I_syn = I_syn * exp(-dt / tau_syn) + W @ spikes
        syn_decay = np.exp(-self.dt / self.tau_syn)
        incoming_input = W @ spikes.astype(np.float64)
        self.i_syn = self.i_syn * syn_decay + incoming_input

        return spikes, self.v.copy()
