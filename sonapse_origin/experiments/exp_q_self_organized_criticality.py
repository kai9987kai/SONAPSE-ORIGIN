"""Experiment Q: Self-Organized Criticality via Dynamic Homeostatic Scaling.

Evaluates whether dynamic synaptic scaling drives neural avalanche dynamics toward
the critical branching ratio (sigma ~ 1.0) compared to static unscaled connectivity.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..substrate.criticality import CriticalitySubstrate
from .runner import ExperimentRunner


def run_experiment_q(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_homeostatic_soc(seed: int) -> float:
        sub = CriticalitySubstrate(num_nodes=32, seed=seed)
        # Induce supercritical initial condition (elevated recurrent gain)
        sub.W *= 2.5
        spikes, _ = sub.simulate(
            timesteps=300,
            homeostatic_adaptation=True,
            adaptation_rate=0.03,
        )
        stats = sub.extract_avalanches(spikes, bin_size=1)
        # Proximity to criticality: 1.0 - abs(1.0 - sigma)
        proximity = max(0.0, 1.0 - abs(1.0 - stats.branching_parameter))
        return float(proximity)

    def control_static_unscaled(seed: int) -> float:
        sub = CriticalitySubstrate(num_nodes=32, seed=seed)
        sub.W *= 2.5
        spikes, _ = sub.simulate(
            timesteps=300,
            homeostatic_adaptation=False,
        )
        stats = sub.extract_avalanches(spikes, bin_size=1)
        proximity = max(0.0, 1.0 - abs(1.0 - stats.branching_parameter))
        return float(proximity)

    return runner.run_controlled_experiment(
        exp_id="EXP_Q_NEURAL_CRITICALITY",
        title="Self-Organized Criticality via Dynamic Homeostatic Synaptic Scaling",
        theory="Neural-Criticality",
        statement="Dynamic homeostatic synaptic scaling tunes the network branching ratio significantly closer to the critical boundary (sigma ~ 1.0) than static unscaled connectivity.",
        target_metric="criticality_proximity",
        treatment_fn=treatment_homeostatic_soc,
        control_fn=control_static_unscaled,
        sample_size=sample_size,
    )
