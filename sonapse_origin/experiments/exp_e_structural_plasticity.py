"""Experiment E: Structural Plasticity.

Compares dynamically evolving topologies against static fixed graphs;
measures adaptation to changing environmental conditions.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..structure.development import StructuralPlasticityEngine
from .runner import ExperimentRunner


def run_experiment_e(sample_size: int = 12) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_plastic(seed: int) -> float:
        rng = np.random.default_rng(seed)
        engine = StructuralPlasticityEngine(pruning_threshold=0.1, seed=seed)
        W = rng.normal(0.2, 0.2, (8, 8))
        # Environmental shift simulation: activity pattern changes
        activity = rng.normal(0.5, 0.2, (8, 20))
        # Dynamic synaptogenesis
        W_adapted, sprouted = engine.sprout_synapses(W, activity)
        # Performance proxy: total effective functional pathways
        return float(np.sum(W_adapted > 0.05))

    def control_static(seed: int) -> float:
        rng = np.random.default_rng(seed)
        W = rng.normal(0.2, 0.2, (8, 8))
        # Fixed static graph: no sprouting or adaptation
        return float(np.sum(W > 0.05))

    res = runner.run_controlled_experiment(
        exp_id="EXP_E_STRUCTURAL_PLASTICITY",
        title="Structural Plasticity Sprouting vs Static Topology",
        theory="Neural-Development",
        statement="Dynamically evolving topologies sprout functional pathways to track changing activity correlations significantly beyond fixed baselines.",
        target_metric="active_synaptic_pathways",
        treatment_fn=treatment_plastic,
        control_fn=control_static,
        sample_size=sample_size,
    )
    return res
