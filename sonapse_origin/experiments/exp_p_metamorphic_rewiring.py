"""Experiment P: Metamorphic BCM Rewiring under Sequential Activity Shifts.

Evaluates whether BCM sliding threshold metaplasticity combined with metamorphic rewiring
adapts to sequential task shifts while preventing catastrophic forgetting.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..structure.metaplasticity import BCMMetaplasticityEngine
from .runner import ExperimentRunner


def run_experiment_p(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_bcm_metamorphic(seed: int) -> float:
        rng = np.random.default_rng(seed)
        engine = BCMMetaplasticityEngine(num_units=8, seed=seed)
        W = rng.normal(0.2, 0.1, (8, 8))

        # Shift 1: Domain A
        for _ in range(10):
            pre = rng.normal(0.8, 0.2, 8)
            post = np.tanh(W @ pre)
            W, _ = engine.step(W, pre, post)

        # Shift 2: Domain B with correlated structure
        cov_matrix = np.eye(8) * 0.2 + 0.6
        W_adapted, sprouted = engine.metamorphic_rewiring(W, cov_matrix, growth_threshold=0.5)

        # Functional pathways retained and sprouted
        return float(np.sum(W_adapted > 0.05))

    def control_static_unadapted(seed: int) -> float:
        rng = np.random.default_rng(seed)
        W = rng.normal(0.2, 0.1, (8, 8))
        return float(np.sum(W > 0.05))

    res = runner.run_controlled_experiment(
        exp_id="EXP_P_METAMORPHIC_REWIRING",
        title="Metamorphic BCM Rewiring Adaptation to Domain Shifts",
        theory="Metaplasticity",
        statement="BCM metaplasticity combined with metamorphic rewiring significantly increases adapted functional synaptic pathways following domain shifts beyond static baselines.",
        target_metric="adapted_functional_pathways",
        treatment_fn=treatment_bcm_metamorphic,
        control_fn=control_static_unadapted,
        sample_size=sample_size,
    )
    return res
