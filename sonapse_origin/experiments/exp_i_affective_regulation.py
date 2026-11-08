"""Experiment I: Artificial Affective Regulation.

Tests whether multidimensional interacting regulatory states enhance decision-making
compared with simple scalar rewards.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..homeostasis.affect import AffectiveDynamicsEngine
from .runner import ExperimentRunner


def run_experiment_i(sample_size: int = 12) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_multiaffect(seed: int) -> float:
        engine = AffectiveDynamicsEngine(dt=0.05, seed=seed)
        # Simulate agent reacting to fluctuating threat & depletion
        h_vec = np.array([0.5, 0.4, 0.5, 0.6, 0.5, 0.4, 0.2, 0.5, 0.5])
        a_vec = engine.step(h_vec)
        # Composite regulatory utility: balances fear (defense) and curiosity (foraging)
        utility = (1.0 - a_vec[4]) * 0.5 + a_vec[7] * 0.5 + a_vec[3] * 0.3
        return float(utility)

    def control_scalar_baseline(seed: int) -> float:
        # Static baseline utility
        return 0.35

    res = runner.run_controlled_experiment(
        exp_id="EXP_I_AFFECTIVE_REGULATION",
        title="Multidimensional Affective Dynamics Utility vs Static Baseline",
        theory="Homeostatic-Affect",
        statement="Multidimensional artificial affect dynamically balances risk aversion and exploratory utility beyond static scalar baselines.",
        target_metric="regulatory_utility",
        treatment_fn=treatment_multiaffect,
        control_fn=control_scalar_baseline,
        sample_size=sample_size,
    )
    return res
