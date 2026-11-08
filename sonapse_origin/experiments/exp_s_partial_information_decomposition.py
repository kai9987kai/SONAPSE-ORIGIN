"""Experiment S: Partial Information Decomposition Synergy in Multi-Modal Binding.

Evaluates whether non-linear sensorimotor integration generates significant information synergy
S(X1, X2 -> Y) exceeding independent linear combinations.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..theories.pid import PartialInformationDecomposition
from .runner import ExperimentRunner


def run_experiment_s(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()
    pid = PartialInformationDecomposition()

    def treatment_nonlinear_binding(seed: int) -> float:
        rng = np.random.default_rng(seed)
        n = 300
        x1 = rng.integers(0, 2, n)
        x2 = rng.integers(0, 2, n)
        # Non-linear XOR-like synergistic target with slight noise
        y = (x1 ^ x2) ^ (rng.random(n) < 0.05).astype(int)
        res = pid.decompose(x1, x2, y)
        return float(res.synergy_fraction)

    def control_linear_additive(seed: int) -> float:
        rng = np.random.default_rng(seed)
        n = 300
        x1 = rng.integers(0, 2, n)
        x2 = rng.integers(0, 2, n)
        # Redundant/Independent linear mapping: Y copied primarily from X1
        y = x1 ^ (rng.random(n) < 0.05).astype(int)
        res = pid.decompose(x1, x2, y)
        return float(res.synergy_fraction)

    return runner.run_controlled_experiment(
        exp_id="EXP_S_PID_SYNERGY",
        title="Partial Information Decomposition Synergy in Multi-Modal Binding",
        theory="Information-Decomposition",
        statement="Non-linear sensorimotor binding generates a significantly higher synergy fraction than linear independent signal transmission.",
        target_metric="synergy_fraction",
        treatment_fn=treatment_nonlinear_binding,
        control_fn=control_linear_additive,
        sample_size=sample_size,
    )
