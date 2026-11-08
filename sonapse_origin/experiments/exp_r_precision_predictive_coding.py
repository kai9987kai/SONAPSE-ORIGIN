"""Experiment R: Precision-Weighted Predictive Coding under Sensory Noise.

Tests whether dynamic precision weighting (attenuating noisy sensory gain and amplifying
top-down prior precision) yields superior sensory reconstruction fidelity under corruption.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..theories.precision_predictive_coding import HierarchicalPredictiveCoder
from .runner import ExperimentRunner


def run_experiment_r(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_dynamic_precision(seed: int) -> float:
        coder = HierarchicalPredictiveCoder(layer_sizes=(16, 8, 4), seed=seed)
        clean = np.sin(np.linspace(0, 2 * np.pi, 16))
        res = coder.evaluate_noise_robustness(clean, noise_level=0.6, dynamic_precision=True)
        fidelity = max(0.0, 1.0 - res["reconstruction_error"])
        return float(fidelity)

    def control_static_precision(seed: int) -> float:
        coder = HierarchicalPredictiveCoder(layer_sizes=(16, 8, 4), seed=seed)
        clean = np.sin(np.linspace(0, 2 * np.pi, 16))
        res = coder.evaluate_noise_robustness(clean, noise_level=0.6, dynamic_precision=False)
        fidelity = max(0.0, 1.0 - res["reconstruction_error"])
        return float(fidelity)

    return runner.run_controlled_experiment(
        exp_id="EXP_R_PRECISION_PREDICTIVE_CODING",
        title="Dynamic Precision Weighting Under Sensory Observation Noise",
        theory="Predictive-Processing",
        statement="Dynamic precision weighting yields significantly higher sensory reconstruction fidelity under severe observation noise compared to static unweighted predictive coding.",
        target_metric="reconstruction_fidelity",
        treatment_fn=treatment_dynamic_precision,
        control_fn=control_static_precision,
        sample_size=sample_size,
    )
