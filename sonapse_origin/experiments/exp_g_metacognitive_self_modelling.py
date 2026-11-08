"""Experiment G: Metacognitive Self-Modelling.

Compares networks with explicit second-order self-state models against first-order baselines;
evaluates Type-2 ROC sensitivity and error awareness.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..theories.higher_order import HigherOrderRepresentationAssay
from .runner import ExperimentRunner


def run_experiment_g(sample_size: int = 12) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_meta_model(seed: int) -> float:
        assay = HigherOrderRepresentationAssay(first_order_dim=12, meta_dim=6, seed=seed)
        res = assay.evaluate_metacognitive_sensitivity(num_trials=80, noise_level=0.3)
        return float(res["type2_roc_auc"])

    def control_chance_baseline(seed: int) -> float:
        # First-order uncalibrated chance baseline
        return 0.50

    res = runner.run_controlled_experiment(
        exp_id="EXP_G_METACOGNITION",
        title="Higher-Order Representation Yields Above-Chance Metacognitive Sensitivity",
        theory="HOT",
        statement="Networks with explicit second-order self-state representations achieve Type-2 ROC sensitivity significantly superior to chance.",
        target_metric="type2_roc_auc",
        treatment_fn=treatment_meta_model,
        control_fn=control_chance_baseline,
        sample_size=sample_size,
    )
    return res
