"""Experiment M: Causal Emergence via Regional Coarse-Graining.

Evaluates Erik Hoel's Causal Emergence hypothesis:
Tests whether macro-scale regional groupings exhibit higher Effective Information (EI)
than microscopic node-level dynamics due to reduced indeterminism and degeneracy.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..structure.genome import TopologyFamily
from ..structure.topology import TopologyGenerator
from ..theories.causal_emergence import CausalEmergenceEngine
from .runner import ExperimentRunner


def run_experiment_m(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()
    engine = CausalEmergenceEngine(seed=42)

    def treatment_macro_ei(seed: int) -> float:
        gen = TopologyGenerator(seed=seed)
        net = gen.generate(TopologyFamily.HIERARCHICAL_MODULAR, num_nodes=16)
        W = net.to_matrix()
        res = engine.evaluate_causal_emergence(W)
        return float(res["ei_macro"])

    def control_micro_ei(seed: int) -> float:
        gen = TopologyGenerator(seed=seed)
        net = gen.generate(TopologyFamily.HIERARCHICAL_MODULAR, num_nodes=16)
        W = net.to_matrix()
        res = engine.evaluate_causal_emergence(W)
        return float(res["ei_micro"])

    res = runner.run_controlled_experiment(
        exp_id="EXP_M_CAUSAL_EMERGENCE",
        title="Macro-Scale Causal Emergence of Effective Information",
        theory="Causal-Emergence",
        statement="Macro-scale regional groupings exhibit significantly higher Effective Information (EI) than microscopic node-level networks.",
        target_metric="effective_information",
        treatment_fn=treatment_macro_ei,
        control_fn=control_micro_ei,
        sample_size=sample_size,
    )
    return res
