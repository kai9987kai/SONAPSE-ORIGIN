"""Experiment A: Architecture and Integration.

Compares Feedforward, Recurrent, Modular, and Hypergraph networks under matched capacity;
measures causal integration proxy (I_partition) and noise robustness.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..structure.genome import TopologyFamily
from ..structure.topology import TopologyGenerator
from ..theories.causal_integration import CausalIntegrationAssay
from .runner import ExperimentRunner


def run_experiment_a(sample_size: int = 12) -> Dict[str, Any]:
    runner = ExperimentRunner()
    assay = CausalIntegrationAssay(num_nodes=16)

    def treatment_recurrent(seed: int) -> float:
        gen = TopologyGenerator(seed=seed)
        rec = gen.generate(TopologyFamily.RECURRENT, num_nodes=16)
        W = rec.to_matrix()
        return assay.partition_sensitivity_proxy(W, num_samples=50)

    def control_feedforward(seed: int) -> float:
        gen = TopologyGenerator(seed=seed)
        ff = gen.generate(TopologyFamily.FEEDFORWARD, num_nodes=16)
        W = ff.to_matrix()
        return assay.partition_sensitivity_proxy(W, num_samples=50)

    res = runner.run_controlled_experiment(
        exp_id="EXP_A_ARCH_INTEGRATION",
        title="Recurrent vs Feedforward Causal Integration",
        theory="IIT-Proxy",
        statement="Recurrent architectures produce significantly higher partition sensitivity (I_partition) than matched feedforward baselines.",
        target_metric="partition_sensitivity",
        treatment_fn=treatment_recurrent,
        control_fn=control_feedforward,
        sample_size=sample_size,
    )
    return res
