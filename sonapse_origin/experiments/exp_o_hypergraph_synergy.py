"""Experiment O: Higher-Order Synergy in Dynamic Hypergraphs.

Tests Rosas et al. (2020) O-Information framework:
Evaluates whether non-linear hypergraph communication generates synergistic O-information (Omega < 0)
beyond parameter-matched pairwise graph projections.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..structure.hypergraph import HypergraphMessagePassing, MatchedGraphBaseline
from ..structure.synergy import HypergraphSynergyAssay
from .runner import ExperimentRunner


def run_experiment_o(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()
    assay = HypergraphSynergyAssay(seed=42)

    def treatment_hypergraph_synergy_boost(seed: int) -> float:
        rng = np.random.default_rng(seed)
        num_nodes = 8
        hyperedges = [[0, 1, 2], [2, 3, 4], [4, 5, 6, 7]]
        hg = HypergraphMessagePassing(hyperedges, node_dim=3)
        base = MatchedGraphBaseline(hyperedges, num_nodes=num_nodes, node_dim=3)

        # Sample joint activity across multiple timesteps
        X_samples_hg = []
        X_samples_base = []
        for _ in range(50):
            X = rng.normal(0, 1.0, (num_nodes, 3))
            out_hg = hg.forward(X)
            out_base = base.forward(X)
            X_samples_hg.append(np.mean(out_hg, axis=-1))
            X_samples_base.append(np.mean(out_base, axis=-1))

        res = assay.evaluate_hyperedge_synergy(
            np.array(X_samples_hg)[:, :4],
            np.array(X_samples_base)[:, :4],
        )
        # Positive synergy boost: Omega_pairwise - Omega_hyperedge
        return float(res["hyperedge_synergy_boost"])

    def control_null_baseline(seed: int) -> float:
        # Zero synergy boost control
        return 0.0

    res = runner.run_controlled_experiment(
        exp_id="EXP_O_HYPERGRAPH_SYNERGY",
        title="Higher-Order Hypergraph Synergy Boost (O-Information)",
        theory="Higher-Order-Topology",
        statement="Non-linear hyperedge interaction generates significantly higher statistical synergy (Omega reduction) than pairwise linear baselines.",
        target_metric="synergy_boost",
        treatment_fn=treatment_hypergraph_synergy_boost,
        control_fn=control_null_baseline,
        sample_size=sample_size,
    )
    return res
