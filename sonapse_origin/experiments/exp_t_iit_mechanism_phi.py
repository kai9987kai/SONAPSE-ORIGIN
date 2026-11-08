"""Experiment T: IIT 4.0 Mechanism Cause-Effect Repertoire Irreducibility.

Evaluates whether integrated recurrent feedback loops generate non-zero mechanism-level
integrated information phi^mech compared to disconnected or purely feedforward networks.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..theories.iit_mechanism import IITMechanismAnalyzer
from .runner import ExperimentRunner


def run_experiment_t(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_recurrent_mechanism(seed: int) -> float:
        rng = np.random.default_rng(seed)
        num_nodes = 4
        # Dense recurrent connectivity with cross-talk
        W = rng.normal(0, 1.2, (num_nodes, num_nodes))
        analyzer = IITMechanismAnalyzer(num_nodes=num_nodes, transition_matrix=W, seed=seed)
        curr_state = np.array([1, 0, 1, 0])
        # Evaluate mechanism [0, 1] over purview [0, 1, 2, 3]
        rep = analyzer.evaluate_mechanism_phi(mechanism=[0, 1], purview=[0, 1, 2, 3], current_state=curr_state)
        return float(rep.phi_mech)

    def control_disconnected_mechanism(seed: int) -> float:
        num_nodes = 4
        # Diagonal uncoupled connections only (strictly partitioned)
        W = np.diag([1.2, 1.2, 1.2, 1.2])
        analyzer = IITMechanismAnalyzer(num_nodes=num_nodes, transition_matrix=W, seed=seed)
        curr_state = np.array([1, 0, 1, 0])
        rep = analyzer.evaluate_mechanism_phi(mechanism=[0, 1], purview=[0, 1, 2, 3], current_state=curr_state)
        return float(rep.phi_mech)

    return runner.run_controlled_experiment(
        exp_id="EXP_T_IIT4_MECHANISM_PHI",
        title="IIT 4.0 Mechanism Cause-Effect Repertoire Irreducibility (Small Phi)",
        theory="IIT-4.0",
        statement="Recurrent interconnected mechanisms produce significantly greater mechanism integrated information (phi^mech) across the Minimum Information Partition than disconnected controls.",
        target_metric="mechanism_phi",
        treatment_fn=treatment_recurrent_mechanism,
        control_fn=control_disconnected_mechanism,
        sample_size=sample_size,
    )
