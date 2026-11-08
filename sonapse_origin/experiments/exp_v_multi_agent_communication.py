"""Experiment V: Multi-Agent Emergent Communication & Dialect Formation.

Evaluates whether ecological viability constraints drive reinforcement-based sender-receiver agents
to discover a shared discrete lexicon with high mutual information I(Symbol; Hazard) exceeding random signaling.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..environments.multi_agent_communication import EmergentCommunicationWorld
from .runner import ExperimentRunner


def run_experiment_v(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_interactive_learning(seed: int) -> float:
        world = EmergentCommunicationWorld(num_hazards=4, alphabet_size=6, seed=seed)
        metrics = world.train_dialects(episodes=350, learning_rate=0.08)
        return float(metrics.communicative_accuracy)

    def control_random_signaling(seed: int) -> float:
        world = EmergentCommunicationWorld(num_hazards=4, alphabet_size=6, seed=seed)
        # 0 learning rate: unadjusted random signaling baseline
        metrics = world.train_dialects(episodes=350, learning_rate=0.0)
        return float(metrics.communicative_accuracy)

    return runner.run_controlled_experiment(
        exp_id="EXP_V_EMERGENT_COMMUNICATION",
        title="Emergent Symbolic Communication Under Environmental Hazards",
        theory="Social-Communication",
        statement="Reinforcement under shared viability threats drives autonomous agents to discover high communicative decoding accuracy significantly exceeding unconstrained baseline signaling.",
        target_metric="communicative_accuracy",
        treatment_fn=treatment_interactive_learning,
        control_fn=control_random_signaling,
        sample_size=sample_size,
    )
