"""Experiment B: Homeostasis and Learning.

Compares agents with and without internal homeostatic regulatory variables;
tests survival-like persistence and resource-seeking adaptation.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..environments.worlds import MinimalRegulatoryWorld
from ..homeostasis.viability import HomeostaticRegulator
from .runner import ExperimentRunner


def run_experiment_b(sample_size: int = 12) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_homeostatic(seed: int) -> float:
        env = MinimalRegulatoryWorld(max_steps=50, seed=seed)
        obs, _ = env.reset()
        reg = HomeostaticRegulator()
        h_vec = np.array([0.8, 1.0, 0.5, 0.1, 0.8, 0.1, 0.05, 0.5, 0.95])
        survival_steps = 0

        for _ in range(50):
            drives = reg.compute_regulatory_drives(h_vec)
            # Action selection biased by homeostatic gradient:
            action = int(np.argmax([0.1, drives[0] * 2.0, abs(drives[2]) * 2.0]))
            step_res = env.step(action)
            # Update h_vec based on environment
            h_vec[0] = float(np.clip(h_vec[0] + (0.1 if action == 1 else -0.02), 0.0, 1.0))
            h_vec[2] = float(step_res.observation[0])
            if not reg.is_viable(h_vec):
                break
            survival_steps += 1
        return float(survival_steps)

    def control_non_homeostatic(seed: int) -> float:
        env = MinimalRegulatoryWorld(max_steps=50, seed=seed)
        env.reset()
        rng = np.random.default_rng(seed)
        reg = HomeostaticRegulator()
        h_vec = np.array([0.8, 1.0, 0.5, 0.1, 0.8, 0.1, 0.05, 0.5, 0.95])
        survival_steps = 0

        # Blind / random action selection without interoceptive feedback
        for _ in range(50):
            action = int(rng.integers(0, 3))
            step_res = env.step(action)
            h_vec[0] = float(np.clip(h_vec[0] + (0.1 if action == 1 else -0.02), 0.0, 1.0))
            h_vec[2] = float(step_res.observation[0])
            if not reg.is_viable(h_vec):
                break
            survival_steps += 1
        return float(survival_steps)

    res = runner.run_controlled_experiment(
        exp_id="EXP_B_HOMEOSTASIS_LEARNING",
        title="Homeostatic Regulation Enhances Viability Longevity",
        theory="Homeostasis",
        statement="Agents equipped with homeostatic regulatory feedback survive significantly longer than open-loop non-homeostatic controls.",
        target_metric="survival_steps",
        treatment_fn=treatment_homeostatic,
        control_fn=control_non_homeostatic,
        sample_size=sample_size,
    )
    return res
