"""Experiment K: Allostatic Active Inference vs Reactive Homeostasis.

Evaluates whether predictive allostasis minimizing Expected Free Energy (EFE) achieves
statistically significant survival longevity compared to reactive homeostatic setpoint control.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..environments.worlds import MinimalRegulatoryWorld
from ..homeostasis.allostasis import AllostaticActiveInferenceController
from ..homeostasis.viability import HomeostaticRegulator
from .runner import ExperimentRunner


def run_experiment_k(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_allostatic_inference(seed: int) -> float:
        env = MinimalRegulatoryWorld(max_steps=80, seed=seed)
        obs, _ = env.reset()
        controller = AllostaticActiveInferenceController(lookahead_steps=5, seed=seed)
        reg = HomeostaticRegulator()
        h_vec = np.array([0.8, 1.0, 0.5, 0.1, 0.8, 0.1, 0.05, 0.5, 0.95])
        survival_steps = 0

        for _ in range(80):
            action, _ = controller.select_action(h_vec[:4], ambient_hazard=float(obs[2]))
            step_res = env.step(action)
            # Update interoceptive state
            h_vec[0] = float(np.clip(h_vec[0] + (0.12 if action == 1 else -0.02), 0.0, 1.0))
            h_vec[2] = float(step_res.observation[0])
            h_vec[3] = float(np.clip(h_vec[3] + (-0.03 if action == 0 else 0.015), 0.0, 1.0))
            obs = step_res.observation
            if not reg.is_viable(h_vec):
                break
            survival_steps += 1
        return float(survival_steps)

    def control_reactive_homeostasis(seed: int) -> float:
        env = MinimalRegulatoryWorld(max_steps=80, seed=seed)
        env.reset()
        reg = HomeostaticRegulator()
        h_vec = np.array([0.8, 1.0, 0.5, 0.1, 0.8, 0.1, 0.05, 0.5, 0.95])
        survival_steps = 0

        for _ in range(80):
            drives = reg.compute_regulatory_drives(h_vec)
            # Reactive greedy action: only acts when drive exceeds threshold
            action = int(np.argmax([0.1, drives[0], abs(drives[2])]))
            step_res = env.step(action)
            h_vec[0] = float(np.clip(h_vec[0] + (0.12 if action == 1 else -0.02), 0.0, 1.0))
            h_vec[2] = float(step_res.observation[0])
            h_vec[3] = float(np.clip(h_vec[3] + (-0.03 if action == 0 else 0.015), 0.0, 1.0))
            if not reg.is_viable(h_vec):
                break
            survival_steps += 1
        return float(survival_steps)

    res = runner.run_controlled_experiment(
        exp_id="EXP_K_ALLOSTATIC_INFERENCE",
        title="Predictive Allostasis vs Reactive Homeostasis Longevity",
        theory="Active-Inference",
        statement="Allostatic active inference minimizing Expected Free Energy sustains viability significantly longer than reactive homeostatic regulation under ambient volatility.",
        target_metric="survival_steps",
        treatment_fn=treatment_allostatic_inference,
        control_fn=control_reactive_homeostasis,
        sample_size=sample_size,
    )
    return res
