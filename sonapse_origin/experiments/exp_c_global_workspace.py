"""Experiment C: Global Workspace Ablation.

Compares modular processing with and without global workspace broadcast access;
measures cross-module information accessibility and reach.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..theories.global_workspace import GlobalWorkspaceExperiment
from .runner import ExperimentRunner


def run_experiment_c(sample_size: int = 12) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_intact_workspace(seed: int) -> float:
        gw = GlobalWorkspaceExperiment(num_modules=4, module_dim=6, workspace_dim=10, seed=seed)
        modules = [np.ones(6) * 1.5, np.zeros(6), np.zeros(6), np.zeros(6)]
        step_res = gw.step(modules, workspace_enabled=True)
        return float(step_res["broadcast_reach"])

    def control_ablated_workspace(seed: int) -> float:
        gw = GlobalWorkspaceExperiment(num_modules=4, module_dim=6, workspace_dim=10, seed=seed)
        modules = [np.ones(6) * 1.5, np.zeros(6), np.zeros(6), np.zeros(6)]
        step_res = gw.step(modules, workspace_enabled=False)
        return float(step_res["broadcast_reach"])

    res = runner.run_controlled_experiment(
        exp_id="EXP_C_GLOBAL_WORKSPACE",
        title="Global Workspace Broadcast Reach vs Ablated Control",
        theory="GWT",
        statement="Intact global workspace hubs achieve complete modular broadcast reach compared with ablated controls.",
        target_metric="broadcast_reach",
        treatment_fn=treatment_intact_workspace,
        control_fn=control_ablated_workspace,
        sample_size=sample_size,
    )
    return res
