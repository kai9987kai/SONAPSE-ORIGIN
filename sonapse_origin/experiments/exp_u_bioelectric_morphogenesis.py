"""Experiment U: Bioelectric Morphogenesis Pattern Restoration Post-Lesion.

Evaluates whether gap-junction mediated somatic bioelectric diffusion enables autonomous
anatomical pattern restoration post severe surgical excision (lesion) compared to uncoupled tissue.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..structure.bioelectric_morphogenesis import BioelectricMorphogeneticTissue
from .runner import ExperimentRunner


def run_experiment_u(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_coupled_gap_junctions(seed: int) -> float:
        tissue = BioelectricMorphogeneticTissue(grid_shape=(8, 8), base_conductance=0.4, seed=seed)
        tissue.step(steps=20)
        tissue.apply_lesion(fraction_cut=0.5)
        # Regeneration phase
        tissue.step(steps=40)
        state = tissue.get_state()
        recovery_score = max(0.0, 1.0 - state.pattern_divergence)
        return float(recovery_score)

    def control_uncoupled_cells(seed: int) -> float:
        # Zero gap junction conductance (isolated independent cells cannot communicate spatial gradients)
        tissue = BioelectricMorphogeneticTissue(grid_shape=(8, 8), base_conductance=0.0, seed=seed)
        tissue.step(steps=20)
        tissue.apply_lesion(fraction_cut=0.5)
        tissue.step(steps=40)
        state = tissue.get_state()
        recovery_score = max(0.0, 1.0 - state.pattern_divergence)
        return float(recovery_score)

    return runner.run_controlled_experiment(
        exp_id="EXP_U_BIOELECTRIC_MORPHOGENESIS",
        title="Bioelectric Gap-Junction Mediated Pattern Restoration Post-Lesion",
        theory="Bioelectric-Morphogenesis",
        statement="Somatic tissue with active bioelectric gap junctions recovers anatomical pattern divergence significantly better post severe lesion than uncoupled tissue.",
        target_metric="morphological_pattern_recovery",
        treatment_fn=treatment_coupled_gap_junctions,
        control_fn=control_uncoupled_cells,
        sample_size=sample_size,
    )
