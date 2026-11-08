"""Temporal Binding by Synchrony Experiments."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TemporalBindingExperiment:
    """Evaluates whether distributed neural processes form stable temporally coordinated representations."""

    def __init__(self, num_features: int = 8, seed: int = 42):
        self.num_features = num_features
        self.rng = np.random.default_rng(seed)

    def measure_phase_locking_value(
        self,
        phase_time_series_1: np.ndarray,
        phase_time_series_2: np.ndarray,
    ) -> float:
        """Computes Phase-Locking Value (PLV):
        
            PLV = (1/T) * | sum_{t=1}^T exp(i * (theta_1(t) - theta_2(t))) |
        """
        phase_diff = phase_time_series_1 - phase_time_series_2
        complex_plv = np.mean(np.exp(1j * phase_diff))
        return float(np.abs(complex_plv))

    def evaluate_binding_coherence(
        self,
        coupled_phases: np.ndarray,
        uncoupled_phases: np.ndarray,
    ) -> Dict[str, float]:
        """Compares coherence between bound features vs unbound control features.
        
        Input shape: (T, num_features)
        """
        T = coupled_phases.shape[0]
        # Cross-feature average PLV
        plv_coupled_list = []
        plv_uncoupled_list = []

        for i in range(self.num_features):
            for j in range(i + 1, self.num_features):
                plv_c = self.measure_phase_locking_value(coupled_phases[:, i], coupled_phases[:, j])
                plv_u = self.measure_phase_locking_value(uncoupled_phases[:, i], uncoupled_phases[:, j])
                plv_coupled_list.append(plv_c)
                plv_uncoupled_list.append(plv_u)

        return {
            "mean_coupled_plv": float(np.mean(plv_coupled_list)),
            "mean_uncoupled_plv": float(np.mean(plv_uncoupled_list)),
            "binding_contrast_ratio": float(np.mean(plv_coupled_list) / (np.mean(plv_uncoupled_list) + 1e-6)),
        }
