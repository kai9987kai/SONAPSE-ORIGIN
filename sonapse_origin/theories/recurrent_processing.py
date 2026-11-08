"""Recurrent Processing Theory (RPT) Experiments."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class RecurrentProcessingExperiment:
    """Compares recurrent processing against parameter-matched feedforward systems and feedback lesions."""

    def __init__(self, num_units: int = 16, seed: int = 42):
        self.num_units = num_units
        self.rng = np.random.default_rng(seed)

        # Upper triangular = feedforward sweep; lower triangular = feedback pathways
        self.W_feedforward = np.triu(self.rng.normal(0, 0.3, (num_units, num_units)), k=1)
        self.W_feedback = np.tril(self.rng.normal(0, 0.3, (num_units, num_units)), k=-1)
        self.W_recurrent = self.W_feedforward + self.W_feedback

    def evaluate_persistence_and_completion(
        self,
        input_stimulus: np.ndarray,
        timesteps: int = 15,
        lesion_feedback: bool = False,
    ) -> Dict[str, Any]:
        """Runs temporal processing with intact vs lesioned feedback pathways."""
        W = self.W_feedforward if lesion_feedback else self.W_recurrent

        x = np.zeros(self.num_units)
        trajectory = []

        # Stimulus presentation for first 3 timesteps only
        for t in range(timesteps):
            stim = input_stimulus if t < 3 else np.zeros_like(input_stimulus)
            x = np.tanh(W @ x + stim)
            trajectory.append(x.copy())

        traj_arr = np.array(trajectory)  # (T, num_units)

        # Metrics:
        # 1. Temporal persistence: activity norm after stimulus offset
        post_stim_norm = float(np.mean(np.linalg.norm(traj_arr[3:], axis=1)))
        # 2. Pattern stability: autocorrelation across time steps
        stability = float(np.corrcoef(traj_arr[-2], traj_arr[-1])[0, 1]) if np.std(traj_arr[-1]) > 1e-6 else 0.0

        return {
            "lesion_feedback": lesion_feedback,
            "post_stimulus_persistence": post_stim_norm,
            "pattern_stability": stability,
            "final_state_norm": float(np.linalg.norm(x)),
        }
