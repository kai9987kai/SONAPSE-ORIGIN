"""Attention Schema Theory (AST) Experiments."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class AttentionSchemaExperiment:
    """Investigates whether an internal model of attention enhances control and distracter resistance."""

    def __init__(self, num_targets: int = 5, seed: int = 42):
        self.num_targets = num_targets
        self.rng = np.random.default_rng(seed)

        # Actual sensory attention weights (physical allocation): shape (num_targets,)
        self.attention = np.ones(num_targets) / num_targets
        # Attention schema (internal descriptive model of where attention is directed): shape (num_targets,)
        self.schema = np.ones(num_targets) / num_targets

    def track_target_under_distraction(
        self,
        target_idx: int = 2,
        distraction_level: float = 0.5,
        has_schema: bool = True,
        timesteps: int = 20,
    ) -> Dict[str, float]:
        """Runs tracking simulation under sensory distracters with vs without attention schema."""
        target_focus_history = []

        for t in range(timesteps):
            # External sensory salience: high on target, but bursts of random distraction
            salience = np.zeros(self.num_targets)
            salience[target_idx] = 1.0
            # Distracter bursts
            if self.rng.random() < 0.4:
                distracter_idx = int(self.rng.choice([i for i in range(self.num_targets) if i != target_idx]))
                salience[distracter_idx] += distraction_level * 1.5

            if has_schema:
                # AST feedback: the internal schema detects discrepancy between intended and actual attention
                # and guides compensatory top-down biasing
                intended_focus = np.zeros(self.num_targets)
                intended_focus[target_idx] = 1.0
                discrepancy = intended_focus - self.schema
                top_down_correction = 0.4 * discrepancy
                # Update schema towards actual attention
                self.schema += 0.2 * (self.attention - self.schema)
            else:
                top_down_correction = np.zeros(self.num_targets)

            # Attention update: driven by bottom-up salience plus top-down schema correction
            raw_att = self.attention * 0.7 + salience * 0.3 + top_down_correction
            # Softmax normalization
            exp_att = np.exp(raw_att - np.max(raw_att))
            self.attention = exp_att / np.sum(exp_att)

            target_focus_history.append(float(self.attention[target_idx]))

        mean_target_focus = float(np.mean(target_focus_history))
        distraction_intrusions = float(np.sum([1.0 for f in target_focus_history if f < 0.4]))

        return {
            "has_schema": float(has_schema),
            "mean_target_focus": mean_target_focus,
            "distraction_intrusions": distraction_intrusions,
            "final_target_focus": target_focus_history[-1],
        }
