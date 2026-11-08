"""Artificial Neuromodulatory Dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class NeuromodulatorLevels:
    dopamine: float = 0.5       # DA: Reward prediction error, vigor
    serotonin: float = 0.5      # 5-HT: Behavioral inhibition, calmness
    norepinephrine: float = 0.3 # NE: Arousal, signal gain, vigilance
    acetylcholine: float = 0.4  # ACh: Expected uncertainty, plasticity gate

    def to_array(self) -> np.ndarray:
        return np.array([self.dopamine, self.serotonin, self.norepinephrine, self.acetylcholine])


class NeuromodulatorySystem:
    """Computes dynamic modulatory baselines from homeostatic & affective states."""

    def __init__(self, baseline: Optional[NeuromodulatorLevels] = None):
        self.levels = baseline or NeuromodulatorLevels()

    def update(
        self,
        reward_error: float,
        threat_level: float,
        uncertainty: float,
        fatigue: float,
    ) -> NeuromodulatorLevels:
        """Dynamically shifts neuromodulator baselines."""
        # DA tracks reward prediction error
        self.levels.dopamine = float(np.clip(0.5 + 0.5 * reward_error, 0.05, 1.0))
        # 5-HT scales inversely with immediate threat and depletion
        self.levels.serotonin = float(np.clip(0.7 - 0.4 * threat_level - 0.2 * fatigue, 0.05, 1.0))
        # NE scales with threat and arousal
        self.levels.norepinephrine = float(np.clip(0.2 + 0.6 * threat_level, 0.05, 1.0))
        # ACh scales with environmental uncertainty
        self.levels.acetylcholine = float(np.clip(0.2 + 0.7 * uncertainty, 0.05, 1.0))

        return self.levels
