"""Sleep, Memory Replay, and Synaptic Renormalisation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ReplayMode(str, Enum):
    SALIENT_REPLAY = "salient_replay"
    RANDOM_REPLAY = "random_replay"
    NO_REPLAY = "no_replay"


@dataclass
class ReplayExperience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    surprise: float  # Absolute prediction error


class SleepReplayEngine:
    """Manages offline consolidation, counterfactual simulation, and synaptic renormalisation."""

    def __init__(
        self,
        buffer_capacity: int = 1000,
        renormalization_factor: float = 0.95,
        seed: int = 42,
    ):
        self.buffer_capacity = buffer_capacity
        self.renormalization_factor = renormalization_factor
        self.buffer: List[ReplayExperience] = []
        self.rng = np.random.default_rng(seed)

    def store(self, experience: ReplayExperience) -> None:
        """Stores experience in circular replay buffer."""
        if len(self.buffer) >= self.buffer_capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample_batch(
        self,
        batch_size: int,
        mode: ReplayMode = ReplayMode.SALIENT_REPLAY,
    ) -> List[ReplayExperience]:
        """Samples experiences according to the specified replay condition."""
        if not self.buffer or mode == ReplayMode.NO_REPLAY:
            return []

        n = min(batch_size, len(self.buffer))

        if mode == ReplayMode.RANDOM_REPLAY:
            indices = self.rng.choice(len(self.buffer), size=n, replace=False)
            return [self.buffer[int(i)] for i in indices]

        elif mode == ReplayMode.SALIENT_REPLAY:
            # Prioritized by surprise / prediction error
            surprises = np.array([e.surprise for e in self.buffer], dtype=np.float64)
            surprises = surprises + 1e-4  # Ensure non-zero
            probs = surprises / np.sum(surprises)
            indices = self.rng.choice(len(self.buffer), size=n, p=probs, replace=False)
            return [self.buffer[int(i)] for i in indices]

        return []

    def synaptic_renormalization(self, W: np.ndarray) -> Tuple[np.ndarray, float]:
        """Tononi-Cirelli synaptic homeostasis hypothesis: downscales synaptic weights during sleep.
        
        Preserves relative potentiation while reducing metabolic energy and noise.
        """
        initial_norm = float(np.linalg.norm(W))
        new_W = W * self.renormalization_factor
        final_norm = float(np.linalg.norm(new_W))
        energy_saved = initial_norm - final_norm
        return new_W, energy_saved
