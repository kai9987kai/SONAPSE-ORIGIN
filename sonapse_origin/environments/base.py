"""Base Environment Protocol and Data Structures for Environment Forge."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class StepResult:
    """Standard transition result returned by all environments."""
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    """Abstract base class for all simulated environments."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to initial state."""
        pass

    @abstractmethod
    def step(self, action: Union[int, np.ndarray]) -> StepResult:
        """Executes one environmental transition."""
        pass

    @abstractmethod
    def intervene(self, intervention: Dict[str, Any]) -> None:
        """Applies causal intervention do(X=x) to environmental state."""
        pass

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass
