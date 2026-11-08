"""Fatigue and Metabolic Workload Dynamics."""
from __future__ import annotations

import numpy as np


class FatigueTracker:
    """Implements fatigue accumulation and recovery dynamics:
    
        f_{t+1} = clip(f_t + alpha * L_t - beta * R_t, 0, 1)
        
    where L_t is workload (neural/computational effort) and R_t is restorative activity.
    """

    def __init__(
        self,
        alpha: float = 0.05,  # Fatigue accumulation rate per unit workload
        beta: float = 0.1,    # Recovery rate per unit rest
    ):
        self.alpha = alpha
        self.beta = beta
        self.fatigue = 0.0

    def step(self, workload: float, resting: bool = False) -> float:
        """Updates fatigue level based on ongoing effort or quiescent recovery."""
        l_t = max(0.0, workload)
        r_t = 1.0 if resting else 0.0
        self.fatigue = float(np.clip(self.fatigue + self.alpha * l_t - self.beta * r_t, 0.0, 1.0))
        return self.fatigue

    def reset(self) -> None:
        self.fatigue = 0.0
