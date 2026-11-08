"""Self-Maintenance Objective and Reward Policies."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np


class SelfMaintenanceRewardPolicy:
    """Computes embodied reward with explicit homeostatic & damage penalties:
    
        r_t = r_task - lambda_H * J_H - lambda_E * C_energy - lambda_D * C_damage
        
    Strictly simulated; self-preservation goals cannot access host processes.
    """

    def __init__(
        self,
        lambda_H: float = 1.0,
        lambda_E: float = 0.5,
        lambda_D: float = 2.0,
    ):
        self.lambda_H = lambda_H
        self.lambda_E = lambda_E
        self.lambda_D = lambda_D

    def compute_reward(
        self,
        task_reward: float,
        homeostatic_cost: float,
        energy_spent: float,
        damage_taken: float,
    ) -> Dict[str, float]:
        """Calculates total composite reward and diagnostic components."""
        h_penalty = self.lambda_H * homeostatic_cost
        e_penalty = self.lambda_E * energy_spent
        d_penalty = self.lambda_D * damage_taken
        total_r = task_reward - h_penalty - e_penalty - d_penalty

        return {
            "total_reward": float(total_r),
            "task_reward": float(task_reward),
            "homeostatic_penalty": float(h_penalty),
            "energy_penalty": float(e_penalty),
            "damage_penalty": float(d_penalty),
        }
