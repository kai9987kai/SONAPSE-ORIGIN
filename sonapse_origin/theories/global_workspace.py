"""Global Workspace Theory (GWT) Experimental Framework.

Operational metrics only; not proof of phenomenal experience.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class GlobalWorkspaceExperiment:
    """Investigates modular information sharing with and without global broadcast."""

    def __init__(
        self,
        num_modules: int = 4,
        module_dim: int = 8,
        workspace_dim: int = 16,
        ignition_threshold: float = 0.6,
        seed: int = 42,
    ):
        self.num_modules = num_modules
        self.module_dim = module_dim
        self.workspace_dim = workspace_dim
        self.ignition_threshold = ignition_threshold
        self.rng = np.random.default_rng(seed)

        # Ingestion projections: module_i -> workspace
        self.W_ingest = [
            self.rng.normal(0, 0.2, (workspace_dim, module_dim)) for _ in range(num_modules)
        ]
        # Broadcast projection: workspace -> module_i
        self.W_broadcast = [
            self.rng.normal(0, 0.2, (module_dim, workspace_dim)) for _ in range(num_modules)
        ]
        self.workspace_state = np.zeros(workspace_dim, dtype=np.float64)

    def step(
        self,
        module_states: List[np.ndarray],
        workspace_enabled: bool = True,
    ) -> Dict[str, Any]:
        """Runs one workspace competition and broadcast cycle."""
        # 1. Local module competition to enter workspace
        salience = []
        ingest_proposals = []
        for i, s in enumerate(module_states):
            prop = self.W_ingest[i] @ s
            sal = float(np.linalg.norm(prop))
            salience.append(sal)
            ingest_proposals.append(prop)

        winning_module = int(np.argmax(salience))
        max_salience = salience[winning_module]

        # 2. Ignition: non-linear threshold activation
        ignited = bool(workspace_enabled and (max_salience >= self.ignition_threshold))

        if ignited:
            # Non-linear all-or-none jump
            self.workspace_state = np.tanh(ingest_proposals[winning_module] * 2.0)
        else:
            self.workspace_state *= 0.2  # rapid decay

        # 3. Global broadcast to all modules
        broadcasted_signals = []
        broadcast_reach = 0
        for i in range(self.num_modules):
            if ignited:
                b_sig = np.tanh(self.W_broadcast[i] @ self.workspace_state)
                broadcasted_signals.append(b_sig)
                broadcast_reach += 1
            else:
                broadcasted_signals.append(np.zeros(self.module_dim))

        reach_fraction = broadcast_reach / self.num_modules if ignited else 0.0

        return {
            "workspace_enabled": workspace_enabled,
            "ignited": ignited,
            "winning_module": winning_module,
            "max_salience": max_salience,
            "broadcast_reach": reach_fraction,
            "workspace_activation_norm": float(np.linalg.norm(self.workspace_state)),
            "broadcasted_signals": broadcasted_signals,
        }
