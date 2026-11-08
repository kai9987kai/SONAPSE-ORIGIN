"""Neural Development Engine and Structural Plasticity."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class StructuralPlasticityEngine:
    """Manages structural modifications, synaptogenesis, pruning, and neurogenesis."""

    def __init__(
        self,
        pruning_threshold: float = 0.05,
        growth_correlation_threshold: float = 0.7,
        target_activity: float = 0.2,
        seed: int = 42,
    ):
        self.pruning_threshold = pruning_threshold
        self.growth_correlation_threshold = growth_correlation_threshold
        self.target_activity = target_activity
        self.rng = np.random.default_rng(seed)

    def prune_connections(self, W: np.ndarray) -> Tuple[np.ndarray, int]:
        """Prunes connections with absolute weight below threshold."""
        mask = np.abs(W) >= self.pruning_threshold
        pruned_count = int(np.sum(~mask & (W != 0)))
        new_W = W * mask
        return new_W, pruned_count

    def sprout_synapses(
        self,
        W: np.ndarray,
        recent_activity: np.ndarray,
        max_new_synapses: int = 5,
    ) -> Tuple[np.ndarray, int]:
        """Sprouts new connections between highly co-active, currently unconnected units."""
        num_nodes = W.shape[0]
        if recent_activity.shape[0] < 2:
            return W, 0

        # Compute Pearson correlation matrix across time
        cov = np.corrcoef(recent_activity)
        np.nan_to_num(cov, copy=False, nan=0.0)

        new_W = W.copy()
        sprouted = 0

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and new_W[i, j] == 0:
                    if cov[i, j] > self.growth_correlation_threshold and self.rng.random() < 0.2:
                        new_W[i, j] = float(self.rng.normal(0.15, 0.05))
                        sprouted += 1
                        if sprouted >= max_new_synapses:
                            return new_W, sprouted

        return new_W, sprouted

    def homeostatic_synaptic_scaling(self, W: np.ndarray, current_firing_rate: np.ndarray) -> np.ndarray:
        """Scales incoming synaptic weights to stabilize target post-synaptic firing rate."""
        # W has shape (num_nodes, num_nodes) where W[i, j] is connection from j to i
        rates = np.clip(current_firing_rate, 1e-4, 10.0)
        # Multiplicative scaling factor: alpha = target / rate
        scaling = (self.target_activity / rates) ** 0.1  # Slow homeostatic adaptation
        new_W = W * scaling[:, np.newaxis]
        return np.clip(new_W, -3.0, 3.0)

    def neurogenesis(
        self,
        W: np.ndarray,
        node_types: List[str],
        donor_node_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[str], int]:
        """Adds a new computational unit to the network."""
        old_n = W.shape[0]
        new_n = old_n + 1

        new_W = np.zeros((new_n, new_n), dtype=np.float64)
        new_W[:old_n, :old_n] = W

        donor = donor_node_idx if donor_node_idx is not None else int(self.rng.integers(0, old_n))
        new_type = node_types[donor]
        new_node_types = list(node_types) + [new_type]

        # Duplicate a subset of incoming and outgoing synapses with noise
        for j in range(old_n):
            if W[donor, j] != 0 and self.rng.random() < 0.5:
                new_W[old_n, j] = float(W[donor, j] + self.rng.normal(0, 0.02))
            if W[j, donor] != 0 and self.rng.random() < 0.5:
                new_W[j, old_n] = float(W[j, donor] + self.rng.normal(0, 0.02))

        return new_W, new_node_types, old_n


class STDPPlasticity:
    """Spike-Timing-Dependent Plasticity rule with explicit causal & anti-causal timing windows."""

    def __init__(
        self,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ):
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus

    def compute_dW(self, spike_times_pre: np.ndarray, spike_times_post: np.ndarray) -> float:
        """Computes weight delta for a pair of spikes at times t_pre and t_post (in ms)."""
        dt = spike_times_post - spike_times_pre
        if dt > 0:
            return float(self.A_plus * np.exp(-dt / self.tau_plus))
        elif dt < 0:
            return float(-self.A_minus * np.exp(dt / self.tau_minus))
        return 0.0
