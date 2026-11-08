"""Integrated Information Theory 4.0: Mechanism Cause-Effect Repertoires and MIP.

Calculates mechanism-level integrated information phi^mech:
- Cause Repertoire p(Past | Present)
- Effect Repertoire p(Future | Present)
- Minimum Information Partition (MIP) across candidate subsets
- Irreducibility metric phi = min(phi^cause, phi^effect)
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


@dataclass
class MechanismRepertoire:
    mechanism_nodes: List[int]
    purview_nodes: List[int]
    cause_repertoire: np.ndarray
    effect_repertoire: np.ndarray
    phi_cause: float
    phi_effect: float
    phi_mech: float
    mip_partition: str


class IITMechanismAnalyzer:
    """Computes cause-effect repertoires and mechanism-level integrated information (small phi)."""

    def __init__(self, num_nodes: int, transition_matrix: np.ndarray, seed: int = 42):
        self.num_nodes = num_nodes
        self.W = np.asarray(transition_matrix, dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _wasserstein_or_l1(p: np.ndarray, q: np.ndarray) -> float:
        """Computes Earth Mover's Distance / Total Variation distance: 0.5 * sum(|p - q|)."""
        return float(0.5 * np.sum(np.abs(p - q)))

    def transition_probability(self, past_state: np.ndarray) -> np.ndarray:
        """Evaluates p(X_{t+1} = 1 | X_t = past_state) via logistic activation."""
        activations = self.W @ past_state
        return 1.0 / (1.0 + np.exp(-activations))

    def compute_cause_repertoire(
        self,
        mechanism: List[int],
        purview: List[int],
        current_state: np.ndarray,
        num_past_samples: int = 256,
    ) -> np.ndarray:
        """Calculates conditional cause distribution p(Purview_{t-1} | Mechanism_t = m)."""
        # Enumerate or sample possible binary past states of purview
        n_purview = len(purview)
        n_states = 2 ** n_purview
        probs = np.zeros(n_states)

        for s_idx in range(n_states):
            # Binary vector for purview
            past_purview_bin = np.array([(s_idx >> b) & 1 for b in range(n_purview)], dtype=np.float64)
            # Full past state with maximum entropy assumption on outside nodes (0.5)
            full_past = np.full(self.num_nodes, 0.5)
            for p_i, node in enumerate(purview):
                full_past[node] = past_purview_bin[p_i]

            # Future probability of mechanism nodes
            future_probs = self.transition_probability(full_past)
            # Likelihood of current mechanism state
            likelihood = 1.0
            for m_node in mechanism:
                p_active = future_probs[m_node]
                likelihood *= p_active if current_state[m_node] == 1 else (1.0 - p_active)
            probs[s_idx] = likelihood

        tot = np.sum(probs)
        if tot > 1e-12:
            return probs / tot
        return np.full(n_states, 1.0 / n_states)

    def compute_effect_repertoire(
        self,
        mechanism: List[int],
        purview: List[int],
        current_state: np.ndarray,
    ) -> np.ndarray:
        """Calculates conditional effect distribution p(Purview_{t+1} | Mechanism_t = m)."""
        # Full current state setting non-mechanism nodes to unconstrained maximum entropy (0.5)
        full_current = np.full(self.num_nodes, 0.5)
        for m_node in mechanism:
            full_current[m_node] = current_state[m_node]

        future_probs = self.transition_probability(full_current)
        n_purview = len(purview)
        n_states = 2 ** n_purview
        probs = np.zeros(n_states)

        for s_idx in range(n_states):
            prob = 1.0
            for b in range(n_purview):
                bit = (s_idx >> b) & 1
                p_active = future_probs[purview[b]]
                prob *= p_active if bit == 1 else (1.0 - p_active)
            probs[s_idx] = prob

        tot = np.sum(probs)
        if tot > 1e-12:
            return probs / tot
        return np.full(n_states, 1.0 / n_states)

    def evaluate_mechanism_phi(
        self,
        mechanism: List[int],
        purview: List[int],
        current_state: np.ndarray,
    ) -> MechanismRepertoire:
        """Finds Minimum Information Partition (MIP) and returns mechanism phi = min(phi^cause, phi^effect)."""
        unpartitioned_cause = self.compute_cause_repertoire(mechanism, purview, current_state)
        unpartitioned_effect = self.compute_effect_repertoire(mechanism, purview, current_state)

        # Generate bipartitions of mechanism
        min_cause_dist = float("inf")
        min_effect_dist = float("inf")
        best_partition = "none"

        # If mechanism has only 1 node, compare against unconstrained uniform prior
        if len(mechanism) <= 1:
            uniform = np.full_like(unpartitioned_cause, 1.0 / len(unpartitioned_cause))
            phi_cause = self._wasserstein_or_l1(unpartitioned_cause, uniform)
            phi_effect = self._wasserstein_or_l1(unpartitioned_effect, uniform)
            best_partition = "1-node-vs-unconstrained"
        else:
            # Search all non-trivial bipartitions of mechanism (part_a, part_b)
            # In IIT 4.0, a partition cuts mechanism and purview: (M1 x P1) U (M2 x P2)
            n_mech = len(mechanism)
            n_purview = len(purview)
            half_p = max(1, n_purview // 2)
            purview_a = purview[:half_p]
            purview_b = purview[half_p:]

            for r in range(1, n_mech // 2 + 1):
                for subset in combinations(mechanism, r):
                    part_a = list(subset)
                    part_b = [n for n in mechanism if n not in part_a]

                    # Directional partition of cause: M1 only constrains P1, M2 only constrains P2
                    cause_a = self.compute_cause_repertoire(part_a, purview_a, current_state)
                    cause_b = self.compute_cause_repertoire(part_b, purview_b, current_state)
                    # Outer product to form joint partitioned distribution over full purview P1 x P2
                    part_cause = np.outer(cause_a, cause_b).flatten()
                    if np.sum(part_cause) > 1e-12:
                        part_cause /= np.sum(part_cause)
                    else:
                        part_cause = np.full_like(unpartitioned_cause, 1.0 / len(unpartitioned_cause))

                    d_cause = self._wasserstein_or_l1(unpartitioned_cause, part_cause)
                    if d_cause < min_cause_dist:
                        min_cause_dist = d_cause
                        best_partition = f"{part_a}:{purview_a}|{part_b}:{purview_b}"

                    # Directional partition of effect: M1 only affects P1, M2 only affects P2
                    eff_a = self.compute_effect_repertoire(part_a, purview_a, current_state)
                    eff_b = self.compute_effect_repertoire(part_b, purview_b, current_state)
                    part_eff = np.outer(eff_a, eff_b).flatten()
                    if np.sum(part_eff) > 1e-12:
                        part_eff /= np.sum(part_eff)
                    else:
                        part_eff = np.full_like(unpartitioned_effect, 1.0 / len(unpartitioned_effect))

                    d_eff = self._wasserstein_or_l1(unpartitioned_effect, part_eff)
                    if d_eff < min_effect_dist:
                        min_effect_dist = d_eff
                        min_effect_dist = d_eff

            phi_cause = min_cause_dist
            phi_effect = min_effect_dist

        phi_mech = float(min(phi_cause, phi_effect))

        return MechanismRepertoire(
            mechanism_nodes=mechanism,
            purview_nodes=purview,
            cause_repertoire=unpartitioned_cause,
            effect_repertoire=unpartitioned_effect,
            phi_cause=float(phi_cause),
            phi_effect=float(phi_effect),
            phi_mech=phi_mech,
            mip_partition=best_partition,
        )
