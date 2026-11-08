"""Causal Emergence: Effective Information (EI) at Micro and Macro Scales.

Implements Erik Hoel's Effective Information framework (Hoel et al., 2013):
    EI(W) = H(<W>) - (1/N) * sum_i H(W_{i*})
and tests whether regional macro-coarse-graining yields higher causal power (causal emergence):
    Delta_EI = EI(macro) - EI(micro) > 0
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CausalEmergenceEngine:
    """Calculates Effective Information (EI) of transition probability matrices at micro and macro scales."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _entropy(self, p: np.ndarray) -> float:
        """Shannon entropy in bits: -sum p log2(p)."""
        p_clean = p[p > 1e-12]
        if len(p_clean) == 0:
            return 0.0
        return float(-np.sum(p_clean * np.log2(p_clean)))

    def effective_information(self, TPM: np.ndarray) -> float:
        """Computes Effective Information of a row-stochastic Transition Probability Matrix (TPM):
        
            EI(W) = H( <W> ) - (1/N) * sum_i H(W_i)
            
        where <W> is the average output distribution under a maximum-entropy (uniform) input perturbation.
        """
        N = TPM.shape[0]
        if N <= 1:
            return 0.0

        # Row-normalize to ensure valid stochastic transition matrix
        row_sums = TPM.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = TPM / row_sums

        # 1. Output distribution under uniform intervention: <W> = (1/N) * sum_i P_i
        mean_output = np.mean(P, axis=0)
        h_mean_output = self._entropy(mean_output)

        # 2. Average entropy of individual transitions (noise / indeterminism)
        mean_row_entropy = float(np.mean([self._entropy(P[i]) for i in range(N)]))

        # Effective Information = Effectiveness - Indeterminism
        ei = max(0.0, h_mean_output - mean_row_entropy)
        return float(ei)

    def coarse_grain_to_macro(
        self,
        micro_TPM: np.ndarray,
        cluster_assignments: List[int],
    ) -> np.ndarray:
        """Constructs a macro-TPM by grouping micro-states into functional clusters."""
        num_macro = len(set(cluster_assignments))
        num_micro = micro_TPM.shape[0]

        # Grouping projection matrix M of shape (num_macro, num_micro)
        M = np.zeros((num_macro, num_micro), dtype=np.float64)
        for micro_idx, macro_idx in enumerate(cluster_assignments):
            M[macro_idx, micro_idx] = 1.0

        # Normalize projection
        cluster_sizes = M.sum(axis=1, keepdims=True)
        cluster_sizes[cluster_sizes == 0] = 1.0
        M_norm = M / cluster_sizes

        # Row-normalize micro_TPM
        row_sums = micro_TPM.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P_micro = micro_TPM / row_sums

        # Macro transition: P_macro = M_norm @ P_micro @ M^T
        P_macro = M_norm @ P_micro @ M.T
        macro_row_sums = P_macro.sum(axis=1, keepdims=True)
        macro_row_sums[macro_row_sums == 0] = 1.0
        return P_macro / macro_row_sums

    def evaluate_causal_emergence(
        self,
        micro_W: np.ndarray,
        clusters: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluates whether macro-scale network exhibits higher Effective Information than micro-scale."""
        num_micro = micro_W.shape[0]
        # Default clustering: pairs of adjacent micro-nodes
        if clusters is None:
            num_macro = max(2, num_micro // 2)
            clusters = [min(num_macro - 1, i // 2) for i in range(num_micro)]

        # Softmax / transition mapping from weights to positive transition tendencies
        exp_w = np.exp(np.clip(np.abs(micro_W), -5.0, 5.0))
        np.fill_diagonal(exp_w, 0.1)

        macro_W = self.coarse_grain_to_macro(exp_w, clusters)

        ei_micro = self.effective_information(exp_w)
        ei_macro = self.effective_information(macro_W)
        delta_ei = ei_macro - ei_micro

        causal_emergence_detected = bool(delta_ei > 0.05)

        return {
            "ei_micro": ei_micro,
            "ei_macro": ei_macro,
            "delta_ei": delta_ei,
            "causal_emergence": causal_emergence_detected,
            "num_micro": num_micro,
            "num_macro": len(set(clusters)),
        }
