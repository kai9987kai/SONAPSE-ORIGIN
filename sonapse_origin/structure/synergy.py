"""Higher-Order Hypergraph Synergy and O-Information Metrics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class HypergraphSynergyAssay:
    """Computes Rosas et al. (2020) O-Information to quantify synergy vs redundancy in hyperedges:
    
        Omega(X_1, ..., X_n) = (n - 2) * H(X_1, ..., X_n) + sum_{i=1}^n [ H(X_i) - H(X_{-i}) ]
        
    where:
        Omega < 0 indicates SYNERGY dominance (irreducible higher-order interaction).
        Omega > 0 indicates REDUNDANCY dominance (shared overlapping information).
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _gaussian_differential_entropy(self, data_matrix: np.ndarray) -> float:
        """Computes differential entropy for multivariate Gaussian: H(X) = 0.5 * ln((2*pi*e)^k * det(Sigma))."""
        N, D = data_matrix.shape
        if D == 1:
            var = float(np.var(data_matrix)) + 1e-6
            return 0.5 * np.log(2.0 * np.pi * np.e * var)

        cov = np.cov(data_matrix, rowvar=False)
        cov = cov + np.eye(D) * 1e-4  # Regularization
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            logdet = float(np.sum(np.log(np.diag(cov))))
        return float(0.5 * (D * np.log(2.0 * np.pi * np.e) + logdet))

    def compute_o_information(self, multivariate_samples: np.ndarray) -> float:
        """Calculates Omega across columns of multivariate_samples (shape: N_samples, n_variables)."""
        N, n = multivariate_samples.shape
        if n < 3:
            return 0.0

        # Total joint entropy: H(X_1, ..., X_n)
        h_joint = self._gaussian_differential_entropy(multivariate_samples)

        sum_individual_diffs = 0.0
        for i in range(n):
            # H(X_i)
            h_i = self._gaussian_differential_entropy(multivariate_samples[:, i : i + 1])
            # H(X_{-i})
            leave_one_out = np.delete(multivariate_samples, i, axis=1)
            h_minus_i = self._gaussian_differential_entropy(leave_one_out)
            sum_individual_diffs += (h_i - h_minus_i)

        omega = (n - 2) * h_joint + sum_individual_diffs
        return float(omega)

    def evaluate_hyperedge_synergy(
        self,
        hyperedge_samples: np.ndarray,
        pairwise_samples: np.ndarray,
    ) -> Dict[str, Any]:
        """Compares O-information between non-linear hyperedge interaction vs linear pairwise graph."""
        omega_hyper = self.compute_o_information(hyperedge_samples)
        omega_pairwise = self.compute_o_information(pairwise_samples)

        return {
            "omega_hyperedge": omega_hyper,
            "omega_pairwise": omega_pairwise,
            "hyperedge_synergy_boost": float(omega_pairwise - omega_hyper),
            "is_synergy_dominated": bool(omega_hyper < 0.0),
        }
