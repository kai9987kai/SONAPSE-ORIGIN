"""Causal Integration and IIT Proxy Experiments.

Provides operational proxies for system-wide causal integration.
These metrics are theory-specific approximations, NOT formal IIT Phi.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class CausalIntegrationAssay:
    """Computes practical proxies of causal integration and partition sensitivity."""

    def __init__(self, num_nodes: int, seed: int = 42):
        self.num_nodes = num_nodes
        self.rng = np.random.default_rng(seed)

    def perturbational_complexity_index_proxy(
        self,
        W: np.ndarray,
        perturbation_amplitude: float = 1.0,
        timesteps: int = 20,
    ) -> float:
        """Measures Perturbational Complexity Index (PCI) proxy via response matrix rank and entropy."""
        responses = np.zeros((timesteps, self.num_nodes), dtype=np.float64)
        x = np.zeros(self.num_nodes)
        # Apply impulse perturbation to node 0
        x[0] = perturbation_amplitude

        for t in range(timesteps):
            x = np.tanh(W @ x)
            responses[t] = x

        # Binarize responses around baseline to compute spatiotemporal diversity
        binary_resp = (responses > 0.1).astype(int)
        # Matrix singular value entropy proxy for complexity
        u, s, vh = np.linalg.svd(responses, full_matrices=False)
        s_norm = s / (np.sum(s) + 1e-8)
        # Shannon entropy of singular values
        pci_proxy = float(-np.sum(s_norm * np.log(np.clip(s_norm, 1e-12, 1.0))))
        return pci_proxy

    def partition_sensitivity_proxy(
        self,
        W: np.ndarray,
        num_samples: int = 500,
        noise_var: float = 0.05,
    ) -> float:
        """Computes I_partition: KL divergence between full joint transition and bipartite product:
        
            I_partition = D_KL( P(X_{t+1} | X_t) || prod_k P(X_{t+1}^(k) | X_t^(k)) )
            
        Measures the degree to which dividing the system into halves loses predictive transition info.
        """
        half = self.num_nodes // 2
        if half == 0:
            return 0.0

        # Sample X_t ~ Normal(0, I)
        X_t = self.rng.normal(0, 1.0, (num_samples, self.num_nodes))
        # Deterministic transition: X_{t+1} = tanh(W @ X_t^T)^T + noise
        mean_full = np.tanh(X_t @ W.T)
        noise = self.rng.normal(0, np.sqrt(noise_var), size=mean_full.shape)
        X_tp1 = mean_full + noise

        # Subsystem transitions: W11 and W22 only (cutting inter-subsystem edges W12 and W21)
        W_part = np.zeros_like(W)
        W_part[:half, :half] = W[:half, :half]
        W_part[half:, half:] = W[half:, half:]
        mean_part = np.tanh(X_t @ W_part.T)

        # Gaussian KL divergence given identical diagonal noise covariance sigma^2 I:
        # D_KL = 1 / (2 * sigma^2) * ||mean_full - mean_part||^2
        diff = mean_full - mean_part
        kl_div = float(np.mean(np.sum(diff ** 2, axis=1)) / (2.0 * noise_var))
        return kl_div

    def directed_effective_connectivity(self, W: np.ndarray) -> Dict[str, float]:
        """Calculates directed graph reciprocity, spectral radius, and mean connection strength."""
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(W))))
        mean_strength = float(np.mean(np.abs(W)))
        # Reciprocity: correlation between W and W^T
        w_flat = W.flatten()
        wt_flat = W.T.flatten()
        reciprocity = float(np.corrcoef(w_flat, wt_flat)[0, 1]) if np.std(w_flat) > 1e-6 else 0.0

        return {
            "spectral_radius": spectral_radius,
            "mean_strength": mean_strength,
            "reciprocity": reciprocity,
        }
