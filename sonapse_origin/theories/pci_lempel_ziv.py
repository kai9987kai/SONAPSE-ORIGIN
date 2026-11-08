"""Perturbational Complexity Index with Lempel-Ziv Compression (PCI_LZ).

Implements the clinical perturbational complexity protocol (Casali et al., Science TM 2013):
    1. Perturb network with localized TMS-like impulse.
    2. Record spatiotemporal response matrix X(t, i).
    3. Binarize by baseline statistical thresholding -> S(t, i) in {0, 1}.
    4. Calculate Lempel-Ziv 1976 complexity C(S).
    5. Normalize by asymptotic source entropy: PCI_LZ = C(S) / (L * H(p) / log2(L)).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class PerturbationalComplexityIndexLZ:
    """Computes normalized Lempel-Ziv perturbational complexity."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _lempel_ziv_complexity(self, binary_string: str) -> int:
        """Calculates standard Lempel-Ziv 1976 vocabulary size complexity."""
        n = len(binary_string)
        if n == 0:
            return 0

        words = set()
        c = 1
        w = binary_string[0]
        words.add(w)

        i = 1
        while i < n:
            curr = binary_string[i]
            w = curr
            while w in words and i + 1 < n:
                i += 1
                w += binary_string[i]
            words.add(w)
            c += 1
            i += 1

        return c

    def compute_pci_lz(
        self,
        W: np.ndarray,
        perturbation_amplitude: float = 2.0,
        timesteps: int = 40,
        baseline_noise: float = 0.05,
        target_node: int = 0,
    ) -> Dict[str, Any]:
        """Simulates perturbation and calculates normalized PCI_LZ."""
        num_nodes = W.shape[0]

        # 1. Baseline simulation before perturbation
        baseline = np.zeros((20, num_nodes))
        x = np.zeros(num_nodes)
        for t in range(20):
            x = np.tanh(W @ x) + self.rng.normal(0, baseline_noise, num_nodes)
            baseline[t] = x

        base_mean = np.mean(baseline, axis=0)
        base_std = np.std(baseline, axis=0) + 1e-4

        # 2. Perturbation: impulse injected at t=0
        response = np.zeros((timesteps, num_nodes))
        x = base_mean.copy()
        x[target_node] += perturbation_amplitude

        for t in range(timesteps):
            x = np.tanh(W @ x) + self.rng.normal(0, baseline_noise, num_nodes)
            response[t] = x

        # 3. Binarize by 2-sigma baseline thresholding
        z_scores = np.abs((response - base_mean) / base_std)
        binary_matrix = (z_scores > 2.0).astype(int)

        # 4. Flatten into 1D bitstring (order by space then time)
        bitstring = "".join(str(b) for b in binary_matrix.flatten())
        L = len(bitstring)

        p = float(np.mean(binary_matrix))
        # If no activation or all-ones, complexity is 0
        if p <= 1e-6 or p >= (1.0 - 1e-6):
            return {
                "pci_lz": 0.0,
                "raw_complexity": 0,
                "activation_density": p,
            }

        # Shannon entropy of the source: H(p) = -p log2(p) - (1-p) log2(1-p)
        h_source = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
        # Asymptotic maximum complexity bound: b(L) = L * H(p) / log2(L)
        b_bound = (L * h_source) / np.log2(L)

        # 5. Raw Lempel-Ziv complexity
        raw_c = self._lempel_ziv_complexity(bitstring)
        pci_lz = float(np.clip(raw_c / (b_bound + 1e-8), 0.0, 1.5))

        return {
            "pci_lz": pci_lz,
            "raw_complexity": raw_c,
            "activation_density": p,
            "source_entropy": float(h_source),
        }
