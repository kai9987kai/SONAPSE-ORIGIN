"""Partial Information Decomposition (PID) Subsystem.

Implements Williams & Beer (2010) bivariate Partial Information Decomposition:
Decomposes total joint mutual information I(X1, X2; Y) into:
  - Redundancy R(X1, X2 -> Y)
  - Unique information U1(X1 -> Y)
  - Unique information U2(X2 -> Y)
  - Synergy S(X1, X2 -> Y)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PIDResult:
    total_information: float
    redundancy: float
    unique_1: float
    unique_2: float
    synergy: float
    synergy_fraction: float


class PartialInformationDecomposition:
    """Calculates bivariate PID using minimum specific information lattice (I_min)."""

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        p_clean = p[p > 1e-12]
        return float(-np.sum(p_clean * np.log2(p_clean)))

    @staticmethod
    def _specific_information(p_joint_xy: np.ndarray, p_y: np.ndarray, p_x: np.ndarray, y_val: int) -> float:
        """Calculates I(Y=y; X) = sum_x p(x|y) log2( p(x|y) / p(x) )."""
        if p_y[y_val] < 1e-12:
            return 0.0
        p_x_given_y = p_joint_xy[:, y_val] / p_y[y_val]
        
        info = 0.0
        for x_val in range(len(p_x)):
            if p_x_given_y[x_val] > 1e-12 and p_x[x_val] > 1e-12:
                info += p_x_given_y[x_val] * np.log2(p_x_given_y[x_val] / p_x[x_val])
        return max(0.0, float(info))

    def decompose(self, X1: np.ndarray, X2: np.ndarray, Y: np.ndarray) -> PIDResult:
        """Decomposes discrete joint data into Redundancy, Unique1, Unique2, Synergy."""
        # Ensure 1D integer categorical representations
        x1_disc = np.asarray(X1, dtype=np.int64).flatten()
        x2_disc = np.asarray(X2, dtype=np.int64).flatten()
        y_disc = np.asarray(Y, dtype=np.int64).flatten()

        n_samples = len(y_disc)
        x1_vals = np.unique(x1_disc)
        x2_vals = np.unique(x2_disc)
        y_vals = np.unique(y_disc)

        k1 = len(x1_vals)
        k2 = len(x2_vals)
        ky = len(y_vals)

        # Map to 0-indexed contiguous states
        x1_map = {val: i for i, val in enumerate(x1_vals)}
        x2_map = {val: i for i, val in enumerate(x2_vals)}
        y_map = {val: i for i, val in enumerate(y_vals)}

        x1_idx = np.array([x1_map[v] for v in x1_disc])
        x2_idx = np.array([x2_map[v] for v in x2_disc])
        y_idx = np.array([y_map[v] for v in y_disc])

        # Estimate joint distributions
        p_joint_all = np.zeros((k1, k2, ky))
        p_joint_x1_y = np.zeros((k1, ky))
        p_joint_x2_y = np.zeros((k2, ky))
        p_x1 = np.zeros(k1)
        p_x2 = np.zeros(k2)
        p_y = np.zeros(ky)

        for i in range(n_samples):
            p_joint_all[x1_idx[i], x2_idx[i], y_idx[i]] += 1.0
            p_joint_x1_y[x1_idx[i], y_idx[i]] += 1.0
            p_joint_x2_y[x2_idx[i], y_idx[i]] += 1.0
            p_x1[x1_idx[i]] += 1.0
            p_x2[x2_idx[i]] += 1.0
            p_y[y_idx[i]] += 1.0

        p_joint_all /= n_samples
        p_joint_x1_y /= n_samples
        p_joint_x2_y /= n_samples
        p_x1 /= n_samples
        p_x2 /= n_samples
        p_y /= n_samples

        # Compute marginal mutual informations:
        # I(X; Y) = H(X) + H(Y) - H(X, Y)
        h_x1 = self._entropy(p_x1)
        h_x2 = self._entropy(p_x2)
        h_y = self._entropy(p_y)
        h_x1_y = self._entropy(p_joint_x1_y.flatten())
        h_x2_y = self._entropy(p_joint_x2_y.flatten())
        h_all = self._entropy(p_joint_all.flatten())

        p_x1_x2 = np.sum(p_joint_all, axis=2)
        h_x1_x2 = self._entropy(p_x1_x2.flatten())

        mi_x1_y = max(0.0, h_x1 + h_y - h_x1_y)
        mi_x2_y = max(0.0, h_x2 + h_y - h_x2_y)
        mi_joint_y = max(0.0, h_x1_x2 + h_y - h_all)

        # Compute Williams & Beer Redundancy I_min
        redundancy = 0.0
        for y_v in range(ky):
            i_y_x1 = self._specific_information(p_joint_x1_y, p_y, p_x1, y_v)
            i_y_x2 = self._specific_information(p_joint_x2_y, p_y, p_x2, y_v)
            redundancy += p_y[y_v] * min(i_y_x1, i_y_x2)

        redundancy = float(np.clip(redundancy, 0.0, min(mi_x1_y, mi_x2_y)))
        unique_1 = float(max(0.0, mi_x1_y - redundancy))
        unique_2 = float(max(0.0, mi_x2_y - redundancy))
        synergy = float(max(0.0, mi_joint_y - redundancy - unique_1 - unique_2))

        tot = redundancy + unique_1 + unique_2 + synergy
        syn_fraction = float(synergy / tot) if tot > 1e-8 else 0.0

        return PIDResult(
            total_information=float(mi_joint_y),
            redundancy=redundancy,
            unique_1=unique_1,
            unique_2=unique_2,
            synergy=synergy,
            synergy_fraction=syn_fraction,
        )
