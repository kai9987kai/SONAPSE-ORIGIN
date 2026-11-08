"""Novelty Search in Behavioral Descriptor Space."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class NoveltySearchArchive:
    """Maintains an archive of discovered behaviors and evaluates novelty:
    
        N(g) = (1/k) * sum_{j=1}^k d(b(g), b(g_j))
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        novelty_threshold: float = 0.25,
        archive_limit: int = 500,
    ):
        self.k_neighbors = k_neighbors
        self.novelty_threshold = novelty_threshold
        self.archive_limit = archive_limit
        self.archive: List[np.ndarray] = []

    def compute_novelty(
        self,
        behavior_descriptor: np.ndarray,
        current_population_descriptors: Optional[List[np.ndarray]] = None,
    ) -> float:
        """Computes average Euclidean distance to the k-nearest neighbors."""
        all_points = list(self.archive)
        if current_population_descriptors:
            all_points.extend(current_population_descriptors)

        if not all_points:
            return 1.0

        dists = [float(np.linalg.norm(behavior_descriptor - p)) for p in all_points]
        dists.sort()
        k = min(self.k_neighbors, len(dists))
        novelty = float(np.mean(dists[:k]))
        return novelty

    def maybe_add_to_archive(self, behavior_descriptor: np.ndarray, novelty: float) -> bool:
        """Adds descriptor to archive if it exceeds the novelty threshold."""
        if novelty >= self.novelty_threshold or len(self.archive) < 10:
            if len(self.archive) >= self.archive_limit:
                self.archive.pop(0)
            self.archive.append(behavior_descriptor.copy())
            return True
        return False
