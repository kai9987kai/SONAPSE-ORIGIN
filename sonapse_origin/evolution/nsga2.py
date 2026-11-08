"""Non-dominated Sorting Genetic Algorithm II (NSGA-II) for Multiobjective Architecture Search."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..structure.genome import ArchitectureGenome


class NSGAParetoOptimizer:
    """Computes Pareto dominance fronts and crowding distances across multiobjective vectors."""

    def __init__(self, objective_names: Optional[List[str]] = None):
        self.objective_names = objective_names or [
            "adaptation",
            "memory",
            "integration",
            "robustness",
            "generalisation",
            "neg_compute",
            "neg_energy",
        ]

    def dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """obj1 dominates obj2 if obj1 >= obj2 in all objectives and strictly > in at least one."""
        return bool(np.all(obj1 >= obj2) and np.any(obj1 > obj2))

    def fast_non_dominated_sort(
        self,
        population_objectives: List[np.ndarray],
    ) -> List[List[int]]:
        """Sorts population indices into Pareto fronts F_1, F_2, ..."""
        pop_size = len(population_objectives)
        domination_counts = [0] * pop_size
        dominated_sets: List[List[int]] = [[] for _ in range(pop_size)]
        fronts: List[List[int]] = [[]]

        for p in range(pop_size):
            for q in range(pop_size):
                if p == q:
                    continue
                if self.dominates(population_objectives[p], population_objectives[q]):
                    dominated_sets[p].append(q)
                elif self.dominates(population_objectives[q], population_objectives[p]):
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_sets[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)

        return [f for f in fronts if f]

    def calculate_crowding_distance(
        self,
        front_indices: List[int],
        population_objectives: List[np.ndarray],
    ) -> Dict[int, float]:
        """Calculates crowding distances within a single Pareto front."""
        if len(front_indices) <= 2:
            return {idx: float("inf") for idx in front_indices}

        num_objectives = len(population_objectives[0])
        distances = {idx: 0.0 for idx in front_indices}

        for m in range(num_objectives):
            sorted_indices = sorted(front_indices, key=lambda idx: population_objectives[idx][m])
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            obj_min = population_objectives[sorted_indices[0]][m]
            obj_max = population_objectives[sorted_indices[-1]][m]
            diff = obj_max - obj_min

            if diff > 1e-8:
                for k in range(1, len(sorted_indices) - 1):
                    distances[sorted_indices[k]] += (
                        population_objectives[sorted_indices[k + 1]][m]
                        - population_objectives[sorted_indices[k - 1]][m]
                    ) / diff

        return distances
