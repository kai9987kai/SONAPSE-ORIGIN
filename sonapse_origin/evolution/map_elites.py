"""Multi-dimensional Archive of Phenotypic Elites (MAP-Elites)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..structure.genome import ArchitectureGenome


@dataclass
class EliteRecord:
    genome: ArchitectureGenome
    fitness: float
    behavior_descriptor: Tuple[float, float]  # e.g. (integration, modularity)
    metrics: Dict[str, float]


class MAPElitesArchive:
    """Discretized 2D behavioral descriptor grid maintaining highest-fitness elite per cell."""

    def __init__(
        self,
        dim_x: int = 10,
        dim_y: int = 10,
        x_bounds: Tuple[float, float] = (0.0, 1.0),
        y_bounds: Tuple[float, float] = (0.0, 1.0),
    ):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.grid: Dict[Tuple[int, int], EliteRecord] = {}

    def _get_cell_index(self, b: Tuple[float, float]) -> Tuple[int, int]:
        bx, by = b
        ix = int(np.clip((bx - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) * self.dim_x, 0, self.dim_x - 1))
        iy = int(np.clip((by - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0]) * self.dim_y, 0, self.dim_y - 1))
        return (ix, iy)

    def add_candidate(
        self,
        genome: ArchitectureGenome,
        fitness: float,
        behavior_descriptor: Tuple[float, float],
        metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Attempts to insert or replace an elite in the corresponding behavioral niche."""
        cell = self._get_cell_index(behavior_descriptor)
        current = self.grid.get(cell)

        if current is None or fitness > current.fitness:
            self.grid[cell] = EliteRecord(
                genome=genome,
                fitness=fitness,
                behavior_descriptor=behavior_descriptor,
                metrics=metrics or {},
            )
            return True
        return False

    def coverage(self) -> float:
        """Fraction of behavioral cells filled."""
        total_cells = self.dim_x * self.dim_y
        return float(len(self.grid) / total_cells)

    def max_fitness(self) -> float:
        if not self.grid:
            return 0.0
        return float(max(e.fitness for e in self.grid.values()))

    def sample_elite(self, rng: np.random.Generator) -> ArchitectureGenome:
        """Uniformly samples a random elite from the filled cells for mutation/crossover."""
        cells = list(self.grid.keys())
        choice = cells[int(rng.integers(0, len(cells)))]
        return self.grid[choice].genome
