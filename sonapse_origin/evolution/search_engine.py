"""Evolution and Architecture Discovery Engine Coordinator."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..structure.genome import ArchitectureGenome, TopologyFamily, CommunicationMode
from ..structure.topology import TopologyGenerator
from .map_elites import MAPElitesArchive
from .nsga2 import NSGAParetoOptimizer
from .novelty_search import NoveltySearchArchive


class ArchitectureDiscoveryEngine:
    """Orchestrates evolutionary quality-diversity and novelty exploration over architectures."""

    def __init__(
        self,
        population_size: int = 20,
        seed: int = 42,
    ):
        self.population_size = population_size
        self.rng = np.random.default_rng(seed)
        self.topology_gen = TopologyGenerator(seed=seed)
        self.map_elites = MAPElitesArchive(dim_x=8, dim_y=8)
        self.nsga = NSGAParetoOptimizer()
        self.novelty = NoveltySearchArchive()

        self.population: List[ArchitectureGenome] = []
        self._initialize_population()

    def _initialize_population(self) -> None:
        """Seeds population with diverse topology families."""
        families = list(TopologyFamily)
        for i in range(self.population_size):
            fam = families[i % len(families)]
            genome = self.topology_gen.generate(fam, num_nodes=16)
            self.population.append(genome)

    def evolve_step(
        self,
        evaluate_fn: Callable[[ArchitectureGenome], Tuple[float, Tuple[float, float], np.ndarray]],
    ) -> Dict[str, Any]:
        """Runs one evolutionary generation:
        
        evaluate_fn returns: (fitness, behavior_descriptor (2D), multiobjective_vector (7D))
        """
        fitnesses = []
        descriptors = []
        objectives_list = []

        # 1. Evaluate current population
        for genome in self.population:
            fit, b_desc, objs = evaluate_fn(genome)
            fitnesses.append(fit)
            descriptors.append(np.array(b_desc))
            objectives_list.append(objs)

            # Insert into MAP-Elites archive
            self.map_elites.add_candidate(genome, fit, b_desc)

            # Evaluate novelty and add to archive if novel
            nov = self.novelty.compute_novelty(np.array(b_desc))
            self.novelty.maybe_add_to_archive(np.array(b_desc), nov)

        # 2. Pareto non-dominated sort
        fronts = self.nsga.fast_non_dominated_sort(objectives_list)

        # 3. Produce offspring via mutation & crossover from elites
        next_population: List[ArchitectureGenome] = []
        # Elitism: retain top Pareto front members
        for idx in fronts[0][: self.population_size // 2]:
            next_population.append(self.population[idx])

        # Fill remaining slots with mutated MAP-Elites or crossover
        while len(next_population) < self.population_size:
            if len(self.map_elites.grid) > 0 and self.rng.random() < 0.7:
                parent = self.map_elites.sample_elite(self.rng)
                child = parent.mutate(self.rng)
            else:
                p1 = self.population[int(self.rng.integers(0, len(self.population)))]
                p2 = self.population[int(self.rng.integers(0, len(self.population)))]
                child = ArchitectureGenome.crossover(p1, p2, self.rng).mutate(self.rng)
            next_population.append(child)

        self.population = next_population

        return {
            "max_fitness": float(max(fitnesses)),
            "mean_fitness": float(np.mean(fitnesses)),
            "map_elites_coverage": self.map_elites.coverage(),
            "pareto_front_size": len(fronts[0]) if fronts else 0,
            "novelty_archive_size": len(self.novelty.archive),
        }
