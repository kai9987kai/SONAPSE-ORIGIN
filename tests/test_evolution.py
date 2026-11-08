import numpy as np
import pytest
from sonapse_origin.evolution.map_elites import MAPElitesArchive
from sonapse_origin.evolution.nsga2 import NSGAParetoOptimizer
from sonapse_origin.evolution.novelty_search import NoveltySearchArchive
from sonapse_origin.evolution.search_engine import ArchitectureDiscoveryEngine
from sonapse_origin.structure.genome import TopologyFamily
from sonapse_origin.structure.topology import TopologyGenerator


def test_map_elites_archive():
    archive = MAPElitesArchive(dim_x=5, dim_y=5)
    gen = TopologyGenerator(seed=1)
    g1 = gen.generate(TopologyFamily.FEEDFORWARD, num_nodes=8)
    g2 = gen.generate(TopologyFamily.RECURRENT, num_nodes=8)

    added1 = archive.add_candidate(g1, fitness=1.5, behavior_descriptor=(0.2, 0.4))
    assert added1
    assert archive.coverage() > 0.0

    # Better candidate in same cell replaces
    added2 = archive.add_candidate(g2, fitness=2.0, behavior_descriptor=(0.22, 0.42))
    assert added2
    assert archive.max_fitness() == 2.0


def test_nsga2_pareto():
    nsga = NSGAParetoOptimizer()
    objs = [
        np.array([1.0, 2.0]),
        np.array([2.0, 1.0]),
        np.array([0.5, 0.5]),  # Dominated by both
    ]
    fronts = nsga.fast_non_dominated_sort(objs)
    assert len(fronts) >= 2
    assert set(fronts[0]) == {0, 1}
    assert fronts[1] == [2]


def test_novelty_archive():
    nov = NoveltySearchArchive(k_neighbors=2, novelty_threshold=0.3)
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 1.0])
    nov.maybe_add_to_archive(p1, novelty=1.0)
    nov.maybe_add_to_archive(p2, novelty=1.0)

    # Point close to p1 has low novelty
    n_close = nov.compute_novelty(np.array([0.05, 0.05]))
    # Point far away has high novelty
    n_far = nov.compute_novelty(np.array([5.0, 5.0]))
    assert n_far > n_close


def test_architecture_discovery_engine():
    engine = ArchitectureDiscoveryEngine(population_size=6, seed=42)

    def dummy_eval(genome):
        mat = genome.to_matrix()
        density = float(np.mean(mat != 0))
        mean_w = float(np.mean(np.abs(mat)))
        fitness = density * 2.0 + mean_w
        b_desc = (float(np.clip(density, 0.0, 1.0)), float(np.clip(mean_w, 0.0, 1.0)))
        objs = np.array([fitness, density, mean_w, 1.0, 1.0, -1.0, -1.0])
        return fitness, b_desc, objs

    res = engine.evolve_step(dummy_eval)
    assert res["max_fitness"] > 0.0
    assert res["map_elites_coverage"] > 0.0
