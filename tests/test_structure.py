import numpy as np
import pytest
from sonapse_origin.structure.genome import ArchitectureGenome, CommunicationMode, TopologyFamily
from sonapse_origin.structure.topology import TopologyGenerator
from sonapse_origin.structure.hypergraph import HypergraphMessagePassing, MatchedGraphBaseline
from sonapse_origin.structure.development import StructuralPlasticityEngine, STDPPlasticity


def test_all_16_topology_families_generation():
    generator = TopologyGenerator(seed=42)
    for family in TopologyFamily:
        genome = generator.generate(family, num_nodes=16)
        assert genome.topology_family == family
        assert genome.num_nodes == 16
        assert len(genome.node_types) == 16
        mat = genome.to_matrix()
        assert mat.shape == (16, 16)
        assert np.all(np.isfinite(mat))


def test_genome_mutation_and_crossover():
    generator = TopologyGenerator(seed=10)
    p1 = generator.generate(TopologyFamily.SMALL_WORLD, num_nodes=12)
    p2 = generator.generate(TopologyFamily.SMALL_WORLD, num_nodes=12)

    rng = np.random.default_rng(99)
    mutated = p1.mutate(rng)
    assert mutated.generation == p1.generation + 1
    assert mutated.parent_ids == [p1.genome_id]

    child = ArchitectureGenome.crossover(p1, p2, rng)
    assert child.num_nodes == 12
    assert len(child.parent_ids) == 2


def test_hypergraph_and_matched_baseline():
    hyperedges = [[0, 1, 2], [2, 3, 4], [0, 4]]
    X = np.ones((5, 3), dtype=np.float64) * 0.5
    hg = HypergraphMessagePassing(hyperedges, node_dim=3)
    out_hg = hg.forward(X)
    assert out_hg.shape == (5, 3)
    assert np.all(np.isfinite(out_hg))

    baseline = MatchedGraphBaseline(hyperedges, num_nodes=5, node_dim=3)
    out_base = baseline.forward(X)
    assert out_base.shape == (5, 3)
    assert np.all(np.isfinite(out_base))


def test_structural_plasticity_prune_and_neurogenesis():
    engine = StructuralPlasticityEngine(pruning_threshold=0.1, seed=42)
    W = np.array([
        [0.0, 0.05, 0.5],
        [0.02, 0.0, 0.8],
        [0.3, 0.4, 0.0],
    ])
    pruned_W, count = engine.prune_connections(W)
    assert count == 2
    assert pruned_W[0, 1] == 0.0
    assert pruned_W[0, 2] == 0.5

    new_W, types, new_idx = engine.neurogenesis(pruned_W, ["sensory", "inter", "motor"])
    assert new_W.shape == (4, 4)
    assert len(types) == 4
    assert new_idx == 3


def test_stdp():
    stdp = STDPPlasticity(A_plus=0.01, A_minus=0.012, tau_plus=20.0, tau_minus=20.0)
    dw_causal = stdp.compute_dW(spike_times_pre=10.0, spike_times_post=20.0)
    assert dw_causal > 0.0

    dw_anticausal = stdp.compute_dW(spike_times_pre=20.0, spike_times_post=10.0)
    assert dw_anticausal < 0.0
