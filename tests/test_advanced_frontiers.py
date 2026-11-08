import numpy as np
import pytest
from sonapse_origin.theories.causal_emergence import CausalEmergenceEngine
from sonapse_origin.theories.pci_lempel_ziv import PerturbationalComplexityIndexLZ
from sonapse_origin.substrate.phase_amplitude_coupling import PhaseAmplitudeCouplingSubstrate
from sonapse_origin.homeostasis.allostasis import AllostaticActiveInferenceController
from sonapse_origin.structure.synergy import HypergraphSynergyAssay
from sonapse_origin.structure.metaplasticity import BCMMetaplasticityEngine
from sonapse_origin.experiments import (
    run_experiment_k,
    run_experiment_l,
    run_experiment_m,
    run_experiment_n,
    run_experiment_o,
    run_experiment_p,
)


def test_causal_emergence_effective_information():
    engine = CausalEmergenceEngine(seed=42)
    # 4-node deterministic cycle with micro-noise
    W = np.array([
        [0.0, 2.0, 0.1, 0.1],
        [0.1, 0.0, 2.0, 0.1],
        [0.1, 0.1, 0.0, 2.0],
        [2.0, 0.1, 0.1, 0.0],
    ])
    res = engine.evaluate_causal_emergence(W, clusters=[0, 0, 1, 1])
    assert "ei_micro" in res
    assert "ei_macro" in res
    assert res["ei_macro"] >= 0.0


def test_pci_lempel_ziv():
    pci_calc = PerturbationalComplexityIndexLZ(seed=42)
    W = np.random.default_rng(1).normal(0, 0.3, (8, 8))
    res = pci_calc.compute_pci_lz(W, perturbation_amplitude=2.0, timesteps=25)
    assert "pci_lz" in res
    assert 0.0 <= res["pci_lz"] <= 1.5


def test_phase_amplitude_coupling():
    pac = PhaseAmplitudeCouplingSubstrate(seed=42)
    res_aligned = pac.evaluate_communication_through_coherence(phase_aligned=True)
    res_misaligned = pac.evaluate_communication_through_coherence(phase_aligned=False)
    assert res_aligned["modulation_index"] >= 0.0
    assert res_aligned["transmission_fidelity"] >= res_misaligned["transmission_fidelity"]


def test_allostatic_active_inference():
    controller = AllostaticActiveInferenceController(lookahead_steps=3, seed=42)
    h_current = np.array([0.4, 0.9, 0.5, 0.3])  # Low energy
    best_action, efe_dict = controller.select_action(h_current, num_actions=3)
    # When energy is low, action 1 (consume) should have lower EFE
    assert best_action == 1


def test_hypergraph_o_information_synergy():
    assay = HypergraphSynergyAssay(seed=42)
    # Synthetic 3-variable synergistic XOR-like Gaussian system
    x1 = np.random.normal(0, 1, 100)
    x2 = np.random.normal(0, 1, 100)
    x3 = x1 + x2 + np.random.normal(0, 0.1, 100)
    samples = np.stack([x1, x2, x3], axis=1)

    omega = assay.compute_o_information(samples)
    assert isinstance(omega, float)


def test_bcm_metaplasticity():
    engine = BCMMetaplasticityEngine(num_units=4, seed=42)
    W = np.eye(4) * 0.5
    pre = np.ones(4) * 0.8
    post = np.ones(4) * 0.9
    new_W, theta = engine.step(W, pre, post)
    assert new_W.shape == (4, 4)
    assert np.all(theta > 0.0)


def test_advanced_experiments_k_through_p():
    experiments = [
        run_experiment_k,
        run_experiment_l,
        run_experiment_m,
        run_experiment_n,
        run_experiment_o,
        run_experiment_p,
    ]
    for exp_fn in experiments:
        res = exp_fn(sample_size=5)
        assert "experiment_id" in res
        assert "status" in res
        assert "effect_size_cohens_d" in res
        assert len(res["receipt_hash"]) == 64
