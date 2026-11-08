"""Unit tests for Frontier Innovations: Criticality, Predictive Coding, PID, IIT 4.0 Mechanism, Bioelectric Morphogenesis, and Communication (Exp Q-V)."""
from __future__ import annotations

import numpy as np
import pytest

from sonapse_origin.substrate.criticality import CriticalitySubstrate
from sonapse_origin.theories.precision_predictive_coding import HierarchicalPredictiveCoder
from sonapse_origin.theories.pid import PartialInformationDecomposition
from sonapse_origin.theories.iit_mechanism import IITMechanismAnalyzer
from sonapse_origin.structure.bioelectric_morphogenesis import BioelectricMorphogeneticTissue
from sonapse_origin.environments.multi_agent_communication import EmergentCommunicationWorld
from sonapse_origin.experiments import (
    run_experiment_q,
    run_experiment_r,
    run_experiment_s,
    run_experiment_t,
    run_experiment_u,
    run_experiment_v,
)


def test_criticality_avalanche_detection():
    sub = CriticalitySubstrate(num_nodes=20, seed=42)
    spikes, weight_hist = sub.simulate(timesteps=150, homeostatic_adaptation=True, adaptation_rate=0.02)
    assert spikes.shape == (150, 20)
    assert weight_hist is not None
    assert len(weight_hist) == 149

    stats = sub.extract_avalanches(spikes, bin_size=1)
    assert stats.num_avalanches >= 0
    assert stats.branching_parameter >= 0.0
    assert stats.size_power_law_alpha > 0.0


def test_hierarchical_predictive_coding_inference():
    coder = HierarchicalPredictiveCoder(layer_sizes=(8, 4, 2), seed=42)
    sensory = np.array([0.5, -0.2, 0.8, -0.5, 0.1, 0.3, -0.4, 0.6])
    state = coder.infer(sensory, adaptation_steps=10)
    assert len(state.mu) == 3
    assert state.free_energy is not None
    assert np.isfinite(state.free_energy)

    noise_res = coder.evaluate_noise_robustness(sensory, noise_level=0.4, dynamic_precision=True)
    assert "reconstruction_error" in noise_res
    assert "free_energy" in noise_res


def test_partial_information_decomposition_canonical_gates():
    pid = PartialInformationDecomposition()

    # Canonical XOR gate: pure synergy
    x1 = np.array([0, 0, 1, 1] * 50)
    x2 = np.array([0, 1, 0, 1] * 50)
    y_xor = x1 ^ x2
    res_xor = pid.decompose(x1, x2, y_xor)
    assert res_xor.synergy > 0.8
    assert res_xor.redundancy < 0.2

    # Canonical COPY gate: pure redundancy
    y_copy = np.copy(x1)
    res_copy = pid.decompose(x1, x1, y_copy)
    assert res_copy.redundancy > 0.8
    assert res_copy.synergy < 0.2


def test_iit_mechanism_cause_effect_repertoires():
    W = np.array([
        [0.8, -0.5, 0.2, 0.0],
        [0.4, 0.9, -0.3, 0.1],
        [-0.2, 0.6, 0.7, -0.4],
        [0.1, -0.1, 0.5, 0.8],
    ])
    analyzer = IITMechanismAnalyzer(num_nodes=4, transition_matrix=W, seed=42)
    state = np.array([1, 0, 1, 0])
    rep = analyzer.evaluate_mechanism_phi(mechanism=[0, 1], purview=[0, 1, 2, 3], current_state=state)
    assert rep.phi_mech >= 0.0
    assert rep.cause_repertoire.shape[0] == 16
    assert np.isclose(np.sum(rep.cause_repertoire), 1.0)
    assert np.isclose(np.sum(rep.effect_repertoire), 1.0)


def test_bioelectric_morphogenesis_regeneration():
    tissue = BioelectricMorphogeneticTissue(grid_shape=(6, 6), base_conductance=0.3, seed=42)
    init_state = tissue.get_state()
    assert init_state.voltages.shape == (6, 6)

    tissue.step(steps=5)
    tissue.apply_lesion(fraction_cut=0.5)
    post_lesion_state = tissue.get_state()
    assert np.all(post_lesion_state.voltages[:, 3:] == 0.0)

    # Regeneration phase
    tissue.step(steps=15)
    post_regen_state = tissue.get_state()
    # Post-regen voltages in damaged section should have revived (no longer strictly 0)
    assert not np.all(post_regen_state.voltages[:, 3:] == 0.0)


def test_emergent_communication_world():
    world = EmergentCommunicationWorld(num_hazards=3, alphabet_size=4, seed=42)
    metrics = world.train_dialects(episodes=60, learning_rate=0.05)
    assert metrics.mutual_information_symbol_hazard >= 0.0
    assert 0.0 <= metrics.communicative_accuracy <= 1.0
    assert metrics.consensus_score >= 0.0


def test_frontier_experiments_q_through_v_execution():
    res_q = run_experiment_q(sample_size=3)
    assert res_q["experiment_id"] == "EXP_Q_NEURAL_CRITICALITY"
    assert "receipt_hash" in res_q

    res_r = run_experiment_r(sample_size=3)
    assert res_r["experiment_id"] == "EXP_R_PRECISION_PREDICTIVE_CODING"

    res_s = run_experiment_s(sample_size=3)
    assert res_s["experiment_id"] == "EXP_S_PID_SYNERGY"

    res_t = run_experiment_t(sample_size=3)
    assert res_t["experiment_id"] == "EXP_T_IIT4_MECHANISM_PHI"

    res_u = run_experiment_u(sample_size=3)
    assert res_u["experiment_id"] == "EXP_U_BIOELECTRIC_MORPHOGENESIS"

    res_v = run_experiment_v(sample_size=3)
    assert res_v["experiment_id"] == "EXP_V_EMERGENT_COMMUNICATION"
