import numpy as np
import pytest
from sonapse_origin.theories.global_workspace import GlobalWorkspaceExperiment
from sonapse_origin.theories.causal_integration import CausalIntegrationAssay
from sonapse_origin.theories.recurrent_processing import RecurrentProcessingExperiment
from sonapse_origin.theories.higher_order import HigherOrderRepresentationAssay
from sonapse_origin.theories.predictive_processing import HierarchicalPredictiveCoding
from sonapse_origin.theories.attention_schema import AttentionSchemaExperiment
from sonapse_origin.theories.temporal_binding import TemporalBindingExperiment


def test_global_workspace_ignition_and_ablation():
    gw = GlobalWorkspaceExperiment(num_modules=4, module_dim=6, workspace_dim=10, ignition_threshold=0.5)
    modules = [np.ones(6) * 1.5, np.zeros(6), np.zeros(6), np.zeros(6)]

    # Intact workspace: should ignite
    res_intact = gw.step(modules, workspace_enabled=True)
    assert res_intact["ignited"]
    assert res_intact["broadcast_reach"] == 1.0

    # Ablated workspace: cannot ignite
    res_ablated = gw.step(modules, workspace_enabled=False)
    assert not res_ablated["ignited"]
    assert res_ablated["broadcast_reach"] == 0.0


def test_causal_integration_proxies():
    assay = CausalIntegrationAssay(num_nodes=8, seed=42)
    # Dense integrated matrix
    W_int = np.random.default_rng(1).normal(0, 0.4, (8, 8))
    pci = assay.perturbational_complexity_index_proxy(W_int)
    assert pci > 0.0

    i_part = assay.partition_sensitivity_proxy(W_int, num_samples=100)
    assert i_part >= 0.0

    eff = assay.directed_effective_connectivity(W_int)
    assert "spectral_radius" in eff


def test_recurrent_processing_lesions():
    rpt = RecurrentProcessingExperiment(num_units=12, seed=42)
    stim = np.ones(12) * 0.8
    res_intact = rpt.evaluate_persistence_and_completion(stim, lesion_feedback=False)
    res_lesioned = rpt.evaluate_persistence_and_completion(stim, lesion_feedback=True)

    # Recurrent intact should have higher post-stimulus persistence than feedforward
    assert res_intact["post_stimulus_persistence"] >= res_lesioned["post_stimulus_persistence"]


def test_higher_order_metacognition():
    hot = HigherOrderRepresentationAssay(first_order_dim=8, meta_dim=4, seed=42)
    res = hot.evaluate_metacognitive_sensitivity(num_trials=100)
    assert "type2_roc_auc" in res
    assert 0.0 <= res["type2_roc_auc"] <= 1.0
    assert 0.0 <= res["expected_calibration_error"] <= 1.0


def test_predictive_coding_error_minimization():
    pp = HierarchicalPredictiveCoding(sensory_dim=6, latent_dim=3, seed=42)
    sensory = np.array([0.5, -0.3, 0.8, -0.1, 0.4, -0.6])
    res = pp.infer_and_minimize_error(sensory, num_inference_steps=15)
    # Free energy should decrease
    assert res["free_energy_reduction"] > 0.0
    assert res["final_free_energy"] < res["initial_free_energy"]


def test_attention_schema():
    ast = AttentionSchemaExperiment(num_targets=4, seed=42)
    res_schema = ast.track_target_under_distraction(target_idx=1, has_schema=True)
    res_no_schema = ast.track_target_under_distraction(target_idx=1, has_schema=False)
    assert res_schema["has_schema"] == 1.0
    assert res_no_schema["has_schema"] == 0.0


def test_temporal_binding():
    tb = TemporalBindingExperiment(num_features=4, seed=42)
    T = 50
    # In-phase synchronized waves
    t = np.linspace(0, 10, T)
    coupled = np.stack([t % (2 * np.pi) for _ in range(4)], axis=1)
    # Random uncoupled phases
    uncoupled = np.random.uniform(0, 2 * np.pi, (T, 4))
    res = tb.evaluate_binding_coherence(coupled, uncoupled)
    assert res["mean_coupled_plv"] > res["mean_uncoupled_plv"]
