"""Experiment N: Perturbational Complexity Index (PCI_LZ) in Integrated vs Anesthetized Networks.

Evaluates the clinical Perturbational Complexity Index (Casali et al., 2013):
Tests whether awake, balanced recurrent networks produce significantly higher normalized PCI_LZ
than anesthetized / disconnected networks.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..structure.genome import TopologyFamily
from ..structure.topology import TopologyGenerator
from ..theories.pci_lempel_ziv import PerturbationalComplexityIndexLZ
from .runner import ExperimentRunner


def run_experiment_n(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()
    pci_calc = PerturbationalComplexityIndexLZ(seed=42)

    def treatment_awake_integrated(seed: int) -> float:
        gen = TopologyGenerator(seed=seed)
        net = gen.generate(TopologyFamily.RICH_CLUB, num_nodes=20)
        W = net.to_matrix()
        # Awake state: balanced excitation/inhibition and recurrent feedback
        res = pci_calc.compute_pci_lz(W, perturbation_amplitude=3.0, timesteps=30)
        return float(res["pci_lz"])

    def control_anesthetized_disconnected(seed: int) -> float:
        gen = TopologyGenerator(seed=seed)
        net = gen.generate(TopologyFamily.FEEDFORWARD, num_nodes=20)
        # Anesthetized / deep sleep state: heavily dampened synaptic efficacy
        W = net.to_matrix() * 0.1
        res = pci_calc.compute_pci_lz(W, perturbation_amplitude=3.0, timesteps=30)
        return float(res["pci_lz"])

    res = runner.run_controlled_experiment(
        exp_id="EXP_N_PCI_LEMPEL_ZIV",
        title="Perturbational Complexity Index (PCI_LZ) in Integrated vs Dampened Networks",
        theory="IIT-PCI",
        statement="Integrated recurrent networks exhibit significantly higher normalized PCI_LZ than feedforward dampened baselines.",
        target_metric="pci_lz",
        treatment_fn=treatment_awake_integrated,
        control_fn=control_anesthetized_disconnected,
        sample_size=sample_size,
    )
    return res
