"""Experiment L: Phase-Amplitude Coupling and Communication-Through-Coherence.

Tests Fries's Communication-Through-Coherence (CTC) hypothesis:
Evaluates whether phase alignment between sender theta bursts and receiver receptive windows
achieves significantly superior decoding fidelity compared to anti-phase controls.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..substrate.phase_amplitude_coupling import PhaseAmplitudeCouplingSubstrate
from .runner import ExperimentRunner


def run_experiment_l(sample_size: int = 15) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_phase_aligned(seed: int) -> float:
        pac = PhaseAmplitudeCouplingSubstrate(seed=seed)
        res = pac.evaluate_communication_through_coherence(phase_aligned=True, noise_level=0.1)
        return float(res["transmission_fidelity"])

    def control_phase_misaligned(seed: int) -> float:
        pac = PhaseAmplitudeCouplingSubstrate(seed=seed)
        res = pac.evaluate_communication_through_coherence(phase_aligned=False, noise_level=0.1)
        return float(res["transmission_fidelity"])

    res = runner.run_controlled_experiment(
        exp_id="EXP_L_PHASE_AMPLITUDE_COUPLING",
        title="Communication Through Coherence via Theta-Gamma Phase Alignment",
        theory="Substrate-Oscillators",
        statement="Phase-aligned receptive windows achieve significantly superior high-frequency envelope decoding fidelity than anti-phase misaligned channels.",
        target_metric="transmission_fidelity",
        treatment_fn=treatment_phase_aligned,
        control_fn=control_phase_misaligned,
        sample_size=sample_size,
    )
    return res
