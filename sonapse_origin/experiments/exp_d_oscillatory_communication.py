"""Experiment D: Oscillatory and Wave-like Communication.

Compares multidimensional wave communication against scalar channels;
evaluates information fidelity and noise resistance.
"""
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from ..substrate.wave_comm import SignalChannelComparator, SignalSubstrateMode
from .runner import ExperimentRunner


def run_experiment_d(sample_size: int = 12) -> Dict[str, Any]:
    runner = ExperimentRunner()

    def treatment_wave(seed: int) -> float:
        comp = SignalChannelComparator(rng_seed=seed)
        signal = np.random.default_rng(seed).normal(0.5, 0.3, 8)
        res = comp.transmit(SignalSubstrateMode.WAVE_MULTIDIM, signal, noise_level=0.1)
        return float(res["fidelity"])

    def control_scalar(seed: int) -> float:
        comp = SignalChannelComparator(rng_seed=seed)
        signal = np.random.default_rng(seed).normal(0.5, 0.3, 8)
        res = comp.transmit(SignalSubstrateMode.SCALAR, signal, noise_level=0.1)
        return float(res["fidelity"])

    res = runner.run_controlled_experiment(
        exp_id="EXP_D_OSCILLATORY_COMM",
        title="Multidimensional Wave Communication Fidelity vs Scalar Baseline",
        theory="Substrate-Dynamics",
        statement="Multidimensional wave communication preserves significantly higher signal fidelity than scalar channels under noisy transmission.",
        target_metric="reconstruction_fidelity",
        treatment_fn=treatment_wave,
        control_fn=control_scalar,
        sample_size=sample_size,
    )
    return res
