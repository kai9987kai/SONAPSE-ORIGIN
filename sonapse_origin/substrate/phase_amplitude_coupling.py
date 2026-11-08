"""Phase-Amplitude Coupling (PAC) and Communication-Through-Coherence (CTC)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class PhaseAmplitudeCouplingSubstrate:
    """Simulates cross-frequency theta-gamma nesting and coherent communication gating."""

    def __init__(
        self,
        num_channels: int = 4,
        theta_freq: float = 6.0,  # Slow carrier (Hz)
        gamma_freq: float = 40.0, # Fast burst carrier (Hz)
        dt: float = 0.002,        # 500 Hz sampling (s)
        num_bins: int = 18,
        seed: int = 42,
    ):
        self.num_channels = num_channels
        self.theta_freq = theta_freq
        self.gamma_freq = gamma_freq
        self.dt = dt
        self.num_bins = num_bins
        self.rng = np.random.default_rng(seed)

    def calculate_modulation_index(
        self,
        theta_phases: np.ndarray,
        gamma_amplitudes: np.ndarray,
    ) -> float:
        """Computes Tort's Modulation Index (MI) via KL divergence from uniform distribution:
        
            MI = D_KL(P || U) / log(N_bins)
        """
        # Bin phases into N_bins in [0, 2pi)
        bin_edges = np.linspace(0.0, 2.0 * np.pi, self.num_bins + 1)
        bin_indices = np.digitize(theta_phases % (2.0 * np.pi), bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        mean_amps = np.zeros(self.num_bins)
        for b in range(self.num_bins):
            in_b = bin_indices == b
            if np.any(in_b):
                mean_amps[b] = np.mean(gamma_amplitudes[in_b])
            else:
                mean_amps[b] = 1e-6

        # Normalize to probability distribution P
        P = mean_amps / (np.sum(mean_amps) + 1e-8)
        # Uniform distribution U
        U = 1.0 / self.num_bins

        # D_KL(P || U) = sum P * log(P / U)
        kl = np.sum(P * np.log(np.clip(P / U, 1e-12, 1e12)))
        mi = float(kl / np.log(self.num_bins))
        return float(np.clip(mi, 0.0, 1.0))

    def evaluate_communication_through_coherence(
        self,
        duration_s: float = 1.0,
        phase_aligned: bool = True,
        noise_level: float = 0.1,
    ) -> Dict[str, Any]:
        """Tests whether phase-matched receiver decodes high-frequency messages with lower error."""
        t = np.arange(0.0, duration_s, self.dt)
        T = len(t)

        # Sender slow theta phase: phi_s(t) = 2 * pi * f_theta * t
        theta_sender = (2.0 * np.pi * self.theta_freq * t) % (2.0 * np.pi)
        # Gamma amplitude modulated by theta phase: peak at phi = pi
        gamma_envelope = 1.0 + 0.8 * np.cos(theta_sender - np.pi)
        gamma_signal = gamma_envelope * np.sin(2.0 * np.pi * self.gamma_freq * t)

        # Receiver theta phase: aligned (in-phase) vs misaligned (anti-phase)
        phase_offset = 0.0 if phase_aligned else np.pi
        theta_receiver = (theta_sender + phase_offset) % (2.0 * np.pi)

        # Gated reception: receiver opens receptive window at peak of its own theta phase
        receptive_window = np.maximum(0.0, np.cos(theta_receiver - np.pi)) ** 2

        # Transmitted message payload
        transmitted = gamma_signal + self.rng.normal(0, noise_level, T)
        received = transmitted * receptive_window

        # Modulation index
        mi = self.calculate_modulation_index(theta_sender, gamma_envelope)

        # Decoding fidelity proxy: correlation between transmitted envelope and received power
        received_power = received ** 2
        fidelity = float(np.corrcoef(gamma_envelope, received_power)[0, 1]) if np.std(received_power) > 1e-6 else 0.0

        return {
            "phase_aligned": phase_aligned,
            "modulation_index": mi,
            "transmission_fidelity": max(0.0, fidelity),
            "effective_bandwidth_gain": float(np.mean(receptive_window)),
        }
