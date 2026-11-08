"""Wave-like Communication Modes and Substrate Signal Channel Comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class SignalSubstrateMode(str, Enum):
    SCALAR = "scalar"
    VECTOR = "vector"
    COMPLEX_VALUED = "complex_valued"
    PHASE_CODED = "phase_coded"
    SPIKE_TIMING = "spike_timing"
    WAVE_MULTIDIM = "wave_multidim"


@dataclass
class WaveMessage:
    """Represents an abstract wave-like computational packet: m_ij = (a, phi, f, p, c).
    
    Not literal electromagnetic or quantum processing; abstract degrees of freedom.
    """
    amplitude: float       # a in [0, 1]
    phase: float           # phi in [0, 2pi)
    frequency: float       # f > 0
    polarization: np.ndarray # p: 2D or 3D unit direction vector
    content: np.ndarray    # c: latent feature payload vector

    def energy_cost(self) -> float:
        """Energy cost proportional to amplitude squared and frequency + payload dimension."""
        payload_dim = len(self.content)
        return float(self.amplitude**2 * (1.0 + self.frequency) + 0.1 * payload_dim)


class SignalChannelComparator:
    """Compares information efficiency, latency, energy cost, and noise robustness."""

    def __init__(self, rng_seed: int = 42):
        self.rng = np.random.default_rng(rng_seed)

    def transmit(
        self,
        mode: SignalSubstrateMode,
        raw_signal: np.ndarray,
        noise_level: float = 0.1,
    ) -> Dict[str, float]:
        """Encodes, transmits over a noisy channel, decodes, and measures metrics."""
        raw_dim = len(raw_signal)

        if mode == SignalSubstrateMode.SCALAR:
            # Average scalar pooling
            tx = np.array([float(np.mean(raw_signal))])
            energy = float(tx[0] ** 2)
            rx = tx + self.rng.normal(0, noise_level, size=tx.shape)
            decoded = np.full(raw_dim, rx[0])
            latency = 1.0

        elif mode == SignalSubstrateMode.VECTOR:
            tx = raw_signal.copy()
            energy = float(np.sum(tx ** 2))
            rx = tx + self.rng.normal(0, noise_level, size=tx.shape)
            decoded = rx
            latency = 1.2

        elif mode == SignalSubstrateMode.COMPLEX_VALUED:
            # Half as real, half as imaginary
            half = max(1, raw_dim // 2)
            re = raw_signal[:half]
            im = raw_signal[half : 2 * half] if 2 * half <= raw_dim else np.zeros(half)
            tx_complex = re + 1j * im
            energy = float(np.sum(np.abs(tx_complex) ** 2))
            rx_complex = tx_complex + (self.rng.normal(0, noise_level, size=half) + 1j * self.rng.normal(0, noise_level, size=half))
            decoded = np.concatenate([np.real(rx_complex), np.imag(rx_complex)])
            if len(decoded) < raw_dim:
                decoded = np.pad(decoded, (0, raw_dim - len(decoded)))
            latency = 1.1

        elif mode == SignalSubstrateMode.PHASE_CODED:
            # Phase modulation: theta = 2 * pi * sigmoid(x)
            phases = 2.0 * np.pi / (1.0 + np.exp(-raw_signal))
            energy = float(len(phases) * 0.5)
            noisy_phases = (phases + self.rng.normal(0, noise_level * np.pi, size=phases.shape)) % (2.0 * np.pi)
            decoded = -np.log(np.clip(2.0 * np.pi / np.clip(noisy_phases, 1e-3, 2.0 * np.pi - 1e-3) - 1.0, 1e-4, 1e4))
            latency = 1.3

        elif mode == SignalSubstrateMode.SPIKE_TIMING:
            # Temporal latency coding: earlier spike = higher value
            norm = (raw_signal - np.min(raw_signal)) / (np.ptp(raw_signal) + 1e-5)
            spike_times = (1.0 - norm) * 10.0 + self.rng.normal(0, noise_level * 2.0, size=raw_dim)
            energy = float(raw_dim * 0.2)
            decoded = 1.0 - np.clip(spike_times / 10.0, 0.0, 1.0)
            latency = 2.5

        elif mode == SignalSubstrateMode.WAVE_MULTIDIM:
            amp = float(np.clip(np.mean(np.abs(raw_signal)), 0.1, 1.0))
            phase = float(self.rng.uniform(0, 2 * np.pi))
            freq = 2.0
            pol = np.array([1.0, 0.0])
            msg = WaveMessage(
                amplitude=amp,
                phase=phase,
                frequency=freq,
                polarization=pol,
                content=raw_signal.copy(),
            )
            energy = msg.energy_cost()
            noisy_content = msg.content + self.rng.normal(0, noise_level / (amp + 0.1), size=raw_dim)
            decoded = noisy_content
            latency = 1.05

        # Calculate reconstruction error (MSE)
        mse = float(np.mean((raw_signal - decoded) ** 2))
        fidelity = 1.0 / (1.0 + mse)
        info_efficiency = float(raw_dim * fidelity / (energy + 1e-5))

        return {
            "mode": mode.value,
            "mse": mse,
            "fidelity": fidelity,
            "energy_cost": energy,
            "latency": latency,
            "info_efficiency": info_efficiency,
        }
