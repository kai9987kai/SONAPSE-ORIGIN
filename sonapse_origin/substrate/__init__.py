"""Dynamical Substrate Lab: Spiking, rate-based, coupled oscillators, neural fields, wave-like channels, and phase-amplitude coupling."""

from .spiking import LIFPopulation
from .rate_based import ContinuousRateRNN
from .oscillators import KuramotoOscillators
from .reaction_diffusion import ContinuousNeuralField
from .wave_comm import SignalChannelComparator, SignalSubstrateMode, WaveMessage
from .simulator import UnifiedDynamicalSimulator
from .phase_amplitude_coupling import PhaseAmplitudeCouplingSubstrate

__all__ = [
    "LIFPopulation",
    "ContinuousRateRNN",
    "KuramotoOscillators",
    "ContinuousNeuralField",
    "SignalChannelComparator",
    "SignalSubstrateMode",
    "WaveMessage",
    "UnifiedDynamicalSimulator",
    "PhaseAmplitudeCouplingSubstrate",
]
