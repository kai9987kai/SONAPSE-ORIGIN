"""Dynamical Substrate Lab: Spiking, rate-based, coupled oscillators, neural fields, and wave-like channels."""

from .spiking import LIFPopulation
from .rate_based import ContinuousRateRNN
from .oscillators import KuramotoOscillators
from .reaction_diffusion import ContinuousNeuralField
from .wave_comm import SignalChannelComparator, SignalSubstrateMode, WaveMessage
from .simulator import UnifiedDynamicalSimulator

__all__ = [
    "LIFPopulation",
    "ContinuousRateRNN",
    "KuramotoOscillators",
    "ContinuousNeuralField",
    "SignalChannelComparator",
    "SignalSubstrateMode",
    "WaveMessage",
    "UnifiedDynamicalSimulator",
]
