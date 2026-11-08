"""Discovery Module: SINDy nonlinear dynamics discovery, symbolic regression, and out-of-distribution law validation."""

from .sindy import SINDyDiscoveryEngine
from .symbolic_regression import SymbolicLawDiscoverer
from .law_validator import CandidateLawValidator

__all__ = [
    "SINDyDiscoveryEngine",
    "SymbolicLawDiscoverer",
    "CandidateLawValidator",
]
