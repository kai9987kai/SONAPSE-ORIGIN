"""SONAPSE-ORIGIN: Autonomous Discovery of Neural Organisation, Homeostasis,
Emergent Cognition and Consciousness-Relevant Dynamics.

Local-first, CPU-first experimental computational neuroscience and artificial
intelligence research platform.
"""

__version__ = "0.1.0"
__author__ = "kai9987kai"

from .mathematical_model import (
    SystemState,
    GraphStructure,
    HomeostaticStateVector,
    AffectiveStateVector,
    StateEvolutionOperator,
    NumericalSolver,
)

__all__ = [
    "SystemState",
    "GraphStructure",
    "HomeostaticStateVector",
    "AffectiveStateVector",
    "StateEvolutionOperator",
    "NumericalSolver",
]
