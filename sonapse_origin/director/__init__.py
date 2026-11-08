"""Research Director: Scientific hypothesis preregistration, adversarial critics, theory registry, and automated evidence management."""

from .hypothesis import ScientificHypothesis, HypothesisStatus, PreregisteredPrediction
from .critics import StatisticianCritic, SkepticCritic, MethodologistCritic
from .theories_registry import TheoryRegistry, TheoryProfile
from .literature import LiteratureIntelligence, ResearchPaper
from .research_director import AutonomousResearchDirector

__all__ = [
    "ScientificHypothesis",
    "HypothesisStatus",
    "PreregisteredPrediction",
    "StatisticianCritic",
    "SkepticCritic",
    "MethodologistCritic",
    "TheoryRegistry",
    "TheoryProfile",
    "LiteratureIntelligence",
    "ResearchPaper",
    "AutonomousResearchDirector",
]
