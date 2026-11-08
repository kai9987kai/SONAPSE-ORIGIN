"""Structure Forge: Architecture generation, hypergraphs, genomes, neural development, O-information synergy, and BCM metaplasticity."""

from .genome import (
    ArchitectureGenome,
    CommunicationMode,
    TopologyFamily,
)
from .topology import TopologyGenerator
from .hypergraph import HypergraphMessagePassing, MatchedGraphBaseline
from .development import StructuralPlasticityEngine, STDPPlasticity
from .synergy import HypergraphSynergyAssay
from .metaplasticity import BCMMetaplasticityEngine

__all__ = [
    "ArchitectureGenome",
    "CommunicationMode",
    "TopologyFamily",
    "TopologyGenerator",
    "HypergraphMessagePassing",
    "MatchedGraphBaseline",
    "StructuralPlasticityEngine",
    "STDPPlasticity",
    "HypergraphSynergyAssay",
    "BCMMetaplasticityEngine",
]
