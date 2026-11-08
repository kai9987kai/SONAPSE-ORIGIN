"""Theory Registry: Competing Theories of Consciousness and Cognitive Architecture."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TheoryProfile:
    name: str
    acronym: str
    primary_authors: List[str]
    core_computational_claim: str
    operational_proxy: str
    falsification_condition: str


class TheoryRegistry:
    """Maintains definitions and comparative predictions across competing theoretical frameworks."""

    def __init__(self):
        self.theories: Dict[str, TheoryProfile] = {
            "GWT": TheoryProfile(
                name="Global Workspace Theory",
                acronym="GWT",
                primary_authors=["Baars", "Dehaene", "Changeux"],
                core_computational_claim="Conscious access corresponds to non-linear ignition and global broadcast across modular specialists.",
                operational_proxy="Broadcast reach, ignition threshold jump, cross-module reportability.",
                falsification_condition="Finding integrated cognition without global broadcast or ignition events.",
            ),
            "IIT": TheoryProfile(
                name="Integrated Information Theory (Proxy)",
                acronym="IIT-Proxy",
                primary_authors=["Tononi", "Massimini", "Koch"],
                core_computational_claim="Consciousness corresponds to intrinsic causal integrated information irreducibility.",
                operational_proxy="Partition sensitivity I_partition, Perturbational Complexity Index (PCI).",
                falsification_condition="Systems with high integration showing completely disintegrated or purely feedforward behavior.",
            ),
            "RPT": TheoryProfile(
                name="Recurrent Processing Theory",
                acronym="RPT",
                primary_authors=["Lamme", "Roelfsema"],
                core_computational_claim="Local recurrent feedback pathways are necessary and sufficient for sensory awareness.",
                operational_proxy="Feedback pathway lesion deficits, post-stimulus temporal persistence.",
                falsification_condition="Preserved perceptual performance when feedback pathways are completely eliminated.",
            ),
            "HOT": TheoryProfile(
                name="Higher-Order Representation Theory",
                acronym="HOT",
                primary_authors=["Rosenthal", "Lau", "Brown"],
                core_computational_claim="Mental states are conscious only when represented by a higher-order metacognitive state.",
                operational_proxy="Type-2 Signal Detection sensitivity (meta-d'), confidence calibration ECE.",
                falsification_condition="Accurate self-monitoring without second-order meta-representation.",
            ),
            "PP": TheoryProfile(
                name="Predictive Processing",
                acronym="PP",
                primary_authors=["Friston", "Clark", "Hohwy"],
                core_computational_claim="Cognition consists of hierarchical minimization of precision-weighted prediction errors.",
                operational_proxy="Variational free energy proxy reduction, top-down predictive suppression.",
                falsification_condition="Feedforward systems matching predictive systems under noisy/occluded stimuli with equal compute.",
            ),
            "AST": TheoryProfile(
                name="Attention Schema Theory",
                acronym="AST",
                primary_authors=["Graziano"],
                core_computational_claim="The brain constructs a descriptive model of its own attention to guide attention control.",
                operational_proxy="Distraction intrusion rate, tracking fidelity with vs without schema.",
                falsification_condition="No performance difference between systems with vs without explicit attention schema.",
            ),
        }

    def get_theory(self, acronym: str) -> Optional[TheoryProfile]:
        return self.theories.get(acronym.upper())

    def list_all(self) -> List[TheoryProfile]:
        return list(self.theories.values())
