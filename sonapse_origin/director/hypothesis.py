"""Scientific Hypothesis Preregistration and Lifecycle Management."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"
    PREREGISTERED = "preregistered"
    TESTING = "testing"
    SUPPORTED = "supported"
    FALSIFIED = "falsified"
    INCONCLUSIVE = "inconclusive"


@dataclass
class PreregisteredPrediction:
    target_metric: str
    expected_direction: str  # "greater_than", "less_than", "equal"
    comparison_condition: str
    baseline_condition: str
    minimum_effect_size: float = 0.2


@dataclass
class ScientificHypothesis:
    """Formal hypothesis record with preregistered predictions and falsification criteria."""
    hypothesis_id: str
    title: str
    statement: str
    theory_association: str  # e.g. "Global Workspace", "IIT Proxy", "Homeostasis", "AST"
    rival_explanations: List[str]
    predictions: List[PreregisteredPrediction]
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    preregistration_time: float = field(default_factory=time.time)
    results: Dict[str, Any] = field(default_factory=dict)
    falsification_notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def preregister(self) -> None:
        """Locks predictions prior to simulation run."""
        self.status = HypothesisStatus.PREREGISTERED
        self.preregistration_time = time.time()

    def evaluate_test_results(
        self,
        effect_size: float,
        p_value: float,
        p_threshold: float = 0.05,
    ) -> HypothesisStatus:
        """Evaluates whether preregistered predictions were supported or falsified."""
        req_effect = self.predictions[0].minimum_effect_size if self.predictions else 0.2
        if p_value < p_threshold and effect_size >= req_effect:
            self.status = HypothesisStatus.SUPPORTED
        elif effect_size < 0.0 or p_value >= p_threshold:
            self.status = HypothesisStatus.FALSIFIED
        else:
            self.status = HypothesisStatus.INCONCLUSIVE
        return self.status
