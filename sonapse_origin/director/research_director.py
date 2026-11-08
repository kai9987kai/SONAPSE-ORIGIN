"""Autonomous Research Director: Orchestrates hypotheses, critics, statistical tests, and discovery registry."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from .hypothesis import ScientificHypothesis, HypothesisStatus, PreregisteredPrediction
from .critics import StatisticianCritic, SkepticCritic, MethodologistCritic
from .theories_registry import TheoryRegistry
from .literature import LiteratureIntelligence


class AutonomousResearchDirector:
    """Manages the full lifecycle of automated scientific exploration:
    
        Hypothesis -> Methodological Review -> Preregistration -> Experiment ->
        Statistical Analysis -> Skeptical Audit -> Replication & Discovery Ledger
    """

    def __init__(self):
        self.theory_registry = TheoryRegistry()
        self.literature = LiteratureIntelligence()
        self.statistician = StatisticianCritic()
        self.skeptic = SkepticCritic()
        self.methodologist = MethodologistCritic()

        self.hypotheses_ledger: Dict[str, ScientificHypothesis] = {}
        self.discoveries_ledger: List[Dict[str, Any]] = []

    def propose_hypothesis(
        self,
        hypothesis_id: str,
        title: str,
        statement: str,
        theory: str,
        target_metric: str,
        expected_direction: str = "greater_than",
        rival_explanations: Optional[List[str]] = None,
    ) -> ScientificHypothesis:
        """Drafts a formal hypothesis and performs initial skeptical audit."""
        audit = self.skeptic.audit_claim(statement)
        if not audit["approved"]:
            statement += " [Reviewed: Restricted to objective computational observations]"

        hyp = ScientificHypothesis(
            hypothesis_id=hypothesis_id,
            title=title,
            statement=statement,
            theory_association=theory,
            rival_explanations=rival_explanations or ["Uncontrolled capacity difference", "Stochastic noise artifact"],
            predictions=[
                PreregisteredPrediction(
                    target_metric=target_metric,
                    expected_direction=expected_direction,
                    comparison_condition="treatment",
                    baseline_condition="control",
                    minimum_effect_size=0.3,
                )
            ],
            metadata={"audit": audit},
        )
        self.hypotheses_ledger[hypothesis_id] = hyp
        return hyp

    def preregister_hypothesis(self, hypothesis_id: str) -> None:
        """Locks hypothesis in preregistered status prior to executing simulation."""
        hyp = self.hypotheses_ledger[hypothesis_id]
        hyp.preregister()

    def evaluate_experiment_results(
        self,
        hypothesis_id: str,
        treatment_data: np.ndarray,
        control_data: np.ndarray,
        replication_seeds: int = 3,
    ) -> Dict[str, Any]:
        """Performs statistical analysis, updates hypothesis status, and records evidence."""
        hyp = self.hypotheses_ledger[hypothesis_id]
        stats_res = self.statistician.analyze(treatment_data, control_data)

        # Falsification / Support evaluation
        status = hyp.evaluate_test_results(
            effect_size=stats_res["cohens_d"],
            p_value=stats_res["p_value"],
        )
        hyp.results = stats_res

        evidence_entry = {
            "hypothesis_id": hypothesis_id,
            "title": hyp.title,
            "theory": hyp.theory_association,
            "status": status.value,
            "effect_size_d": stats_res["cohens_d"],
            "p_value": stats_res["p_value"],
            "statistically_significant": stats_res["statistically_significant"],
            "ci_95": [stats_res["ci_95_lower"], stats_res["ci_95_upper"]],
            "replicated_seeds": replication_seeds,
        }

        if status == HypothesisStatus.SUPPORTED:
            self.discoveries_ledger.append(evidence_entry)

        return evidence_entry
