import numpy as np
import pytest
from sonapse_origin.director import (
    ScientificHypothesis,
    HypothesisStatus,
    StatisticianCritic,
    SkepticCritic,
    TheoryRegistry,
    AutonomousResearchDirector,
)


def test_statistician_critic():
    stat = StatisticianCritic()
    treatment = np.array([5.2, 5.5, 5.8, 5.1, 5.6, 5.4, 5.7])
    control = np.array([3.1, 3.4, 3.2, 3.5, 3.0, 3.3, 3.2])
    res = stat.analyze(treatment, control)
    assert res["statistically_significant"]
    assert res["p_value"] < 0.001
    assert res["cohens_d"] > 2.0


def test_skeptic_critic_banned_phrases():
    skeptic = SkepticCritic()
    bad_claim = "The agent feels pain and is conscious of its existence."
    audit_bad = skeptic.audit_claim(bad_claim)
    assert not audit_bad["approved"]
    assert len(audit_bad["flags"]) >= 1

    good_claim = "The agent exhibits adaptive avoidance driven by homeostatic cost."
    audit_good = skeptic.audit_claim(good_claim)
    assert audit_good["approved"]


def test_theory_registry():
    registry = TheoryRegistry()
    gwt = registry.get_theory("GWT")
    assert gwt is not None
    assert "Global Workspace" in gwt.name


def test_autonomous_research_director_cycle():
    director = AutonomousResearchDirector()
    hyp = director.propose_hypothesis(
        hypothesis_id="HYP_001",
        title="Global Workspace Broadens Information Access",
        statement="Global workspace hub enhances cross-module information broadcast.",
        theory="GWT",
        target_metric="broadcast_reach",
    )
    assert hyp.status == HypothesisStatus.PROPOSED
    director.preregister_hypothesis("HYP_001")
    assert hyp.status == HypothesisStatus.PREREGISTERED

    # High-performance treatment vs low-performance control
    treatment = np.random.normal(0.9, 0.05, 10)
    control = np.random.normal(0.2, 0.05, 10)
    ev = director.evaluate_experiment_results("HYP_001", treatment, control)
    assert ev["status"] == "supported"
    assert len(director.discoveries_ledger) == 1
