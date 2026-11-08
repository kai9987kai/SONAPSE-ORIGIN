import os
import pytest
from sonapse_origin.infrastructure import ExperimentDatabase, ExperimentReceipt, SimulationContainmentGate
from sonapse_origin.adapters import SonapseLabAdapter, PoseidonAdapter, AetherAdapter, SupermixModelRouter


def test_experiment_database_and_receipt(tmp_path):
    db_file = tmp_path / "test.db"
    db = ExperimentDatabase(str(db_file))

    db.record_genome("g1", "feedforward", 8, "hash1", 0, {"weights": []})
    db.record_hypothesis("h1", "Test Hyp", "GWT", "preregistered", {})
    db.record_experiment("exp1", "Run 1", "h1", "g1", "Level1", 42, "completed", {})
    db.log_measurement("exp1", 1, "accuracy", 0.95)

    receipt = ExperimentReceipt.create(
        experiment_id="exp1",
        git_commit="test_commit",
        architecture_hash="hash1",
        seed=42,
        environment_level="Level1",
        solver="rk4",
        measurements={"accuracy": 0.95},
        status="supported",
    )
    assert len(receipt.receipt_hash) == 64


def test_simulation_containment_gate():
    gate = SimulationContainmentGate()
    safe_action = {"thrust": 0.5, "consume": True}
    ok, _ = gate.verify_action_safety(safe_action)
    assert ok

    dangerous_action = {"thrust": 0.5, "os_execute": "rm -rf /"}
    ok, err = gate.verify_action_safety(dangerous_action)
    assert not ok
    assert "SECURITY GATE BLOCKED" in err


def test_adapters():
    sonapse_ad = SonapseLabAdapter()
    bridge_info = sonapse_ad.create_homeostatic_bridge()
    assert "connected" in bridge_info

    pos_ad = PoseidonAdapter()
    critique = pos_ad.critique_hypothesis("Causal test")
    assert critique["passed_causal_check"]

    router = SupermixModelRouter()
    r_res = router.route_reasoning_task("Evaluate causal intervention")
    assert "Poseidon" in r_res.get("source", "")
