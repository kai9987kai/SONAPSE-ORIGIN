import numpy as np
import pytest
from sonapse_origin.homeostasis.viability import HomeostaticRegulator, ViabilityEnvelope
from sonapse_origin.homeostasis.affect import AffectiveDynamicsEngine
from sonapse_origin.homeostasis.neuromodulation import NeuromodulatorySystem
from sonapse_origin.homeostasis.fatigue import FatigueTracker
from sonapse_origin.homeostasis.sleep_replay import SleepReplayEngine, ReplayExperience, ReplayMode
from sonapse_origin.homeostasis.policy import SelfMaintenanceRewardPolicy


def test_homeostatic_regulator():
    reg = HomeostaticRegulator()
    h = np.array([0.9, 1.0, 0.5, 0.1, 0.8, 0.1, 0.05, 0.5, 0.95])
    cost = reg.compute_cost(h)
    assert cost == pytest.approx(0.0, abs=1e-5)
    assert reg.is_viable(h)

    # Extreme depletion
    h_depleted = np.array([0.0, 0.2, 0.5, 0.9, 0.1, 0.5, 0.5, 0.1, 0.1])
    assert not reg.is_viable(h_depleted)
    assert reg.compute_cost(h_depleted) > 1.0


def test_affective_dynamics_stability():
    engine = AffectiveDynamicsEngine(dt=0.05, seed=42)
    stable, max_real = engine.is_lyapunov_stable()
    assert stable
    assert max_real < 0.0

    h_vec = np.ones(9) * 0.5
    a_next = engine.step(h_vec)
    assert len(a_next) == 9
    assert -1.0 <= a_next[0] <= 1.0
    assert np.all(a_next[1:] >= 0.0)


def test_neuromodulatory_system():
    nm = NeuromodulatorySystem()
    levels = nm.update(reward_error=0.8, threat_level=0.9, uncertainty=0.7, fatigue=0.4)
    assert levels.dopamine > 0.5
    assert levels.norepinephrine > 0.5


def test_fatigue_and_recovery():
    ft = FatigueTracker(alpha=0.1, beta=0.2)
    f1 = ft.step(workload=2.0, resting=False)
    assert f1 > 0.0
    f2 = ft.step(workload=0.0, resting=True)
    assert f2 < f1


def test_sleep_replay_and_renormalization():
    engine = SleepReplayEngine(buffer_capacity=10, renormalization_factor=0.9)
    for i in range(5):
        engine.store(ReplayExperience(
            state=np.zeros(2), action=i, reward=float(i), next_state=np.ones(2), surprise=float(i * 0.5)
        ))

    salient = engine.sample_batch(3, mode=ReplayMode.SALIENT_REPLAY)
    assert len(salient) == 3

    random_b = engine.sample_batch(3, mode=ReplayMode.RANDOM_REPLAY)
    assert len(random_b) == 3

    no_b = engine.sample_batch(3, mode=ReplayMode.NO_REPLAY)
    assert len(no_b) == 0

    W = np.ones((4, 4))
    new_W, saved = engine.synaptic_renormalization(W)
    assert saved > 0.0
    assert np.allclose(new_W, W * 0.9)


def test_self_maintenance_policy():
    policy = SelfMaintenanceRewardPolicy(lambda_H=1.0, lambda_E=0.5, lambda_D=2.0)
    res = policy.compute_reward(task_reward=10.0, homeostatic_cost=2.0, energy_spent=1.0, damage_taken=1.0)
    # 10 - 1*2 - 0.5*1 - 2*1 = 5.5
    assert res["total_reward"] == pytest.approx(5.5)
