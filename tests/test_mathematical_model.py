import numpy as np
import pytest
from sonapse_origin.mathematical_model import (
    SystemState,
    GraphStructure,
    HomeostaticStateVector,
    AffectiveStateVector,
    StateEvolutionOperator,
    NumericalSolver,
)


def test_system_state_initialization_and_snapshot():
    num_nodes = 5
    node_dim = 4
    edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    edge_weights = np.ones(4, dtype=np.float64)

    G = GraphStructure(num_nodes=num_nodes, edge_index=edge_index, edge_weights=edge_weights)
    X = np.zeros((num_nodes, node_dim), dtype=np.float64)
    W = np.eye(num_nodes, dtype=np.float64) * 0.5
    H = HomeostaticStateVector()
    A = AffectiveStateVector()

    state = SystemState(
        step=0,
        time=0.0,
        dt=0.05,
        G=G,
        X=X,
        W=W,
        H=H,
        A=A,
        M={"memory_buffer": []},
        B={"pos": [0.0, 0.0]},
        E={"food_items": 10},
    )

    stable, issues = state.verify_numerical_stability()
    assert stable
    assert len(issues) == 0

    snapshot = state.snapshot()
    restored = SystemState.restore(snapshot)

    assert restored.step == state.step
    assert np.allclose(restored.X, state.X)
    assert np.allclose(restored.W, state.W)
    assert restored.H.energy == state.H.energy
    assert restored.A.calmness == state.A.calmness


def test_rk4_and_euler_evolution():
    num_nodes = 4
    node_dim = 2
    G = GraphStructure(
        num_nodes=num_nodes,
        edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        edge_weights=np.ones(3, dtype=np.float64),
    )
    X = np.ones((num_nodes, node_dim), dtype=np.float64) * 0.2
    W = np.random.default_rng(123).normal(0, 0.2, (num_nodes, num_nodes))
    state = SystemState(
        step=0,
        time=0.0,
        dt=0.01,
        G=G,
        X=X,
        W=W,
        H=HomeostaticStateVector(),
        A=AffectiveStateVector(),
        M={},
        B={},
        E={},
    )

    operator_rk4 = StateEvolutionOperator(solver=NumericalSolver.RUNGE_KUTTA_4, seed=42)
    next_rk4 = operator_rk4.step(state)
    assert next_rk4.step == 1
    assert next_rk4.time == pytest.approx(0.01)
    assert np.all(np.isfinite(next_rk4.X))

    operator_euler = StateEvolutionOperator(solver=NumericalSolver.EULER, seed=42)
    next_euler = operator_euler.step(state)
    assert next_euler.step == 1
    assert np.all(np.isfinite(next_euler.X))


def test_causal_intervention_clamping():
    num_nodes = 3
    node_dim = 2
    G = GraphStructure(
        num_nodes=num_nodes,
        edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
        edge_weights=np.ones(2, dtype=np.float64),
    )
    X = np.zeros((num_nodes, node_dim), dtype=np.float64)
    W = np.ones((num_nodes, num_nodes), dtype=np.float64) * 0.1
    state = SystemState(
        step=0,
        time=0.0,
        dt=0.05,
        G=G,
        X=X,
        W=W,
        H=HomeostaticStateVector(),
        A=AffectiveStateVector(),
        M={},
        B={},
        E={},
    )

    operator = StateEvolutionOperator(solver=NumericalSolver.RUNGE_KUTTA_4, seed=1)
    clamped_state = operator.step(state, intervention={"clamp_nodes": {0: np.array([5.0, 5.0])}})
    assert np.all(np.isfinite(clamped_state.X))
