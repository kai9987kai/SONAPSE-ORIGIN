"""General Mathematical Model for SONAPSE-ORIGIN.

Implements the formal time-varying heterogeneous dynamical system:
    S_t = (G_t, X_t, W_t, H_t, A_t, M_t, B_t, E_t)

And its state evolution operator:
    S_{t + dt} = F_theta(S_t, u_t, xi_t)

Under deterministic seeding, snapshot restoration, and explicit numerical solver selection.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


class NumericalSolver(str, Enum):
    EULER = "euler"
    RUNGE_KUTTA_4 = "rk4"
    EXPONENTIAL_RELAX = "exponential_relax"


@dataclass
class GraphStructure:
    """Represents G_t: structural graph or hypergraph topology."""
    num_nodes: int
    edge_index: np.ndarray  # Shape: (2, num_edges), int64
    edge_weights: np.ndarray  # Shape: (num_edges,), float64
    node_types: List[str] = field(default_factory=list)
    hyperedges: List[List[int]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "edge_index": self.edge_index.tolist(),
            "edge_weights": self.edge_weights.tolist(),
            "node_types": self.node_types,
            "hyperedges": self.hyperedges,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GraphStructure:
        return cls(
            num_nodes=data["num_nodes"],
            edge_index=np.array(data["edge_index"], dtype=np.int64),
            edge_weights=np.array(data["edge_weights"], dtype=np.float64),
            node_types=data.get("node_types", []),
            hyperedges=data.get("hyperedges", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class HomeostaticStateVector:
    """Represents H_t: internal viability variables."""
    energy: float = 1.0          # Available metabolic energy [0, 1]
    integrity: float = 1.0       # Structural / physical integrity [0, 1]
    temperature: float = 0.5     # Regulated operating temperature [0, 1]
    fatigue: float = 0.0         # Computational & metabolic fatigue [0, 1]
    resource_reserve: float = 0.8# Stored reservoir [0, 1]
    uncertainty: float = 0.2     # Internal state estimation variance [0, 1]
    prediction_error: float = 0.1# Ongoing sensory/predictive divergence [0, 1]
    social_cohesion: float = 0.5 # Inter-agent interaction baseline [0, 1]
    internal_stability: float = 0.9# Global Lyapunov stability proxy [0, 1]

    def to_array(self) -> np.ndarray:
        return np.array([
            self.energy,
            self.integrity,
            self.temperature,
            self.fatigue,
            self.resource_reserve,
            self.uncertainty,
            self.prediction_error,
            self.social_cohesion,
            self.internal_stability,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> HomeostaticStateVector:
        c = np.clip(arr, 0.0, 1.0)
        return cls(
            energy=float(c[0]),
            integrity=float(c[1]),
            temperature=float(c[2]),
            fatigue=float(c[3]),
            resource_reserve=float(c[4]),
            uncertainty=float(c[5]),
            prediction_error=float(c[6]),
            social_cohesion=float(c[7]),
            internal_stability=float(c[8]),
        )

    def calculate_cost(self, setpoints: Optional[HomeostaticStateVector] = None) -> float:
        """Quadratic deviation cost: J_H = (h - h*)^T Q (h - h*)."""
        h = self.to_array()
        target = setpoints.to_array() if setpoints else np.array([1.0, 1.0, 0.5, 0.0, 0.8, 0.1, 0.05, 0.5, 0.95])
        weights = np.array([2.0, 3.0, 1.0, 1.5, 1.0, 0.5, 0.5, 0.3, 1.5])
        diff = h - target
        return float(np.sum(weights * (diff ** 2)))


@dataclass
class AffectiveStateVector:
    """Represents A_t: multidimensional artificial affective dynamics.
    
    Control states used for adaptive regulation; not subjective experiences.
    """
    valence: float = 0.0       # [-1.0 (aversive), +1.0 (appetitive)]
    arousal: float = 0.2       # [0.0 (quiescent), 1.0 (maximal)]
    dread: float = 0.0         # Anticipated prospective catastrophe [0, 1]
    calmness: float = 0.8      # Relaxation & equilibrium tendency [0, 1]
    fear: float = 0.0          # Immediate acute threat signal [0, 1]
    joy: float = 0.2           # Goal congruence & positive reinforcement [0, 1]
    surprise: float = 0.0      # Instantaneous divergence from expectation [0, 1]
    curiosity: float = 0.5     # Epistemic drive / novelty seeking [0, 1]
    restorative_drive: float = 0.1 # Need for sleep/quiescent consolidation [0, 1]

    def to_array(self) -> np.ndarray:
        return np.array([
            self.valence,
            self.arousal,
            self.dread,
            self.calmness,
            self.fear,
            self.joy,
            self.surprise,
            self.curiosity,
            self.restorative_drive,
        ], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> AffectiveStateVector:
        return cls(
            valence=float(np.clip(arr[0], -1.0, 1.0)),
            arousal=float(np.clip(arr[1], 0.0, 1.0)),
            dread=float(np.clip(arr[2], 0.0, 1.0)),
            calmness=float(np.clip(arr[3], 0.0, 1.0)),
            fear=float(np.clip(arr[4], 0.0, 1.0)),
            joy=float(np.clip(arr[5], 0.0, 1.0)),
            surprise=float(np.clip(arr[6], 0.0, 1.0)),
            curiosity=float(np.clip(arr[7], 0.0, 1.0)),
            restorative_drive=float(np.clip(arr[8], 0.0, 1.0)),
        )


@dataclass
class SystemState:
    """Represents S_t = (G_t, X_t, W_t, H_t, A_t, M_t, B_t, E_t)."""
    step: int
    time: float
    dt: float
    G: GraphStructure
    X: np.ndarray                 # Neural/field activations, shape (num_nodes, node_dim)
    W: np.ndarray                 # Synaptic parameter matrix, shape (num_nodes, num_nodes)
    H: HomeostaticStateVector
    A: AffectiveStateVector
    M: Dict[str, Any]             # Memory traces and self-model representations
    B: Dict[str, Any]             # Embodied agent pose/velocity/actuators
    E: Dict[str, Any]             # Environment state
    metadata: Dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> Dict[str, Any]:
        """Serialize state for exact checkpoint restoration and replay."""
        return {
            "step": self.step,
            "time": self.time,
            "dt": self.dt,
            "G": self.G.to_dict(),
            "X": self.X.tolist(),
            "W": self.W.tolist(),
            "H": asdict(self.H),
            "A": asdict(self.A),
            "M": copy.deepcopy(self.M),
            "B": copy.deepcopy(self.B),
            "E": copy.deepcopy(self.E),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def restore(cls, snapshot_data: Dict[str, Any]) -> SystemState:
        return cls(
            step=snapshot_data["step"],
            time=snapshot_data["time"],
            dt=snapshot_data["dt"],
            G=GraphStructure.from_dict(snapshot_data["G"]),
            X=np.array(snapshot_data["X"], dtype=np.float64),
            W=np.array(snapshot_data["W"], dtype=np.float64),
            H=HomeostaticStateVector(**snapshot_data["H"]),
            A=AffectiveStateVector(**snapshot_data["A"]),
            M=copy.deepcopy(snapshot_data["M"]),
            B=copy.deepcopy(snapshot_data["B"]),
            E=copy.deepcopy(snapshot_data["E"]),
            metadata=copy.deepcopy(snapshot_data.get("metadata", {})),
        )

    def verify_numerical_stability(self) -> Tuple[bool, List[str]]:
        """Validates that no numerical overflow, NaN, or non-finite values exist."""
        issues = []
        if not np.all(np.isfinite(self.X)):
            issues.append("Non-finite values detected in neural state X")
        if not np.all(np.isfinite(self.W)):
            issues.append("Non-finite values detected in weight matrix W")
        h_arr = self.H.to_array()
        if not np.all(np.isfinite(h_arr)):
            issues.append("Non-finite values detected in homeostatic state H")
        a_arr = self.A.to_array()
        if not np.all(np.isfinite(a_arr)):
            issues.append("Non-finite values detected in affective state A")
        return len(issues) == 0, issues


class StateEvolutionOperator:
    """F_theta: Evolves S_t into S_{t+dt} under intervention u_t and noise xi_t."""

    def __init__(
        self,
        solver: NumericalSolver = NumericalSolver.RUNGE_KUTTA_4,
        seed: int = 42,
    ):
        self.solver = solver
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def step(
        self,
        state: SystemState,
        intervention: Optional[Dict[str, Any]] = None,
        stochastic_scale: float = 0.01,
    ) -> SystemState:
        """Executes a single discrete forward integration step."""
        dt = state.dt
        u = intervention or {}

        # 1. Controlled Stochasticity xi_t
        noise_X = self.rng.normal(0, stochastic_scale, size=state.X.shape)

        # 2. Derivative calculation for neural dynamics: dX/dt = -X/tau + tanh(W X + b) + I_ext
        def compute_dX(x_val: np.ndarray) -> np.ndarray:
            tau = 1.0
            # Interaction: W @ x_val
            # If x_val is (N, D), message passing:
            net_input = state.W @ x_val
            # Apply intervention if any node clamped: do(X_i = c)
            if "clamp_nodes" in u:
                for n_idx, clamp_val in u["clamp_nodes"].items():
                    net_input[int(n_idx)] = clamp_val
            dx = (-x_val + np.tanh(net_input)) / tau
            return dx

        # 3. Numerical Integration (Euler or RK4)
        if self.solver == NumericalSolver.RUNGE_KUTTA_4:
            k1 = compute_dX(state.X)
            k2 = compute_dX(state.X + 0.5 * dt * k1)
            k3 = compute_dX(state.X + 0.5 * dt * k2)
            k4 = compute_dX(state.X + dt * k3)
            new_X = state.X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4) + noise_X
        else:
            # Euler
            k1 = compute_dX(state.X)
            new_X = state.X + dt * k1 + noise_X

        # Clip activations for biological plausibility and numerical safety
        new_X = np.clip(new_X, -10.0, 10.0)

        # 4. Homeostatic Evolution dH/dt
        h_vec = state.H.to_array()
        # Workload / energy consumption proportional to neural activity norm
        activity_energy = float(np.mean(np.abs(new_X))) * 0.05
        dH = np.zeros_like(h_vec)
        dH[0] = -activity_energy * dt                     # energy drains
        dH[1] = 0.0                                        # integrity
        dH[3] = (activity_energy * 2.0 - 0.02) * dt        # fatigue grows
        dH[6] = (float(np.std(new_X)) * 0.1 - h_vec[6]) * 0.1 * dt # prediction error

        # Environmental or intervention updates to H
        if "replenish_energy" in u:
            dH[0] += float(u["replenish_energy"])
        if "inflict_damage" in u:
            dH[1] -= float(u["inflict_damage"])

        new_H = HomeostaticStateVector.from_array(h_vec + dH)

        # 5. Affective Dynamics Evolution:
        # dA/dt = -Lambda A + W_a A + B_h H + eta
        a_vec = state.A.to_array()
        dA = np.zeros_like(a_vec)
        # Fear tracks threat and low integrity
        dA[4] = ((1.0 - new_H.integrity) * 2.0 - a_vec[4]) * 0.2 * dt
        # Joy tracks energy and recovery
        dA[5] = (new_H.energy * 0.8 - a_vec[5]) * 0.1 * dt
        # Valence tracks joy minus fear
        dA[0] = (dA[5] - dA[4] - a_vec[0] * 0.1) * dt
        # Arousal tracks activity
        dA[1] = (float(np.mean(np.abs(new_X))) * 0.5 - a_vec[1]) * 0.2 * dt

        new_A = AffectiveStateVector.from_array(a_vec + dA)

        # 6. Weights W_t: Hebbian / Plasticity update if enabled
        new_W = state.W.copy()
        if u.get("enable_plasticity", False):
            # Outer product of mean node activities
            act_mean = np.mean(new_X, axis=-1)
            delta_W = 0.001 * np.outer(act_mean, act_mean) - 0.0001 * new_W
            new_W = np.clip(new_W + delta_W, -2.0, 2.0)

        # Construct evolved system state S_{t+dt}
        evolved = SystemState(
            step=state.step + 1,
            time=state.time + dt,
            dt=dt,
            G=copy.deepcopy(state.G),
            X=new_X,
            W=new_W,
            H=new_H,
            A=new_A,
            M=copy.deepcopy(state.M),
            B=copy.deepcopy(state.B),
            E=copy.deepcopy(state.E),
            metadata=copy.deepcopy(state.metadata),
        )

        stable, issues = evolved.verify_numerical_stability()
        if not stable:
            raise FloatingPointError(f"Numerical instability in state evolution: {issues}")

        return evolved
