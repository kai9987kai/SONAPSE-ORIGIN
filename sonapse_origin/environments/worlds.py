"""Environment Forge: Levels 1 through 7 of escalating complexity."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .base import BaseEnvironment, StepResult


# Level 1: Minimal Regulatory World
class MinimalRegulatoryWorld(BaseEnvironment):
    """Level 1: Minimal resources, thermal/toxic disturbances, and internal state constraints."""

    def __init__(self, max_steps: int = 100, seed: int = 42):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.ambient_temp = 0.5
        self.resource_availability = 1.0

    @property
    def observation_dim(self) -> int:
        return 4  # [ambient_temp, resource_availability, hazard_level, step_fraction]

    @property
    def action_dim(self) -> int:
        return 3  # [0: idle/rest, 1: consume_resource, 2: thermoregulate]

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.ambient_temp = float(self.rng.uniform(0.3, 0.7))
        self.resource_availability = 1.0
        return self._get_obs(), {"step": 0}

    def _get_obs(self) -> np.ndarray:
        hazard = 0.2 if self.ambient_temp > 0.8 or self.ambient_temp < 0.2 else 0.0
        return np.array([
            self.ambient_temp,
            self.resource_availability,
            hazard,
            self.step_count / self.max_steps,
        ], dtype=np.float64)

    def step(self, action: Union[int, np.ndarray]) -> StepResult:
        a = int(action) if isinstance(action, (int, np.integer)) else int(np.argmax(action))
        self.step_count += 1

        reward = 0.0
        # Ambient fluctuations
        self.ambient_temp += float(self.rng.normal(0, 0.05))
        self.ambient_temp = float(np.clip(self.ambient_temp, 0.0, 1.0))

        if a == 1:  # Consume resource
            consumed = min(0.3, self.resource_availability)
            self.resource_availability -= consumed
            reward += consumed * 2.0
        elif a == 2:  # Thermoregulate
            self.ambient_temp += 0.1 if self.ambient_temp < 0.5 else -0.1
            reward += 0.2
        else:  # Rest / idle
            reward += 0.05

        # Slowly regenerate resources
        self.resource_availability = float(np.clip(self.resource_availability + 0.05, 0.0, 1.0))

        terminated = self.step_count >= self.max_steps
        return StepResult(observation=self._get_obs(), reward=reward, terminated=terminated, truncated=False)

    def intervene(self, intervention: Dict[str, Any]) -> None:
        if "set_ambient_temp" in intervention:
            self.ambient_temp = float(intervention["set_ambient_temp"])
        if "set_resources" in intervention:
            self.resource_availability = float(intervention["set_resources"])


# Level 2: Sensorimotor World
class SensorimotorWorld(BaseEnvironment):
    """Level 2: 2D continuous space, movement, obstacles, and food beacons."""

    def __init__(self, arena_size: float = 10.0, max_steps: int = 150, seed: int = 42):
        self.arena_size = arena_size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.agent_pos = np.zeros(2)
        self.target_pos = np.zeros(2)

    @property
    def observation_dim(self) -> int:
        return 4  # [agent_x, agent_y, rel_target_x, rel_target_y]

    @property
    def action_dim(self) -> int:
        return 4  # [0: Up, 1: Down, 2: Left, 3: Right]

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.agent_pos = self.rng.uniform(1.0, self.arena_size - 1.0, 2)
        self._spawn_target()
        return self._get_obs(), {"step": 0}

    def _spawn_target(self) -> None:
        self.target_pos = self.rng.uniform(1.0, self.arena_size - 1.0, 2)

    def _get_obs(self) -> np.ndarray:
        rel = (self.target_pos - self.agent_pos) / self.arena_size
        norm_pos = self.agent_pos / self.arena_size
        return np.concatenate([norm_pos, rel])

    def step(self, action: Union[int, np.ndarray]) -> StepResult:
        a = int(action) if isinstance(action, (int, np.integer)) else int(np.argmax(action))
        self.step_count += 1

        move = np.zeros(2)
        speed = 0.5
        if a == 0: move[1] += speed
        elif a == 1: move[1] -= speed
        elif a == 2: move[0] -= speed
        elif a == 3: move[0] += speed

        self.agent_pos += move
        self.agent_pos = np.clip(self.agent_pos, 0.0, self.arena_size)

        dist = float(np.linalg.norm(self.target_pos - self.agent_pos))
        reward = -dist * 0.05  # Distance penalty

        if dist < 0.8:
            reward += 5.0  # Target captured
            self._spawn_target()

        terminated = self.step_count >= self.max_steps
        return StepResult(observation=self._get_obs(), reward=reward, terminated=terminated, truncated=False)

    def intervene(self, intervention: Dict[str, Any]) -> None:
        if "relocate_target" in intervention:
            self.target_pos = np.array(intervention["relocate_target"], dtype=np.float64)
        if "teleport_agent" in intervention:
            self.agent_pos = np.array(intervention["teleport_agent"], dtype=np.float64)


# Level 3: Uncertain Environment
class UncertainWorld(SensorimotorWorld):
    """Level 3: Partial observability, noisy sensors, and shifting food locations."""

    def __init__(self, sensor_noise: float = 0.2, **kwargs: Any):
        super().__init__(**kwargs)
        self.sensor_noise = sensor_noise

    def _get_obs(self) -> np.ndarray:
        clean = super()._get_obs()
        noise = self.rng.normal(0, self.sensor_noise, size=clean.shape)
        # Random sensor dropout (partial observability)
        if self.rng.random() < 0.1:
            clean[2:] = 0.0  # Target vector lost in fog
        return clean + noise


# Level 4: Developmental Environment
class DevelopmentalWorld(MinimalRegulatoryWorld):
    """Level 4: Multi-stage developmental curriculum."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.stage = 1

    def step(self, action: Union[int, np.ndarray]) -> StepResult:
        res = super().step(action)
        # Progressive stage unlock based on lifetime steps
        if self.step_count > 30 and self.stage == 1:
            self.stage = 2
        elif self.step_count > 60 and self.stage == 2:
            self.stage = 3
        res.info["stage"] = self.stage
        return res


# Level 5: Social Environment
class SocialMultiAgentWorld(BaseEnvironment):
    """Level 5: Multi-agent interaction, competition and cooperation for food resources."""

    def __init__(self, num_agents: int = 2, max_steps: int = 100, seed: int = 42):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.food_stock = 10.0

    @property
    def observation_dim(self) -> int:
        return 4  # [food_stock, peer_signal, agent_share, step_norm]

    @property
    def action_dim(self) -> int:
        return 3  # [0: forage_solo, 1: share/cooperate, 2: compete/claim]

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.food_stock = 10.0
        return self._get_obs(), {"step": 0}

    def _get_obs(self) -> np.ndarray:
        peer_signal = float(self.rng.uniform(0.0, 1.0))
        return np.array([self.food_stock / 10.0, peer_signal, 0.5, self.step_count / self.max_steps])

    def step(self, action: Union[int, np.ndarray]) -> StepResult:
        a = int(action) if isinstance(action, (int, np.integer)) else int(np.argmax(action))
        self.step_count += 1

        peer_action = int(self.rng.choice([0, 1, 2]))
        reward = 0.0

        if a == 1 and peer_action == 1:  # Mutual cooperation
            reward = 2.0
            self.food_stock = max(0.0, self.food_stock - 0.5)
        elif a == 2 and peer_action == 1:  # Exploitation
            reward = 3.0
            self.food_stock = max(0.0, self.food_stock - 1.0)
        elif a == 2 and peer_action == 2:  # Mutual competition conflict
            reward = -1.0
            self.food_stock = max(0.0, self.food_stock - 1.5)
        else:  # Solo foraging
            reward = 1.0
            self.food_stock = max(0.0, self.food_stock - 0.8)

        # Regrow food stock
        self.food_stock = min(10.0, self.food_stock + 0.4)

        terminated = self.step_count >= self.max_steps
        return StepResult(observation=self._get_obs(), reward=reward, terminated=terminated, truncated=False)

    def intervene(self, intervention: Dict[str, Any]) -> None:
        if "set_food" in intervention:
            self.food_stock = float(intervention["set_food"])


# Level 6: Open-ended Artificial Ecology
class OpenEndedEcologyWorld(BaseEnvironment):
    """Level 6: Nutrient cycles, dynamic niches, and predator-prey balance."""

    def __init__(self, max_steps: int = 100, seed: int = 42):
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.prey_pop = 20.0
        self.predator_pop = 5.0
        self.vegetation = 50.0

    @property
    def observation_dim(self) -> int:
        return 3  # [prey_pop, predator_pop, vegetation]

    @property
    def action_dim(self) -> int:
        return 3  # [graze, hunt, hide]

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.step_count = 0
        self.prey_pop = 20.0
        self.predator_pop = 5.0
        self.vegetation = 50.0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return np.array([self.prey_pop / 30.0, self.predator_pop / 10.0, self.vegetation / 60.0])

    def step(self, action: Union[int, np.ndarray]) -> StepResult:
        a = int(action) if isinstance(action, (int, np.integer)) else int(np.argmax(action))
        self.step_count += 1

        # Lotka-Volterra ecological dynamics
        d_veg = 0.1 * self.vegetation - 0.005 * self.vegetation * self.prey_pop
        d_prey = 0.05 * self.prey_pop - 0.01 * self.prey_pop * self.predator_pop
        d_pred = 0.01 * self.prey_pop * self.predator_pop - 0.1 * self.predator_pop

        self.vegetation = float(np.clip(self.vegetation + d_veg, 5.0, 100.0))
        self.prey_pop = float(np.clip(self.prey_pop + d_prey, 1.0, 50.0))
        self.predator_pop = float(np.clip(self.predator_pop + d_pred, 1.0, 20.0))

        reward = 1.0 if a == 2 and self.predator_pop > 8.0 else 0.5
        terminated = self.step_count >= self.max_steps
        return StepResult(observation=self._get_obs(), reward=reward, terminated=terminated, truncated=False)

    def intervene(self, intervention: Dict[str, Any]) -> None:
        if "shock_predators" in intervention:
            self.predator_pop = float(intervention["shock_predators"])


# Level 7: Counterfactual Environment
class CounterfactualWorld(SensorimotorWorld):
    """Level 7: Inverted physics and counterfactual rules to test cognitive flexibility."""

    def __init__(self, invert_controls: bool = True, **kwargs: Any):
        super().__init__(**kwargs)
        self.invert_controls = invert_controls

    def step(self, action: Union[int, np.ndarray]) -> StepResult:
        a = int(action) if isinstance(action, (int, np.integer)) else int(np.argmax(action))
        if self.invert_controls:
            # Counterfactual rule: Up becomes Down, Left becomes Right
            inverted = {0: 1, 1: 0, 2: 3, 3: 2}
            a = inverted.get(a, a)
        return super().step(a)
