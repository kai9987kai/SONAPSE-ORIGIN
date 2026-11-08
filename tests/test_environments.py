import numpy as np
import pytest
from sonapse_origin.environments import (
    MinimalRegulatoryWorld,
    SensorimotorWorld,
    UncertainWorld,
    DevelopmentalWorld,
    SocialMultiAgentWorld,
    OpenEndedEcologyWorld,
    CounterfactualWorld,
)


def test_level1_minimal_regulatory():
    env = MinimalRegulatoryWorld(max_steps=10)
    obs, info = env.reset()
    assert obs.shape == (4,)
    res = env.step(1)
    assert not res.terminated
    env.intervene({"set_ambient_temp": 0.9})
    assert env.ambient_temp == 0.9


def test_level2_sensorimotor():
    env = SensorimotorWorld(max_steps=10)
    obs, info = env.reset()
    assert obs.shape == (4,)
    res = env.step(0)
    assert isinstance(res.reward, float)


def test_level3_uncertain():
    env = UncertainWorld(max_steps=10, sensor_noise=0.1)
    obs, info = env.reset()
    assert obs.shape == (4,)
    res = env.step(2)
    assert not res.terminated


def test_level4_developmental():
    env = DevelopmentalWorld(max_steps=50)
    env.reset()
    res = env.step(0)
    assert "stage" in res.info


def test_level5_social():
    env = SocialMultiAgentWorld(max_steps=10)
    obs, _ = env.reset()
    assert obs.shape == (4,)
    res = env.step(1)
    assert not res.terminated


def test_level6_ecology():
    env = OpenEndedEcologyWorld(max_steps=10)
    obs, _ = env.reset()
    assert obs.shape == (3,)
    res = env.step(0)
    assert res.reward > 0.0


def test_level7_counterfactual():
    env = CounterfactualWorld(invert_controls=True, max_steps=10)
    obs, _ = env.reset()
    assert obs.shape == (4,)
    res = env.step(0)
    assert not res.terminated
