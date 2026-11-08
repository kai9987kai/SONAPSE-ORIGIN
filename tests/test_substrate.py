import numpy as np
import pytest
from sonapse_origin.substrate.spiking import LIFPopulation
from sonapse_origin.substrate.rate_based import ContinuousRateRNN
from sonapse_origin.substrate.oscillators import KuramotoOscillators
from sonapse_origin.substrate.reaction_diffusion import ContinuousNeuralField
from sonapse_origin.substrate.wave_comm import SignalChannelComparator, SignalSubstrateMode
from sonapse_origin.substrate.simulator import UnifiedDynamicalSimulator


def test_lif_population():
    lif = LIFPopulation(num_neurons=5, dt=0.5)
    W = np.ones((5, 5)) * 20.0
    ext = np.ones(5) * 50.0  # Strong current to trigger spikes
    spikes, v = lif.step(W, external_current=ext)
    assert len(spikes) == 5
    assert len(v) == 5
    assert np.all(np.isfinite(v))


def test_rate_based_rnn():
    rnn = ContinuousRateRNN(num_units=6, dt=1.0)
    W = np.random.normal(0, 0.1, (6, 6))
    out = rnn.step(W, external_input=np.ones(6) * 0.5)
    assert out.shape == (6,)
    assert np.all(np.isfinite(out))


def test_kuramoto_oscillators():
    osc = KuramotoOscillators(num_oscillators=8, dt=0.05, seed=42)
    K = np.ones((8, 8)) * 0.5
    phases, r = osc.step(coupling_matrix=K)
    assert phases.shape == (8,)
    assert 0.0 <= r <= 1.0


def test_continuous_neural_field():
    field = ContinuousNeuralField(grid_size=16, dt=0.5)
    u = field.step()
    assert u.shape == (16, 16)
    assert np.all(np.isfinite(u))


def test_wave_comm_comparison():
    comparator = SignalChannelComparator(rng_seed=123)
    raw = np.array([0.2, -0.5, 0.8, -0.1, 0.4])
    for mode in SignalSubstrateMode:
        res = comparator.transmit(mode, raw, noise_level=0.05)
        assert res["fidelity"] > 0.0
        assert res["energy_cost"] >= 0.0
        assert res["latency"] > 0.0


def test_unified_simulator():
    for sub in ["rate", "lif", "kuramoto", "field"]:
        sim = UnifiedDynamicalSimulator(substrate_type=sub, num_units=9, dt=0.5)
        W = np.eye(9) * 0.2
        step_res = sim.step(W, external_input=np.ones(9) * 0.1)
        assert step_res["step"] == 1
        assert "state" in step_res
