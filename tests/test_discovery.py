import numpy as np
import pytest
from sonapse_origin.discovery.sindy import SINDyDiscoveryEngine
from sonapse_origin.discovery.symbolic_regression import SymbolicLawDiscoverer
from sonapse_origin.discovery.law_validator import CandidateLawValidator


def test_sindy_linear_oscillator():
    # True system: dx0/dt = -x1, dx1/dt = x0
    t = np.linspace(0, 10, 200)
    dt = t[1] - t[0]
    x0 = np.cos(t)
    x1 = np.sin(t)
    X = np.stack([x0, x1], axis=1)

    dx0 = -np.sin(t)
    dx1 = np.cos(t)
    dX = np.stack([dx0, dx1], axis=1)

    sindy = SINDyDiscoveryEngine(poly_order=1, include_trig=False, threshold=0.1)
    Xi, equations = sindy.fit(X, dX, var_names=["u", "v"])
    assert len(equations) == 2
    # Check that discovered derivatives match
    pred_dX = sindy.predict_derivative(X, Xi)
    assert np.allclose(pred_dX, dX, atol=0.2)


def test_symbolic_power_law():
    discoverer = SymbolicLawDiscoverer(seed=42)
    # y = 2.5 * x^0.5
    x = np.linspace(1.0, 20.0, 30)
    y = 2.5 * (x ** 0.5)

    res = discoverer.fit_power_law(x, y, "Integration", "Stability")
    assert res["r_squared"] > 0.95
    assert res["a"] == pytest.approx(2.5, rel=0.1)
    assert res["exponent_b"] == pytest.approx(0.5, rel=0.1)

    validator = CandidateLawValidator(r2_promotion_threshold=0.8)
    test_x = np.linspace(25.0, 40.0, 10)
    test_y = 2.5 * (test_x ** 0.5)
    val_res = validator.validate_power_law(res, test_x, test_y)
    assert val_res["promoted"]
    assert val_res["status"] == "CANDIDATE_DISCOVERY"
