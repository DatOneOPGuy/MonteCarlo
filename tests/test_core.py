"""
Tests for core Monte Carlo simulation functionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from montecarlo.core import Simulation, monte_carlo_se, quantiles


def test_monte_carlo_se() -> None:
    """Test standard error computation."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    se = monte_carlo_se(values)
    expected = np.std(values, ddof=1) / np.sqrt(len(values))
    assert abs(se - expected) < 1e-10

    # Empty array should raise
    with pytest.raises(ValueError):
        monte_carlo_se(np.array([]))


def test_quantiles() -> None:
    """Test quantile computation."""
    values = np.arange(100)
    qs = quantiles(values)
    assert qs["p05"] == 5.0
    assert qs["p50"] == 50.0
    assert qs["p95"] == 95.0

    # Empty array should raise
    with pytest.raises(ValueError):
        quantiles(np.array([]))


def test_simulation_fixed_seed() -> None:
    """Test that fixed seed produces deterministic results."""
    def model(x: np.ndarray) -> np.ndarray:
        return x[:, 0]

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(0, 1, size=(size, 1))

    sim1 = Simulation(model, sampler, seed=42)
    sim2 = Simulation(model, sampler, seed=42)

    result1 = sim1.run(1000)
    result2 = sim2.run(1000)

    assert abs(result1["mean"] - result2["mean"]) < 1e-10
    assert abs(result1["std"] - result2["std"]) < 1e-10


def test_simulation_shape_validation() -> None:
    """Test that shape validation works correctly."""
    def model(x: np.ndarray) -> np.ndarray:
        return x[:, 0]

    # Wrong sampler output shape
    def bad_sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(0, 1, size=size)  # 1D instead of 2D

    sim = Simulation(model, bad_sampler, seed=42)
    with pytest.raises(ValueError, match="2D array"):
        sim.run(100)

    # Wrong model output shape
    def bad_model(x: np.ndarray) -> np.ndarray:
        return x  # 2D instead of 1D

    def good_sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(0, 1, size=(size, 1))

    sim = Simulation(bad_model, good_sampler, seed=42)
    with pytest.raises(ValueError, match="1D array"):
        sim.run(100)


def test_simulation_pi_estimate() -> None:
    """Test π estimation with reasonable tolerance."""
    def model(x: np.ndarray) -> np.ndarray:
        inside = (x[:, 0]**2 + x[:, 1]**2) <= 1.0
        return inside.astype(float)

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(0, 1, size=(size, 2))

    sim = Simulation(model, sampler, seed=42)
    result = sim.run(100_000, keep=True)

    pi_estimate = 4 * result["mean"]
    se = 4 * result["se"]

    # Should be within 3 standard errors of true π
    assert abs(pi_estimate - np.pi) < 3 * se


def test_risk_calculation() -> None:
    """Test risk probability calculation."""
    def model(x: np.ndarray) -> np.ndarray:
        return x[:, 0]

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(0, 1, size=(size, 1))

    sim = Simulation(model, sampler, seed=42)
    sim.run(1000, keep=True)

    # Risk without keep=True should raise
    sim2 = Simulation(model, sampler, seed=42)
    sim2.run(1000, keep=False)
    with pytest.raises(ValueError):
        sim2.risk(lambda y: y > 0)

    # Risk with keep=True should work
    risk_result = sim.risk(lambda y: y > 1.0)
    assert 0 <= risk_result["p"] <= 1
    assert risk_result["se"] >= 0
    assert risk_result["n"] == 1000


def test_confint_mean() -> None:
    """Test confidence interval computation."""
    def model(x: np.ndarray) -> np.ndarray:
        return x[:, 0]

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(0, 1, size=(size, 1))

    sim = Simulation(model, sampler, seed=42)
    sim.run(1000, keep=True)

    ci = sim.confint_mean(0.95)
    assert len(ci) == 2
    assert ci[0] < ci[1]

    # Without keep=True should raise
    sim2 = Simulation(model, sampler, seed=42)
    sim2.run(1000, keep=False)
    with pytest.raises(ValueError):
        sim2.confint_mean()


def test_invalid_n() -> None:
    """Test that invalid n raises error."""
    def model(x: np.ndarray) -> np.ndarray:
        return x[:, 0]

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.normal(0, 1, size=(size, 1))

    sim = Simulation(model, sampler, seed=42)
    with pytest.raises(ValueError):
        sim.run(0)
    with pytest.raises(ValueError):
        sim.run(-1)

