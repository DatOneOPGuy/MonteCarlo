"""
Tests for distribution samplers.
"""

from __future__ import annotations

import numpy as np
import pytest

from montecarlo.distributions import (
    bernoulli,
    binomial,
    lognormal,
    normal,
    uniform,
)


def test_normal_distribution() -> None:
    """Test normal distribution sampler."""
    rng = np.random.default_rng(42)
    sampler = normal(0.0, 1.0)
    samples = sampler(rng, 10000)

    assert len(samples) == 10000
    assert samples.ndim == 1
    # Mean should be close to 0
    assert abs(np.mean(samples)) < 0.1
    # Std should be close to 1
    assert abs(np.std(samples, ddof=1) - 1.0) < 0.1

    # Invalid sigma should raise
    with pytest.raises(ValueError):
        normal(0.0, -1.0)
    with pytest.raises(ValueError):
        normal(0.0, 0.0)


def test_lognormal_distribution() -> None:
    """Test lognormal distribution sampler."""
    rng = np.random.default_rng(42)
    sampler = lognormal(0.0, 1.0)
    samples = sampler(rng, 10000)

    assert len(samples) == 10000
    assert samples.ndim == 1
    # All values should be positive
    assert np.all(samples > 0)

    # Invalid sigma should raise
    with pytest.raises(ValueError):
        lognormal(0.0, -1.0)


def test_uniform_distribution() -> None:
    """Test uniform distribution sampler."""
    rng = np.random.default_rng(42)
    sampler = uniform(0.0, 1.0)
    samples = sampler(rng, 10000)

    assert len(samples) == 10000
    assert samples.ndim == 1
    # All values should be in [0, 1]
    assert np.all(samples >= 0)
    assert np.all(samples <= 1)

    # Invalid bounds should raise
    with pytest.raises(ValueError):
        uniform(1.0, 0.0)
    with pytest.raises(ValueError):
        uniform(0.0, 0.0)


def test_bernoulli_distribution() -> None:
    """Test Bernoulli distribution sampler."""
    rng = np.random.default_rng(42)
    sampler = bernoulli(0.5)
    samples = sampler(rng, 10000)

    assert len(samples) == 10000
    assert samples.ndim == 1
    # All values should be 0 or 1
    assert np.all((samples == 0) | (samples == 1))
    # Mean should be close to 0.5
    assert abs(np.mean(samples) - 0.5) < 0.05

    # Invalid p should raise
    with pytest.raises(ValueError):
        bernoulli(-0.1)
    with pytest.raises(ValueError):
        bernoulli(1.1)


def test_binomial_distribution() -> None:
    """Test binomial distribution sampler."""
    rng = np.random.default_rng(42)
    sampler = binomial(10, 0.5)
    samples = sampler(rng, 10000)

    assert len(samples) == 10000
    assert samples.ndim == 1
    # All values should be in [0, 10]
    assert np.all(samples >= 0)
    assert np.all(samples <= 10)
    # Mean should be close to n*p = 5
    assert abs(np.mean(samples) - 5.0) < 0.2

    # Invalid parameters should raise
    with pytest.raises(ValueError):
        binomial(0, 0.5)
    with pytest.raises(ValueError):
        binomial(-1, 0.5)
    with pytest.raises(ValueError):
        binomial(10, -0.1)
    with pytest.raises(ValueError):
        binomial(10, 1.1)


def test_large_n_sanity() -> None:
    """Test that large-N simulations produce reasonable statistics."""
    rng = np.random.default_rng(42)

    # Normal: N(0, 1)
    sampler = normal(0.0, 1.0)
    samples = sampler(rng, 100_000)
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)
    # Should be within 3 standard errors
    se_mean = std / np.sqrt(len(samples))
    assert abs(mean - 0.0) < 3 * se_mean
    assert abs(std - 1.0) < 0.01

    # Uniform: U(0, 1)
    sampler = uniform(0.0, 1.0)
    samples = sampler(rng, 100_000)
    mean = np.mean(samples)
    # Theoretical mean is 0.5
    assert abs(mean - 0.5) < 0.01

