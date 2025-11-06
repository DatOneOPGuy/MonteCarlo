"""Tests for distribution sampling functions."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator, PCG64

from app.core.distributions import (
    sample,
    sample_beta,
    sample_lognormal,
    sample_normal,
    sample_student_t,
    sample_triangular,
    sample_uniform,
)


def test_sample_normal() -> None:
    """Test normal distribution sampling."""
    rng = Generator(PCG64(42))
    samples = sample_normal(rng, 10000, mu=0.0, sigma=1.0)

    assert len(samples) == 10000
    assert np.abs(np.mean(samples)) < 0.1
    assert np.abs(np.std(samples) - 1.0) < 0.1


def test_sample_lognormal() -> None:
    """Test lognormal distribution sampling."""
    rng = Generator(PCG64(42))
    samples = sample_lognormal(rng, 10000, mu=0.0, sigma=1.0)

    assert len(samples) == 10000
    assert np.all(samples > 0)


def test_sample_student_t() -> None:
    """Test Student-t distribution sampling."""
    rng = Generator(PCG64(42))
    samples = sample_student_t(rng, 10000, df=5.0, mu=0.0, sigma=1.0)

    assert len(samples) == 10000
    # Student-t should have fatter tails than normal
    assert np.abs(np.mean(samples)) < 0.2


def test_sample_triangular() -> None:
    """Test triangular distribution sampling."""
    rng = Generator(PCG64(42))
    samples = sample_triangular(rng, 10000, a=0.0, b=1.0, c=0.5)

    assert len(samples) == 10000
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_sample_uniform() -> None:
    """Test uniform distribution sampling."""
    rng = Generator(PCG64(42))
    samples = sample_uniform(rng, 10000, low=0.0, high=1.0)

    assert len(samples) == 10000
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_sample_beta() -> None:
    """Test beta distribution sampling."""
    rng = Generator(PCG64(42))
    samples = sample_beta(rng, 10000, alpha=2.0, beta=2.0, low=0.0, high=1.0)

    assert len(samples) == 10000
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_sample_dispatcher() -> None:
    """Test unified sample dispatcher."""
    rng = Generator(PCG64(42))

    # Normal
    dist_config = {"type": "normal", "mu": 0.0, "sigma": 1.0}
    samples = sample(dist_config, rng, 1000)
    assert len(samples) == 1000

    # Lognormal
    dist_config = {"type": "lognormal", "mu": 0.0, "sigma": 1.0}
    samples = sample(dist_config, rng, 1000)
    assert len(samples) == 1000
    assert np.all(samples > 0)

    # With bounds
    dist_config = {
        "type": "normal",
        "mu": 0.0,
        "sigma": 1.0,
        "min_bound": -1.0,
        "max_bound": 1.0,
    }
    samples = sample(dist_config, rng, 1000)
    assert np.all(samples >= -1.0)
    assert np.all(samples <= 1.0)


def test_deterministic_seed() -> None:
    """Test that fixed seed produces deterministic results."""
    rng1 = Generator(PCG64(42))
    rng2 = Generator(PCG64(42))

    samples1 = sample_normal(rng1, 1000, mu=0.0, sigma=1.0)
    samples2 = sample_normal(rng2, 1000, mu=0.0, sigma=1.0)

    np.testing.assert_array_equal(samples1, samples2)

