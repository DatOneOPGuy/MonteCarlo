"""Tests for simulation engine."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.random import Generator, PCG64

from app.core.simulate import apply_correlation, create_rng, simulate_growth_paths
from app.core.types import CorrelationConfig, DistConfig, SimConfig, YearConfig


def test_create_rng() -> None:
    """Test RNG creation."""
    rng1 = create_rng(42)
    rng2 = create_rng(42)
    rng3 = create_rng(None)

    # Same seed should produce same generator state
    assert rng1.bit_generator.state == rng2.bit_generator.state
    # Different seeds should produce different states
    assert rng1.bit_generator.state != rng3.bit_generator.state


def test_apply_correlation() -> None:
    """Test correlation application."""
    n_sims = 1000
    n_years = 2

    # Independent samples
    samples = np.random.randn(n_sims, n_years)

    # Identity correlation (no change)
    identity = np.eye(n_years)
    correlated = apply_correlation(samples, identity)

    # Should be approximately the same (within numerical precision)
    np.testing.assert_allclose(samples, correlated, rtol=1e-1)

    # Positive correlation
    corr_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
    correlated = apply_correlation(samples, corr_matrix)

    # Check shape preserved
    assert correlated.shape == (n_sims, n_years)


def test_simulate_growth_paths() -> None:
    """Test basic simulation."""
    dist_config: DistConfig = {"type": "normal", "mu": 0.05, "sigma": 0.02}
    year_config = YearConfig(year=1, dist=dist_config)

    config = SimConfig(
        n_sims=1000,
        seed=42,
        years=[year_config],
        correlation=CorrelationConfig(matrix=np.eye(1), enabled=False),
    )

    results = simulate_growth_paths(config)

    assert results.paths.shape == (1000, 1)
    assert results.config == config
    assert results.seed_used == 42


def test_multi_year_simulation() -> None:
    """Test multi-year simulation."""
    dist1: DistConfig = {"type": "normal", "mu": 0.05, "sigma": 0.02}
    dist2: DistConfig = {"type": "normal", "mu": 0.06, "sigma": 0.03}

    year1 = YearConfig(year=1, dist=dist1)
    year2 = YearConfig(year=2, dist=dist2)

    config = SimConfig(
        n_sims=1000,
        seed=42,
        years=[year1, year2],
        correlation=CorrelationConfig(matrix=np.eye(2), enabled=False),
    )

    results = simulate_growth_paths(config)

    assert results.paths.shape == (1000, 2)


def test_deterministic_simulation() -> None:
    """Test simulation with deterministic values."""
    dist_config: DistConfig = {"type": "normal", "mu": 0.05, "sigma": 0.02}
    year_config = YearConfig(year=1, dist=dist_config, deterministic_value=0.10)

    config = SimConfig(
        n_sims=1000,
        seed=42,
        years=[year_config],
        correlation=CorrelationConfig(matrix=np.eye(1), enabled=False),
    )

    results = simulate_growth_paths(config)

    # All values should be the deterministic value
    assert np.allclose(results.paths, 0.10)


def test_reproducibility() -> None:
    """Test that same seed produces same results."""
    dist_config: DistConfig = {"type": "normal", "mu": 0.05, "sigma": 0.02}
    year_config = YearConfig(year=1, dist=dist_config)

    config = SimConfig(
        n_sims=1000,
        seed=42,
        years=[year_config],
        correlation=CorrelationConfig(matrix=np.eye(1), enabled=False),
    )

    results1 = simulate_growth_paths(config)
    results2 = simulate_growth_paths(config)

    np.testing.assert_array_equal(results1.paths, results2.paths)

