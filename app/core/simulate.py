"""
Core simulation engine for multi-year growth paths with correlation.

Uses vectorized NumPy operations and centralized RNG for reproducibility.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, PCG64
from scipy.stats import qmc

from app.core.distributions import sample
from app.core.types import CorrelationConfig, SimConfig, SimResults, YearConfig


def create_rng(seed: int | None) -> Generator:
    """
    Create a numpy.random.Generator with PCG64 algorithm.

    Parameters
    ----------
    seed : int or None
        Random seed (None for non-deterministic)

    Returns
    -------
    Generator
        NumPy random number generator
    """
    if seed is not None:
        return Generator(PCG64(seed))
    return Generator(PCG64())


def apply_correlation(
    samples: np.ndarray, correlation_matrix: np.ndarray
) -> np.ndarray:
    """
    Apply correlation structure using Cholesky decomposition.

    Parameters
    ----------
    samples : np.ndarray
        Independent samples, shape (n_sims, n_years)
    correlation_matrix : np.ndarray
        Correlation matrix, shape (n_years, n_years)

    Returns
    -------
    np.ndarray
        Correlated samples, same shape as input
    """
    # Standardize samples to have mean 0, std 1
    standardized = (samples - samples.mean(axis=0)) / (samples.std(axis=0) + 1e-10)

    # Cholesky decomposition
    L = np.linalg.cholesky(correlation_matrix)

    # Apply transformation
    correlated = standardized @ L.T

    # Rescale to original distribution
    correlated = (
        correlated * samples.std(axis=0, keepdims=True)
        + samples.mean(axis=0, keepdims=True)
    )

    return correlated


def simulate_growth_paths(
    config: SimConfig, rng: Generator | None = None
) -> SimResults:
    """
    Simulate multi-year growth paths with optional correlation.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration
    rng : Generator or None
        Random number generator (if None, creates one from seed)

    Returns
    -------
    SimResults
        Simulation results container

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    if rng is None:
        rng = create_rng(config.seed)

    n_sims = config.n_sims
    n_years = len(config.years)

    # Initialize paths array
    paths = np.zeros((n_sims, n_years))

    # Sample each year
    for i, year_config in enumerate(config.years):
        if year_config.deterministic_value is not None:
            # Use deterministic value
            paths[:, i] = year_config.deterministic_value
        else:
            # Sample from distribution
            paths[:, i] = sample(year_config.dist, rng, n_sims)

    # Apply correlation if enabled
    if config.correlation.enabled and n_years > 1:
        paths = apply_correlation(paths, config.correlation.matrix)

    # Create results
    from datetime import datetime

    timestamp = datetime.now().isoformat()
    parameter_hash = "placeholder"  # Will be computed from config

    # Extract seed used
    if config.seed is not None:
        seed_used = config.seed
    else:
        # Try to extract from generator state, fallback to 0
        try:
            seed_used = rng.bit_generator.state.get("state", {}).get("state", 0)
        except (AttributeError, KeyError, TypeError):
            seed_used = 0

    return SimResults(
        paths=paths,
        config=config,
        seed_used=seed_used,
        timestamp=timestamp,
        parameter_hash=parameter_hash,
    )


def simulate_with_lhs(
    config: SimConfig, rng: Generator | None = None
) -> SimResults:
    """
    Simulate using Latin Hypercube Sampling (LHS) for better coverage.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration
    rng : Generator or None
        Random number generator

    Returns
    -------
    SimResults
        Simulation results
    """
    if rng is None:
        rng = create_rng(config.seed)

    n_sims = config.n_sims
    n_years = len(config.years)

    # Generate LHS samples in [0, 1]^n_years
    sampler = qmc.LatinHypercube(d=n_years, seed=config.seed)
    lhs_samples = sampler.random(n=n_sims)

    # Convert to uniform [0, 1] then apply inverse CDF for each distribution
    paths = np.zeros((n_sims, n_years))

    for i, year_config in enumerate(config.years):
        if year_config.deterministic_value is not None:
            paths[:, i] = year_config.deterministic_value
        else:
            # Use LHS sample for this year
            u = lhs_samples[:, i]
            dist_config = year_config.dist
            dist_type = dist_config.get("type")

            # Apply inverse CDF (PPF) based on distribution type
            if dist_type == "normal":
                from scipy.stats import norm

                paths[:, i] = norm.ppf(
                    u, loc=dist_config["mu"], scale=dist_config["sigma"]
                )
            elif dist_type == "lognormal":
                from scipy.stats import lognorm

                paths[:, i] = lognorm.ppf(
                    u, s=dist_config["sigma"], scale=np.exp(dist_config["mu"])
                )
            elif dist_type == "uniform":
                paths[:, i] = dist_config["low"] + u * (
                    dist_config["high"] - dist_config["low"]
                )
            else:
                # Fall back to regular sampling for distributions without easy PPF
                paths[:, i] = sample(dist_config, rng, n_sims)

    # Apply correlation if enabled
    if config.correlation.enabled and n_years > 1:
        paths = apply_correlation(paths, config.correlation.matrix)

    from datetime import datetime

    timestamp = datetime.now().isoformat()
    return SimResults(
        paths=paths,
        config=config,
        seed_used=config.seed if config.seed is not None else 0,
        timestamp=timestamp,
        parameter_hash="lhs",
    )

