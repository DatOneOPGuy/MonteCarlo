"""
Distribution sampling functions.

All functions are pure and vectorized, using numpy.random.Generator
for reproducibility.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy import stats

from app.core.types import DistConfig


def sample_normal(rng: Generator, size: int, mu: float, sigma: float) -> np.ndarray:
    """Sample from normal distribution."""
    return rng.normal(mu, sigma, size=size)


def sample_lognormal(rng: Generator, size: int, mu: float, sigma: float) -> np.ndarray:
    """Sample from lognormal distribution."""
    return rng.lognormal(mu, sigma, size=size)


def sample_student_t(
    rng: Generator, size: int, df: float, mu: float = 0.0, sigma: float = 1.0
) -> np.ndarray:
    """Sample from Student-t distribution."""
    # Use scipy for Student-t with location and scale
    t_samples = stats.t.rvs(df=df, size=size, random_state=rng)
    return mu + sigma * t_samples


def sample_triangular(
    rng: Generator, size: int, a: float, b: float, c: float
) -> np.ndarray:
    """Sample from triangular distribution."""
    return stats.triang.rvs(
        c=(c - a) / (b - a), loc=a, scale=b - a, size=size, random_state=rng
    )


def sample_uniform(rng: Generator, size: int, low: float, high: float) -> np.ndarray:
    """Sample from uniform distribution."""
    return rng.uniform(low, high, size=size)


def sample_beta(
    rng: Generator,
    size: int,
    alpha: float,
    beta: float,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    """Sample from beta distribution, scaled to [low, high]."""
    beta_samples = rng.beta(alpha, beta, size=size)
    return low + (high - low) * beta_samples


def sample_custom(rng: Generator, size: int, values: np.ndarray) -> np.ndarray:
    """Sample from custom distribution (bootstrap from provided values)."""
    if len(values) == 0:
        raise ValueError("Custom distribution values cannot be empty")
    indices = rng.integers(0, len(values), size=size)
    return values[indices]


def clamp_values(
    values: np.ndarray, min_bound: float | None, max_bound: float | None
) -> np.ndarray:
    """Clamp values to bounds if provided."""
    if min_bound is not None:
        values = np.maximum(values, min_bound)
    if max_bound is not None:
        values = np.minimum(values, max_bound)
    return values


def sample(
    dist_config: DistConfig, rng: Generator, size: int
) -> np.ndarray:
    """
    Unified distribution sampler dispatcher.

    Parameters
    ----------
    dist_config : DistConfig
        Distribution configuration dictionary
    rng : Generator
        NumPy random number generator
    size : int
        Number of samples to generate

    Returns
    -------
    np.ndarray
        Array of samples

    Raises
    ------
    ValueError
        If distribution type is unknown or required parameters are missing
    """
    dist_type = dist_config.get("type")
    if dist_type is None:
        raise ValueError("Distribution type must be specified")

    # Sample based on type
    if dist_type == "normal":
        samples = sample_normal(
            rng, size, dist_config["mu"], dist_config["sigma"]
        )
    elif dist_type == "lognormal":
        samples = sample_lognormal(
            rng, size, dist_config["mu"], dist_config["sigma"]
        )
    elif dist_type == "student_t":
        mu = dist_config.get("mu", 0.0)
        sigma = dist_config.get("sigma", 1.0)
        samples = sample_student_t(rng, size, dist_config["df"], mu, sigma)
    elif dist_type == "triangular":
        samples = sample_triangular(
            rng, size, dist_config["a"], dist_config["b"], dist_config["c"]
        )
    elif dist_type == "uniform":
        samples = sample_uniform(rng, size, dist_config["low"], dist_config["high"])
    elif dist_type == "beta":
        alpha = dist_config["alpha"]
        beta = dist_config["beta"]
        low = dist_config.get("low", 0.0)
        high = dist_config.get("high", 1.0)
        samples = sample_beta(rng, size, alpha, beta, low, high)
    elif dist_type == "custom":
        if "values" not in dist_config:
            raise ValueError("Custom distribution requires 'values' array")
        samples = sample_custom(rng, size, dist_config["values"])
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    # Apply bounds if specified
    min_bound = dist_config.get("min_bound")
    max_bound = dist_config.get("max_bound")
    if min_bound is not None or max_bound is not None:
        samples = clamp_values(samples, min_bound, max_bound)

    return samples

