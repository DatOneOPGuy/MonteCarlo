"""
Distribution samplers for Monte Carlo simulations.

Each function returns a vectorized sampler that takes a numpy.random.Generator
and a size parameter, returning a numpy array of samples.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator


def normal(mu: float, sigma: float) -> callable[[Generator, int], np.ndarray]:
    """
    Create a normal distribution sampler.

    Parameters
    ----------
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation of the normal distribution (must be > 0).

    Returns
    -------
    sampler : callable
        Function that takes (rng: Generator, size: int) and returns
        np.ndarray of shape (size,).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> sampler = normal(0.0, 1.0)
    >>> samples = sampler(rng, 1000)
    >>> np.abs(np.mean(samples)) < 0.1
    True
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    def sampler(rng: Generator, size: int) -> np.ndarray:
        return rng.normal(mu, sigma, size=size)

    return sampler


def lognormal(mu: float, sigma: float) -> callable[[Generator, int], np.ndarray]:
    """
    Create a lognormal distribution sampler.

    Parameters
    ----------
    mu : float
        Mean of the underlying normal distribution (log-space).
    sigma : float
        Standard deviation of the underlying normal distribution (must be > 0).

    Returns
    -------
    sampler : callable
        Function that takes (rng: Generator, size: int) and returns
        np.ndarray of shape (size,).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> sampler = lognormal(0.0, 1.0)
    >>> samples = sampler(rng, 1000)
    >>> np.all(samples > 0)
    True
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    def sampler(rng: Generator, size: int) -> np.ndarray:
        return rng.lognormal(mu, sigma, size=size)

    return sampler


def uniform(a: float, b: float) -> callable[[Generator, int], np.ndarray]:
    """
    Create a uniform distribution sampler.

    Parameters
    ----------
    a : float
        Lower bound of the uniform distribution.
    b : float
        Upper bound of the uniform distribution (must be > a).

    Returns
    -------
    sampler : callable
        Function that takes (rng: Generator, size: int) and returns
        np.ndarray of shape (size,).

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> sampler = uniform(0.0, 1.0)
    >>> samples = sampler(rng, 1000)
    >>> np.all((samples >= 0) & (samples <= 1))
    True
    """
    if b <= a:
        raise ValueError(f"b must be greater than a, got a={a}, b={b}")

    def sampler(rng: Generator, size: int) -> np.ndarray:
        return rng.uniform(a, b, size=size)

    return sampler


def bernoulli(p: float) -> callable[[Generator, int], np.ndarray]:
    """
    Create a Bernoulli distribution sampler.

    Parameters
    ----------
    p : float
        Probability of success (must be in [0, 1]).

    Returns
    -------
    sampler : callable
        Function that takes (rng: Generator, size: int) and returns
        np.ndarray of shape (size,) with values 0 or 1.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> sampler = bernoulli(0.5)
    >>> samples = sampler(rng, 1000)
    >>> np.all((samples == 0) | (samples == 1))
    True
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")

    def sampler(rng: Generator, size: int) -> np.ndarray:
        return rng.binomial(1, p, size=size)

    return sampler


def binomial(n: int, p: float) -> callable[[Generator, int], np.ndarray]:
    """
    Create a binomial distribution sampler.

    Parameters
    ----------
    n : int
        Number of trials (must be > 0).
    p : float
        Probability of success per trial (must be in [0, 1]).

    Returns
    -------
    sampler : callable
        Function that takes (rng: Generator, size: int) and returns
        np.ndarray of shape (size,) with integer values in [0, n].

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> sampler = binomial(10, 0.5)
    >>> samples = sampler(rng, 1000)
    >>> np.all((samples >= 0) & (samples <= 10))
    True
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0, 1], got {p}")

    def sampler(rng: Generator, size: int) -> np.ndarray:
        return rng.binomial(n, p, size=size)

    return sampler

