"""
Core Monte Carlo simulation engine.

Provides the Simulation class and helper functions for running Monte Carlo
simulations with vectorized operations.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.random import Generator, default_rng


def monte_carlo_se(values: np.ndarray) -> float:
    """
    Compute the standard error of the Monte Carlo mean estimate.

    Parameters
    ----------
    values : np.ndarray
        Array of simulation outputs.

    Returns
    -------
    se : float
        Standard error: std(values) / sqrt(len(values))

    Examples
    --------
    >>> values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> se = monte_carlo_se(values)
    >>> abs(se - np.std(values, ddof=1) / np.sqrt(len(values))) < 1e-10
    True
    """
    n = len(values)
    if n == 0:
        raise ValueError("values array is empty")
    return np.std(values, ddof=1) / np.sqrt(n)


def quantiles(
    values: np.ndarray, qs: tuple[int, ...] = (5, 50, 95)
) -> dict[str, float]:
    """
    Compute quantiles of the simulation outputs.

    Parameters
    ----------
    values : np.ndarray
        Array of simulation outputs.
    qs : tuple of int, optional
        Quantile percentiles to compute (default: (5, 50, 95)).

    Returns
    -------
    quantiles : dict
        Dictionary mapping percentile names (e.g., "p05") to values.

    Examples
    --------
    >>> values = np.arange(100)
    >>> qs_dict = quantiles(values)
    >>> qs_dict["p50"] == 50.0
    True
    """
    if len(values) == 0:
        raise ValueError("values array is empty")
    return {f"p{q:02d}": float(np.percentile(values, q)) for q in qs}


class Simulation:
    """
    Monte Carlo simulation engine.

    This class orchestrates Monte Carlo simulations by sampling from input
    distributions and evaluating a user-defined model function.

    Parameters
    ----------
    model : callable
        Function that takes a 2D array of shape (n, k) where n is the number
        of trials and k is the number of input variables, and returns a 1D
        array of shape (n,) with the model outputs.
    sampler : callable
        Function that takes (rng: Generator, size: int) and returns a 2D
        array of shape (size, k) where k is the number of input variables.
    seed : int or None, optional
        Random seed for reproducibility (default: None).

    Examples
    --------
    >>> def model(x):
    ...     return x[:, 0] + 2 * x[:, 1]
    >>> sampler = lambda rng, size: np.column_stack([
    ...     rng.normal(0, 1, size),
    ...     rng.normal(0, 1, size)
    ... ])
    >>> sim = Simulation(model, sampler, seed=42)
    >>> result = sim.run(1000)
    >>> "mean" in result
    True
    """

    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        sampler: Callable[[Generator, int], np.ndarray],
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.rng = default_rng(seed)
        self._last_results: np.ndarray | None = None

    def run(self, n: int, keep: bool = False) -> dict:
        """
        Run a Monte Carlo simulation.

        Parameters
        ----------
        n : int
            Number of simulation trials (must be > 0).
        keep : bool, optional
            If True, store all simulation outputs for later access
            (default: False).

        Returns
        -------
        result : dict
            Dictionary containing:
            - "mean": float, sample mean
            - "std": float, sample standard deviation
            - "p05": float, 5th percentile
            - "p50": float, 50th percentile (median)
            - "p95": float, 95th percentile
            - "se": float, standard error of the mean
            - "n": int, number of trials
            - "all": np.ndarray (only if keep=True), all outputs
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        # Sample inputs
        X = self.sampler(self.rng, n)

        # Validate sampler output shape
        if X.ndim != 2:
            raise ValueError(
                f"sampler must return 2D array, got {X.ndim}D array"
            )
        if X.shape[0] != n:
            raise ValueError(
                f"sampler returned {X.shape[0]} rows, expected {n}"
            )

        # Evaluate model
        y = self.model(X)

        # Validate model output shape
        if y.ndim != 1:
            raise ValueError(
                f"model must return 1D array, got {y.ndim}D array"
            )
        if len(y) != n:
            raise ValueError(
                f"model returned {len(y)} values, expected {n}"
            )

        # Compute statistics
        result = {
            "mean": float(np.mean(y)),
            "std": float(np.std(y, ddof=1)),
            "se": monte_carlo_se(y),
            "n": n,
        }
        result.update(quantiles(y))

        if keep:
            result["all"] = y
            self._last_results = y

        return result

    def risk(self, predicate: Callable[[np.ndarray], np.ndarray | bool]) -> dict:
        """
        Compute the probability that a predicate is true.

        Uses binomial standard error for the probability estimate.

        Parameters
        ----------
        predicate : callable
            Function that takes a 1D array of outputs and returns a boolean
            array or a single boolean indicating which outcomes satisfy the
            condition.

        Returns
        -------
        result : dict
            Dictionary containing:
            - "p": float, estimated probability
            - "se": float, binomial standard error sqrt(p * (1-p) / n)
            - "n": int, number of trials used

        Examples
        --------
        >>> def model(x):
        ...     return x[:, 0]
        >>> sampler = lambda rng, size: rng.normal(0, 1, (size, 1))
        >>> sim = Simulation(model, sampler, seed=42)
        >>> sim.run(1000)
        >>> risk_result = sim.risk(lambda y: y > 1.0)
        >>> 0 <= risk_result["p"] <= 1
        True
        """
        if self._last_results is None:
            raise ValueError(
                "must call run(keep=True) before calling risk()"
            )

        y = self._last_results
        mask = predicate(y)

        # Convert to boolean array if needed
        if isinstance(mask, bool):
            mask = np.full(len(y), mask, dtype=bool)
        else:
            mask = np.asarray(mask, dtype=bool)

        if len(mask) != len(y):
            raise ValueError(
                f"predicate returned {len(mask)} values, expected {len(y)}"
            )

        p = float(np.mean(mask))
        n = len(y)
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0

        return {"p": p, "se": se, "n": n}

    def confint_mean(self, level: float = 0.95) -> tuple[float, float]:
        """
        Compute a confidence interval for the mean.

        Parameters
        ----------
        level : float, optional
            Confidence level (default: 0.95).

        Returns
        -------
        ci : tuple[float, float]
            (lower, upper) confidence interval bounds.

        Examples
        --------
        >>> def model(x):
        ...     return x[:, 0]
        >>> sampler = lambda rng, size: rng.normal(0, 1, (size, 1))
        >>> sim = Simulation(model, sampler, seed=42)
        >>> result = sim.run(1000)
        >>> ci = sim.confint_mean(0.95)
        >>> ci[0] < ci[1]
        True
        """
        if self._last_results is None:
            raise ValueError(
                "must call run(keep=True) before calling confint_mean()"
            )

        y = self._last_results
        mean = np.mean(y)
        se = monte_carlo_se(y)

        # Use normal approximation (CLT)
        from scipy import stats

        z = stats.norm.ppf((1 + level) / 2)
        margin = z * se

        return (float(mean - margin), float(mean + margin))

