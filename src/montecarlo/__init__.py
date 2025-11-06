"""
Monte Carlo simulation toolkit.

A minimal, fast framework for propagating uncertainty from input distributions
through user-defined models using vectorized NumPy operations.
"""

from __future__ import annotations

from montecarlo.core import Simulation, monte_carlo_se, quantiles
from montecarlo.distributions import (
    bernoulli,
    binomial,
    lognormal,
    normal,
    uniform,
)

__version__ = "0.1.0"
__all__ = [
    "Simulation",
    "monte_carlo_se",
    "quantiles",
    "normal",
    "lognormal",
    "uniform",
    "bernoulli",
    "binomial",
]

