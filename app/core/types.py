"""
Type definitions for Monte Carlo simulation.

Provides TypedDicts, dataclasses, and protocols for type-safe
configuration and results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np


# Distribution types
DistributionType = Literal[
    "normal",
    "lognormal",
    "student_t",
    "triangular",
    "uniform",
    "beta",
    "custom",
]

ScenarioType = Literal["base", "bull", "bear"]


class DistConfig(TypedDict, total=False):
    """Distribution configuration dictionary."""

    type: DistributionType
    # Normal/Lognormal
    mu: float
    sigma: float
    # Student-t
    df: float  # degrees of freedom
    # Triangular
    a: float  # lower bound
    b: float  # upper bound
    c: float  # mode
    # Uniform
    low: float
    high: float
    # Beta
    alpha: float
    beta: float
    # Custom (CSV)
    values: np.ndarray
    # Common
    min_bound: float | None  # optional clamping
    max_bound: float | None


@dataclass
class YearConfig:
    """Configuration for a single year in multi-year simulation."""

    year: int
    dist: DistConfig
    deterministic_value: float | None = None  # if set, use this instead of sampling


@dataclass
class CorrelationConfig:
    """Correlation matrix configuration."""

    matrix: np.ndarray  # n_years x n_years
    enabled: bool = False


@dataclass
class SimConfig:
    """Complete simulation configuration."""

    n_sims: int
    seed: int | None
    years: list[YearConfig]
    correlation: CorrelationConfig
    scenario_name: str = "base"
    scenario_weight: float = 1.0


@dataclass
class SimResults:
    """Simulation results container."""

    paths: np.ndarray  # shape: (n_sims, n_years)
    config: SimConfig
    seed_used: int
    timestamp: str
    parameter_hash: str


@dataclass
class AnalyticsResults:
    """Analytics computed from simulation results."""

    summary_stats: dict[str, float]
    quantiles: dict[str, float]
    var: dict[float, float]  # alpha -> VaR value
    cvar: dict[float, float]  # alpha -> CVaR value
    prob_loss: float | None
    prob_target: float | None
    target_value: float | None
    sensitivity: dict[str, float] | None  # tornado chart data
    prcc: dict[str, float] | None  # partial rank correlation coefficients


@dataclass
class DCFConfig:
    """DCF-specific configuration."""

    initial_cash_flow: float
    wacc: float  # weighted average cost of capital
    terminal_growth_rate: float | DistConfig | None
    terminal_multiple: float | None
    years_to_project: int = 2
    discount_years: list[int] | None = None  # which years to discount

