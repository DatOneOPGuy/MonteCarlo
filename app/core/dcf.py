"""
Discounted Cash Flow (DCF) model implementation.

Provides DCF valuation with Monte Carlo simulation for growth rate uncertainty.
"""

from __future__ import annotations

import numpy as np

from app.core.analytics import compute_analytics
from app.core.simulate import SimResults
from app.core.types import DistConfig, DCFConfig, YearConfig


def calculate_dcf_value(
    initial_fcf: float,
    growth_rates: np.ndarray,
    wacc: float,
    terminal_growth: float | None = None,
    terminal_multiple: float | None = None,
    years: int = 2,
    fcf_in_billions: bool = True,
) -> np.ndarray:
    """
    Calculate DCF value for each simulation path.

    Parameters
    ----------
    initial_fcf : float
        Initial free cash flow (Year 0)
    growth_rates : np.ndarray
        Growth rates for each year, shape (n_sims, n_years)
    wacc : float
        Weighted average cost of capital (discount rate)
    terminal_growth : float or None
        Perpetual growth rate for terminal value (Gordon Growth Model)
    terminal_multiple : float or None
        Terminal multiple (EV/EBITDA or similar) - alternative to terminal growth
    years : int
        Number of projection years
    fcf_in_billions : bool
        If True, assumes FCF is in billions (for display purposes)

    Returns
    -------
    np.ndarray
        DCF values for each simulation, shape (n_sims,)
    """
    n_sims = growth_rates.shape[0]

    # Project cash flows
    fcf_paths = np.zeros((n_sims, years))
    fcf_paths[:, 0] = initial_fcf * (1 + growth_rates[:, 0])

    for year in range(1, years):
        fcf_paths[:, year] = fcf_paths[:, year - 1] * (1 + growth_rates[:, year])

    # Discount projected cash flows to present value
    pv_projected = np.zeros((n_sims, years))
    for year in range(years):
        pv_projected[:, year] = fcf_paths[:, year] / ((1 + wacc) ** (year + 1))

    # Calculate terminal value
    if terminal_growth is not None:
        # Gordon Growth Model: TV = FCF_n * (1 + g) / (WACC - g)
        terminal_fcf = fcf_paths[:, -1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
    elif terminal_multiple is not None:
        # Terminal multiple approach
        terminal_value = fcf_paths[:, -1] * terminal_multiple
    else:
        # No terminal value
        terminal_value = np.zeros(n_sims)

    # Discount terminal value to present
    pv_terminal = terminal_value / ((1 + wacc) ** years)

    # Total DCF value = sum of discounted cash flows + terminal value
    dcf_values = pv_projected.sum(axis=1) + pv_terminal

    return dcf_values


def dcf_valuation_from_results(
    results: SimResults,
    initial_fcf: float,
    wacc: float,
    terminal_growth: float | None = None,
    terminal_multiple: float | None = None,
    fcf_in_billions: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Perform DCF valuation from simulation results.

    Parameters
    ----------
    results : SimResults
        Simulation results with growth rate paths
    initial_fcf : float
        Initial free cash flow (in billions if fcf_in_billions=True)
    wacc : float
        Weighted average cost of capital
    terminal_growth : float or None
        Perpetual growth rate
    terminal_multiple : float or None
        Terminal multiple
    fcf_in_billions : bool
        If True, assumes FCF is in billions (output will also be in billions)

    Returns
    -------
    tuple
        (dcf_values, analytics_dict) - both in billions if fcf_in_billions=True
    """
    # Use growth rate paths as input
    growth_rates = results.paths
    n_years = growth_rates.shape[1]

    # Calculate DCF values
    dcf_values = calculate_dcf_value(
        initial_fcf=initial_fcf,
        growth_rates=growth_rates,
        wacc=wacc,
        terminal_growth=terminal_growth,
        terminal_multiple=terminal_multiple,
        years=n_years,
        fcf_in_billions=fcf_in_billions,
    )

    # Compute analytics on DCF values
    # Create a temporary results object for analytics
    temp_results = SimResults(
        paths=dcf_values.reshape(-1, 1),
        config=results.config,
        seed_used=results.seed_used,
        timestamp=results.timestamp,
        parameter_hash=results.parameter_hash,
    )

    analytics = compute_analytics(
        temp_results,
        var_alphas=[0.01, 0.05, 0.10],
        compute_sensitivity=False,
    )

    analytics_dict = {
        "mean": analytics.summary_stats["mean"],
        "median": analytics.summary_stats["median"],
        "std": analytics.summary_stats["std"],
        "p05": analytics.quantiles.get("p05", 0),
        "p50": analytics.quantiles.get("p50", 0),
        "p95": analytics.quantiles.get("p95", 0),
        "var_95": analytics.var.get(0.05, 0),
        "cvar_95": analytics.cvar.get(0.05, 0),
    }

    return dcf_values, analytics_dict


def calculate_equity_value(
    dcf_value: float,
    cash: float = 0.0,
    debt: float = 0.0,
    shares_outstanding: float = 1.0,
) -> float:
    """
    Calculate equity value per share from enterprise value.

    Parameters
    ----------
    dcf_value : float
        Enterprise value (DCF value)
    cash : float
        Cash and cash equivalents
    debt : float
        Total debt
    shares_outstanding : float
        Number of shares outstanding

    Returns
    -------
    float
        Equity value per share
    """
    equity_value = (dcf_value + cash - debt) / shares_outstanding
    return equity_value

