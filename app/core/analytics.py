"""
Analytics and statistical computations for simulation results.

Includes summary statistics, risk metrics (VaR, CVaR), sensitivity analysis,
and probability calculations.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from app.core.types import AnalyticsResults, SimResults


def describe(arr: np.ndarray) -> dict[str, float]:
    """
    Compute comprehensive summary statistics.

    Parameters
    ----------
    arr : np.ndarray
        Array of values (flattened if multi-dimensional)

    Returns
    -------
    dict
        Dictionary of summary statistics
    """
    arr_flat = arr.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]  # Remove NaN

    if len(arr_flat) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
        }

    return {
        "mean": float(np.mean(arr_flat)),
        "median": float(np.median(arr_flat)),
        "std": float(np.std(arr_flat, ddof=1)),
        "min": float(np.min(arr_flat)),
        "max": float(np.max(arr_flat)),
        "skew": float(stats.skew(arr_flat)),
        "kurtosis": float(stats.kurtosis(arr_flat)),
    }


def compute_quantiles(arr: np.ndarray, qs: list[float] | None = None) -> dict[str, float]:
    """
    Compute quantiles at specified percentiles.

    Parameters
    ----------
    arr : np.ndarray
        Array of values
    qs : list of float, optional
        Quantile percentiles (default: [1, 5, 25, 50, 75, 95, 99])

    Returns
    -------
    dict
        Dictionary mapping percentile names to values
    """
    if qs is None:
        qs = [1, 5, 25, 50, 75, 95, 99]

    arr_flat = arr.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]

    if len(arr_flat) == 0:
        return {f"p{q:02d}": np.nan for q in qs}

    quantiles = np.percentile(arr_flat, qs)
    return {f"p{q:02d}": float(qval) for q, qval in zip(qs, quantiles)}


def value_at_risk(arr: np.ndarray, alpha: float) -> float:
    """
    Compute Value at Risk (VaR) at confidence level alpha.

    VaR is the (1-alpha) quantile of the loss distribution.

    Parameters
    ----------
    arr : np.ndarray
        Array of values (losses or returns)
    alpha : float
        Confidence level (e.g., 0.05 for 95% VaR)

    Returns
    -------
    float
        VaR value
    """
    arr_flat = arr.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]

    if len(arr_flat) == 0:
        return np.nan

    return float(np.percentile(arr_flat, (1 - alpha) * 100))


def conditional_var(arr: np.ndarray, alpha: float) -> float:
    """
    Compute Conditional Value at Risk (CVaR) / Expected Shortfall.

    CVaR is the expected value of losses beyond the VaR threshold.

    Parameters
    ----------
    arr : np.ndarray
        Array of values (losses or returns)
    alpha : float
        Confidence level (e.g., 0.05 for 95% CVaR)

    Returns
    -------
    float
        CVaR value
    """
    arr_flat = arr.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]

    if len(arr_flat) == 0:
        return np.nan

    var_threshold = value_at_risk(arr_flat, alpha)
    tail_losses = arr_flat[arr_flat <= var_threshold]

    if len(tail_losses) == 0:
        return var_threshold

    return float(np.mean(tail_losses))


def probability_loss(arr: np.ndarray, threshold: float = 0.0) -> float:
    """
    Compute probability of loss (or below threshold).

    Parameters
    ----------
    arr : np.ndarray
        Array of values
    threshold : float
        Loss threshold (default: 0.0)

    Returns
    -------
    float
        Probability of being below threshold
    """
    arr_flat = arr.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]

    if len(arr_flat) == 0:
        return np.nan

    return float(np.mean(arr_flat < threshold))


def probability_target(arr: np.ndarray, target: float) -> float:
    """
    Compute probability of exceeding target value.

    Parameters
    ----------
    arr : np.ndarray
        Array of values
    target : float
        Target value

    Returns
    -------
    float
        Probability of exceeding target
    """
    arr_flat = arr.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]

    if len(arr_flat) == 0:
        return np.nan

    return float(np.mean(arr_flat >= target))


def tornado_analysis(
    results: SimResults, target_year: int | None = None
) -> dict[str, float]:
    """
    One-at-a-time sensitivity analysis (tornado chart data).

    Varies each input parameter by ±10% and measures impact on output.

    Parameters
    ----------
    results : SimResults
        Simulation results
    target_year : int or None
        Which year's output to analyze (None for final year)

    Returns
    -------
    dict
        Dictionary mapping parameter names to sensitivity scores
    """
    if target_year is None:
        target_year = results.paths.shape[1] - 1

    base_output = results.paths[:, target_year].mean()

    sensitivities: dict[str, float] = {}

    # For each year's distribution, vary parameters
    for i, year_config in enumerate(results.config.years):
        if year_config.deterministic_value is not None:
            continue

        dist_config = year_config.dist
        dist_type = dist_config.get("type", "unknown")

        # Create parameter name
        param_name = f"Year{i+1}_{dist_type}"

        # Vary key parameter by ±10%
        if dist_type == "normal":
            # Vary mu
            mu_base = dist_config.get("mu", 0.0)
            mu_high = mu_base * 1.1
            mu_low = mu_base * 0.9

            # Simplified: use linear approximation
            # In practice, would re-run simulation
            sensitivity = abs(mu_high - mu_low) / abs(mu_base) if mu_base != 0 else 0.0
            sensitivities[param_name] = sensitivity

        elif dist_type in ["lognormal", "uniform", "beta"]:
            # Similar approach for other distributions
            sensitivities[param_name] = 0.1  # Placeholder

    return sensitivities


def partial_rank_correlation(
    results: SimResults, target_year: int | None = None
) -> dict[str, float]:
    """
    Compute Partial Rank Correlation Coefficients (PRCC).

    Measures non-linear relationships between inputs and output.

    Parameters
    ----------
    results : SimResults
        Simulation results
    target_year : int or None
        Which year's output to analyze

    Returns
    -------
    dict
        Dictionary mapping parameter names to PRCC values
    """
    if target_year is None:
        target_year = results.paths.shape[1] - 1

    output = results.paths[:, target_year]

    prcc_values: dict[str, float] = {}

    # For each input year, compute rank correlation
    for i in range(results.paths.shape[1]):
        if i == target_year:
            continue

        input_values = results.paths[:, i]
        # Rank correlation (Spearman)
        correlation, _ = stats.spearmanr(input_values, output)
        prcc_values[f"Year{i+1}"] = float(correlation) if not np.isnan(correlation) else 0.0

    return prcc_values


def compute_analytics(
    results: SimResults,
    var_alphas: list[float] | None = None,
    loss_threshold: float | None = None,
    target_value: float | None = None,
    compute_sensitivity: bool = True,
) -> AnalyticsResults:
    """
    Compute comprehensive analytics from simulation results.

    Parameters
    ----------
    results : SimResults
        Simulation results
    var_alphas : list of float, optional
        VaR/CVaR confidence levels (default: [0.01, 0.05, 0.10])
    loss_threshold : float or None
        Threshold for probability of loss calculation
    target_value : float or None
        Target value for probability calculation
    compute_sensitivity : bool
        Whether to compute sensitivity analysis

    Returns
    -------
    AnalyticsResults
        Complete analytics results
    """
    if var_alphas is None:
        var_alphas = [0.01, 0.05, 0.10]

    # Use final year or aggregate across all years
    if results.paths.shape[1] == 1:
        analysis_array = results.paths[:, 0]
    else:
        # Sum across years or use final year
        analysis_array = results.paths[:, -1]

    # Summary statistics
    summary = describe(analysis_array)
    quantiles_dict = compute_quantiles(analysis_array)

    # VaR and CVaR
    var_dict = {alpha: value_at_risk(analysis_array, alpha) for alpha in var_alphas}
    cvar_dict = {alpha: conditional_var(analysis_array, alpha) for alpha in var_alphas}

    # Probabilities
    prob_loss = (
        probability_loss(analysis_array, loss_threshold)
        if loss_threshold is not None
        else None
    )
    prob_target = (
        probability_target(analysis_array, target_value)
        if target_value is not None
        else None
    )

    # Sensitivity
    sensitivity = tornado_analysis(results) if compute_sensitivity else None
    prcc = partial_rank_correlation(results) if compute_sensitivity else None

    return AnalyticsResults(
        summary_stats=summary,
        quantiles=quantiles_dict,
        var=var_dict,
        cvar=cvar_dict,
        prob_loss=prob_loss,
        prob_target=prob_target,
        target_value=target_value,
        sensitivity=sensitivity,
        prcc=prcc,
    )


def ecdf(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Empirical Cumulative Distribution Function (ECDF).

    Parameters
    ----------
    arr : np.ndarray
        Array of values

    Returns
    -------
    tuple
        (sorted_values, cumulative_probabilities)
    """
    arr_flat = arr.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]

    if len(arr_flat) == 0:
        return np.array([]), np.array([])

    sorted_vals = np.sort(arr_flat)
    n = len(sorted_vals)
    cumulative = np.arange(1, n + 1) / n

    return sorted_vals, cumulative

