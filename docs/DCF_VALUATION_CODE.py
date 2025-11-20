"""
DCF Valuation Code
Shows how the program receives n growth rate pairs (Year 1 and Year 2)
and calculates estimated enterprise value using calculate_dcf_value().
"""

from __future__ import annotations

import numpy as np
from app.core.simulate import SimResults
from app.core.analytics import compute_analytics


# ============================================================================
# Core DCF Calculation Function
# ============================================================================

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
    Calculate DCF value (Enterprise Value) for each simulation path.
    
    This function receives n growth rate pairs (Year 1 and Year 2) and
    calculates the estimated enterprise value for each pair using the DCF process.
    
    Parameters:
    - initial_fcf: Initial free cash flow (Year 0), in billions
    - growth_rates: Growth rates for each year, shape (n_sims, n_years)
                    Each row is one growth rate pair (Year 1, Year 2)
    - wacc: Weighted average cost of capital (discount rate)
    - terminal_growth: Perpetual growth rate for terminal value (Gordon Growth Model)
    - terminal_multiple: Terminal multiple (alternative to terminal growth)
    - years: Number of projection years
    - fcf_in_billions: If True, assumes FCF is in billions
    
    Returns:
    - dcf_values: Enterprise values for each simulation, shape (n_sims,)
                  Stored similar to how growth pairs were stored
    """
    n_sims = growth_rates.shape[0]  # Number of simulation trials (n)
    
    # Project cash flows for each simulation
    # fcf_paths shape: (n_sims, years)
    # Each row represents one simulation's projected cash flows
    fcf_paths = np.zeros((n_sims, years))
    
    # Year 1: FCF_1 = FCF_0 * (1 + growth_rate_1)
    fcf_paths[:, 0] = initial_fcf * (1 + growth_rates[:, 0])
    
    # Year 2 and beyond: FCF_year = FCF_year-1 * (1 + growth_rate_year)
    for year in range(1, years):
        fcf_paths[:, year] = fcf_paths[:, year - 1] * (1 + growth_rates[:, year])
    
    # Discount projected cash flows to present value
    # pv_projected shape: (n_sims, years)
    pv_projected = np.zeros((n_sims, years))
    for year in range(years):
        # PV = FCF / (1 + WACC)^(year + 1)
        pv_projected[:, year] = fcf_paths[:, year] / ((1 + wacc) ** (year + 1))
    
    # Calculate terminal value
    if terminal_growth is not None:
        # Gordon Growth Model: TV = FCF_n * (1 + g) / (WACC - g)
        terminal_fcf = fcf_paths[:, -1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
    elif terminal_multiple is not None:
        # Terminal multiple approach: TV = FCF_n * multiple
        terminal_value = fcf_paths[:, -1] * terminal_multiple
    else:
        # No terminal value
        terminal_value = np.zeros(n_sims)
    
    # Discount terminal value to present
    # PV_terminal = TV / (1 + WACC)^years
    pv_terminal = terminal_value / ((1 + wacc) ** years)
    
    # Total Enterprise Value = sum of discounted cash flows + terminal value
    # dcf_values shape: (n_sims,) - one enterprise value per simulation
    dcf_values = pv_projected.sum(axis=1) + pv_terminal
    
    return dcf_values


# ============================================================================
# DCF Valuation from Simulation Results
# ============================================================================

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
    
    This function receives the simulation results containing n growth rate pairs
    and calculates enterprise values for each pair.
    
    Parameters:
    - results: Simulation results with growth rate paths
               results.paths shape: (n_sims, n_years)
               Each row is one growth rate pair (Year 1, Year 2)
    - initial_fcf: Initial free cash flow, in billions
    - wacc: Weighted average cost of capital
    - terminal_growth: Perpetual growth rate
    - terminal_multiple: Terminal multiple
    - fcf_in_billions: If True, assumes FCF is in billions
    
    Returns:
    - tuple: (dcf_values, analytics_dict)
      - dcf_values: Enterprise values for each simulation, shape (n_sims,)
                    Stored similar to how growth pairs were stored
      - analytics_dict: Summary statistics and analytics
    """
    # Extract growth rate paths from simulation results
    # growth_rates shape: (n_sims, n_years)
    # Each row is one growth rate pair (Year 1, Year 2)
    growth_rates = results.paths
    n_years = growth_rates.shape[1]
    
    # Calculate DCF values using each growth rate pair
    # This calls calculate_dcf_value() for all n simulations at once (vectorized)
    dcf_values = calculate_dcf_value(
        initial_fcf=initial_fcf,
        growth_rates=growth_rates,  # n growth rate pairs
        wacc=wacc,
        terminal_growth=terminal_growth,
        terminal_multiple=terminal_multiple,
        years=n_years,
        fcf_in_billions=fcf_in_billions,
    )
    
    # Compute analytics on enterprise values
    # Create temporary results object for analytics computation
    temp_results = SimResults(
        paths=dcf_values.reshape(-1, 1),  # Reshape to (n_sims, 1) for analytics
        config=results.config,
        seed_used=results.seed_used,
        timestamp=results.timestamp,
        parameter_hash=results.parameter_hash,
    )
    
    # Compute summary statistics and risk metrics
    analytics = compute_analytics(
        temp_results,
        var_alphas=[0.01, 0.05, 0.10],
        compute_sensitivity=False,
    )
    
    # Extract analytics into dictionary
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


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: DCF valuation with n growth rate pairs
    
    # Simulate growth rates (from previous simulation step)
    # This would come from simulate_growth_paths() results
    n_sims = 10000  # n = 10,000 simulations
    n_years = 2     # Year 1 and Year 2
    
    # Example growth rate pairs (Year 1, Year 2)
    # Each row is one growth rate pair
    growth_rates = np.array([
        [0.08, 0.15],   # Simulation 1: Year 1 = 8%, Year 2 = 15%
        [0.12, 0.20],   # Simulation 2: Year 1 = 12%, Year 2 = 20%
        [-0.05, 0.10],  # Simulation 3: Year 1 = -5%, Year 2 = 10%
        # ... n more pairs
    ])
    
    # DCF parameters
    initial_fcf = 107.509  # Initial FCF in billions (Apple)
    wacc = 0.0821          # WACC = 8.21%
    terminal_growth = 0.03 # Terminal growth = 3%
    
    # Calculate enterprise value for each growth rate pair
    # This processes all n pairs at once (vectorized)
    dcf_values = calculate_dcf_value(
        initial_fcf=initial_fcf,
        growth_rates=growth_rates,  # n growth rate pairs
        wacc=wacc,
        terminal_growth=terminal_growth,
        years=n_years,
        fcf_in_billions=True,
    )
    
    # dcf_values shape: (n_sims,)
    # Each element is the enterprise value for one simulation
    # Stored similar to how growth pairs were stored
    
    print(f"Enterprise values calculated: {dcf_values.shape[0]} values")
    print(f"First 10 enterprise values (billions USD): {dcf_values[:10]}")
    print(f"Mean enterprise value: ${dcf_values.mean():.2f}B")
    print(f"Median enterprise value: ${np.median(dcf_values):.2f}B")
    
    # Example with full results object
    # (In actual usage, this would come from simulate_growth_paths())
    from app.core.types import SimConfig, YearConfig, CorrelationConfig
    
    # Create mock results object
    mock_results = SimResults(
        paths=growth_rates,  # n growth rate pairs
        config=SimConfig(
            n_sims=n_sims,
            seed=42,
            years=[YearConfig(year=1, dist={}), YearConfig(year=2, dist={})],
            correlation=CorrelationConfig(enabled=False, matrix=np.eye(2)),
        ),
        seed_used=42,
        timestamp="2024-01-01T00:00:00",
        parameter_hash="mock",
    )
    
    # Calculate DCF values from results
    dcf_values_full, analytics = dcf_valuation_from_results(
        results=mock_results,
        initial_fcf=initial_fcf,
        wacc=wacc,
        terminal_growth=terminal_growth,
        fcf_in_billions=True,
    )
    
    print(f"\nFull DCF valuation results:")
    print(f"Mean EV: ${analytics['mean']:.2f}B")
    print(f"Median EV: ${analytics['median']:.2f}B")
    print(f"5th percentile: ${analytics['p05']:.2f}B")
    print(f"95th percentile: ${analytics['p95']:.2f}B")

