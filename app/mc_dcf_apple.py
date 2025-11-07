"""
Monte Carlo DCF Model for Apple Valuation

This script implements a Discounted Cash Flow (DCF) model with Monte Carlo
simulation to value Apple Inc. based on uncertain growth rates in FY2026 and FY2027.

Model Structure:
- Year 1 (FY2026): Variable growth rate from triangular distribution
- Year 2 (FY2027): Variable growth rate from triangular distribution  
- Year 3+ (FY2028+): Perpetual growth at terminal rate

All values are in billions of USD for readability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple


# ============================================================================
# INPUTS & CONSTANTS (Edit these as needed)
# ============================================================================

# Base free cash flow (FY2025)
fcf_base_2025 = 107.509395  # billions USD

# Discount rate (WACC)
wacc = 0.0821  # 8.21%

# Terminal growth rate (perpetual growth from Year 3 onward)
terminal_growth = 0.03  # 3%

# Growth rate distributions (triangular)
# Format: (min, mode/base, max)
g1_params = (-0.15, 0.08, 0.34)   # FY2026 FCF growth: min=-15%, mode=8%, max=34%
g2_params = (-0.15, 0.15, 0.41)   # FY2027 FCF growth: min=-15%, mode=15%, max=41%

# Simulation parameters
n_iter = 50000  # Number of Monte Carlo iterations
random_seed = 42  # Random seed for reproducibility

# Market cap for comparison (optional - set to None if not available)
# TODO: Paste Apple's market cap at 2024 close here (in billions USD)
market_cap_2024_close = None  # e.g., 3500.0


# ============================================================================
# VALIDATION
# ============================================================================

def validate_inputs() -> None:
    """
    Validate that inputs are reasonable and satisfy model constraints.
    
    Raises
    ------
    ValueError
        If inputs are invalid
    """
    if wacc <= terminal_growth:
        raise ValueError(
            f"WACC ({wacc}) must be greater than terminal growth ({terminal_growth}) "
            "for Gordon Growth Model to be valid."
        )
    
    if fcf_base_2025 <= 0:
        raise ValueError(f"Base FCF must be positive, got {fcf_base_2025}")
    
    if n_iter <= 0:
        raise ValueError(f"Number of iterations must be positive, got {n_iter}")
    
    # Validate triangular distribution parameters
    for name, params in [("g1", g1_params), ("g2", g2_params)]:
        min_val, mode, max_val = params
        if not (min_val <= mode <= max_val):
            raise ValueError(
                f"{name} parameters invalid: min ({min_val}) <= mode ({mode}) <= max ({max_val})"
            )


# ============================================================================
# SIMULATION FUNCTIONS
# ============================================================================

def run_simulation(
    fcf_base: float,
    wacc: float,
    terminal_growth: float,
    g1_params: Tuple[float, float, float],
    g2_params: Tuple[float, float, float],
    n_iter: int,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation of DCF model.
    
    Model Structure:
    - Year 1 (FY2026): FCF_1 = FCF_base * (1 + g1)
    - Year 2 (FY2027): FCF_2 = FCF_1 * (1 + g2)
    - Year 3+ (FY2028+): FCF_3 = FCF_2 * (1 + terminal_growth)
    
    Discounting:
    - PV_1 = FCF_1 / (1 + WACC)^1
    - PV_2 = FCF_2 / (1 + WACC)^2
    - TV_2 = FCF_3 / (WACC - terminal_growth)  [Terminal value at end of Year 2]
    - PV_TV = TV_2 / (1 + WACC)^2  [Terminal value discounted to present]
    - Enterprise Value = PV_1 + PV_2 + PV_TV
    
    Parameters
    ----------
    fcf_base : float
        Base year free cash flow (FY2025), billions USD
    wacc : float
        Weighted average cost of capital (discount rate)
    terminal_growth : float
        Perpetual growth rate from Year 3 onward
    g1_params : tuple
        (min, mode, max) for Year 1 growth rate distribution
    g2_params : tuple
        (min, mode, max) for Year 2 growth rate distribution
    n_iter : int
        Number of Monte Carlo iterations
    random_seed : int or None
        Random seed for reproducibility
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: g1, g2, fcf1, fcf2, fcf3, ev
        All values in billions USD
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Sample growth rates from triangular distributions
    # numpy.random.triangular(left, mode, right, size)
    # Note: NumPy's triangular uses (left, mode, right) where:
    #   - left = minimum value
    #   - mode = most likely value (peak)
    #   - right = maximum value
    g1_min, g1_mode, g1_max = g1_params
    g2_min, g2_mode, g2_max = g2_params
    
    # Generate triangular samples (vectorized)
    g1 = np.random.triangular(g1_min, g1_mode, g1_max, size=n_iter)
    g2 = np.random.triangular(g2_min, g2_mode, g2_max, size=n_iter)
    
    # Verify triangular distribution properties
    # The mode should be the most likely value, creating a peak in the distribution
    assert np.all(g1 >= g1_min) and np.all(g1 <= g1_max), "g1 out of bounds"
    assert np.all(g2 >= g2_min) and np.all(g2 <= g2_max), "g2 out of bounds"
    
    # Compute cash flow paths (vectorized)
    # Year 1 (FY2026): FCF_1 = FCF_base * (1 + g1)
    fcf1 = fcf_base * (1 + g1)
    
    # Year 2 (FY2027): FCF_2 = FCF_1 * (1 + g2)
    fcf2 = fcf1 * (1 + g2)
    
    # Year 3 (FY2028): First year of perpetual growth
    # FCF_3 = FCF_2 * (1 + terminal_growth)
    fcf3 = fcf2 * (1 + terminal_growth)
    
    # Discount projected cash flows to present value
    # PV = FCF_n / (1 + r)^n where r = WACC
    pv1 = fcf1 / ((1 + wacc) ** 1)  # Year 1 discounted
    pv2 = fcf2 / ((1 + wacc) ** 2)  # Year 2 discounted
    
    # Terminal value calculation (Gordon Growth Model)
    # TV_2 = FCF_3 / (WACC - terminal_growth)
    # This gives terminal value as of end of Year 2
    tv2 = fcf3 / (wacc - terminal_growth)
    
    # Discount terminal value to present (end of Year 2, so discount by 2 years)
    # PV_TV = TV_2 / (1 + WACC)^2
    pv_tv = tv2 / ((1 + wacc) ** 2)
    
    # Enterprise Value = sum of all present values
    ev = pv1 + pv2 + pv_tv
    
    # Create DataFrame with all results
    results = pd.DataFrame({
        'g1': g1,
        'g2': g2,
        'fcf1': fcf1,
        'fcf2': fcf2,
        'fcf3': fcf3,
        'ev': ev,
    })
    
    return results


def compute_summary_stats(ev: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics for enterprise values.
    
    Parameters
    ----------
    ev : np.ndarray
        Array of enterprise values (billions USD)
    
    Returns
    -------
    dict
        Dictionary with mean, median, std, min, max, p5, p25, p75, p95
    """
    return {
        'mean': float(np.mean(ev)),
        'median': float(np.median(ev)),
        'std': float(np.std(ev, ddof=1)),
        'min': float(np.min(ev)),
        'max': float(np.max(ev)),
        'p5': float(np.percentile(ev, 5)),
        'p25': float(np.percentile(ev, 25)),
        'p75': float(np.percentile(ev, 75)),
        'p95': float(np.percentile(ev, 95)),
    }


def compare_to_market_cap(
    ev: np.ndarray, market_cap: float
) -> Dict[str, float]:
    """
    Compare enterprise values to market capitalization.
    
    Parameters
    ----------
    ev : np.ndarray
        Array of enterprise values (billions USD)
    market_cap : float
        Market capitalization (billions USD)
    
    Returns
    -------
    dict
        Dictionary with diff, pct_diff, prob_ev_gt_mkt
    """
    mean_ev = np.mean(ev)
    diff = mean_ev - market_cap
    pct_diff = (diff / market_cap) * 100
    prob_ev_gt_mkt = float(np.mean(ev > market_cap))
    
    return {
        'mean_ev': mean_ev,
        'market_cap': market_cap,
        'diff': diff,
        'pct_diff': pct_diff,
        'prob_ev_gt_mkt': prob_ev_gt_mkt,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ev_distribution(
    ev: np.ndarray,
    stats: Dict[str, float],
    market_cap: float | None = None,
    save_path: str = 'ev_distribution.png',
) -> None:
    """
    Plot histogram of enterprise values with summary statistics.
    
    Parameters
    ----------
    ev : np.ndarray
        Array of enterprise values (billions USD)
    stats : dict
        Summary statistics dictionary
    market_cap : float or None
        Market capitalization for comparison (optional)
    save_path : str
        Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    n, bins, patches = ax.hist(
        ev,
        bins=60,
        density=True,
        alpha=0.7,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5,
        label='Enterprise Value Distribution',
    )
    
    # Vertical lines for key statistics
    ax.axvline(
        stats['mean'],
        color='green',
        linestyle='--',
        linewidth=2,
        label=f"Mean: ${stats['mean']:.2f}B",
    )
    ax.axvline(
        stats['median'],
        color='blue',
        linestyle='--',
        linewidth=2,
        label=f"Median: ${stats['median']:.2f}B",
    )
    ax.axvline(
        stats['p5'],
        color='orange',
        linestyle='--',
        linewidth=2,
        label=f"5th %ile: ${stats['p5']:.2f}B",
    )
    ax.axvline(
        stats['p95'],
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"95th %ile: ${stats['p95']:.2f}B",
    )
    
    # Market cap line if provided
    if market_cap is not None:
        ax.axvline(
            market_cap,
            color='purple',
            linestyle='-',
            linewidth=2,
            label=f"Market Cap: ${market_cap:.2f}B",
        )
    
    ax.set_xlabel('Enterprise Value (Billions USD)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        'Monte Carlo DCF: Apple Enterprise Value Distribution',
        fontsize=14,
        fontweight='bold',
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    # plt.show()  # Commented out for headless execution - uncomment if you want to display


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """Main execution function."""
    print("=" * 70)
    print("Monte Carlo DCF Model for Apple Valuation")
    print("=" * 70)
    
    # Validate inputs
    validate_inputs()
    
    print(f"\nSimulation Parameters:")
    print(f"  Base FCF (FY2025): ${fcf_base_2025:.3f}B")
    print(f"  WACC: {wacc:.4f} ({wacc*100:.2f}%)")
    print(f"  Terminal Growth: {terminal_growth:.4f} ({terminal_growth*100:.2f}%)")
    print(f"  Year 1 Growth: Triangular({g1_params[0]:.2f}, {g1_params[1]:.2f}, {g1_params[2]:.2f})")
    print(f"  Year 2 Growth: Triangular({g2_params[0]:.2f}, {g2_params[1]:.2f}, {g2_params[2]:.2f})")
    print(f"  Iterations: {n_iter:,}")
    print(f"  Random Seed: {random_seed}")
    
    # Run simulation
    print(f"\nRunning Monte Carlo simulation...")
    results = run_simulation(
        fcf_base=fcf_base_2025,
        wacc=wacc,
        terminal_growth=terminal_growth,
        g1_params=g1_params,
        g2_params=g2_params,
        n_iter=n_iter,
        random_seed=random_seed,
    )
    
    # Compute summary statistics
    stats = compute_summary_stats(results['ev'].values)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Enterprise Value Summary Statistics (Billions USD)")
    print("=" * 70)
    print(f"  Mean:   ${stats['mean']:>12.2f}B")
    print(f"  Median: ${stats['median']:>12.2f}B")
    print(f"  Std Dev: ${stats['std']:>11.2f}B")
    print(f"  Min:    ${stats['min']:>12.2f}B")
    print(f"  Max:    ${stats['max']:>12.2f}B")
    print(f"  5th %ile:  ${stats['p5']:>10.2f}B")
    print(f"  25th %ile: ${stats['p25']:>10.2f}B")
    print(f"  75th %ile: ${stats['p75']:>10.2f}B")
    print(f"  95th %ile: ${stats['p95']:>10.2f}B")
    
    # Market cap comparison if provided
    if market_cap_2024_close is not None:
        comparison = compare_to_market_cap(
            results['ev'].values, market_cap_2024_close
        )
        print("\n" + "=" * 70)
        print("Comparison to Market Capitalization")
        print("=" * 70)
        print(f"  Mean EV:        ${comparison['mean_ev']:>12.2f}B")
        print(f"  Market Cap:     ${comparison['market_cap']:>12.2f}B")
        print(f"  Difference:     ${comparison['diff']:>12.2f}B")
        print(f"  % Difference:   {comparison['pct_diff']:>12.2f}%")
        print(
            f"  P(EV > Market): {comparison['prob_ev_gt_mkt']:>12.2%}"
        )
        
        if comparison['diff'] > 0:
            print(f"\n  → Model suggests Apple is UNDERVALUED by ${comparison['diff']:.2f}B")
        else:
            print(f"\n  → Model suggests Apple is OVERVALUED by ${abs(comparison['diff']):.2f}B")
    else:
        comparison = None
        print("\n  (Market cap not provided - set 'market_cap_2024_close' to enable comparison)")
    
    # Create visualization
    print("\nGenerating visualization...")
    plot_ev_distribution(
        results['ev'].values,
        stats,
        market_cap=market_cap_2024_close,
        save_path='ev_distribution.png',
    )
    
    # Save results to CSV
    csv_path = 'mc_dcf_results.csv'
    results.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    print("\n" + "=" * 70)
    print("Simulation Complete")
    print("=" * 70)


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_shapes() -> None:
    """Test that simulation returns correct shapes."""
    results = run_simulation(
        fcf_base=100.0,
        wacc=0.08,
        terminal_growth=0.03,
        g1_params=(-0.1, 0.05, 0.2),
        g2_params=(-0.1, 0.1, 0.3),
        n_iter=1000,
        random_seed=42,
    )
    assert results.shape == (1000, 6), f"Expected shape (1000, 6), got {results.shape}"
    assert list(results.columns) == ['g1', 'g2', 'fcf1', 'fcf2', 'fcf3', 'ev']
    print("✓ Shape test passed")


def test_seed_reproducibility() -> None:
    """Test that different seeds produce different results."""
    results1 = run_simulation(
        fcf_base=100.0,
        wacc=0.08,
        terminal_growth=0.03,
        g1_params=(-0.1, 0.05, 0.2),
        g2_params=(-0.1, 0.1, 0.3),
        n_iter=1000,
        random_seed=42,
    )
    results2 = run_simulation(
        fcf_base=100.0,
        wacc=0.08,
        terminal_growth=0.03,
        g1_params=(-0.1, 0.05, 0.2),
        g2_params=(-0.1, 0.1, 0.3),
        n_iter=1000,
        random_seed=43,
    )
    assert not np.allclose(
        results1['ev'].values, results2['ev'].values
    ), "Different seeds should produce different results"
    print("✓ Seed reproducibility test passed")


def test_wacc_validation() -> None:
    """Test that WACC <= terminal_growth raises error."""
    try:
        validate_inputs()
        # Temporarily set invalid WACC
        global wacc, terminal_growth
        old_wacc, old_terminal = wacc, terminal_growth
        wacc = 0.02
        terminal_growth = 0.03
        try:
            validate_inputs()
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        finally:
            wacc, terminal_growth = old_wacc, old_terminal
        print("✓ WACC validation test passed")
    except Exception as e:
        print(f"✗ WACC validation test failed: {e}")


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    test_shapes()
    test_seed_reproducibility()
    test_wacc_validation()
    print("\n")
    
    # Run main simulation
    main()

