"""
Monte Carlo Simulation Code
Shows how validated user inputs are used to run the Monte Carlo Simulation
using simulate_growth_paths() to create n possibilities for FCF growth.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, PCG64
from scipy import stats
from datetime import datetime

from app.core.types import SimConfig, SimResults, YearConfig, CorrelationConfig, DistConfig


# ============================================================================
# Random Number Generator Creation
# ============================================================================

def create_rng(seed: int | None) -> Generator:
    """Create a numpy.random.Generator with PCG64 algorithm for reproducibility."""
    if seed is not None:
        return Generator(PCG64(seed))
    return Generator(PCG64())


# ============================================================================
# Distribution Sampling Functions
# ============================================================================

def sample_triangular(
    rng: Generator, size: int, a: float, b: float, c: float
) -> np.ndarray:
    """
    Sample from triangular distribution.
    
    Parameters:
    - rng: Random number generator (ensures reproducibility)
    - size: Number of samples to generate (n_sims)
    - a: Minimum value (e.g., -0.15 for -15% growth)
    - b: Maximum value (e.g., 0.34 for 34% growth)
    - c: Mode value (e.g., 0.08 for 8% growth)
    
    Returns:
    - Array of size 'size' containing random samples from triangular distribution
    """
    return stats.triang.rvs(
        c=(c - a) / (b - a),  # Normalized mode position
        loc=a,                 # Location parameter (minimum)
        scale=b - a,           # Scale parameter (range)
        size=size,             # Number of samples
        random_state=rng       # Random number generator for reproducibility
    )


def sample(
    dist_config: DistConfig, rng: Generator, size: int
) -> np.ndarray:
    """
    Unified distribution sampler dispatcher.
    Routes to the appropriate sampling function based on distribution type.
    """
    dist_type = dist_config.get("type")
    if dist_type is None:
        raise ValueError("Distribution type must be specified")

    # Sample based on type
    if dist_type == "triangular":
        # For Year 1: a=-0.15, b=0.34, c=0.08
        samples = sample_triangular(
            rng, size, 
            dist_config["a"],  # Minimum
            dist_config["b"],  # Maximum
            dist_config["c"]   # Mode
        )
    elif dist_type == "normal":
        samples = rng.normal(dist_config["mu"], dist_config["sigma"], size=size)
    elif dist_type == "lognormal":
        samples = rng.lognormal(dist_config["mu"], dist_config["sigma"], size=size)
    elif dist_type == "uniform":
        samples = rng.uniform(dist_config["low"], dist_config["high"], size=size)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    # Apply bounds if specified
    min_bound = dist_config.get("min_bound")
    max_bound = dist_config.get("max_bound")
    if min_bound is not None or max_bound is not None:
        samples = np.maximum(samples, min_bound) if min_bound is not None else samples
        samples = np.minimum(samples, max_bound) if max_bound is not None else samples

    return samples


# ============================================================================
# Correlation Application
# ============================================================================

def apply_correlation(
    samples: np.ndarray, correlation_matrix: np.ndarray
) -> np.ndarray:
    """
    Apply correlation structure using Cholesky decomposition.
    
    Parameters:
    - samples: Independent samples, shape (n_sims, n_years)
    - correlation_matrix: Correlation matrix, shape (n_years, n_years)
    
    Returns:
    - Correlated samples, same shape as input
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


# ============================================================================
# Main Simulation Function
# ============================================================================

def simulate_growth_paths(
    config: SimConfig, rng: Generator | None = None
) -> SimResults:
    """
    Simulate multi-year growth paths with optional correlation.
    
    This function:
    1. Creates a random number generator (RNG) with the specified seed
    2. For each year, samples n_sims values from the specified distribution
    3. Applies correlation structure if enabled
    4. Returns SimResults containing all simulation paths
    
    Parameters:
    - config: Simulation configuration (n_sims, seed, years, correlation)
    - rng: Random number generator (if None, creates one from seed)
    
    Returns:
    - SimResults: Simulation results container with paths array
    """
    # Create RNG if not provided
    if rng is None:
        rng = create_rng(config.seed)

    n_sims = config.n_sims      # e.g., 10,000
    n_years = len(config.years)  # e.g., 2 (Year 1 and Year 2)

    # Initialize paths array: shape (n_sims, n_years)
    # Each row is one simulation trial, each column is one year
    paths = np.zeros((n_sims, n_years))

    # Sample each year's growth rate distribution
    for i, year_config in enumerate(config.years):
        if year_config.deterministic_value is not None:
            # Use fixed value (not stochastic)
            paths[:, i] = year_config.deterministic_value
        else:
            # Sample from distribution (e.g., triangular for Year 1)
            # This is where random sampling happens for each simulation trial
            paths[:, i] = sample(year_config.dist, rng, n_sims)

    # Apply correlation if enabled (e.g., Year 1 ↔ Year 2 correlation = 0.6)
    if config.correlation.enabled and n_years > 1:
        paths = apply_correlation(paths, config.correlation.matrix)

    # Create and return results
    timestamp = datetime.now().isoformat()
    seed_used = config.seed if config.seed is not None else 0

    return SimResults(
        paths=paths,              # Array of shape (n_sims, n_years)
        config=config,            # Original configuration
        seed_used=seed_used,      # Seed used for reproducibility
        timestamp=timestamp,
        parameter_hash="placeholder",
    )


# ============================================================================
# Configuration Building (from Streamlit UI)
# ============================================================================

def build_simulation_config(
    n_sims: int,
    seed: int | None,
    year_configs: list[tuple[dict, float | None]],
    correlation: CorrelationConfig,
) -> SimConfig:
    """
    Build SimConfig from validated user inputs.
    
    Example year_configs:
    [
        ({"type": "triangular", "a": -0.15, "b": 0.34, "c": 0.08}, None),  # Year 1
        ({"type": "triangular", "a": -0.15, "b": 0.41, "c": 0.15}, None),  # Year 2
    ]
    """
    years = []
    for i, (dist_config, det_value) in enumerate(year_configs):
        year_config = YearConfig(
            year=i + 1, 
            dist=dist_config, 
            deterministic_value=det_value
        )
        years.append(year_config)

    config = SimConfig(
        n_sims=n_sims,
        seed=seed,
        years=years,
        correlation=correlation,
        scenario_name=None,
        scenario_weight=None,
    )
    
    return config


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Year 1 FCF growth sampling with Apple preset parameters
    year1_dist_config = {
        "type": "triangular",
        "a": -0.15,  # Minimum growth rate: -15%
        "b": 0.34,   # Maximum growth rate: 34%
        "c": 0.08,   # Mode (most likely): 8%
    }
    
    year2_dist_config = {
        "type": "triangular",
        "a": -0.15,  # Minimum growth rate: -15%
        "b": 0.41,   # Maximum growth rate: 41%
        "c": 0.15,   # Mode (most likely): 15%
    }
    
    # Build configuration
    correlation_matrix = np.array([
        [1.0, 0.6],  # Year 1 ↔ Year 2 correlation = 0.6
        [0.6, 1.0],
    ])
    
    correlation = CorrelationConfig(
        enabled=True,
        matrix=correlation_matrix,
    )
    
    config = build_simulation_config(
        n_sims=10000,  # Run 10,000 simulations
        seed=42,       # Reproducible seed
        year_configs=[
            (year1_dist_config, None),  # Year 1: stochastic
            (year2_dist_config, None),  # Year 2: stochastic
        ],
        correlation=correlation,
    )
    
    # Run simulation
    results = simulate_growth_paths(config)
    
    # Results contain:
    # - results.paths: shape (10000, 2) array
    #   - Each row is one simulation trial
    #   - Column 0: Year 1 FCF growth rates (sampled from triangular distribution)
    #   - Column 1: Year 2 FCF growth rates (sampled from triangular distribution)
    # - results.seed_used: 42 (for reproducibility)
    # - results.config: Original configuration
    
    print(f"Simulation completed: {results.paths.shape[0]} trials, {results.paths.shape[1]} years")
    print(f"Year 1 growth rates (first 10): {results.paths[:10, 0]}")
    print(f"Year 2 growth rates (first 10): {results.paths[:10, 1]}")

