# Monte Carlo Simulation Code Flow

This document shows the complete code flow for how validated user inputs are used to run the Monte Carlo Simulation, specifically demonstrating the `simulate_growth_paths()` function that creates n possibilities for FCF growth using random sampling from the Year 1 FCF growth triangular distribution.

## Overview

The simulation process follows these steps:
1. **User Inputs** → Validated through Pydantic models
2. **Configuration Building** → Creates `SimConfig` object
3. **Random Number Generator** → Creates reproducible RNG with seed
4. **Distribution Sampling** → Samples from Year 1 triangular distribution (and Year 2, etc.)
5. **Correlation Application** → Applies correlation structure if enabled
6. **Results** → Returns `SimResults` with all simulation paths

---

## Step 1: User Inputs and Configuration Building

**Location:** `app/streamlit_app.py` (lines 304-325)

```python
# Build configuration from validated user inputs
from app.core.types import YearConfig, SimConfig

years = []
for i, (dist_config, det_value) in enumerate(year_configs):
    # dist_config contains the distribution parameters
    # For Year 1 triangular: {"type": "triangular", "a": -0.15, "b": 0.34, "c": 0.08}
    year_config = YearConfig(
        year=i + 1, 
        dist=dist_config, 
        deterministic_value=det_value
    )
    years.append(year_config)

# Create simulation configuration
config = SimConfig(
    n_sims=n_sims,           # Number of simulations (e.g., 10,000)
    seed=seed,                # Random seed for reproducibility (e.g., 42)
    years=years,              # List of YearConfig objects (Year 1, Year 2, etc.)
    correlation=correlation,  # Correlation configuration
    scenario_name=scenario_name,
    scenario_weight=scenario_weight,
)
```

**Example Year 1 Configuration (Apple Preset):**
```python
dist_config = {
    "type": "triangular",
    "a": -0.15,  # Minimum growth rate: -15%
    "b": 0.34,   # Maximum growth rate: 34%
    "c": 0.08,   # Mode (most likely): 8%
}
```

---

## Step 2: Main Simulation Function

**Location:** `app/core/simulate.py` (lines 72-139)

```python
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
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    
    return SimResults(
        paths=paths,              # Array of shape (n_sims, n_years)
        config=config,            # Original configuration
        seed_used=config.seed,    # Seed used for reproducibility
        timestamp=timestamp,
        parameter_hash="placeholder",
    )
```

---

## Step 3: Distribution Sampling (Year 1 Triangular)

**Location:** `app/core/distributions.py` (lines 36-42, 82-149)

### Triangular Distribution Sampler

```python
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
    from scipy import stats
    
    # Calculate triangular distribution parameters
    # c_normalized = (c - a) / (b - a)  # Mode position in [0, 1]
    # loc = a                            # Location (minimum)
    # scale = b - a                       # Scale (range)
    
    return stats.triang.rvs(
        c=(c - a) / (b - a),  # Normalized mode position
        loc=a,                 # Location parameter (minimum)
        scale=b - a,           # Scale parameter (range)
        size=size,             # Number of samples
        random_state=rng       # Random number generator for reproducibility
    )
```

### Unified Distribution Dispatcher

```python
def sample(
    dist_config: DistConfig, rng: Generator, size: int
) -> np.ndarray:
    """
    Unified distribution sampler dispatcher.
    
    This function routes to the appropriate sampling function based on
    the distribution type specified in dist_config.
    """
    dist_type = dist_config.get("type")
    
    if dist_type == "triangular":
        # For Year 1: a=-0.15, b=0.34, c=0.08
        samples = sample_triangular(
            rng, size, 
            dist_config["a"],  # Minimum
            dist_config["b"],  # Maximum
            dist_config["c"]   # Mode
        )
    elif dist_type == "normal":
        samples = sample_normal(rng, size, dist_config["mu"], dist_config["sigma"])
    # ... other distribution types ...
    
    # Apply bounds if specified
    min_bound = dist_config.get("min_bound")
    max_bound = dist_config.get("max_bound")
    if min_bound is not None or max_bound is not None:
        samples = clamp_values(samples, min_bound, max_bound)
    
    return samples
```

---

## Step 4: Random Number Generator Creation

**Location:** `app/core/simulate.py` (lines 17-33)

```python
def create_rng(seed: int | None) -> Generator:
    """
    Create a numpy.random.Generator with PCG64 algorithm.
    
    This ensures reproducibility: same seed → same random numbers → same results.
    """
    if seed is not None:
        return Generator(PCG64(seed))
    return Generator(PCG64())
```

---

## Step 5: Calling the Simulation

**Location:** `app/streamlit_app.py` (lines 336-342)

```python
# Run button clicked
if st.sidebar.button("▶️ Run Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        try:
            # Call the main simulation function
            results = simulate_growth_paths(config)
            
            # Store results in session state
            st.session_state["results"] = results
            st.session_state["config"] = config
            st.success("✅ Simulation completed!")
            
        except Exception as e:
            st.error(f"Simulation failed: {e}")
```

---

## Complete Example: Year 1 FCF Growth Sampling

Here's a complete example showing how Year 1 FCF growth rates are sampled:

```python
# 1. User input (Apple preset)
year1_dist_config = {
    "type": "triangular",
    "a": -0.15,  # Minimum: -15% growth
    "b": 0.34,   # Maximum: 34% growth
    "c": 0.08,   # Mode: 8% growth (most likely)
}

# 2. Configuration
config = SimConfig(
    n_sims=10000,  # Run 10,000 simulations
    seed=42,       # Reproducible seed
    years=[YearConfig(year=1, dist=year1_dist_config)],
    correlation=CorrelationConfig(enabled=False),
)

# 3. Create RNG
rng = create_rng(seed=42)

# 4. Sample Year 1 growth rates
# This calls: sample_triangular(rng, size=10000, a=-0.15, b=0.34, c=0.08)
year1_growth_rates = sample(year1_dist_config, rng, n_sims=10000)

# Result: year1_growth_rates is an array of 10,000 values
# Each value is a random sample from the triangular distribution
# Example values: [0.08, 0.12, -0.05, 0.25, 0.08, ...]
# These represent 10,000 possible Year 1 FCF growth rates
```

---

## Data Flow Diagram

```
User Inputs (Sidebar)
    ↓
Validated Parameters
    ↓
SimConfig Object
    ├── n_sims: 10,000
    ├── seed: 42
    └── years: [
          YearConfig(year=1, dist={"type": "triangular", "a": -0.15, "b": 0.34, "c": 0.08}),
          YearConfig(year=2, dist={"type": "triangular", "a": -0.15, "b": 0.41, "c": 0.15})
        ]
    ↓
simulate_growth_paths(config)
    ↓
create_rng(seed=42) → Generator(PCG64(42))
    ↓
For each year:
    sample(dist_config, rng, n_sims=10000)
        ↓
    sample_triangular(rng, size=10000, a=-0.15, b=0.34, c=0.08)
        ↓
    stats.triang.rvs(...) → Array of 10,000 random samples
    ↓
paths[:, 0] = [0.08, 0.12, -0.05, 0.25, ...]  # Year 1 growth rates
paths[:, 1] = [0.15, 0.20, -0.10, 0.30, ...]  # Year 2 growth rates
    ↓
apply_correlation(paths, correlation_matrix)  # If enabled
    ↓
SimResults(paths=paths, config=config, seed_used=42)
    ↓
DCF Valuation (if enabled)
    ↓
Enterprise Value Distribution
```

---

## Key Points

1. **Reproducibility**: The same seed always produces the same random numbers, ensuring identical results across runs.

2. **Vectorized Operations**: All sampling is done using NumPy arrays, making it fast even for 10,000+ simulations.

3. **Distribution Flexibility**: The `sample()` function dispatches to the appropriate sampler based on distribution type (triangular, normal, uniform, etc.).

4. **Correlation Support**: After sampling, correlation can be applied between years using Cholesky decomposition.

5. **Each Simulation Trial**: One row in the `paths` array represents one complete simulation trial with growth rates for all years.

---

## Appendix: Full Function Code

### `simulate_growth_paths()` - Complete Code

```python
def simulate_growth_paths(
    config: SimConfig, rng: Generator | None = None
) -> SimResults:
    """
    Simulate multi-year growth paths with optional correlation.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration
    rng : Generator or None
        Random number generator (if None, creates one from seed)

    Returns
    -------
    SimResults
        Simulation results container
    """
    if rng is None:
        rng = create_rng(config.seed)

    n_sims = config.n_sims
    n_years = len(config.years)

    # Initialize paths array: (n_sims, n_years)
    paths = np.zeros((n_sims, n_years))

    # Sample each year
    for i, year_config in enumerate(config.years):
        if year_config.deterministic_value is not None:
            # Use deterministic value
            paths[:, i] = year_config.deterministic_value
        else:
            # Sample from distribution (e.g., triangular for Year 1)
            paths[:, i] = sample(year_config.dist, rng, n_sims)

    # Apply correlation if enabled
    if config.correlation.enabled and n_years > 1:
        paths = apply_correlation(paths, config.correlation.matrix)

    # Create results
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    seed_used = config.seed if config.seed is not None else 0

    return SimResults(
        paths=paths,
        config=config,
        seed_used=seed_used,
        timestamp=timestamp,
        parameter_hash="placeholder",
    )
```

---

This code demonstrates the complete process of how validated user inputs are used to run the Monte Carlo Simulation, specifically showing how `simulate_growth_paths()` creates n possibilities for FCF growth using random sampling from the Year 1 FCF growth triangular distribution.

