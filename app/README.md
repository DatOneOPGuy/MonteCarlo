# Monte Carlo DCF / Growth Rate Uncertainty Simulator

A production-quality Streamlit application for modeling discounted cash flow (DCF) and growth-rate uncertainty using advanced Monte Carlo simulation techniques.

## Overview

This application provides a comprehensive toolkit for:
- **Multi-year growth path simulation** with optional correlation structures
- **7+ distribution types**: Normal, Lognormal, Student-t, Triangular, Uniform, Beta, and Custom (CSV)
- **Advanced analytics**: VaR, CVaR, sensitivity analysis, tornado diagrams
- **Interactive visualizations**: Distribution histograms, fan charts, ECDF, convergence plots
- **Reproducible results**: Centralized RNG with PCG64 algorithm and seed tracking
- **Export capabilities**: CSV data, JSON config, and parameter hashing

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Run the Application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`.

## Architecture

The application follows a modular, type-safe architecture:

```
app/
├── streamlit_app.py          # Main Streamlit application
├── core/
│   ├── types.py              # Type definitions (TypedDicts, dataclasses)
│   ├── validation.py         # Pydantic validation models
│   ├── distributions.py     # Distribution sampling functions
│   ├── simulate.py           # Simulation engine
│   └── analytics.py          # Statistical analytics
├── ui/
│   ├── controls.py           # Streamlit sidebar controls
│   └── plots.py              # Plotly visualization functions
└── tests/
    ├── test_distributions.py
    └── test_simulate.py
```

## Features

### Distributions

1. **Normal**: `N(μ, σ)` - Standard normal distribution
2. **Lognormal**: `LogN(μ, σ)` - For positive, skewed values
3. **Student-t**: `t(df, μ, σ)` - Fat-tailed distributions
4. **Triangular**: `Tri(a, b, c)` - Bounded with mode
5. **Uniform**: `U(low, high)` - Equal probability over range
6. **Beta**: `Beta(α, β)` - Flexible bounded distribution
7. **Custom**: Upload CSV or paste values for bootstrap sampling

### Correlation

- Optional correlation matrix between years
- Validated for symmetry and positive definiteness
- Applied via Cholesky decomposition
- Falls back to identity matrix if invalid

### Analytics

- **Summary Statistics**: Mean, median, std, skewness, kurtosis
- **Quantiles**: 1%, 5%, 25%, 50%, 75%, 95%, 99%
- **Risk Metrics**: VaR and CVaR at multiple confidence levels
- **Probabilities**: P(Loss) and P(Target)
- **Sensitivity**: One-at-a-time tornado analysis
- **PRCC**: Partial Rank Correlation Coefficients

### Visualizations

1. **Distribution Histogram + KDE**: Output distribution with kernel density estimate
2. **Fan Chart**: Multi-year quantile bands showing uncertainty over time
3. **ECDF**: Empirical cumulative distribution function
4. **Tornado Diagram**: Sensitivity analysis visualization
5. **Convergence Plot**: Cumulative mean vs. simulation number

## Usage Examples

### Basic Single-Year Simulation

1. Set **Number of years** to 1
2. Select **Year 1 Distribution** (e.g., Normal)
3. Enter parameters (e.g., μ=0.05, σ=0.02 for 5% growth with 2% volatility)
4. Set **Number of simulations** (e.g., 10,000)
5. Set **Random seed** (e.g., 42 for reproducibility)
6. Click **Run Simulation**

### Multi-Year with Correlation

1. Set **Number of years** to 2
2. Configure distributions for each year
3. Enable **Correlation between years**
4. Enter correlation matrix values (e.g., 0.7 for Year 1 ↔ Year 2)
5. Run simulation

### Custom Distribution

1. Select **Custom** distribution type
2. Either:
   - Upload a CSV file with one column of values
   - Paste comma/space-separated values
3. Run simulation

## Reproducibility

### Fixed Seed

Set a **Random seed** (e.g., 42) to ensure reproducible results. The same seed with the same parameters will produce identical simulation outputs.

### Parameter Hash

Each simulation run includes a parameter hash in the footer, computed from the configuration. This allows tracking and reproducing specific runs.

### Example: Reproduce Figure 1

To reproduce a specific visualization:

1. **Seed**: 42
2. **Years**: 2
3. **Year 1**: Normal(μ=0.05, σ=0.02)
4. **Year 2**: Normal(μ=0.06, σ=0.03)
5. **Correlation**: Enabled, matrix = [[1.0, 0.7], [0.7, 1.0]]
6. **Simulations**: 10,000
7. **LHS**: Disabled

This configuration will produce consistent results across runs.

## Distribution Selection Guide

### Normal
- **Use when**: Symmetric uncertainty, central limit theorem applies
- **Parameters**: μ (mean), σ (standard deviation)
- **Example**: Growth rate with symmetric uncertainty

### Lognormal
- **Use when**: Positive values, right-skewed (e.g., returns, prices)
- **Parameters**: μ (log-space mean), σ (log-space std)
- **Example**: Asset prices, revenue growth

### Student-t
- **Use when**: Fat tails, extreme events more likely
- **Parameters**: df (degrees of freedom), μ, σ
- **Example**: Financial returns during crises

### Triangular
- **Use when**: Bounded range with most likely value
- **Parameters**: a (min), b (max), c (mode)
- **Example**: Expert estimates with min/max/mode

### Uniform
- **Use when**: Equal probability over range, no mode
- **Parameters**: low, high
- **Example**: Complete uncertainty within bounds

### Beta
- **Use when**: Flexible bounded distribution, various shapes
- **Parameters**: α, β (shape), low, high (bounds)
- **Example**: Probabilities, percentages

## Best Practices

1. **Sample Size**: Use at least 1,000 simulations for basic analysis, 10,000+ for risk metrics
2. **LHS Sampling**: Enable for better coverage with fewer samples (especially for sensitivity analysis)
3. **Correlation**: Validate correlation matrices are positive definite before running
4. **Bounds**: Use min/max bounds to enforce realistic constraints (e.g., growth rates between -50% and +200%)
5. **Seed Management**: Use fixed seeds for reproducibility, random seeds for exploration

## Testing

Run the test suite:

```bash
pytest app/tests/ -v
```

Tests cover:
- Distribution sampling determinism
- Simulation reproducibility
- Shape and NaN validation
- Correlation application
- Analytics computations

## Type Checking

Run mypy for type checking:

```bash
mypy app/
```

Configuration is in `pyproject.toml` under `[tool.mypy]`.

## Limitations & Future Enhancements

### Current Limitations
- Correlation matrix input is manual (could be improved with matrix editor)
- Sensitivity analysis is simplified (one-at-a-time)
- PRCC computation is basic (could use more sophisticated methods)

### Nice-to-Haves (Future)
- Latin Hypercube Sampling (LHS) - ✅ Implemented
- Sobol sequence QMC option
- Bootstrap option for empirical returns
- Session-state scenario comparison (side-by-side)
- Dark mode charts synced to Streamlit theme
- DCF-specific workflow templates

## License

MIT License (see LICENSE file)

## Version

1.0.0

