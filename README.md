# Monte Carlo Simulation Toolkit

A production-quality Monte Carlo simulation toolkit with a modern, interactive UI and clear visualizations. This toolkit provides a minimal, fast framework for propagating uncertainty from input distributions through user-defined models using vectorized NumPy operations.

## What is Monte Carlo Simulation?

Monte Carlo simulation is a computational technique that uses random sampling to estimate the behavior of complex systems or compute numerical results. By repeatedly sampling from input probability distributions and evaluating a model, we can approximate the distribution of outputs and compute statistics like means, percentiles, and probabilities of events.

**Key principles:**
- **Law of Large Numbers**: As the number of trials increases, sample statistics converge to true values
- **Standard Error**: Uncertainty decreases as \(SE \approx \frac{s}{\sqrt{N}}\), where \(s\) is the sample standard deviation and \(N\) is the number of trials

## Quickstart

### Installation

```bash
pip install -e .
```

### Run the API

Start the FastAPI server:

```bash
uvicorn montecarlo.api:app --reload
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

### Run the UI

In a separate terminal, start the Streamlit dashboard:

```bash
streamlit run ui/streamlit_app.py
```

The UI will open in your browser with interactive controls and visualizations.

### CLI Examples

Estimate π:

```bash
python -m montecarlo.cli pi --n 200000 --seed 0
```

Run a linear model:

```bash
python -m montecarlo.cli linear --mu1 10 --sigma1 2 --mu2 1 --sigma2 0.25 --n 200000 --seed 1 --threshold 20
```

## Features

### Core Library

- **Vectorized operations**: All sampling and model evaluation uses NumPy for performance
- **Flexible distributions**: Normal, lognormal, uniform, Bernoulli, binomial
- **Statistics**: Mean, std, percentiles, standard error, confidence intervals
- **Risk analysis**: Probability of events with binomial standard errors

### REST API (FastAPI)

- `POST /simulate/linear`: Linear model with configurable parameters
- `POST /simulate/pi`: π estimation
- `POST /simulate/custom`: Custom models with expression evaluation
- Interactive OpenAPI documentation at `/docs`

### Interactive UI (Streamlit)

- **Sidebar controls**: Adjust simulation parameters in real-time
- **Visualizations**: Interactive Plotly charts (histogram + KDE, convergence plots, quantiles)
- **Metrics dashboard**: Summary statistics with metric cards
- **Multiple tabs**: Distribution, convergence, and detailed results

### CLI

- Quick command-line interface using Typer
- Pretty-printed results with Rich tables
- JSON output option for scripting

## Project Structure

```
montecarlo/
├── src/montecarlo/
│   ├── __init__.py
│   ├── core.py              # Simulation engine
│   ├── distributions.py     # Distribution samplers
│   ├── cli.py               # Command-line interface
│   ├── api.py               # FastAPI REST API
│   └── viz.py               # Plotly visualization helpers
├── ui/
│   ├── streamlit_app.py     # Streamlit dashboard
│   └── assets/
│       └── logo.svg
├── examples/
│   ├── estimate_pi.py       # π estimation example
│   └── portfolio_risk.py    # Portfolio risk analysis
├── notebooks/
│   └── intro_monte_carlo.ipynb  # Tutorial notebook
├── tests/
│   ├── test_core.py
│   ├── test_distributions.py
│   └── test_api.py
├── pyproject.toml
└── README.md
```

## Usage

### Python API

```python
from montecarlo import Simulation
from montecarlo.distributions import normal, lognormal
import numpy as np

# Define model
def model(x):
    return x[:, 0] + 2 * x[:, 1]

# Define sampler
def sampler(rng, size):
    x1 = normal(10.0, 2.0)(rng, size)
    x2 = lognormal(1.0, 0.25)(rng, size)
    return np.column_stack([x1, x2])

# Run simulation
sim = Simulation(model, sampler, seed=42)
result = sim.run(100_000, keep=True)

print(f"Mean: {result['mean']:.4f}")
print(f"Std: {result['std']:.4f}")
print(f"5th percentile: {result['p05']:.4f}")
print(f"95th percentile: {result['p95']:.4f}")

# Risk analysis
risk = sim.risk(lambda y: y > 20.0)
print(f"P(y > 20): {risk['p']:.4f}")
```

### REST API

```python
import requests

response = requests.post(
    "http://localhost:8000/simulate/linear",
    json={
        "mu1": 10.0,
        "sigma1": 2.0,
        "mu2": 1.0,
        "sigma2": 0.25,
        "n": 100000,
        "seed": 42,
        "threshold": 20.0
    }
)
print(response.json())
```

## Reproducibility

Set a `seed` parameter in the `Simulation` constructor or API requests to ensure reproducible results. The same seed will produce identical random sequences.

## Accuracy vs. Number of Trials

The standard error of the mean estimate decreases as \(SE = \frac{s}{\sqrt{N}}\), where:
- \(s\) is the sample standard deviation
- \(N\) is the number of trials

**Rule of thumb**: To halve the standard error, quadruple the number of trials.

## Limitations & Tips

1. **Vectorize your model**: Avoid Python loops inside the model function. Use NumPy operations that work on entire arrays.
2. **Pass RNG explicitly**: For advanced use cases, you can pass a custom `numpy.random.Generator` to samplers.
3. **Check shapes**: The sampler must return a 2D array of shape `(n, k)` where `k` is the number of input variables. The model must return a 1D array of shape `(n,)`.
4. **Large N warnings**: For very large simulations (>5M trials), consider using smaller batches or increasing memory limits.
5. **CI/CD**: Run `pytest` to ensure tests pass before deploying.

## Testing

Run the test suite:

```bash
pytest -q
```

All tests should pass with deterministic results when using fixed seeds.

## Examples

See the `examples/` directory for:
- **estimate_pi.py**: Classic π estimation
- **portfolio_risk.py**: Portfolio value simulation with VaR/CVaR computation

## Notebook

See `notebooks/intro_monte_carlo.ipynb` for a tutorial with convergence analysis and visualizations.

## License

MIT License (see LICENSE file)

## Screenshots

The Streamlit UI provides:
- Interactive parameter controls in the sidebar
- Real-time metric cards showing summary statistics
- Distribution tab with histogram + KDE overlay
- Convergence tab showing cumulative mean vs. iterations
- Details tab with JSON summary and CSV download

---

**Version**: 0.1.0  
**Python**: 3.10+  
**Dependencies**: numpy, scipy, plotly, pandas, typer, rich, fastapi, uvicorn, pydantic, streamlit, pytest


