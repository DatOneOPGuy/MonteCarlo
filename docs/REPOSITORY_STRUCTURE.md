# Repository Structure

This document describes the organization of the Monte Carlo repository after cleanup.

## Directory Structure

```
MonteCarlo/
├── .gitignore              # Git ignore rules for generated files
├── README.md               # Main project README
├── LICENSE                 # MIT License
├── pyproject.toml          # Python project configuration
├── run_services.sh         # Service startup script
│
├── src/                    # Core Monte Carlo library (installable package)
│   └── montecarlo/
│       ├── __init__.py
│       ├── core.py         # Simulation engine
│       ├── distributions.py
│       ├── cli.py          # CLI interface
│       ├── api.py          # FastAPI REST API
│       └── viz.py          # Visualization helpers
│
├── app/                    # DCF Valuation Application
│   ├── streamlit_app.py    # Main DCF Streamlit app
│   ├── mc_dcf_apple.py     # Apple-specific DCF code
│   ├── core/               # DCF application core modules
│   │   ├── types.py
│   │   ├── distributions.py
│   │   ├── simulate.py
│   │   ├── analytics.py
│   │   ├── dcf.py
│   │   └── validation.py
│   ├── ui/                 # DCF UI components
│   │   ├── controls.py
│   │   └── plots.py
│   └── tests/              # DCF application tests
│       ├── test_distributions.py
│       └── test_simulate.py
│
├── ui/                     # General Monte Carlo Toolkit UI
│   ├── streamlit_app.py    # Basic Streamlit dashboard
│   └── assets/
│       └── logo.svg
│
├── tests/                  # Core library tests
│   ├── test_core.py
│   ├── test_distributions.py
│   └── test_api.py
│
├── examples/               # Example scripts
│   ├── estimate_pi.py
│   ├── portfolio_risk.py
│   └── test_triangular.py
│
├── notebooks/              # Jupyter notebooks
│   └── intro_monte_carlo.ipynb
│
├── docs/                   # Documentation
│   ├── README.md           # Documentation index
│   ├── assets/             # Images and diagrams
│   │   ├── ev_distribution.png
│   │   └── triangular_distribution_test.png
│   ├── APPLE_PROJECT_GUIDE.md
│   ├── DCF_CODE_GUIDE.md
│   ├── METHODOLOGY_EXPLANATION.md
│   ├── MONTE_CARLO_SIMULATION_CODE.md
│   ├── GITHUB_SETUP_TUTORIAL.md
│   ├── CURSOR_PROMPT.md
│   ├── DCF_VALUATION_CODE.py    # Example code
│   └── SIMULATION_CODE.py       # Example code
│
├── data/                   # Output data files (gitignored)
│   ├── .gitkeep
│   └── mc_dcf_results.csv
│
└── venv/                   # Virtual environment (gitignored)
```

## Key Changes Made

### 1. Created `.gitignore`
- Excludes `__pycache__/`, `*.pyc`, `*.egg-info/`
- Excludes `venv/` and other virtual environments
- Excludes output files (CSV, PNG, etc.) except in `docs/` and `examples/`
- Excludes IDE and OS-specific files

### 2. Organized Documentation
- Created `docs/` directory for all documentation
- Moved all markdown files from `app/` to `docs/`
- Moved example code files (`DCF_VALUATION_CODE.py`, `SIMULATION_CODE.py`) to `docs/`
- Created `docs/assets/` for images and diagrams

### 3. Organized Assets
- Moved image files (`ev_distribution.png`, `triangular_distribution_test.png`) to `docs/assets/`
- Kept UI assets in their respective directories (`ui/assets/`, `app/ui/`)

### 4. Organized Output Files
- Created `data/` directory for output CSV files
- Added `.gitkeep` to preserve directory structure
- Output files are gitignored but directory structure is preserved

### 5. Organized Examples
- Moved `test_triangular.py` from `app/` to `examples/` (it's a test/example script)

### 6. Cleaned Up Generated Files
- Removed all `__pycache__/` directories
- Removed `.egg-info/` directories

### 7. Updated README
- Updated main `README.md` to reflect new structure
- Clarified the two different Streamlit applications
- Added links to documentation

## Two Main Applications

### 1. Core Monte Carlo Toolkit
- **Location**: `src/montecarlo/` + `ui/streamlit_app.py`
- **Purpose**: General-purpose Monte Carlo simulation library
- **Run**: `streamlit run ui/streamlit_app.py`

### 2. DCF Valuation Application
- **Location**: `app/`
- **Purpose**: Specialized DCF valuation with advanced Monte Carlo simulation
- **Run**: `streamlit run app/streamlit_app.py`
- **Documentation**: See `docs/README.md` and other files in `docs/`

## Testing

- **Core library tests**: `tests/` directory
- **DCF application tests**: `app/tests/` directory
- Run all tests: `pytest`

## Best Practices

1. **Documentation**: Add new documentation to `docs/`
2. **Examples**: Add example scripts to `examples/`
3. **Output files**: Save output data to `data/` (gitignored)
4. **Images**: Place documentation images in `docs/assets/`
5. **Tests**: Keep tests close to the code they test

