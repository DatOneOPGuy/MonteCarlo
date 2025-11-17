"""
Streamlit UI controls for simulation configuration.

Provides sidebar widgets for distribution parameters, correlation,
scenarios, and export options.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.core.types import CorrelationConfig, DistConfig, ScenarioType, SimConfig, YearConfig
from app.core.validation import CorrelationMatrix, validate_distribution_params


def render_distribution_selector(year: int, default_type: str = "normal") -> DistConfig:
    """
    Render distribution selector and parameter inputs for a year.

    Parameters
    ----------
    year : int
        Year number (1-indexed)
    default_type : str
        Default distribution type

    Returns
    -------
    DistConfig
        Distribution configuration dictionary
    """
    dist_type = st.selectbox(
        f"Year {year} Distribution",
        ["normal", "lognormal", "student_t", "triangular", "uniform", "beta", "custom"],
        index=["normal", "lognormal", "student_t", "triangular", "uniform", "beta", "custom"].index(default_type),
        key=f"dist_type_{year}",
        help=f"Select distribution type for Year {year} growth rate",
    )

    dist_config: DistConfig = {"type": dist_type}

    if dist_type == "normal":
        dist_config["mu"] = st.number_input(
            "μ (mean)",
            value=0.05,
            step=0.01,
            format="%.3f",
            key=f"normal_mu_{year}",
            help="Mean growth rate (e.g., 0.05 for 5%)",
        )
        dist_config["sigma"] = st.number_input(
            "σ (std dev)",
            min_value=0.001,
            value=0.02,
            step=0.01,
            format="%.3f",
            key=f"normal_sigma_{year}",
            help="Standard deviation of growth rate",
        )

    elif dist_type == "lognormal":
        dist_config["mu"] = st.number_input(
            "μ (log-space mean)",
            value=0.04,
            step=0.01,
            format="%.3f",
            key=f"lognormal_mu_{year}",
            help="Mean in log space",
        )
        dist_config["sigma"] = st.number_input(
            "σ (log-space std)",
            min_value=0.001,
            value=0.15,
            step=0.01,
            format="%.3f",
            key=f"lognormal_sigma_{year}",
            help="Standard deviation in log space",
        )

    elif dist_type == "student_t":
        dist_config["df"] = st.number_input(
            "Degrees of freedom",
            min_value=1.0,
            value=5.0,
            step=1.0,
            key=f"student_df_{year}",
            help="Degrees of freedom (lower = fatter tails)",
        )
        dist_config["mu"] = st.number_input(
            "μ (location)",
            value=0.05,
            step=0.01,
            format="%.3f",
            key=f"student_mu_{year}",
        )
        dist_config["sigma"] = st.number_input(
            "σ (scale)",
            min_value=0.001,
            value=0.02,
            step=0.01,
            format="%.3f",
            key=f"student_sigma_{year}",
        )

    elif dist_type == "triangular":
        st.markdown("**Triangular Distribution Parameters**")
        st.caption("The mode (c) is the most likely value, creating a peak in the distribution")
        
        # Display in columns for better visibility
        col_a, col_c, col_b = st.columns(3)
        
        with col_a:
            dist_config["a"] = st.number_input(
                "Lower bound (a)",
                value=-0.10,
                step=0.01,
                format="%.3f",
                key=f"tri_a_{year}",
                help="Minimum value (left bound)",
            )
        
        with col_c:
            dist_config["c"] = st.number_input(
                "Mode (c)",
                value=0.05,
                step=0.01,
                format="%.3f",
                key=f"tri_c_{year}",
                help="Most likely value (peak of distribution)",
            )
        
        with col_b:
            dist_config["b"] = st.number_input(
                "Upper bound (b)",
                value=0.20,
                step=0.01,
                format="%.3f",
                key=f"tri_b_{year}",
                help="Maximum value (right bound)",
            )
        
        # Validation message
        if dist_config["a"] <= dist_config["c"] <= dist_config["b"]:
            st.success(f"✓ Valid: {dist_config['a']:.3f} ≤ {dist_config['c']:.3f} ≤ {dist_config['b']:.3f}")
        else:
            st.error(f"⚠ Invalid: Must have a ≤ c ≤ b")

    elif dist_type == "uniform":
        dist_config["low"] = st.number_input(
            "Lower bound",
            value=-0.05,
            step=0.01,
            format="%.3f",
            key=f"uniform_low_{year}",
        )
        dist_config["high"] = st.number_input(
            "Upper bound",
            value=0.15,
            step=0.01,
            format="%.3f",
            key=f"uniform_high_{year}",
        )

    elif dist_type == "beta":
        dist_config["alpha"] = st.number_input(
            "α (alpha)",
            min_value=0.001,
            value=2.0,
            step=0.1,
            key=f"beta_alpha_{year}",
        )
        dist_config["beta"] = st.number_input(
            "β (beta)",
            min_value=0.001,
            value=2.0,
            step=0.1,
            key=f"beta_beta_{year}",
        )
        dist_config["low"] = st.number_input(
            "Lower bound",
            value=-0.10,
            step=0.01,
            format="%.3f",
            key=f"beta_low_{year}",
        )
        dist_config["high"] = st.number_input(
            "Upper bound",
            value=0.20,
            step=0.01,
            format="%.3f",
            key=f"beta_high_{year}",
        )

    elif dist_type == "custom":
        uploaded_file = st.file_uploader(
            f"Upload CSV for Year {year}",
            type=["csv"],
            key=f"custom_csv_{year}",
            help="CSV file with one column of values",
        )
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if df.shape[1] != 1:
                    st.error("CSV must have exactly one column")
                    dist_config["values"] = np.array([])
                else:
                    values = df.iloc[:, 0].values
                    values = values[~np.isnan(values)]
                    dist_config["values"] = values
                    st.success(f"Loaded {len(values)} values")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                dist_config["values"] = np.array([])
        else:
            # Allow paste
            paste_text = st.text_area(
                f"Or paste values (comma/space separated) for Year {year}",
                key=f"custom_paste_{year}",
            )
            if paste_text:
                try:
                    values = np.array([float(x.strip()) for x in paste_text.replace(",", " ").split()])
                    dist_config["values"] = values
                    st.success(f"Loaded {len(values)} values")
                except Exception as e:
                    st.error(f"Error parsing values: {e}")
                    dist_config["values"] = np.array([])
            else:
                dist_config["values"] = np.array([])

    # Optional bounds
    use_bounds = st.checkbox(
        f"Apply bounds for Year {year}",
        value=False,
        key=f"use_bounds_{year}",
    )
    if use_bounds:
        dist_config["min_bound"] = st.number_input(
            "Min bound",
            value=None,
            key=f"min_bound_{year}",
        )
        dist_config["max_bound"] = st.number_input(
            "Max bound",
            value=None,
            key=f"max_bound_{year}",
        )

    # Validate
    try:
        validate_distribution_params(dist_type, {k: v for k, v in dist_config.items() if k != "type"})
    except Exception as e:
        st.warning(f"Validation warning: {e}")

    return dist_config


def render_correlation_controls(n_years: int) -> CorrelationConfig:
    """Render correlation matrix input controls."""
    enabled = st.checkbox(
        "Enable correlation between years",
        value=False,
        help="Apply correlation structure across years using Cholesky decomposition",
    )

    if not enabled:
        return CorrelationConfig(matrix=np.eye(n_years), enabled=False)

    st.markdown("**Correlation Matrix**")
    st.caption("Enter correlation values (must be symmetric, positive definite)")

    # Build matrix input
    matrix_input = []
    for i in range(n_years):
        row = []
        for j in range(n_years):
            if i == j:
                value = 1.0
            else:
                value = st.number_input(
                    f"Year {i+1} ↔ Year {j+1}",
                    min_value=-1.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    format="%.2f",
                    key=f"corr_{i}_{j}",
                )
            row.append(value)
        matrix_input.append(row)

    matrix = np.array(matrix_input)

    # Validate
    try:
        CorrelationMatrix(matrix=matrix_input.tolist())
        st.success("✓ Valid correlation matrix")
    except Exception as e:
        st.error(f"Invalid correlation matrix: {e}")
        st.info("Falling back to identity matrix (no correlation)")
        matrix = np.eye(n_years)

    return CorrelationConfig(matrix=matrix, enabled=enabled)


def render_simulation_controls() -> tuple[int, int | None, bool]:
    """Render basic simulation controls."""
    n_sims = st.number_input(
        "Number of simulations",
        min_value=100,
        max_value=10_000_000,
        value=10_000,
        step=1_000,
        help="More simulations = better accuracy but slower",
    )

    seed = st.number_input(
        "Random seed",
        min_value=0,
        value=42,
        help="Set seed for reproducibility (0 = random)",
    )
    seed = None if seed == 0 else int(seed)

    use_lhs = st.checkbox(
        "Use Latin Hypercube Sampling (LHS)",
        value=False,
        help="LHS provides better coverage with fewer samples",
    )

    return n_sims, seed, use_lhs


def render_scenario_controls() -> tuple[ScenarioType, float]:
    """Render scenario selection and weighting controls."""
    scenario = st.selectbox(
        "Scenario",
        ["base", "bull", "bear"],
        help="Select scenario type",
    )

    weight = st.slider(
        "Scenario weight",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Weight for blending scenarios",
    )

    return scenario, weight


def build_sim_config(
    n_years: int,
    n_sims: int,
    seed: int | None,
    correlation: CorrelationConfig,
    scenario_name: str = "base",
    scenario_weight: float = 1.0,
    year_configs: list[tuple[DistConfig, float | None]] | None = None,
) -> SimConfig:
    """
    Build simulation configuration from UI state.

    Parameters
    ----------
    n_years : int
        Number of years to simulate
    n_sims : int
        Number of simulations
    seed : int or None
        Random seed
    correlation : CorrelationConfig
        Correlation configuration
    scenario_name : str
        Scenario name
    scenario_weight : float
        Scenario weight
    year_configs : list of tuples or None
        Pre-computed (dist_config, deterministic_value) tuples for each year

    Returns
    -------
    SimConfig
        Complete simulation configuration
    """
    years = []
    if year_configs is None:
        # Render controls for each year
        for i in range(n_years):
            dist_config = render_distribution_selector(i + 1)
            year_config = YearConfig(year=i + 1, dist=dist_config)
            years.append(year_config)
    else:
        # Use provided configs
        for i, (dist_config, det_value) in enumerate(year_configs):
            year_config = YearConfig(
                year=i + 1, dist=dist_config, deterministic_value=det_value
            )
            years.append(year_config)

    return SimConfig(
        n_sims=n_sims,
        seed=seed,
        years=years,
        correlation=correlation,
        scenario_name=scenario_name,
        scenario_weight=scenario_weight,
    )

