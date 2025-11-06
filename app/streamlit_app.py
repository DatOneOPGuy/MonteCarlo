"""
Main Streamlit application for Monte Carlo DCF/Growth Rate Uncertainty Simulator.

Provides interactive UI for configuring and running simulations with
advanced analytics and visualizations.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import streamlit as st

from app.core.analytics import compute_analytics
from app.core.simulate import simulate_growth_paths, simulate_with_lhs
from app.core.types import SimConfig
from app.core.validation import compute_parameter_hash
from app.ui.controls import (
    render_correlation_controls,
    render_distribution_selector,
    render_scenario_controls,
    render_simulation_controls,
)
from app.ui.plots import (
    plot_convergence,
    plot_distribution_histogram,
    plot_ecdf,
    plot_fan_chart,
    plot_quantile_band,
    plot_tornado,
)

# Page configuration
st.set_page_config(
    page_title="Monte Carlo DCF Simulator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cached_simulation(
    config_dict: dict, use_lhs: bool
) -> tuple[np.ndarray, int, str, str]:
    """
    Cached simulation function.

    Parameters
    ----------
    config_dict : dict
        Serialized configuration
    use_lhs : bool
        Whether to use LHS sampling

    Returns
    -------
    tuple
        (paths, seed_used, timestamp, parameter_hash)
    """
    # Reconstruct config (simplified - in practice would deserialize properly)
    # For now, return placeholder
    return np.array([]), 0, "", ""


def main() -> None:
    """Main application entry point."""
    st.title("📊 Monte Carlo DCF / Growth Rate Uncertainty Simulator")
    st.markdown(
        "Model discounted cash flow and growth-rate uncertainty with advanced analytics"
    )

    # Sidebar controls
    st.sidebar.header("Simulation Configuration")

    # Number of years
    n_years = st.sidebar.number_input(
        "Number of years",
        min_value=1,
        max_value=10,
        value=2,
        help="Number of years to simulate",
    )

    # Simulation controls
    n_sims, seed, use_lhs = render_simulation_controls()

    # Scenario controls
    scenario_name, scenario_weight = render_scenario_controls()

    # Correlation
    correlation = render_correlation_controls(n_years)

    # Build year configurations
    st.sidebar.subheader("Year Configurations")
    year_configs = []
    for i in range(n_years):
        with st.sidebar.expander(f"Year {i+1}", expanded=i==0):
            dist_config = render_distribution_selector(i + 1)
            deterministic = st.checkbox(
                f"Use deterministic value for Year {i+1}",
                value=False,
                key=f"det_{i}",
            )
            det_value = None
            if deterministic:
                det_value = st.number_input(
                    f"Year {i+1} value",
                    value=0.05,
                    step=0.01,
                    format="%.3f",
                    key=f"det_val_{i}",
                )
            year_configs.append((dist_config, det_value))

    # Build configuration
    try:
        from app.core.types import YearConfig

        years = []
        for i, (dist_config, det_value) in enumerate(year_configs):
            year_config = YearConfig(year=i + 1, dist=dist_config, deterministic_value=det_value)
            years.append(year_config)

        config = SimConfig(
            n_sims=n_sims,
            seed=seed,
            years=years,
            correlation=correlation,
            scenario_name=scenario_name,
            scenario_weight=scenario_weight,
        )
    except Exception as e:
        st.sidebar.error(f"Configuration error: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    # Run button
    if st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running simulation..."):
            try:
                if use_lhs:
                    results = simulate_with_lhs(config)
                else:
                    results = simulate_growth_paths(config)

                # Compute parameter hash
                config_dict = {
                    "n_sims": config.n_sims,
                    "seed": config.seed,
                    "n_years": len(config.years),
                }
                param_hash = compute_parameter_hash(config_dict)

                # Store in session state
                st.session_state["results"] = results
                st.session_state["config"] = config
                st.session_state["param_hash"] = param_hash
                st.success("✅ Simulation completed!")

            except Exception as e:
                st.error(f"Simulation failed: {e}")
                import traceback

                st.code(traceback.format_exc())

    # Display results
    if "results" in st.session_state:
        results = st.session_state["results"]
        config = st.session_state["config"]

        # Compute analytics
        var_alphas = [0.01, 0.05, 0.10]
        loss_threshold = st.sidebar.number_input(
            "Loss threshold",
            value=0.0,
            step=0.01,
            format="%.3f",
            help="Threshold for probability of loss calculation",
        )
        target_value = st.sidebar.number_input(
            "Target value",
            value=None,
            step=0.01,
            format="%.3f",
            help="Target value for probability calculation",
        )

        analytics = compute_analytics(
            results,
            var_alphas=var_alphas,
            loss_threshold=loss_threshold if loss_threshold != 0 else None,
            target_value=target_value,
            compute_sensitivity=True,
        )

        # Metrics cards
        st.header("Summary Statistics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Mean", f"{analytics.summary_stats['mean']:.4f}")
        with col2:
            st.metric("Median", f"{analytics.summary_stats['median']:.4f}")
        with col3:
            st.metric("Std Dev", f"{analytics.summary_stats['std']:.4f}")
        with col4:
            st.metric("Skewness", f"{analytics.summary_stats['skew']:.4f}")
        with col5:
            st.metric("Kurtosis", f"{analytics.summary_stats['kurtosis']:.4f}")

        # Risk metrics
        st.subheader("Risk Metrics")
        risk_col1, risk_col2, risk_col3 = st.columns(3)

        with risk_col1:
            st.metric("VaR (95%)", f"{analytics.var.get(0.05, 0):.4f}")
        with risk_col2:
            st.metric("CVaR (95%)", f"{analytics.cvar.get(0.05, 0):.4f}")
        with risk_col3:
            if analytics.prob_loss is not None:
                st.metric("P(Loss)", f"{analytics.prob_loss:.4f}")

        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 Distribution", "📈 Fan Chart", "📉 ECDF", "🌪️ Tornado", "🔄 Convergence"]
        )

        with tab1:
            # Use final year or aggregate
            if results.paths.shape[1] == 1:
                plot_data = results.paths[:, 0]
            else:
                plot_data = results.paths[:, -1]

            fig = plot_distribution_histogram(plot_data, title="Output Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # Quantile band
            fig2 = plot_quantile_band(plot_data, analytics.quantiles)
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            if results.paths.shape[1] > 1:
                fig = plot_fan_chart(results, title="Multi-Year Fan Chart")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Fan chart requires multiple years")

        with tab3:
            if results.paths.shape[1] == 1:
                plot_data = results.paths[:, 0]
            else:
                plot_data = results.paths[:, -1]
            fig = plot_ecdf(plot_data, title="Empirical Cumulative Distribution")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            fig = plot_tornado(analytics, title="Sensitivity Analysis (Tornado)")
            st.plotly_chart(fig, use_container_width=True)

        with tab5:
            fig = plot_convergence(results, title="Convergence Analysis")
            st.plotly_chart(fig, use_container_width=True)

        # Data export
        st.subheader("Export Results")
        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            # CSV export
            df = pd.DataFrame(results.paths, columns=[f"Year {i+1}" for i in range(results.paths.shape[1])])
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        with export_col2:
            # JSON export (config)
            config_dict = {
                "n_sims": config.n_sims,
                "seed": config.seed,
                "scenario": config.scenario_name,
            }
            json_str = json.dumps(config_dict, indent=2)
            st.download_button(
                label="📥 Download Config (JSON)",
                data=json_str,
                file_name=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        # Footer with metadata
        st.markdown("---")
        st.caption(
            f"Seed: {results.seed_used} | "
            f"Parameter Hash: {st.session_state.get('param_hash', 'N/A')} | "
            f"Timestamp: {results.timestamp}"
        )

    else:
        st.info("👈 Configure simulation parameters in the sidebar and click 'Run Simulation'")


if __name__ == "__main__":
    main()

