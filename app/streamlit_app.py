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
from app.core.dcf import dcf_valuation_from_results, calculate_equity_value
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
    plot_distribution_preview,
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
        "Model discounted cash flow and growth-rate uncertainty with advanced analytics. "
        "**Perfect for Apple valuation analysis!**"
    )

    # Sidebar controls
    st.sidebar.header("Simulation Configuration")

    # DCF Mode toggle
    dcf_mode = st.sidebar.checkbox(
        "DCF Valuation Mode",
        value=False,
        help="Enable DCF valuation with terminal value calculation",
    )
    
    # Apple Project Preset
    apple_preset = st.sidebar.checkbox(
        "Use Apple Project Presets",
        value=False,
        help="Load Apple-specific parameters (FCF, WACC, triangular distributions)",
    )
    
    # Show Apple preset info if enabled
    if apple_preset and dcf_mode:
        st.info(
            "🍎 **Apple Project Preset Active**: Using FY2025 FCF ($105.0B), WACC (8.21%), "
            "and triangular growth distributions for Year 1 & Year 2."
        )

    # Number of years
    n_years = st.sidebar.number_input(
        "Number of years",
        min_value=1,
        max_value=10,
        value=2,
        help="Number of years to simulate",
    )

    # DCF-specific inputs
    if dcf_mode:
        st.sidebar.subheader("DCF Parameters")
        
        # Apple preset values
        if apple_preset:
            preset_fcf = 105.0  # FY2025 FCF in billions
            preset_wacc = 0.0821  # 8.21%
            preset_terminal = 0.03  # 3%
        else:
            preset_fcf = 100.0
            preset_wacc = 0.0766
            preset_terminal = 0.03
        
        initial_fcf = st.sidebar.number_input(
            "Initial Free Cash Flow (Year 0, Billions USD)",
            min_value=0.0,
            value=preset_fcf,
            step=1.0,
            format="%.3f",
            help="Free cash flow in Year 0 (before projections)",
        )
        wacc = st.sidebar.number_input(
            "WACC (Discount Rate)",
            min_value=0.0,
            max_value=1.0,
            value=preset_wacc,
            step=0.0001,
            format="%.4f",
            help="Weighted Average Cost of Capital",
        )
        terminal_method = st.sidebar.radio(
            "Terminal Value Method",
            ["Perpetual Growth", "Terminal Multiple", "None"],
            help="Method for calculating terminal value",
        )
        terminal_growth = None
        terminal_multiple = None
        if terminal_method == "Perpetual Growth":
            terminal_growth = st.sidebar.number_input(
                "Terminal Growth Rate",
                min_value=-0.1,
                max_value=0.1,
                value=preset_terminal if apple_preset else 0.03,
                step=0.001,
                format="%.3f",
                help="Perpetual growth rate (Gordon Growth Model)",
            )
        elif terminal_method == "Terminal Multiple":
            terminal_multiple = st.sidebar.number_input(
                "Terminal Multiple",
                min_value=0.0,
                value=15.0,
                step=0.5,
                help="Terminal multiple (e.g., EV/EBITDA)",
            )
        
        # Market cap for comparison
        market_cap = st.sidebar.number_input(
            "Market Cap (Billions USD, Optional)",
            min_value=0.0,
            value=None,
            step=10.0,
            format="%.2f",
            help="Apple's market cap for comparison (set to None to disable)",
        )
        if market_cap == 0:
            market_cap = None

        # Equity value calculation (optional, hidden by default)
        calculate_equity = False  # Disabled - focus on Enterprise Value only
        # Uncomment below if you want to enable per-share calculations
        # st.sidebar.subheader("Equity Value (Optional)")
        # calculate_equity = st.sidebar.checkbox("Calculate Equity Value per Share", value=False)
        if False:  # calculate_equity:
            cash = st.sidebar.number_input(
                "Cash & Equivalents",
                value=0.0,
                step=1_000_000_000.0,
                format="%.0f",
            )
            debt = st.sidebar.number_input(
                "Total Debt",
                value=0.0,
                step=1_000_000_000.0,
                format="%.0f",
            )
            shares = st.sidebar.number_input(
                "Shares Outstanding",
                min_value=0.0,
                value=1.0,
                step=0.1,
                format="%.2f",
            )
        else:
            cash = debt = shares = 0.0

    # Simulation controls
    n_sims, seed, use_lhs = render_simulation_controls()

    # Scenario controls
    scenario_name, scenario_weight = render_scenario_controls()

    # Correlation - auto-enable with 0.6 for Apple preset
    if apple_preset and dcf_mode and n_years >= 2:
        # Pre-set correlation for Apple preset
        st.sidebar.info("🍎 **Apple Preset**: Correlation automatically enabled (Year 1 ↔ Year 2 = 0.6)")
    correlation = render_correlation_controls(n_years, apple_preset=apple_preset and dcf_mode)

    # Build year configurations
    st.sidebar.subheader("Year Configurations")
    year_configs = []
    for i in range(n_years):
        with st.sidebar.expander(f"Year {i+1}", expanded=i==0):
            # Apple preset: use triangular distributions
            if apple_preset and dcf_mode and i < 2:
                if i == 0:
                    # Year 1: min=-0.15, mode=0.08, max=0.34
                    st.markdown("**🍎 Apple Preset: Year 1 (FY2026) Growth Rate**")
                    st.markdown("**Triangular Distribution Parameters:**")
                    col_a, col_c, col_b = st.columns(3)
                    with col_a:
                        st.metric("Min (a)", "-0.15", help="Lower bound: -15% growth")
                    with col_c:
                        st.metric("Mode (c)", "0.08", help="Most likely: 8% growth")
                    with col_b:
                        st.metric("Max (b)", "0.34", help="Upper bound: 34% growth")
                    
                    dist_config = {
                        "type": "triangular",
                        "a": -0.15,
                        "b": 0.34,
                        "c": 0.08,
                    }
                else:
                    # Year 2: min=-0.15, mode=0.15, max=0.41
                    st.markdown("**🍎 Apple Preset: Year 2 (FY2027) Growth Rate**")
                    st.markdown("**Triangular Distribution Parameters:**")
                    col_a, col_c, col_b = st.columns(3)
                    with col_a:
                        st.metric("Min (a)", "-0.15", help="Lower bound: -15% growth")
                    with col_c:
                        st.metric("Mode (c)", "0.15", help="Most likely: 15% growth")
                    with col_b:
                        st.metric("Max (b)", "0.41", help="Upper bound: 41% growth")
                    
                    dist_config = {
                        "type": "triangular",
                        "a": -0.15,
                        "b": 0.41,
                        "c": 0.15,
                    }
                # Show preset values but allow override
                st.markdown("---")
                override = st.checkbox(f"Override Year {i+1} preset", value=False, key=f"override_{i}")
                if override:
                    dist_config = render_distribution_selector(i + 1)
            else:
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

    # Distribution Preview Section (moved to sidebar, but will also show in main area)
    st.sidebar.markdown("---")
    show_preview = st.sidebar.checkbox(
        "Show Distribution Previews",
        value=True,
        help="Preview what each year's distribution looks like based on current parameters",
    )

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
                st.session_state["dcf_mode"] = dcf_mode
                if dcf_mode:
                    st.session_state["initial_fcf"] = initial_fcf
                    st.session_state["wacc"] = wacc
                    st.session_state["terminal_growth"] = terminal_growth
                    st.session_state["terminal_multiple"] = terminal_multiple
                    st.session_state["market_cap"] = market_cap
                    st.session_state["calculate_equity"] = calculate_equity
                    st.session_state["cash"] = cash
                    st.session_state["debt"] = debt
                    st.session_state["shares"] = shares
                    st.session_state["apple_preset"] = apple_preset
                st.success("✅ Simulation completed!")

            except Exception as e:
                st.error(f"Simulation failed: {e}")
                import traceback

                st.code(traceback.format_exc())

    # Display results
    if "results" in st.session_state:
        results = st.session_state["results"]
        config = st.session_state["config"]
        
        # Get DCF mode from session state if available, otherwise from current state
        dcf_mode_active = st.session_state.get("dcf_mode", dcf_mode)
        
        # Initialize DCF variables with defaults
        initial_fcf = st.session_state.get("initial_fcf", 105.0)  # Billions
        wacc = st.session_state.get("wacc", 0.0821)
        terminal_growth = st.session_state.get("terminal_growth", None)
        terminal_multiple = st.session_state.get("terminal_multiple", None)
        market_cap = st.session_state.get("market_cap", None)
        calculate_equity = st.session_state.get("calculate_equity", False)
        cash = st.session_state.get("cash", 0.0)
        debt = st.session_state.get("debt", 0.0)
        shares = st.session_state.get("shares", 0.0)
        apple_preset_used = st.session_state.get("apple_preset", False)

        # DCF valuation if enabled
        if dcf_mode_active:
            # Convert initial_fcf to actual value (it's stored in billions)
            # The growth rates from simulation are already in the right format
            dcf_values, dcf_analytics = dcf_valuation_from_results(
                results,
                initial_fcf=initial_fcf,  # Already in billions
                wacc=wacc,
                terminal_growth=terminal_growth,
                terminal_multiple=terminal_multiple,
                fcf_in_billions=True,  # All values in billions
            )

            # Calculate equity value if requested
            if calculate_equity and shares > 0:
                equity_values = np.array([
                    calculate_equity_value(dcf_val, cash, debt, shares)
                    for dcf_val in dcf_values
                ])
                equity_mean = np.mean(equity_values)
                equity_median = np.median(equity_values)
                equity_p05 = np.percentile(equity_values, 5)
                equity_p95 = np.percentile(equity_values, 95)
            else:
                equity_values = None
                equity_mean = equity_median = equity_p05 = equity_p95 = None

            # Display DCF results - Enterprise Value is the primary output
            st.header("📊 DCF Valuation Results")
            st.subheader("Enterprise Value (Billions USD)")
            
            # Show distribution parameters used
            if apple_preset_used:
                st.markdown("**Growth Rate Distributions Used:**")
                dist_col1, dist_col2 = st.columns(2)
                with dist_col1:
                    st.markdown("**Year 1 (FY2026):**")
                    st.code("Triangular(min=-0.15, mode=0.08, max=0.34)")
                with dist_col2:
                    st.markdown("**Year 2 (FY2027):**")
                    st.code("Triangular(min=-0.15, mode=0.15, max=0.41)")
            
            # Primary Enterprise Value metrics
            ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)
            with ev_col1:
                st.metric("Mean Enterprise Value", f"${dcf_analytics['mean']:.2f}B", help="Average Enterprise Value across all simulations")
            with ev_col2:
                st.metric("Median Enterprise Value", f"${dcf_analytics['median']:.2f}B", help="50th percentile Enterprise Value")
            with ev_col3:
                st.metric("5th Percentile EV", f"${dcf_analytics['p05']:.2f}B", help="5% of simulations have EV below this value")
            with ev_col4:
                st.metric("95th Percentile EV", f"${dcf_analytics['p95']:.2f}B", help="95% of simulations have EV below this value")
            
            # Additional Enterprise Value statistics
            st.markdown("**Additional Enterprise Value Statistics:**")
            ev_stats_col1, ev_stats_col2, ev_stats_col3, ev_stats_col4 = st.columns(4)
            with ev_stats_col1:
                st.metric("Std Deviation", f"${dcf_analytics.get('std', 0):.2f}B")
            with ev_stats_col2:
                st.metric("1st Percentile", f"${dcf_analytics.get('p01', 0):.2f}B")
            with ev_stats_col3:
                st.metric("25th Percentile", f"${dcf_analytics.get('p25', 0):.2f}B")
            with ev_stats_col4:
                st.metric("75th Percentile", f"${dcf_analytics.get('p75', 0):.2f}B")
            
            # Market cap comparison
            if market_cap is not None and market_cap > 0:
                mean_ev = dcf_analytics['mean']
                diff = mean_ev - market_cap
                pct_diff = (diff / market_cap) * 100
                prob_ev_gt_mkt = float(np.mean(dcf_values > market_cap))
                
                st.subheader("Comparison to Market Capitalization")
                comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                with comp_col1:
                    st.metric("Mean EV", f"${mean_ev:.2f}B")
                with comp_col2:
                    st.metric("Market Cap", f"${market_cap:.2f}B")
                with comp_col3:
                    st.metric(
                        "Difference",
                        f"${diff:.2f}B",
                        delta=f"{pct_diff:.2f}%",
                    )
                with comp_col4:
                    st.metric("P(EV > Market)", f"{prob_ev_gt_mkt:.2%}")
                
                if diff > 0:
                    st.success(f"✓ Model suggests **UNDERVALUED** by ${diff:.2f}B ({pct_diff:.2f}%)")
                else:
                    st.warning(f"⚠ Model suggests **OVERVALUED** by ${abs(diff):.2f}B ({abs(pct_diff):.2f}%)")

            # Equity Value per Share section removed - focus on Enterprise Value only
            # Uncomment below if you want to show per-share calculations
            # if calculate_equity and equity_values is not None:
            #     st.subheader("Equity Value per Share")
            #     eq_col1, eq_col2, eq_col3, eq_col4 = st.columns(4)
            #     with eq_col1:
            #         st.metric("Mean", f"${equity_mean:.2f}")
            #     with eq_col2:
            #         st.metric("Median", f"${equity_median:.2f}")
            #     with eq_col3:
            #         st.metric("5th Percentile", f"${equity_p05:.2f}")
            #     with eq_col4:
            #         st.metric("95th Percentile", f"${equity_p95:.2f}")

            # Use DCF values for visualization
            analysis_array = dcf_values
            analytics = None  # Analytics computed in dcf_analytics
        else:
            # Standard growth rate analysis
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
            analysis_array = results.paths[:, -1] if results.paths.shape[1] > 1 else results.paths[:, 0]

        # Compute analytics for display
        if not dcf_mode_active:
            analytics = compute_analytics(
                results,
                var_alphas=[0.01, 0.05, 0.10],
                compute_sensitivity=True,
            )
        else:
            # Analytics already computed in dcf_analytics, but we need it for risk metrics
            analytics = None

        # Metrics cards (only show if not in DCF mode, or show both)
        if not dcf_mode_active and analytics is not None:
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
        if not dcf_mode_active or analytics is not None:
            st.subheader("Risk Metrics")
            risk_col1, risk_col2, risk_col3 = st.columns(3)

            if analytics is not None:
                with risk_col1:
                    st.metric("VaR (95%)", f"{analytics.var.get(0.05, 0):.4f}")
                with risk_col2:
                    st.metric("CVaR (95%)", f"{analytics.cvar.get(0.05, 0):.4f}")
                with risk_col3:
                    if analytics.prob_loss is not None:
                        st.metric("P(Loss)", f"{analytics.prob_loss:.4f}")
            elif dcf_mode_active:
                # Compute risk metrics from DCF values
                from app.core.analytics import value_at_risk, conditional_var, probability_loss
                var_95 = value_at_risk(dcf_values, 0.05)
                cvar_95 = conditional_var(dcf_values, 0.05)
                prob_loss = probability_loss(dcf_values, initial_fcf)  # Loss if below initial FCF
                
                with risk_col1:
                    st.metric("VaR (95%)", f"${var_95:.2f}B")
                with risk_col2:
                    st.metric("CVaR (95%)", f"${cvar_95:.2f}B")
                with risk_col3:
                    st.metric("P(EV < Initial FCF)", f"{prob_loss:.4f}")

        # Visualization tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📊 Distribution", "📈 Fan Chart", "📉 ECDF", "🌪️ Tornado", "🔄 Convergence", "📋 Growth Rate Distributions"]
        )

        with tab1:
            # Always show Year 1 and Year 2 growth rate distributions if available
            if results.paths.shape[1] >= 2:
                st.subheader("Growth Rate Distributions")
                growth_col1, growth_col2 = st.columns(2)
                
                with growth_col1:
                    st.markdown("**Year 1 Growth Rate**")
                    fig_g1 = plot_distribution_histogram(
                        results.paths[:, 0],
                        title="Year 1 Growth Rate Distribution",
                    )
                    st.plotly_chart(fig_g1, use_container_width=True)
                
                with growth_col2:
                    st.markdown("**Year 2 Growth Rate**")
                    fig_g2 = plot_distribution_histogram(
                        results.paths[:, 1],
                        title="Year 2 Growth Rate Distribution",
                    )
                    st.plotly_chart(fig_g2, use_container_width=True)
                
                st.markdown("---")
            
            # Show main output distribution (Enterprise Value in DCF mode, or final year growth rate)
            if dcf_mode_active:
                plot_data = dcf_values
                title = "Enterprise Value Distribution (Billions USD)"
                st.subheader("Enterprise Value Distribution (Output)")
            else:
                if results.paths.shape[1] == 1:
                    plot_data = results.paths[:, 0]
                    title = "Growth Rate Distribution"
                else:
                    plot_data = results.paths[:, -1]
                    title = "Final Year Growth Rate Distribution"

            fig = plot_distribution_histogram(plot_data, title=title)
            st.plotly_chart(fig, use_container_width=True)

            # Quantile band
            st.subheader("Quantile Analysis")
            if dcf_mode_active:
                from app.core.analytics import compute_quantiles
                quantiles_dict = compute_quantiles(plot_data)
            elif analytics is not None:
                quantiles_dict = analytics.quantiles
            else:
                from app.core.analytics import compute_quantiles
                quantiles_dict = compute_quantiles(plot_data)
            fig2 = plot_quantile_band(plot_data, quantiles_dict)
            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            st.markdown("""
            **Fan Chart Explanation:**
            A fan chart shows the uncertainty in projections over time. The colored bands represent different probability ranges:
            - **Red band (outer)**: 5th to 95th percentile range (90% of simulations fall within this range)
            - **Orange band (middle)**: 25th to 75th percentile range (50% of simulations fall within this range)
            - **Blue line**: Median (50th percentile) - the middle value across all simulations
            
            Wider bands indicate greater uncertainty, while narrower bands suggest more confidence in the projections.
            """)
            if results.paths.shape[1] > 1:
                fig = plot_fan_chart(results, title="Multi-Year Fan Chart")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Fan chart requires multiple years")

        with tab3:
            st.markdown("""
            **ECDF (Empirical Cumulative Distribution Function) Explanation:**
            The ECDF shows the cumulative probability of values. For any value on the x-axis:
            - The y-axis shows the **percentage of simulations** that resulted in a value less than or equal to that value
            - For example, if the curve is at 0.5 (50%) for a value of $100B, it means 50% of simulations produced Enterprise Values ≤ $100B
            - Steeper curves indicate values are more concentrated, while flatter curves suggest wider spread
            
            This helps you understand: "What's the probability that Enterprise Value will be below a certain threshold?"
            """)
            if dcf_mode_active:
                plot_data = dcf_values
                title = "Enterprise Value ECDF (Billions USD)"
            else:
                if results.paths.shape[1] == 1:
                    plot_data = results.paths[:, 0]
                else:
                    plot_data = results.paths[:, -1]
                title = "Empirical Cumulative Distribution"
            fig = plot_ecdf(plot_data, title=title)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            if analytics is not None and analytics.sensitivity is not None:
                fig = plot_tornado(analytics, title="Sensitivity Analysis (Tornado)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sensitivity analysis not available in DCF mode or for this configuration")

        with tab5:
            st.markdown("""
            **Convergence Analysis Explanation:**
            The convergence plot shows how the mean value stabilizes as more simulations are added. This helps verify:
            - **Stability**: Does the mean converge to a stable value, or does it keep changing?
            - **Sample size adequacy**: Have you run enough simulations to get reliable results?
            - **Confidence**: If the line flattens out, you can be more confident in your results
            
            The **red dashed line** shows the final mean value. If the blue line is still trending up or down at the end, you may need more simulations.
            """)
            fig = plot_convergence(results, title="Convergence Analysis")
            st.plotly_chart(fig, use_container_width=True)

        with tab6:
            st.header("📋 FCF Growth Rate Distributions")
            st.markdown(
                "**Visualization of what each year's growth rate distribution looks like based on your configured parameters.**"
            )
            try:
                preview_fig = plot_distribution_preview(year_configs, n_preview_samples=10_000, seed=42)
                st.plotly_chart(preview_fig, use_container_width=True)
                
                # Show parameter summary
                st.subheader("Distribution Parameters Summary")
                param_cols = st.columns(min(len(year_configs), 3))
                for idx, (dist_config, det_value) in enumerate(year_configs):
                    with param_cols[idx % len(param_cols)]:
                        st.markdown(f"**Year {idx+1}:**")
                        if det_value is not None:
                            st.code(f"Deterministic: {det_value:.3f}")
                        else:
                            dist_type = dist_config.get("type", "unknown")
                            if dist_type == "normal":
                                st.code(f"Normal(μ={dist_config.get('mu', 0):.3f}, σ={dist_config.get('sigma', 1):.3f})")
                            elif dist_type == "triangular":
                                st.code(f"Triangular(a={dist_config.get('a', 0):.3f}, c={dist_config.get('c', 0):.3f}, b={dist_config.get('b', 1):.3f})")
                            elif dist_type == "uniform":
                                st.code(f"Uniform(low={dist_config.get('low', 0):.3f}, high={dist_config.get('high', 1):.3f})")
                            elif dist_type == "lognormal":
                                st.code(f"Lognormal(μ={dist_config.get('mu', 0):.3f}, σ={dist_config.get('sigma', 1):.3f})")
                            elif dist_type == "student_t":
                                st.code(f"Student-t(df={dist_config.get('df', 5):.1f}, μ={dist_config.get('mu', 0):.3f}, σ={dist_config.get('sigma', 1):.3f})")
                            elif dist_type == "beta":
                                st.code(f"Beta(α={dist_config.get('alpha', 1):.2f}, β={dist_config.get('beta', 1):.2f})")
                            else:
                                st.code(f"Type: {dist_type}")
            except Exception as e:
                st.error(f"Error generating distribution preview: {e}")
                import traceback
                st.code(traceback.format_exc())

        # Data export
        st.subheader("Export Results")
        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            # CSV export - Enterprise Value is the primary output
            if dcf_mode_active:
                df = pd.DataFrame({
                    **{f"Year {i+1} Growth Rate": results.paths[:, i] for i in range(results.paths.shape[1])},
                    "Enterprise Value (Billions USD)": dcf_values,
                })
                # Per-share values removed - focus on Enterprise Value
                # if equity_values is not None:
                #     df["Equity Value per Share"] = equity_values
            else:
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

