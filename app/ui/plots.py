"""
Plotly visualization functions for simulation results.

Provides interactive charts for distributions, convergence, fan charts,
tornado diagrams, and ECDF plots.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats

from app.core.analytics import ecdf
from app.core.types import AnalyticsResults, SimResults


def plot_distribution_histogram(
    arr: np.ndarray, bins: int = 60, title: str = "Distribution"
) -> go.Figure:
    """
    Create histogram with KDE overlay.

    Parameters
    ----------
    arr : np.ndarray
        Array of values
    bins : int
        Number of histogram bins
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    arr_flat = arr.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]

    if len(arr_flat) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No data to plot", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=arr_flat,
            nbinsx=bins,
            name="Histogram",
            marker_color="rgba(55, 128, 191, 0.7)",
            opacity=0.7,
            histnorm="probability density",
        )
    )

    # KDE overlay
    x_min, x_max = np.min(arr_flat), np.max(arr_flat)
    x_kde = np.linspace(x_min, x_max, 200)
    kde = stats.gaussian_kde(arr_flat)
    y_kde = kde(x_kde)

    fig.add_trace(
        go.Scatter(
            x=x_kde,
            y=y_kde,
            mode="lines",
            name="KDE",
            line=dict(color="red", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Density",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_ecdf(arr: np.ndarray, title: str = "Empirical CDF") -> go.Figure:
    """
    Plot Empirical Cumulative Distribution Function.

    Parameters
    ----------
    arr : np.ndarray
        Array of values
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    sorted_vals, cumulative = ecdf(arr)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=sorted_vals,
            y=cumulative,
            mode="lines",
            name="ECDF",
            line=dict(color="blue", width=2),
            fill="tozeroy",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Cumulative Probability",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_fan_chart(
    results: SimResults, title: str = "Fan Chart"
) -> go.Figure:
    """
    Plot fan chart showing quantile bands over time.

    Parameters
    ----------
    results : SimResults
        Simulation results
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    paths = results.paths
    n_years = paths.shape[1]
    years = np.arange(1, n_years + 1)

    # Compute quantiles for each year
    quantiles = [5, 25, 50, 75, 95]
    q_values = {q: [] for q in quantiles}

    for year_idx in range(n_years):
        year_data = paths[:, year_idx]
        for q in quantiles:
            q_values[q].append(np.percentile(year_data, q))

    fig = go.Figure()

    # Add quantile bands (filled areas)
    colors = ["rgba(255,0,0,0.1)", "rgba(255,165,0,0.2)", "rgba(0,255,0,0.3)"]

    # 5th-95th percentile band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([q_values[95], q_values[5][::-1]]),
            fill="toself",
            fillcolor="rgba(255,0,0,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="5th-95th percentile",
            showlegend=True,
        )
    )

    # 25th-75th percentile band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([q_values[75], q_values[25][::-1]]),
            fill="toself",
            fillcolor="rgba(255,165,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="25th-75th percentile",
            showlegend=True,
        )
    )

    # Median line
    fig.add_trace(
        go.Scatter(
            x=years,
            y=q_values[50],
            mode="lines",
            name="Median",
            line=dict(color="blue", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_tornado(
    analytics: AnalyticsResults, title: str = "Tornado Diagram"
) -> go.Figure:
    """
    Plot tornado diagram for sensitivity analysis.

    Parameters
    ----------
    analytics : AnalyticsResults
        Analytics results with sensitivity data
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    if analytics.sensitivity is None or len(analytics.sensitivity) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No sensitivity data available", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    # Sort by absolute sensitivity
    sorted_items = sorted(
        analytics.sensitivity.items(), key=lambda x: abs(x[1]), reverse=True
    )
    param_names = [name for name, _ in sorted_items]
    sensitivities = [val for _, val in sorted_items]

    fig = go.Figure()

    # Horizontal bar chart
    colors = ["red" if s < 0 else "green" for s in sensitivities]

    fig.add_trace(
        go.Bar(
            x=sensitivities,
            y=param_names,
            orientation="h",
            marker_color=colors,
            name="Sensitivity",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Sensitivity Score",
        yaxis_title="Parameter",
        template="plotly_white",
        height=max(400, len(param_names) * 30),
    )

    return fig


def plot_convergence(
    results: SimResults, target_year: int | None = None, title: str = "Convergence"
) -> go.Figure:
    """
    Plot convergence of cumulative mean over simulations.

    Parameters
    ----------
    results : SimResults
        Simulation results
    target_year : int or None
        Which year to plot (None for final year)
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    if target_year is None:
        target_year = results.paths.shape[1] - 1

    values = results.paths[:, target_year]
    n = len(values)
    iterations = np.arange(1, n + 1)
    cumulative_mean = np.cumsum(values) / iterations

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=cumulative_mean,
            mode="lines",
            name="Cumulative mean",
            line=dict(color="blue", width=2),
        )
    )

    # Add final mean line
    final_mean = cumulative_mean[-1]
    fig.add_hline(
        y=final_mean,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Final mean: {final_mean:.4f}",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Simulation Number",
        yaxis_title="Cumulative Mean",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def plot_quantile_band(
    arr: np.ndarray, quantiles: dict[str, float] | None = None, title: str = "Quantiles"
) -> go.Figure:
    """
    Plot distribution with quantile markers.

    Parameters
    ----------
    arr : np.ndarray
        Array of values
    quantiles : dict or None
        Quantile dictionary (if None, computes default quantiles)
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    arr_flat = arr.flatten()
    arr_flat = arr_flat[~np.isnan(arr_flat)]

    if quantiles is None:
        quantiles = {
            "p05": np.percentile(arr_flat, 5),
            "p50": np.percentile(arr_flat, 50),
            "p95": np.percentile(arr_flat, 95),
        }

    fig = plot_distribution_histogram(arr_flat, title=title)

    # Add quantile lines
    colors = {"p05": "orange", "p50": "green", "p95": "red"}
    labels = {"p05": "5th percentile", "p50": "Median", "p95": "95th percentile"}

    for q_name, q_val in quantiles.items():
        if q_name in colors:
            fig.add_vline(
                x=q_val,
                line_dash="dash",
                line_color=colors[q_name],
                annotation_text=f"{labels[q_name]}: {q_val:.4f}",
                annotation_position="top",
            )

    return fig

