"""
Visualization helpers for Monte Carlo simulations.

Provides Plotly figure factories for interactive charts.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy import stats


def hist_with_kde(
    y: np.ndarray, bins: int = 60, title: str = "Distribution"
) -> go.Figure:
    """
    Create a histogram with KDE overlay using Plotly.

    Parameters
    ----------
    y : np.ndarray
        Array of simulation outputs.
    bins : int, optional
        Number of histogram bins (default: 60).
    title : str, optional
        Plot title (default: "Distribution").

    Returns
    -------
    fig : plotly.graph_objs.Figure
        Interactive Plotly figure with histogram and KDE curve.
    """
    if len(y) == 0:
        raise ValueError("y array is empty")

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=y,
            nbinsx=bins,
            name="Histogram",
            marker_color="rgba(55, 128, 191, 0.7)",
            opacity=0.7,
            histnorm="probability density",
        )
    )

    # KDE overlay
    y_min, y_max = np.min(y), np.max(y)
    x_kde = np.linspace(y_min, y_max, 200)
    kde = stats.gaussian_kde(y)
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


def convergence_plot(
    estimates: np.ndarray,
    true_value: float | None = None,
    title: str = "Convergence",
) -> go.Figure:
    """
    Create a convergence plot showing cumulative mean vs. iterations.

    Parameters
    ----------
    estimates : np.ndarray
        Array of simulation outputs (will compute cumulative mean).
    true_value : float or None, optional
        True/reference value to plot as a horizontal line (default: None).
    title : str, optional
        Plot title (default: "Convergence").

    Returns
    -------
    fig : plotly.graph_objs.Figure
        Interactive Plotly figure with convergence curve.
    """
    if len(estimates) == 0:
        raise ValueError("estimates array is empty")

    n = len(estimates)
    iterations = np.arange(1, n + 1)
    cumulative_mean = np.cumsum(estimates) / iterations

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

    if true_value is not None:
        fig.add_hline(
            y=true_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"True value: {true_value:.4f}",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Cumulative mean",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )

    return fig


def quantile_band(y: np.ndarray, title: str = "Quantiles") -> go.Figure:
    """
    Create a visualization showing quantiles as vertical lines.

    Parameters
    ----------
    y : np.ndarray
        Array of simulation outputs.
    title : str, optional
        Plot title (default: "Quantiles").

    Returns
    -------
    fig : plotly.graph_objs.Figure
        Interactive Plotly figure with quantile markers.
    """
    if len(y) == 0:
        raise ValueError("y array is empty")

    p05 = np.percentile(y, 5)
    p50 = np.percentile(y, 50)
    p95 = np.percentile(y, 95)

    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=y,
            nbinsx=60,
            name="Distribution",
            marker_color="rgba(55, 128, 191, 0.7)",
            opacity=0.7,
            histnorm="probability density",
        )
    )

    # Quantile lines
    y_max = np.max(np.histogram(y, bins=60)[0])
    fig.add_vline(
        x=p05,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"p05: {p05:.4f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=p50,
        line_dash="dash",
        line_color="green",
        annotation_text=f"p50: {p50:.4f}",
        annotation_position="top",
    )
    fig.add_vline(
        x=p95,
        line_dash="dash",
        line_color="red",
        annotation_text=f"p95: {p95:.4f}",
        annotation_position="top",
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

