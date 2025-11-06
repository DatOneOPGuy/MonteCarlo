"""
Example: Portfolio risk analysis using Monte Carlo simulation.

Simulates a 2-asset portfolio with correlated returns and computes
Value at Risk (VaR) and Conditional VaR (CVaR) at 95% confidence.
"""

from __future__ import annotations

import numpy as np
from montecarlo.core import Simulation
from montecarlo.viz import hist_with_kde
import plotly.io as pio


def main() -> None:
    """Run portfolio risk example."""
    # Portfolio parameters
    initial_value = 100_000  # $100k portfolio
    w1, w2 = 0.6, 0.4  # 60% asset 1, 40% asset 2
    mu1, sigma1 = 0.08, 0.15  # Asset 1: 8% mean, 15% volatility
    mu2, sigma2 = 0.12, 0.20  # Asset 2: 12% mean, 20% volatility
    rho = 0.3  # Correlation coefficient

    # Model: simulate 1-year portfolio value
    def model(x: np.ndarray) -> np.ndarray:
        # x is (n, 2): [r1, r2] annual returns
        # Portfolio return: w1*r1 + w2*r2
        portfolio_return = w1 * x[:, 0] + w2 * x[:, 1]
        # Portfolio value after 1 year
        portfolio_value = initial_value * (1 + portfolio_return)
        return portfolio_value

    # Sampler: correlated normal returns
    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        # Generate correlated returns using Cholesky decomposition
        cov_matrix = np.array([
            [sigma1**2, rho * sigma1 * sigma2],
            [rho * sigma1 * sigma2, sigma2**2],
        ])
        L = np.linalg.cholesky(cov_matrix)
        z = rng.standard_normal(size=(size, 2))
        returns = np.array([mu1, mu2]) + z @ L.T
        return returns

    # Run simulation
    print("Running portfolio risk simulation...")
    sim = Simulation(model, sampler, seed=42)
    result = sim.run(200_000, keep=True)

    portfolio_values = result["all"]

    # Compute VaR and CVaR at 95%
    var_95 = np.percentile(portfolio_values, 5)  # 5th percentile (loss)
    cvar_95 = np.mean(portfolio_values[portfolio_values <= var_95])

    print(f"\nPortfolio Risk Analysis:")
    print(f"  Initial value: ${initial_value:,.2f}")
    print(f"  Mean value: ${result['mean']:,.2f}")
    print(f"  Std dev: ${result['std']:,.2f}")
    print(f"  VaR(95%): ${var_95:,.2f} (5th percentile)")
    print(f"  CVaR(95%): ${cvar_95:,.2f} (expected loss given VaR)")
    print(f"  Probability of loss: {sim.risk(lambda y: y < initial_value)['p']:.4f}")

    # Create visualization
    fig = hist_with_kde(
        portfolio_values,
        bins=60,
        title="Portfolio Value Distribution (1 Year)",
    )
    fig.add_vline(
        x=initial_value,
        line_dash="dash",
        line_color="green",
        annotation_text="Initial Value",
    )
    fig.add_vline(
        x=var_95,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR(95%): ${var_95:,.0f}",
    )
    fig.update_layout(
        xaxis_title="Portfolio Value ($)",
        yaxis_title="Density",
    )

    # Save to HTML
    output_file = "portfolio_risk.html"
    pio.write_html(fig, output_file)
    print(f"\nVisualization saved to: {output_file}")


if __name__ == "__main__":
    main()

