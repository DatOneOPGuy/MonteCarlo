"""
Example: Estimate π using Monte Carlo simulation.

This script demonstrates the basic usage of the Monte Carlo toolkit
to estimate π by sampling random points in a unit square.
"""

from __future__ import annotations

import numpy as np

from montecarlo.core import Simulation


def main() -> None:
    """Run π estimation example."""
    # Model: check if point (x, y) is inside unit circle
    def model(x: np.ndarray) -> np.ndarray:
        x_coords = x[:, 0]
        y_coords = x[:, 1]
        # Check if inside unit circle: x^2 + y^2 <= 1
        inside = (x_coords**2 + y_coords**2) <= 1.0
        return inside.astype(float)

    # Sampler: uniform points in [0,1] x [0,1]
    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(0, 1, size=(size, 2))

    # Run simulation
    print("Estimating π using Monte Carlo simulation...")
    sim = Simulation(model, sampler, seed=42)
    result = sim.run(1_000_000, keep=True)

    # Estimate π = 4 * (fraction inside circle)
    pi_estimate = 4 * result["mean"]
    se_pi = 4 * result["se"]

    print(f"\nResults:")
    print(f"  π estimate: {pi_estimate:.6f}")
    print(f"  Standard error: {se_pi:.6f}")
    print(f"  True π: {np.pi:.6f}")
    print(f"  Error: {abs(pi_estimate - np.pi):.6f}")
    print(f"  Trials: {result['n']:,}")


if __name__ == "__main__":
    main()

