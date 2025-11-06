"""
Command-line interface for Monte Carlo simulations.

Provides convenient commands for common simulation tasks.
"""

from __future__ import annotations

import json

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from montecarlo.core import Simulation
from montecarlo.distributions import lognormal, normal

app = typer.Typer(help="Monte Carlo simulation toolkit CLI")
console = Console()


def _print_table(result: dict, title: str = "Simulation Results") -> None:
    """Pretty print simulation results as a Rich table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Statistic", style="cyan")
    table.add_column("Value", justify="right", style="green")

    for key, value in result.items():
        if key != "all":
            if isinstance(value, float):
                table.add_row(key, f"{value:.6f}")
            else:
                table.add_row(key, str(value))

    console.print(table)


@app.command()
def pi(
    n: int = typer.Option(1000000, "--n", "-n", help="Number of trials"),
    seed: int = typer.Option(None, "--seed", "-s", help="Random seed"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """
    Estimate π using Monte Carlo simulation.

    Generates random points in a unit square and estimates π by counting
    the fraction that fall inside a unit circle.
    """
    def model(x: np.ndarray) -> np.ndarray:
        # x is (n, 2) with uniform [0,1] x [0,1]
        x_coords = x[:, 0]
        y_coords = x[:, 1]
        # Check if inside unit circle
        inside = (x_coords**2 + y_coords**2) <= 1.0
        return inside.astype(float)

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(0, 1, size=(size, 2))

    sim = Simulation(model, sampler, seed=seed)
    result = sim.run(n, keep=True)

    # Estimate π = 4 * (fraction inside circle)
    pi_estimate = 4 * result["mean"]
    se_pi = 4 * result["se"]

    output = {
        "pi_estimate": pi_estimate,
        "se": se_pi,
        "n": n,
        "seed": seed,
    }

    if json_output:
        console.print(json.dumps(output, indent=2))
    else:
        table = Table(title="π Estimation", show_header=True, header_style="bold magenta")
        table.add_column("Statistic", style="cyan")
        table.add_column("Value", justify="right", style="green")
        table.add_row("π estimate", f"{pi_estimate:.6f}")
        table.add_row("Standard error", f"{se_pi:.6f}")
        table.add_row("True π", "3.141593")
        table.add_row("Error", f"{abs(pi_estimate - np.pi):.6f}")
        table.add_row("Trials", str(n))
        if seed is not None:
            table.add_row("Seed", str(seed))
        console.print(table)


@app.command()
def linear(
    mu1: float = typer.Option(10.0, "--mu1", help="Mean of first normal"),
    sigma1: float = typer.Option(2.0, "--sigma1", help="Std of first normal"),
    mu2: float = typer.Option(1.0, "--mu2", help="Mean (log-space) of lognormal"),
    sigma2: float = typer.Option(0.25, "--sigma2", help="Std (log-space) of lognormal"),
    n: int = typer.Option(200000, "--n", help="Number of trials"),
    seed: int = typer.Option(None, "--seed", "-s", help="Random seed"),
    threshold: float = typer.Option(None, "--threshold", help="Threshold for risk calculation"),
    json_output: bool = typer.Option(False, "--json", help="JSON output"),
) -> None:
    """
    Run a linear model simulation: y = x1 + 2*x2.

    Where x1 ~ N(mu1, sigma1) and x2 ~ LogNormal(mu2, sigma2).
    """
    def model(x: np.ndarray) -> np.ndarray:
        # x is (n, 2): [x1, x2]
        return x[:, 0] + 2 * x[:, 1]

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        x1 = normal(mu1, sigma1)(rng, size)
        x2 = lognormal(mu2, sigma2)(rng, size)
        return np.column_stack([x1, x2])

    sim = Simulation(model, sampler, seed=seed)
    result = sim.run(n, keep=True)

    output = {
        "mean": result["mean"],
        "std": result["std"],
        "p05": result["p05"],
        "p50": result["p50"],
        "p95": result["p95"],
        "se": result["se"],
        "n": n,
        "seed": seed,
    }

    if threshold is not None:
        risk_result = sim.risk(lambda y: y > threshold)
        output["threshold"] = threshold
        output["P(y > threshold)"] = risk_result["p"]
        output["P_se"] = risk_result["se"]

    if json_output:
        console.print(json.dumps(output, indent=2))
    else:
        _print_table(output, "Linear Model Results")


if __name__ == "__main__":
    app()

