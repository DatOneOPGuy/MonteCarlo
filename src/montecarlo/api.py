"""
FastAPI REST API for Monte Carlo simulations.

Provides HTTP endpoints for running simulations programmatically.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from montecarlo.core import Simulation
from montecarlo.distributions import (
    bernoulli,
    binomial,
    lognormal,
    normal,
    uniform,
)

app = FastAPI(
    title="Monte Carlo Simulation API",
    description="REST API for Monte Carlo simulations",
    version="0.1.0",
)

# Safe math operators for expression evaluation
_SAFE_DICT = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "pi": np.pi,
    "e": np.e,
    "np": np,  # Allow numpy namespace
}
# Add safe operators
for name in ["add", "sub", "mul", "truediv", "floordiv", "mod", "pow", "lt", "le", "eq", "ne", "ge", "gt"]:
    if hasattr(operator, name):
        _SAFE_DICT[name] = getattr(operator, name)


def _safe_eval(expr: str, variables: dict[str, np.ndarray]) -> np.ndarray:
    """
    Safely evaluate a mathematical expression with variable substitution.

    Parameters
    ----------
    expr : str
        Expression string (e.g., "x1 + 2*x2").
    variables : dict
        Dictionary mapping variable names to arrays.

    Returns
    -------
    result : np.ndarray
        Evaluated result array.
    """
    # Validate variable names (alphanumeric + underscore only)
    for var_name in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", expr):
        if var_name not in variables and var_name not in _SAFE_DICT:
            raise ValueError(f"Unknown variable or function: {var_name}")

    # Replace variable names with their values in a safe context
    safe_vars = {**_SAFE_DICT, **variables}
    
    try:
        result = eval(expr, {"__builtins__": {}}, safe_vars)
        if not isinstance(result, np.ndarray):
            # If scalar, broadcast to match variable length
            n = len(next(iter(variables.values())))
            result = np.full(n, result)
        return result
    except Exception as e:
        raise ValueError(f"Expression evaluation failed: {e}")


class LinearSimRequest(BaseModel):
    """Request model for linear simulation endpoint."""

    mu1: float = Field(..., description="Mean of first normal distribution")
    sigma1: float = Field(..., gt=0, description="Std of first normal distribution")
    mu2: float = Field(..., description="Mean (log-space) of lognormal distribution")
    sigma2: float = Field(..., gt=0, description="Std (log-space) of lognormal distribution")
    n: int = Field(..., gt=0, description="Number of trials")
    seed: int | None = Field(None, description="Random seed")
    threshold: float | None = Field(None, description="Threshold for risk calculation")


class PiSimRequest(BaseModel):
    """Request model for π estimation endpoint."""

    n: int = Field(..., gt=0, description="Number of trials")
    seed: int | None = Field(None, description="Random seed")


class DistributionSpec(BaseModel):
    """Specification for a distribution in custom simulation."""

    name: str = Field(..., description="Variable name (e.g., 'x1')")
    type: str = Field(..., description="Distribution type: normal, lognormal, uniform, bernoulli, binomial")
    params: dict[str, float] = Field(..., description="Distribution parameters")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = ["normal", "lognormal", "uniform", "bernoulli", "binomial"]
        if v not in allowed:
            raise ValueError(f"type must be one of {allowed}")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
            raise ValueError("name must be a valid identifier (alphanumeric + underscore)")
        return v


class CustomSimRequest(BaseModel):
    """Request model for custom simulation endpoint."""

    n: int = Field(..., gt=0, description="Number of trials")
    seed: int | None = Field(None, description="Random seed")
    dists: list[DistributionSpec] = Field(..., description="List of distribution specifications")
    model: str = Field(..., description="Model expression (e.g., 'x1 + 2*x2')")
    threshold: float | None = Field(None, description="Threshold for risk calculation")


class SimulationResponse(BaseModel):
    """Response model for simulation endpoints."""

    mean: float
    std: float
    p05: float
    p50: float
    p95: float
    se: float
    n: int
    seed: int | None
    extras: dict | None = None


@app.post("/simulate/linear", response_model=SimulationResponse)
async def simulate_linear(request: LinearSimRequest) -> SimulationResponse:
    """
    Run a linear model simulation: y = x1 + 2*x2.

    Where x1 ~ N(mu1, sigma1) and x2 ~ LogNormal(mu2, sigma2).
    """
    def model(x: np.ndarray) -> np.ndarray:
        return x[:, 0] + 2 * x[:, 1]

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        x1 = normal(request.mu1, request.sigma1)(rng, size)
        x2 = lognormal(request.mu2, request.sigma2)(rng, size)
        return np.column_stack([x1, x2])

    sim = Simulation(model, sampler, seed=request.seed)
    result = sim.run(request.n, keep=True)

    extras = {}
    if request.threshold is not None:
        risk_result = sim.risk(lambda y: y > request.threshold)
        extras["threshold"] = request.threshold
        extras["P(y > threshold)"] = risk_result["p"]
        extras["P_se"] = risk_result["se"]

    return SimulationResponse(
        mean=result["mean"],
        std=result["std"],
        p05=result["p05"],
        p50=result["p50"],
        p95=result["p95"],
        se=result["se"],
        n=result["n"],
        seed=request.seed,
        extras=extras if extras else None,
    )


@app.post("/simulate/pi", response_model=dict)
async def simulate_pi(request: PiSimRequest) -> dict:
    """
    Estimate π using Monte Carlo simulation.
    """
    def model(x: np.ndarray) -> np.ndarray:
        x_coords = x[:, 0]
        y_coords = x[:, 1]
        inside = (x_coords**2 + y_coords**2) <= 1.0
        return inside.astype(float)

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(0, 1, size=(size, 2))

    sim = Simulation(model, sampler, seed=request.seed)
    result = sim.run(request.n)

    pi_estimate = 4 * result["mean"]
    se_pi = 4 * result["se"]

    return {
        "pi_estimate": pi_estimate,
        "se": se_pi,
        "n": request.n,
        "seed": request.seed,
    }


@app.post("/simulate/custom", response_model=SimulationResponse)
async def simulate_custom(request: CustomSimRequest) -> SimulationResponse:
    """
    Run a custom simulation with user-defined distributions and model expression.
    """
    # Build samplers
    dist_factories = {
        "normal": lambda p: normal(p["mu"], p["sigma"]),
        "lognormal": lambda p: lognormal(p["mu"], p["sigma"]),
        "uniform": lambda p: uniform(p["a"], p["b"]),
        "bernoulli": lambda p: bernoulli(p["p"]),
        "binomial": lambda p: binomial(int(p["n"]), p["p"]),
    }

    samplers = []
    var_names = []
    for dist_spec in request.dists:
        factory = dist_factories.get(dist_spec.type)
        if factory is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown distribution type: {dist_spec.type}",
            )
        try:
            sampler = factory(dist_spec.params)
            samplers.append(sampler)
            var_names.append(dist_spec.name)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid parameters for {dist_spec.name}: {e}",
            )

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        samples = [s(rng, size) for s in samplers]
        return np.column_stack(samples)

    # Build model from expression
    def model(x: np.ndarray) -> np.ndarray:
        variables = {name: x[:, i] for i, name in enumerate(var_names)}
        try:
            return _safe_eval(request.model, variables)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    sim = Simulation(model, sampler, seed=request.seed)
    result = sim.run(request.n, keep=True)

    extras = {}
    if request.threshold is not None:
        risk_result = sim.risk(lambda y: y > request.threshold)
        extras["threshold"] = request.threshold
        extras["P(y > threshold)"] = risk_result["p"]
        extras["P_se"] = risk_result["se"]

    return SimulationResponse(
        mean=result["mean"],
        std=result["std"],
        p05=result["p05"],
        p50=result["p50"],
        p95=result["p95"],
        se=result["se"],
        n=result["n"],
        seed=request.seed,
        extras=extras if extras else None,
    )


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "message": "Monte Carlo Simulation API",
        "version": "0.1.0",
        "docs": "/docs",
    }

