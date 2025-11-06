"""
Input validation using Pydantic models.

Validates distribution parameters, correlation matrices, and simulation configs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class NormalDistParams(BaseModel):
    """Normal distribution parameters."""

    mu: float = Field(..., description="Mean")
    sigma: float = Field(..., gt=0, description="Standard deviation (must be > 0)")


class LognormalDistParams(BaseModel):
    """Lognormal distribution parameters."""

    mu: float = Field(..., description="Log-space mean")
    sigma: float = Field(..., gt=0, description="Log-space standard deviation (must be > 0)")


class StudentTDistParams(BaseModel):
    """Student-t distribution parameters."""

    df: float = Field(..., gt=0, description="Degrees of freedom (must be > 0)")
    mu: float = Field(0.0, description="Location parameter")
    sigma: float = Field(1.0, gt=0, description="Scale parameter (must be > 0)")


class TriangularDistParams(BaseModel):
    """Triangular distribution parameters."""

    a: float = Field(..., description="Lower bound")
    b: float = Field(..., description="Upper bound")
    c: float = Field(..., description="Mode")

    @model_validator(mode="after")
    def validate_bounds(self) -> TriangularDistParams:
        """Validate that a <= c <= b."""
        if not (self.a <= self.c <= self.b):
            raise ValueError(f"Must have a <= c <= b, got a={self.a}, c={self.c}, b={self.b}")
        return self


class UniformDistParams(BaseModel):
    """Uniform distribution parameters."""

    low: float = Field(..., description="Lower bound")
    high: float = Field(..., description="Upper bound")

    @model_validator(mode="after")
    def validate_bounds(self) -> UniformDistParams:
        """Validate that low < high."""
        if self.low >= self.high:
            raise ValueError(f"Must have low < high, got low={self.low}, high={self.high}")
        return self


class BetaDistParams(BaseModel):
    """Beta distribution parameters."""

    alpha: float = Field(..., gt=0, description="Alpha parameter (must be > 0)")
    beta: float = Field(..., gt=0, description="Beta parameter (must be > 0)")
    low: float = Field(0.0, description="Lower bound for scaling")
    high: float = Field(1.0, description="Upper bound for scaling")

    @model_validator(mode="after")
    def validate_bounds(self) -> BetaDistParams:
        """Validate that low < high."""
        if self.low >= self.high:
            raise ValueError(f"Must have low < high, got low={self.low}, high={self.high}")
        return self


class CorrelationMatrix(BaseModel):
    """Correlation matrix validation."""

    matrix: list[list[float]] = Field(..., description="Correlation matrix")

    @field_validator("matrix")
    @classmethod
    def validate_matrix(cls, v: list[list[float]]) -> list[list[float]]:
        """Validate matrix structure and convert to numpy array."""
        if not v:
            raise ValueError("Matrix cannot be empty")
        n = len(v)
        if not all(len(row) == n for row in v):
            raise ValueError("Matrix must be square")
        return v

    @model_validator(mode="after")
    def validate_symmetric_positive_definite(self) -> CorrelationMatrix:
        """Validate that matrix is symmetric and positive definite."""
        arr = np.array(self.matrix)
        n = arr.shape[0]

        # Check symmetry
        if not np.allclose(arr, arr.T):
            raise ValueError("Matrix must be symmetric")

        # Check diagonal is 1.0
        if not np.allclose(np.diag(arr), 1.0):
            raise ValueError("Diagonal elements must be 1.0")

        # Check off-diagonal in [-1, 1]
        mask = ~np.eye(n, dtype=bool)
        if np.any(arr[mask] < -1) or np.any(arr[mask] > 1):
            raise ValueError("Off-diagonal elements must be in [-1, 1]")

        # Check positive definite
        try:
            np.linalg.cholesky(arr)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix must be positive definite (Cholesky decomposition failed)")

        return self


class SimConfigValidator(BaseModel):
    """Simulation configuration validator."""

    n_sims: int = Field(..., gt=0, le=10_000_000, description="Number of simulations")
    seed: int | None = Field(None, ge=0, description="Random seed (None for random)")
    scenario_name: str = Field("base", description="Scenario name")
    scenario_weight: float = Field(1.0, ge=0, le=1, description="Scenario weight for blending")


def validate_distribution_params(
    dist_type: str, params: dict[str, Any]
) -> BaseModel:
    """
    Validate distribution parameters based on type.

    Parameters
    ----------
    dist_type : str
        Distribution type name
    params : dict
        Parameter dictionary

    Returns
    -------
    BaseModel
        Validated Pydantic model

    Raises
    ------
    ValueError
        If distribution type is unknown or parameters are invalid
    """
    validators = {
        "normal": NormalDistParams,
        "lognormal": LognormalDistParams,
        "student_t": StudentTDistParams,
        "triangular": TriangularDistParams,
        "uniform": UniformDistParams,
        "beta": BetaDistParams,
    }

    if dist_type not in validators:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    validator = validators[dist_type]
    return validator(**params)


def compute_parameter_hash(config: dict[str, Any]) -> str:
    """
    Compute a hash of the configuration for reproducibility tracking.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    str
        Hexadecimal hash string
    """
    # Sort keys and convert to JSON for consistent hashing
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]

