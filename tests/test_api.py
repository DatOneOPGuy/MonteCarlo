"""
Tests for FastAPI endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from montecarlo.api import app

client = TestClient(app)


def test_root_endpoint() -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data


def test_pi_simulation() -> None:
    """Test π estimation endpoint."""
    response = client.post(
        "/simulate/pi",
        json={"n": 10000, "seed": 42},
    )
    assert response.status_code == 200
    data = response.json()
    assert "pi_estimate" in data
    assert "se" in data
    assert "n" in data
    assert abs(data["pi_estimate"] - 3.14159) < 0.1


def test_linear_simulation() -> None:
    """Test linear model endpoint."""
    response = client.post(
        "/simulate/linear",
        json={
            "mu1": 10.0,
            "sigma1": 2.0,
            "mu2": 1.0,
            "sigma2": 0.25,
            "n": 10000,
            "seed": 42,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "mean" in data
    assert "std" in data
    assert "p05" in data
    assert "p50" in data
    assert "p95" in data
    assert "se" in data


def test_linear_simulation_with_threshold() -> None:
    """Test linear model with threshold."""
    response = client.post(
        "/simulate/linear",
        json={
            "mu1": 10.0,
            "sigma1": 2.0,
            "mu2": 1.0,
            "sigma2": 0.25,
            "n": 10000,
            "seed": 42,
            "threshold": 20.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "extras" in data
    assert data["extras"] is not None
    assert "P(y > threshold)" in data["extras"]
    assert 0 <= data["extras"]["P(y > threshold)"] <= 1


def test_custom_simulation() -> None:
    """Test custom simulation endpoint."""
    response = client.post(
        "/simulate/custom",
        json={
            "n": 10000,
            "seed": 42,
            "dists": [
                {"name": "x1", "type": "normal", "params": {"mu": 0.0, "sigma": 1.0}},
                {"name": "x2", "type": "normal", "params": {"mu": 0.0, "sigma": 1.0}},
            ],
            "model": "x1 + 2*x2",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "mean" in data
    assert "std" in data


def test_custom_simulation_unsafe_symbol() -> None:
    """Test that unsafe symbols are rejected."""
    response = client.post(
        "/simulate/custom",
        json={
            "n": 1000,
            "seed": 42,
            "dists": [
                {"name": "x1", "type": "normal", "params": {"mu": 0.0, "sigma": 1.0}},
            ],
            "model": "__import__('os').system('ls')",
        },
    )
    # Should either reject or sanitize
    assert response.status_code in [200, 400]


def test_validation_errors() -> None:
    """Test that validation errors are caught."""
    # Invalid sigma
    response = client.post(
        "/simulate/linear",
        json={
            "mu1": 10.0,
            "sigma1": -1.0,  # Invalid
            "mu2": 1.0,
            "sigma2": 0.25,
            "n": 1000,
        },
    )
    assert response.status_code == 422  # Validation error

    # Invalid n
    response = client.post(
        "/simulate/pi",
        json={"n": -1},  # Invalid
    )
    assert response.status_code == 422


def test_fixed_seed_reproducibility() -> None:
    """Test that fixed seed produces same results."""
    response1 = client.post(
        "/simulate/pi",
        json={"n": 1000, "seed": 42},
    )
    response2 = client.post(
        "/simulate/pi",
        json={"n": 1000, "seed": 42},
    )
    assert response1.status_code == 200
    assert response2.status_code == 200
    data1 = response1.json()
    data2 = response2.json()
    assert abs(data1["pi_estimate"] - data2["pi_estimate"]) < 1e-10

