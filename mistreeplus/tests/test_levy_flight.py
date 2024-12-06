import pytest
import numpy as np
from mistreeplus.levy import generate_user_flight, generate_levy_flight, generate_adj_levy_flight

# Test for generate_user_flight
def test_generate_user_flight():
    # Test 2D mode
    steps = np.array([1.0, 2.0, 3.0])
    pos = generate_user_flight(steps, mode="2D", boxsize=10.0, periodic=True)
    assert pos.shape == (len(steps) + 1, 2), f"Expected shape {len(steps) + 1, 2}, got {pos.shape}"

    # Test 3D mode
    pos = generate_user_flight(steps, mode="3D", boxsize=10.0, periodic=True)
    assert pos.shape == (len(steps) + 1, 3), f"Expected shape {len(steps) + 1, 3}, got {pos.shape}"

    # Test unit sphere mode
    pos = generate_user_flight(steps, mode="usphere")
    assert pos.shape == (len(steps) + 1, 2), f"Expected shape {len(steps) + 1, 2}, got {pos.shape}"

# Test for generate_levy_flight
def test_generate_levy_flight():
    size = 10
    pos = generate_levy_flight(size, t0=0.2, alpha=1.5, mode="2D")
    assert pos.shape == (size, 2), f"Expected shape ({size}, 2), got {pos.shape}"

    pos = generate_levy_flight(size, t0=0.5, alpha=1.8, mode="3D")
    assert pos.shape == (size, 3), f"Expected shape ({size}, 3), got {pos.shape}"

# Test for generate_adj_levy_flight
def test_generate_adj_levy_flight():
    size = 15
    pos = generate_adj_levy_flight(size, t0=0.325, ts=0.01, alpha=1.5, beta=0.43, gamma=1.3, mode="2D")
    assert pos.shape == (size, 2), f"Expected shape ({size}, 2), got {pos.shape}"

    pos = generate_adj_levy_flight(size, t0=0.3, ts=0.02, alpha=1.6, beta=0.5, gamma=1.2, mode="3D")
    assert pos.shape == (size, 3), f"Expected shape ({size}, 3), got {pos.shape}"

    pos = generate_adj_levy_flight(size, t0=0.4, ts=0.015, alpha=1.4, beta=0.4, gamma=1.5, mode="usphere")
    assert pos.shape == (size, 2), f"Expected shape ({size}, 2), got {pos.shape}"
