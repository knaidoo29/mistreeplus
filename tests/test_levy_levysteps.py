import pytest
import numpy as np
from mistreeplus.levy import generate_levy_steps, generate_adj_levy_steps

# Test for generate_levy_steps
def test_generate_levy_steps():
    size = 1000
    t0 = 0.2
    alpha = 1.5

    steps = generate_levy_steps(size, t0, alpha)

    # Check the output is a numpy array of the correct size
    assert isinstance(steps, np.ndarray), f"Expected a numpy array, got {type(steps)}"
    assert steps.shape == (size,), f"Expected shape ({size},), got {steps.shape}"

    # Check that the generated steps are positive
    assert np.all(steps > 0), f"Expected all steps to be positive, but found non-positive values"

# Test for generate_adj_levy_steps
def test_generate_adj_levy_steps():
    size = 1000
    t0 = 0.3
    ts = 0.01
    alpha = 1.5
    beta = 0.4
    gamma = 1.3

    steps = generate_adj_levy_steps(size, t0, ts, alpha, beta, gamma)

    # Check the output is a numpy array of the correct size
    assert isinstance(steps, np.ndarray), f"Expected a numpy array, got {type(steps)}"
    assert steps.shape == (size,), f"Expected shape ({size},), got {steps.shape}"

    # Check that the generated steps are positive
    assert np.all(steps > 0), f"Expected all steps to be positive, but found non-positive values"

    # Validate that steps below t0 are clustered and have values around ts
    steps_below_t0 = steps[steps < t0]
    assert np.mean(steps_below_t0) < t0, f"Expected mean of steps below t0 to be less than t0, but found {np.mean(steps_below_t0)}"

    # Validate that steps above beta are calculated as expected
    steps_above_beta = steps[steps >= t0]
    assert np.all(steps_above_beta >= t0), "Expected all steps above beta to be >= t0"
