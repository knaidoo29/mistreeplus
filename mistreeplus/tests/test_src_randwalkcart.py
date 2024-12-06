import numpy as np
import pytest
from mistreeplus.src import periodicboundary, randwalkcart2d, randwalkcart3d


# Test periodicboundary
def test_periodicboundary_within_bounds():
    assert periodicboundary(5.0, 10.0) == 5.0, "Value within bounds should remain unchanged"
    assert periodicboundary(0.0, 10.0) == 0.0, "Value at lower bound should remain unchanged"
    assert periodicboundary(10.0, 10.0) == 10.0, "Value at upper bound should remain unchanged"


def test_periodicboundary_below_bounds():
    assert periodicboundary(-1.0, 10.0) == 9.0, "Value below bounds should wrap around to the end of the range"
    assert periodicboundary(-11.0, 10.0) == 9.0, "Multiple wraps below bounds should work correctly"


def test_periodicboundary_above_bounds():
    assert periodicboundary(11.0, 10.0) == 1.0, "Value above bounds should wrap around to the start of the range"
    assert periodicboundary(21.0, 10.0) == 1.0, "Multiple wraps above bounds should work correctly"


# Test randwalkcart2d
def test_randwalkcart2d_no_periodic():
    steps = np.array([1.0, 1.0])
    prand = np.array([0.0, np.pi / 2])
    boxsize = 10.0
    x0, y0 = 0.0, 0.0
    useperiodic = 0

    x, y = randwalkcart2d(steps, prand, boxsize, x0, y0, useperiodic)

    expected_x = np.array([0.0, 1.0, 1.0])
    expected_y = np.array([0.0, 0.0, 1.0])

    assert np.allclose(x, expected_x), "2D random walk without periodic boundaries (x) is incorrect"
    assert np.allclose(y, expected_y), "2D random walk without periodic boundaries (y) is incorrect"


def test_randwalkcart2d_with_periodic():
    steps = np.array([11.0])
    prand = np.array([0.0])
    boxsize = 10.0
    x0, y0 = 0.0, 0.0
    useperiodic = 1

    x, y = randwalkcart2d(steps, prand, boxsize, x0, y0, useperiodic)

    expected_x = np.array([0.0, 1.0])  # 11.0 wraps to 1.0
    expected_y = np.array([0.0, 0.0])  # No vertical movement

    assert np.allclose(x, expected_x), "2D random walk with periodic boundaries (x) is incorrect"
    assert np.allclose(y, expected_y), "2D random walk with periodic boundaries (y) is incorrect"


def test_randwalkcart2d_edge_cases():
    steps = np.array([])
    prand = np.array([])
    boxsize = 10.0
    x0, y0 = 5.0, 5.0
    useperiodic = 0

    x, y = randwalkcart2d(steps, prand, boxsize, x0, y0, useperiodic)

    assert np.allclose(x, [5.0]), "2D random walk with no steps (x) is incorrect"
    assert np.allclose(y, [5.0]), "2D random walk with no steps (y) is incorrect"


# Test randwalkcart3d
def test_randwalkcart3d_no_periodic():
    steps = np.array([1.0])
    prand = np.array([0.0])  # Move along +x-axis
    trand = np.array([np.pi / 2])  # Flat in x-y plane
    boxsize = 10.0
    x0, y0, z0 = 0.0, 0.0, 0.0
    useperiodic = 0

    x, y, z = randwalkcart3d(steps, prand, trand, boxsize, x0, y0, z0, useperiodic)

    expected_x = np.array([0.0, 1.0])
    expected_y = np.array([0.0, 0.0])
    expected_z = np.array([0.0, 0.0])

    assert np.allclose(x, expected_x), "3D random walk without periodic boundaries (x) is incorrect"
    assert np.allclose(y, expected_y), "3D random walk without periodic boundaries (y) is incorrect"
    assert np.allclose(z, expected_z), "3D random walk without periodic boundaries (z) is incorrect"


def test_randwalkcart3d_with_periodic():
    steps = np.array([11.0])
    prand = np.array([0.0])
    trand = np.array([np.pi / 2])
    boxsize = 10.0
    x0, y0, z0 = 0.0, 0.0, 0.0
    useperiodic = 1

    x, y, z = randwalkcart3d(steps, prand, trand, boxsize, x0, y0, z0, useperiodic)

    expected_x = np.array([0.0, 1.0])  # 11.0 wraps to 1.0
    expected_y = np.array([0.0, 0.0])
    expected_z = np.array([0.0, 0.0])

    assert np.allclose(x, expected_x), "3D random walk with periodic boundaries (x) is incorrect"
    assert np.allclose(y, expected_y), "3D random walk with periodic boundaries (y) is incorrect"
    assert np.allclose(z, expected_z), "3D random walk with periodic boundaries (z) is incorrect"


def test_randwalkcart3d_edge_cases():
    steps = np.array([])
    prand = np.array([])
    trand = np.array([])
    boxsize = 10.0
    x0, y0, z0 = 5.0, 5.0, 5.0
    useperiodic = 0

    x, y, z = randwalkcart3d(steps, prand, trand, boxsize, x0, y0, z0, useperiodic)

    assert np.allclose(x, [5.0]), "3D random walk with no steps (x) is incorrect"
    assert np.allclose(y, [5.0]), "3D random walk with no steps (y) is incorrect"
    assert np.allclose(z, [5.0]), "3D random walk with no steps (z) is incorrect"
