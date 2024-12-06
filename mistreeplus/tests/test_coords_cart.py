import pytest
import numpy as np
from mistreeplus.coords import dist2D, dist3D  # Replace 'your_module' with the actual module name

def test_dist2D():
    # Valid scalar cases
    assert dist2D(0, 3, 0, 4) == 5.0  # 3-4-5 triangle
    assert dist2D(1, 1, 1, 1) == 0.0  # Same point

    # Valid array cases
    np.testing.assert_array_almost_equal(
        dist2D(np.array([0, 1]), np.array([3, 1]), np.array([0, 1]), np.array([4, 1])),
        np.array([5.0, 0.0])
    )

    # Edge cases: ensure correct dtype handling
    assert dist2D(0.0, 3.0, 0.0, 4.0) == 5.0  # Float inputs
    np.testing.assert_array_almost_equal(
        dist2D(np.array([0.0]), np.array([3.0]), np.array([0.0]), np.array([4.0])),
        np.array([5.0])
    )


def test_dist3D():
    # Valid scalar cases
    assert dist3D(0, 3, 0, 4, 0, 12) == 13.0  # 5-12-13 triangle
    assert dist3D(1, 1, 1, 1, 1, 1) == 0.0  # Same point

    # Valid array cases
    np.testing.assert_array_almost_equal(
        dist3D(
            np.array([0, 1]),
            np.array([3, 1]),
            np.array([0, 1]),
            np.array([4, 1]),
            np.array([0, 1]),
            np.array([12, 1]),
        ),
        np.array([13.0, 0.0])
    )

    # Edge cases: ensure correct dtype handling
    assert dist3D(0.0, 3.0, 0.0, 4.0, 0.0, 12.0) == 13.0  # Float inputs
    np.testing.assert_array_almost_equal(
        dist3D(
            np.array([0.0]),
            np.array([3.0]),
            np.array([0.0]),
            np.array([4.0]),
            np.array([0.0]),
            np.array([12.0]),
        ),
        np.array([13.0])
    )
