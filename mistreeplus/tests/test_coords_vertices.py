import pytest
import numpy as np
from mistreeplus.coords import xy2vert, vert2xy, xyz2vert, vert2xyz

# xy2vert tests
def test_xy2vert():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    result = xy2vert(x, y)
    assert np.array_equal(result, expected)

# vert2xy tests
def test_vert2xy():
    vert = np.array([[1, 4], [2, 5], [3, 6]])
    expected_x = np.array([1, 2, 3])
    expected_y = np.array([4, 5, 6])
    result_x, result_y = vert2xy(vert)
    assert np.array_equal(result_x, expected_x)
    assert np.array_equal(result_y, expected_y)

# xyz2vert tests
def test_xyz2vert():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = np.array([7, 8, 9])
    expected = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    result = xyz2vert(x, y, z)
    assert np.array_equal(result, expected)

# vert2xyz tests
def test_vert2xyz():
    vert = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
    expected_x = np.array([1, 2, 3])
    expected_y = np.array([4, 5, 6])
    expected_z = np.array([7, 8, 9])
    result_x, result_y, result_z = vert2xyz(vert)
    assert np.array_equal(result_x, expected_x)
    assert np.array_equal(result_y, expected_y)
    assert np.array_equal(result_z, expected_z)
