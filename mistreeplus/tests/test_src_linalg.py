import numpy as np
import pytest
from mistreeplus.src import (
    dotvector3,
    dot3by3mat3vec,
    crossvector3,
    normalisevector,
    inv3by3,
)


def test_dotvector3():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    result = dotvector3(a, b)

    # Expected dot product
    expected = np.dot(a, b)
    assert np.isclose(result, expected), "Dot product calculation is incorrect"


def test_dot3by3mat3vec():
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    b = np.array([1.0, 0.0, -1.0])
    result = dot3by3mat3vec(a, b)

    # Expected dot product
    expected = np.dot(a, b)
    assert np.allclose(result, expected), "Dot product between matrix and vector is incorrect"


def test_crossvector3():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    result = crossvector3(a, b)

    # Expected cross product
    expected = np.cross(a, b)
    assert np.allclose(result, expected), "Cross product calculation is incorrect"


def test_normalisevector():
    vecin = np.array([3.0, 4.0, 0.0])
    result = normalisevector(vecin)

    # Expected normalized vector
    expected = vecin / np.linalg.norm(vecin)
    assert np.allclose(result, expected), "Vector normalization is incorrect"

    # Check if normalized vector has unit magnitude
    assert np.isclose(np.linalg.norm(result), 1.0), "Normalized vector magnitude is not 1"


def test_inv3by3():
    mat = np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
    result = inv3by3(mat)

    # Expected inverse matrix
    expected = np.linalg.inv(mat)
    assert np.allclose(result, expected), "Matrix inversion is incorrect"

    # Check if the result is indeed the inverse
    identity = np.dot(mat, result)
    assert np.allclose(identity, np.eye(3)), "Matrix inverse does not satisfy identity property"


def test_inv3by3_singular():
    mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # Singular matrix

    with pytest.raises(ValueError, match="Matrix is singular and cannot be inverted"):
        inv3by3(mat)


# Example of parameterized tests for robustness
@pytest.mark.parametrize(
    "vec1, vec2, expected_dot",
    [
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0),  # Orthogonal vectors
        ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], 32.0),  # Standard case
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0.0),  # Zero vector
    ],
)
def test_dotvector3_parametrized(vec1, vec2, expected_dot):
    result = dotvector3(np.array(vec1), np.array(vec2))
    assert np.isclose(result, expected_dot), "Parameterized dot product test failed"


@pytest.mark.parametrize(
    "matrix, vector",
    [
        ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 2, 3]),  # Identity matrix
        ([[0, 1, 0], [1, 0, 0], [0, 0, 1]], [1, 0, 0]),  # Permutation matrix
    ],
)
def test_dot3by3mat3vec_parametrized(matrix, vector):
    result = dot3by3mat3vec(np.array(matrix), np.array(vector))
    expected = np.dot(np.array(matrix), np.array(vector))
    assert np.allclose(result, expected), "Parameterized matrix-vector dot product test failed"
