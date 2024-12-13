import pytest
import numpy as np
from mistreeplus.index import cantor_pair, uncantor_pair

# Test for cantor_pair
def test_cantor_pair():
    # Test scalar inputs
    k1, k2 = 3, 5
    result = cantor_pair(k1, k2)
    expected = 41  # Pre-calculated result for (3, 5)
    assert result == expected, f"Expected {expected}, got {result} for input ({k1}, {k2})"

    # Test array inputs
    k1 = np.array([1, 2, 3])
    k2 = np.array([4, 5, 6])
    result = cantor_pair(k1, k2)
    expected = np.array([19, 33, 51])  # Pre-calculated results for the array inputs
    np.testing.assert_array_equal(result, expected)

# Test for uncantor_pair
def test_uncantor_pair():
    # Test scalar input
    pi = 41  # Pre-calculated input
    k1, k2 = uncantor_pair(pi)
    expected_k1, expected_k2 = 3, 5  # The original values for (3, 5)
    assert k1 == expected_k1, f"Expected {expected_k1}, got {k1}"
    assert k2 == expected_k2, f"Expected {expected_k2}, got {k2}"

    # Test array input
    pi = np.array([19, 33, 51])  # Pre-calculated inputs
    k1, k2 = uncantor_pair(pi)
    expected_k1 = np.array([1, 2, 3])
    expected_k2 = np.array([4, 5, 6])
    np.testing.assert_array_equal(k1, expected_k1)
    np.testing.assert_array_equal(k2, expected_k2)
