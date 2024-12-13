import pytest
import numpy as np
from mistreeplus.check import (
    check_isscalar,
    check_length,
    check_positive,
    check_finite,
)


def test_check_isscalar_with_scalars():
    """Test check_isscalar with Python scalar types."""
    assert check_isscalar(42) is True  # Integer
    assert check_isscalar(3.14) is True  # Float
    assert check_isscalar(True) is True  # Boolean


def test_check_isscalar_with_numpy_scalars():
    """Test check_isscalar with NumPy scalar types."""
    assert check_isscalar(np.int32(42)) is True
    assert check_isscalar(np.float64(3.14)) is True
    assert check_isscalar(np.bool_(True)) is True


def test_check_isscalar_with_0d_numpy_array():
    """Test check_isscalar with 0-dimensional NumPy arrays."""
    zero_dim_array = np.array(42)  # Creates a 0-dimensional array
    assert check_isscalar(zero_dim_array) is True


def test_check_isscalar_with_1d_numpy_array():
    """Test check_isscalar with 1-dimensional NumPy arrays."""
    one_dim_array = np.array([42])
    assert check_isscalar(one_dim_array) is False


def test_check_isscalar_with_multidimensional_numpy_array():
    """Test check_isscalar with multi-dimensional NumPy arrays."""
    multi_dim_array = np.array([[42]])
    assert check_isscalar(multi_dim_array) is False


def test_check_isscalar_with_non_numpy_iterables():
    """Test check_isscalar with non-NumPy iterable types."""
    assert check_isscalar([42]) is False  # List
    assert check_isscalar((42,)) is False  # Tuple


def test_check_isscalar_with_none():
    """Test check_isscalar with None."""
    assert check_isscalar(None) is False


def test_check_isscalar_with_complex_numbers():
    """Test check_isscalar with complex numbers."""
    assert check_isscalar(3 + 4j) is True  # Python complex number
    assert check_isscalar(np.complex64(3 + 4j)) is True  # NumPy complex number


def test_check_length():
    # Valid cases
    check_length(np.array([1, 2, 3]), 3)
    check_length(np.array([]), 0)

    # Invalid cases
    with pytest.raises(AssertionError):
        check_length(np.array([1, 2, 3]), 2)
    with pytest.raises(AssertionError):
        check_length(np.array([1, 2]), 3)


def test_check_positive():
    # Valid cases
    check_positive(3.5)
    check_positive(np.array([1, 2, 3]))
    check_positive(np.array([0]))  # Zero is non-negative

    # Invalid cases
    with pytest.raises(AssertionError):
        check_positive(-1.0)
    with pytest.raises(AssertionError):
        check_positive(np.array([-1, 2, 3]))
    with pytest.raises(AssertionError):
        check_positive(np.array([1, -2, 3]))


def test_check_finite():
    # Valid cases
    check_finite(3.5)
    check_finite(np.array([1, 2, 3]))
    check_finite(np.array([np.pi, np.e, 0]))

    # Invalid cases
    with pytest.raises(AssertionError):
        check_finite(np.inf)
    with pytest.raises(AssertionError):
        check_finite(-np.inf)
    with pytest.raises(AssertionError):
        check_finite(np.array([1, 2, np.nan]))
    with pytest.raises(AssertionError):
        check_finite(np.array([1, np.inf, 3]))
