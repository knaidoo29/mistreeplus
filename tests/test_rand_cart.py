import pytest
import numpy as np
from mistreeplus.randoms import cart1d, cart2d, cart3d


def test_cart1d():
    """Test the cart1d function."""
    size = 10
    xmin = 5.0
    xmax = 10.0

    result = cart1d(size, xmin=xmin, xmax=xmax)

    # Check the size of the output array
    assert len(result) == size, "cart1d did not return the expected number of samples."

    # Check that all values are within the specified range
    assert (result >= xmin).all() and (result <= xmax).all(), \
        "cart1d values are not within the expected range."


def test_cart2d():
    """Test the cart2d function."""
    size = 10
    mins = [0.0, 1.0]
    maxs = [5.0, 10.0]

    xrand, yrand = cart2d(size, mins=mins, maxs=maxs)

    # Check the size of the output arrays
    assert len(xrand) == size and len(yrand) == size, \
        "cart2d did not return the expected number of samples."

    # Check that all xrand values are within the range [mins[0], maxs[0]]
    assert (xrand >= mins[0]).all() and (xrand <= maxs[0]).all(), \
        "cart2d x values are not within the expected range."

    # Check that all yrand values are within the range [mins[1], maxs[1]]
    assert (yrand >= mins[1]).all() and (yrand <= maxs[1]).all(), \
        "cart2d y values are not within the expected range."


def test_cart3d():
    """Test the cart3d function."""
    size = 15
    mins = [0.0, 2.0, 3.0]
    maxs = [5.0, 10.0, 8.0]

    xrand, yrand, zrand = cart3d(size, mins=mins, maxs=maxs)

    # Check the size of the output arrays
    assert len(xrand) == size and len(yrand) == size and len(zrand) == size, \
        "cart3d did not return the expected number of samples."

    # Check that all xrand values are within the range [mins[0], maxs[0]]
    assert (xrand >= mins[0]).all() and (xrand <= maxs[0]).all(), \
        "cart3d x values are not within the expected range."

    # Check that all yrand values are within the range [mins[1], maxs[1]]
    assert (yrand >= mins[1]).all() and (yrand <= maxs[1]).all(), \
        "cart3d y values are not within the expected range."

    # Check that all zrand values are within the range [mins[2], maxs[2]]
    assert (zrand >= mins[2]).all() and (zrand <= maxs[2]).all(), \
        "cart3d z values are not within the expected range."


def test_cart_functions_with_defaults():
    """Test the cart1d, cart2d, and cart3d functions with default parameters."""
    size = 5

    # Test cart1d with default parameters
    result1d = cart1d(size)
    assert len(result1d) == size, "cart1d with default parameters did not return the expected number of samples."
    assert (result1d >= 0.0).all() and (result1d <= 1.0).all(), \
        "cart1d with default parameters returned values outside the range [0.0, 1.0]."

    # Test cart2d with default parameters
    xrand, yrand = cart2d(size)
    assert len(xrand) == size and len(yrand) == size, \
        "cart2d with default parameters did not return the expected number of samples."
    assert (xrand >= 0.0).all() and (xrand <= 1.0).all(), \
        "cart2d x values with default parameters are not within the range [0.0, 1.0]."
    assert (yrand >= 0.0).all() and (yrand <= 1.0).all(), \
        "cart2d y values with default parameters are not within the range [0.0, 1.0]."

    # Test cart3d with default parameters
    xrand, yrand, zrand = cart3d(size)
    assert len(xrand) == size and len(yrand) == size and len(zrand) == size, \
        "cart3d with default parameters did not return the expected number of samples."
    assert (xrand >= 0.0).all() and (xrand <= 1.0).all(), \
        "cart3d x values with default parameters are not within the range [0.0, 1.0]."
    assert (yrand >= 0.0).all() and (yrand <= 1.0).all(), \
        "cart3d y values with default parameters are not within the range [0.0, 1.0]."
    assert (zrand >= 0.0).all() and (zrand <= 1.0).all(), \
        "cart3d z values with default parameters are not within the range [0.0, 1.0]."
