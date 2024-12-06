import pytest
import numpy as np
from mistreeplus.randoms import polar_r, polar_phi, polar_rphi

@pytest.fixture
def mock_cart1d(monkeypatch):
    """Mock the cart1d function to simplify tests."""
    def mock_cart1d(size, xmin=0., xmax=1.):
        return np.linspace(xmin, xmax, size)
    monkeypatch.setattr("mistreeplus.randoms.cart.cart1d", mock_cart1d)


def test_polar_r(mock_cart1d):
    """Test the polar_r function."""
    size = 10
    rmin = 1.0
    rmax = 5.0

    # Ensure the function handles proper positive checks
    with pytest.raises(AssertionError):
        polar_r(size, rmin=-1.0, rmax=rmax)

    with pytest.raises(AssertionError):
        polar_r(size, rmin=rmin, rmax=-1.0)

    result = polar_r(size, rmin=rmin, rmax=rmax)

    # Check output size
    assert len(result) == size, "polar_r did not return the expected number of samples."

    # Check that all radial values are within the expected range
    assert (result >= rmin).all() and (result <= rmax).all(), \
        "polar_r values are not within the expected range."


def test_polar_phi(mock_cart1d):
    """Test the polar_phi function."""
    size = 10
    phimin = 0.0
    phimax = np.pi

    # Ensure valid angle units
    with pytest.raises(AssertionError):
        polar_phi(size, phimin=phimin, phimax=phimax, units="invalid_units")

    # Ensure valid phi range checks
    with pytest.raises(AssertionError):
        polar_phi(size, phimin=-10, phimax=phimax, units="rads")

    with pytest.raises(AssertionError):
        polar_phi(size, phimin=phimin, phimax=400, units="degs")

    # Test the actual function
    result = polar_phi(size, phimin=phimin, phimax=phimax, units="rads")

    # Check output size
    assert len(result) == size, "polar_phi did not return the expected number of samples."

    # Check that all phi values are within the expected range
    assert (result >= phimin).all() and (result <= phimax).all(), \
        "polar_phi values are not within the expected range."


def test_polar_rphi(mock_cart1d):
    """Test the polar_rphi function."""
    size = 10
    mins = [1.0, 0.0]
    maxs = [5.0, np.pi]

    # Test the function with valid inputs
    rrand, prand = polar_rphi(size, mins=mins, maxs=maxs, units="rads")

    # Check output sizes
    assert len(rrand) == size, "polar_rphi did not return the expected number of radial samples."
    assert len(prand) == size, "polar_rphi did not return the expected number of angular samples."

    # Check that radial values are within the expected range
    assert (rrand >= mins[0]).all() and (rrand <= maxs[0]).all(), \
        "polar_rphi radial values are not within the expected range."

    # Check that angular values are within the expected range
    assert (prand >= mins[1]).all() and (prand <= maxs[1]).all(), \
        "polar_rphi angular values are not within the expected range."

    # Test units conversion
    prand_degs = polar_phi(size, phimin=0.0, phimax=180.0, units="degs")
    assert (prand_degs >= 0.0).all() and (prand_degs <= 180.0).all(), \
        "polar_rphi angular values are not correctly generated in degrees."
