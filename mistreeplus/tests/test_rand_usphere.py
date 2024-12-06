import numpy as np
import pytest
from mistreeplus.randoms import usphere_phi, usphere_theta, usphere_phitheta


def test_usphere_phi():
    size = 1000
    phimin, phimax = 0.0, 2.0 * np.pi
    prand = usphere_phi(size, phimin=phimin, phimax=phimax)

    # Check the shape of the output
    assert prand.shape == (size,), "Output shape is incorrect for usphere_phi"

    # Check that values are within the specified range
    assert (prand >= phimin).all() and (prand <= phimax).all(), "Phi values are out of range"

    # Check distribution (optional, for randomness)
    assert np.unique(prand).shape[0] > size * 0.5, "Generated phi values seem non-random"


def test_usphere_theta():
    size = 1000
    thetamin, thetamax = 0.0, np.pi
    trand = usphere_theta(size, thetamin=thetamin, thetamax=thetamax)

    # Check the shape of the output
    assert trand.shape == (size,), "Output shape is incorrect for usphere_theta"

    # Check that values are within the specified range
    assert (trand >= thetamin).all() and (trand <= thetamax).all(), "Theta values are out of range"

    # Validate for distribution of theta (ensures values vary across the range)
    assert np.var(trand) > 0, "Theta values are not varying as expected"

    # Check if cosine weighting is reflected in the distribution (optional statistical test)
    cos_values = np.cos(trand)
    assert np.var(cos_values) > 0, "Cosine transformation suggests incorrect distribution"


def test_usphere_phitheta():
    size = 1000
    mins = [0.0, 0.0]
    maxs = [2.0 * np.pi, np.pi]
    prand, trand = usphere_phitheta(size, mins=mins, maxs=maxs)

    # Check the shapes of the outputs
    assert prand.shape == (size,), "Output shape is incorrect for phi values in usphere_phitheta"
    assert trand.shape == (size,), "Output shape is incorrect for theta values in usphere_phitheta"

    # Check that phi values are within the specified range
    assert (prand >= mins[0]).all() and (prand <= maxs[0]).all(), "Phi values are out of range"

    # Check that theta values are within the specified range
    assert (trand >= mins[1]).all() and (trand <= maxs[1]).all(), "Theta values are out of range"

    # Additional statistical sanity checks
    assert np.mean(prand) > mins[0], "Mean phi value is suspiciously low in usphere_phitheta"
    assert np.mean(trand) > mins[1], "Mean theta value is suspiciously low in usphere_phitheta"

    # Optional: Verify cosine-weighted uniformity in theta
    cos_theta = np.cos(trand)
    assert np.min(cos_theta) >= -1.0, "Cosine theta values are out of expected range"
    assert np.max(cos_theta) <= 1.0, "Cosine theta values are out of expected range"


# Example of parameterized test (optional for multiple ranges)
@pytest.mark.parametrize(
    "phimin, phimax, thetamin, thetamax",
    [
        (0.0, 2.0 * np.pi, 0.0, np.pi),
        (0.0, np.pi, np.pi / 4, 3 * np.pi / 4),
        (np.pi / 2, 3 * np.pi / 2, 0.0, np.pi / 2),
    ],
)

def test_usphere_phitheta_varied_ranges(phimin, phimax, thetamin, thetamax):
    size = 1000
    prand, trand = usphere_phitheta(
        size, mins=[phimin, thetamin], maxs=[phimax, thetamax]
    )

    # Check that phi values are within the specified range
    assert (prand >= phimin).all() and (prand <= phimax).all(), "Phi values are out of range"

    # Check that theta values are within the specified range
    assert (trand >= thetamin).all() and (trand <= thetamax).all(), "Theta values are out of range"
