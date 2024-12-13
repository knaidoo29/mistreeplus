import numpy as np
import pytest
from mistreeplus.randoms import sphere_r, sphere_phi, sphere_theta, sphere_rphitheta


def test_sphere_r():
    size = 1000
    rmin, rmax = 1.0, 5.0
    rrand = sphere_r(size, rmin=rmin, rmax=rmax)

    # Check the shape of the output
    assert rrand.shape == (size,), "Output shape is incorrect for sphere_r"

    # Check that values are within the specified range
    assert (rrand >= rmin).all() and (rrand <= rmax).all(), "Values are out of range"

    # Check distribution characteristics (not essential, but useful for QA)
    assert np.mean(rrand) > rmin, "Mean radial value is suspiciously low"
    assert np.mean(rrand) < rmax, "Mean radial value is suspiciously high"


def test_sphere_phi():
    size = 1000
    phimin, phimax = 0.0, 2.0 * np.pi
    prand = sphere_phi(size, phimin=phimin, phimax=phimax)

    # Check the shape of the output
    assert prand.shape == (size,), "Output shape is incorrect for sphere_phi"

    # Check that values are within the specified range
    assert (prand >= phimin).all() and (prand <= phimax).all(), "Values are out of range"

    # Check periodicity for angles (optional, since we are generating uniform)
    unique_values = np.unique(prand)
    assert len(unique_values) > size * 0.5, "Generated phi values seem non-random"


def test_sphere_theta():
    size = 1000
    thetamin, thetamax = 0.0, np.pi
    trand = sphere_theta(size, thetamin=thetamin, thetamax=thetamax)

    # Check the shape of the output
    assert trand.shape == (size,), "Output shape is incorrect for sphere_theta"

    # Check that values are within the specified range
    assert (trand >= thetamin).all() and (trand <= thetamax).all(), "Values are out of range"

    # Validate for uniformity of the generated angles
    assert np.var(trand) > 0, "Theta values are not varying as expected"


def test_sphere_rphitheta():
    size = 1000
    mins = [1.0, 0.0, 0.0]
    maxs = [5.0, 2.0 * np.pi, np.pi]
    rrand, prand, trand = sphere_rphitheta(size, mins=mins, maxs=maxs)

    # Check the shape of the outputs
    assert rrand.shape == (size,), "Output shape is incorrect for radial distances in sphere_rphitheta"
    assert prand.shape == (size,), "Output shape is incorrect for phi values in sphere_rphitheta"
    assert trand.shape == (size,), "Output shape is incorrect for theta values in sphere_rphitheta"

    # Check that all values are within the specified ranges
    assert (rrand >= mins[0]).all() and (rrand <= maxs[0]).all(), "Radial distances are out of range"
    assert (prand >= mins[1]).all() and (prand <= maxs[1]).all(), "Phi values are out of range"
    assert (trand >= mins[2]).all() and (trand <= maxs[2]).all(), "Theta values are out of range"

    # Additional statistical tests
    assert np.mean(rrand) > mins[0], "Mean radial value is suspiciously low in sphere_rphitheta"
    assert np.mean(rrand) < maxs[0], "Mean radial value is suspiciously high in sphere_rphitheta"
    assert np.mean(prand) > mins[1], "Mean phi value is suspiciously low in sphere_rphitheta"
    assert np.mean(trand) > mins[2], "Mean theta value is suspiciously low in sphere_rphitheta"
