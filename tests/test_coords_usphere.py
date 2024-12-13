import pytest
import numpy as np
from mistreeplus.coords import usphere2cart, usphere2cart_radec, cart2usphere, cart2usphere_radec, usphere_dist2ang

# Mocking sphere functions for standalone testing
def mock_sphere2cart(r, phi, theta, units="rads"):
    phi = np.deg2rad(phi) if units == "degs" else phi
    theta = np.deg2rad(theta) if units == "degs" else theta
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z

def mock_sphere2cart_radec(r, ra, dec, units="rads"):
    phi = np.deg2rad(ra) if units == "degs" else ra
    theta = np.deg2rad(90 - dec) if units == "degs" else np.pi / 2 - dec
    return mock_sphere2cart(r, phi, theta, units)

def mock_cart2sphere(x, y, z, units="rads"):
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    phi = np.rad2deg(phi) if units == "degs" else phi
    phi = np.where(phi < 0, phi + (360 if units == "degs" else 2 * np.pi), phi)
    theta = np.arccos(z / r)
    theta = np.rad2deg(theta) if units == "degs" else theta
    return r, phi, theta

def mock_cart2sphere_radec(x, y, z, units="rads"):
    r, phi, theta = mock_cart2sphere(x, y, z, units)
    ra = phi
    dec = 90 - theta if units == "degs" else np.pi / 2 - theta
    return r, ra, dec

@pytest.fixture
def mock_sphere_module(monkeypatch):
    monkeypatch.setattr("mistreeplus.coords.sphere.sphere2cart", mock_sphere2cart)
    monkeypatch.setattr("mistreeplus.coords.sphere.sphere2cart_radec", mock_sphere2cart_radec)
    monkeypatch.setattr("mistreeplus.coords.sphere.cart2sphere", mock_cart2sphere)
    monkeypatch.setattr("mistreeplus.coords.sphere.cart2sphere_radec", mock_cart2sphere_radec)

# usphere2cart tests
def test_usphere2cart_scalar(mock_sphere_module):
    x, y, z = usphere2cart(0, 0, units="rads")
    assert (x, y, z) == (0.0, 0.0, 1.0)

def test_usphere2cart_array(mock_sphere_module):
    phi = np.array([0, np.pi / 2, np.pi])
    theta = np.array([0, np.pi / 2, np.pi])
    x, y, z = usphere2cart(phi, theta, units="rads")
    assert x.shape == phi.shape
    assert y.shape == phi.shape
    assert z.shape == phi.shape

# usphere2cart_radec tests
def test_usphere2cart_radec_scalar(mock_sphere_module):
    x, y, z = usphere2cart_radec(0, 0, units="rads")    
    np.testing.assert_array_almost_equal(np.array([x, y, z]), np.array([1.0, 0.0, 0.0]), decimal=4)

def test_usphere2cart_radec_array(mock_sphere_module):
    ra = np.array([0, np.pi / 2, np.pi])
    dec = np.array([0, np.pi / 4, np.pi / 2])
    x, y, z = usphere2cart_radec(ra, dec, units="rads")
    assert x.shape == ra.shape
    assert y.shape == ra.shape
    assert z.shape == ra.shape

# cart2usphere tests
def test_cart2usphere_scalar(mock_sphere_module):
    phi, theta = cart2usphere(1, 0, 0, units="rads")
    assert phi == 0.0
    assert theta == np.pi / 2

def test_cart2usphere_array(mock_sphere_module):
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    phi, theta = cart2usphere(x, y, z, units="rads")
    assert phi.shape == x.shape
    assert theta.shape == x.shape

# cart2usphere_radec tests
def test_cart2usphere_radec_scalar(mock_sphere_module):
    ra, dec = cart2usphere_radec(1, 0, 0, units="rads")
    assert ra == 0.0
    assert dec == 0.0

def test_cart2usphere_radec_array(mock_sphere_module):
    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    ra, dec = cart2usphere_radec(x, y, z, units="rads")
    assert ra.shape == x.shape
    assert dec.shape == x.shape

# usphere_dist2ang tests
def test_usphere_dist2ang():
    dist = np.array([0, 1, 2])
    ang = usphere_dist2ang(dist)
    expected = 2 * np.arcsin(dist / 2.0)
    assert np.allclose(ang, expected)

def test_usphere_dist2ang_scalar():
    dist = 1
    ang = usphere_dist2ang(dist)
    expected = 2 * np.arcsin(dist / 2.0)
    assert ang == pytest.approx(expected)
