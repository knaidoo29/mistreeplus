import pytest
import numpy as np
from mistreeplus.coords import (
    theta2dec,
    dec2theta,
    sphere2cart,
    sphere2cart_radec,
    cart2sphere,
    cart2sphere_radec,
)

def test_theta2dec():
    # Scalar cases
    assert theta2dec(np.pi / 2., units="rads") == 0.0
    assert theta2dec(90., units="degs") == 0.0

    # Array cases
    np.testing.assert_array_almost_equal(
        theta2dec(np.array([np.pi / 2., 0.]), units="rads"),
        np.array([0.0, np.pi / 2.])
    )
    np.testing.assert_array_almost_equal(
        theta2dec(np.array([90., 0.]), units="degs"),
        np.array([0.0, 90.])
    )


def test_dec2theta():
    # Scalar cases
    assert dec2theta(0., units="rads") == np.pi / 2.
    assert dec2theta(0., units="degs") == 90.

    # Array cases
    np.testing.assert_array_almost_equal(
        dec2theta(np.array([0.0, -90]), units="degs"),
        np.array([90.0, 180.0])
    )
    np.testing.assert_array_almost_equal(
        dec2theta(np.array([0.0, -np.pi / 2]), units="rads"),
        np.array([np.pi / 2, np.pi])
    )


def test_sphere2cart():
    # Scalar cases
    x, y, z = sphere2cart(1., 0., np.pi / 2., units="rads")
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.0)

    x, y, z = sphere2cart(1., 0., 90., units="degs")
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.0)

    # Array cases
    x, y, z = sphere2cart(
        np.array([1., 1.]),
        np.array([0., np.pi]),
        np.array([np.pi / 2., np.pi / 2.]),
        units="rads"
    )
    np.testing.assert_array_almost_equal(x, np.array([1.0, -1.0]))
    np.testing.assert_array_almost_equal(y, np.array([0.0, 0.0]))
    np.testing.assert_array_almost_equal(z, np.array([0.0, 0.0]))


def test_sphere2cart_radec():
    # Scalar cases
    x, y, z = sphere2cart_radec(1., 0., 0., units="rads")
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.0)

    x, y, z = sphere2cart_radec(1., 0., 0., units="degs")
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)
    assert z == pytest.approx(0.0)


def test_cart2sphere():
    # Scalar cases
    r, phi, theta = cart2sphere(1., 0., 0., units="rads")
    assert r == pytest.approx(1.0)
    assert phi == pytest.approx(0.0)
    assert theta == pytest.approx(np.pi / 2)

    r, phi, theta = cart2sphere(1., 0., 0., units="degs")
    assert r == pytest.approx(1.0)
    assert phi == pytest.approx(0.0)
    assert theta == pytest.approx(90.0)


def test_cart2sphere_radec():
    # Scalar cases
    r, ra, dec = cart2sphere_radec(1., 0., 0., units="rads")
    assert r == pytest.approx(1.0)
    assert ra == pytest.approx(0.0)
    assert dec == pytest.approx(0.0)

    r, ra, dec = cart2sphere_radec(1., 0., 0., units="degs")
    assert r == pytest.approx(1.0)
    assert ra == pytest.approx(0.0)
    assert dec == pytest.approx(0.0)

    # Array cases
    r, ra, dec = cart2sphere_radec(
        np.array([1., 1.]),
        np.array([1., -1.]),
        np.array([1., 0.]),
        units="rads"
    )
    np.testing.assert_array_almost_equal(r, np.array([1.73205, 1.4142]), decimal=4)
    np.testing.assert_array_almost_equal(ra, np.array([0.7854, 5.4978]), decimal=4)
    np.testing.assert_array_almost_equal(dec, np.array([0.6155, 0.0]), decimal=4)
