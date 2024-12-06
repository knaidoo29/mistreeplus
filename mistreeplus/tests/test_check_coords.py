import pytest
import numpy as np
from mistreeplus.check import (
    check_angle_units,
    check_phi_in_range,
    check_ra_in_range,
    check_theta_in_range,
    check_dec_in_range,
    check_r_unit_sphere,
)

def test_check_angle_units():
    # Valid cases
    check_angle_units("degs")
    check_angle_units("rads")

    # Invalid cases
    with pytest.raises(AssertionError):
        check_angle_units("degrees")
    with pytest.raises(AssertionError):
        check_angle_units("invalid")

def test_check_phi_in_range():
    # Valid cases
    check_phi_in_range(180.0, "degs")
    check_phi_in_range(3.14, "rads")
    check_phi_in_range(np.array([0, 180, 360]), "degs")
    check_phi_in_range(np.array([0, np.pi, 2 * np.pi]), "rads")

    # Invalid cases
    with pytest.raises(AssertionError):
        check_phi_in_range(-1.0, "degs")
    with pytest.raises(AssertionError):
        check_phi_in_range(400.0, "degs")
    with pytest.raises(AssertionError):
        check_phi_in_range(-0.1, "rads")
    with pytest.raises(AssertionError):
        check_phi_in_range(7.0, "rads")
    with pytest.raises(AssertionError):
        check_phi_in_range(np.array([-10, 180, 360]), "degs")
    with pytest.raises(AssertionError):
        check_phi_in_range(np.array([0, np.pi, 3 * np.pi]), "rads")

def test_check_ra_in_range():
    # Reuse test cases from check_phi_in_range since it calls the same function
    test_check_phi_in_range()

def test_check_theta_in_range():
    # Valid cases
    check_theta_in_range(90.0, "degs")
    check_theta_in_range(np.pi / 2, "rads")
    check_theta_in_range(np.array([0, 90, 180]), "degs")
    check_theta_in_range(np.array([0, np.pi / 2, np.pi]), "rads")

    # Invalid cases
    with pytest.raises(AssertionError):
        check_theta_in_range(-10.0, "degs")
    with pytest.raises(AssertionError):
        check_theta_in_range(200.0, "degs")
    with pytest.raises(AssertionError):
        check_theta_in_range(-0.1, "rads")
    with pytest.raises(AssertionError):
        check_theta_in_range(4.0, "rads")
    with pytest.raises(AssertionError):
        check_theta_in_range(np.array([-10, 90, 180]), "degs")
    with pytest.raises(AssertionError):
        check_theta_in_range(np.array([0, np.pi, 4.0]), "rads")

def test_check_dec_in_range():
    # Valid cases
    check_dec_in_range(45.0, "degs")
    check_dec_in_range(-np.pi / 4, "rads")
    check_dec_in_range(np.array([-90, 0, 90]), "degs")
    check_dec_in_range(np.array([-np.pi / 2, 0, np.pi / 2]), "rads")

    # Invalid cases
    with pytest.raises(AssertionError):
        check_dec_in_range(-100.0, "degs")
    with pytest.raises(AssertionError):
        check_dec_in_range(100.0, "degs")
    with pytest.raises(AssertionError):
        check_dec_in_range(-np.pi, "rads")
    with pytest.raises(AssertionError):
        check_dec_in_range(np.pi, "rads")
    with pytest.raises(AssertionError):
        check_dec_in_range(np.array([-100, 0, 90]), "degs")
    with pytest.raises(AssertionError):
        check_dec_in_range(np.array([-np.pi / 2, np.pi, np.pi / 2]), "rads")

def test_check_r_unit_sphere():
    # Valid cases
    check_r_unit_sphere(np.array([1.0, 1.000001]))
    check_r_unit_sphere(1.0)

    # Invalid cases
    with pytest.raises(AssertionError):
        check_r_unit_sphere(0.999)
    with pytest.raises(AssertionError):
        check_r_unit_sphere(np.array([1.0, 1.1]))
    with pytest.raises(AssertionError):
        check_r_unit_sphere(np.array([0.99, 1.0, 1.000001]))
