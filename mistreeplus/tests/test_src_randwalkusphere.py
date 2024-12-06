import numpy as np
import pytest
from unittest.mock import patch
from mistreeplus.src import usphererotate, randwalkusphere


# Mock linalg functions (if not provided in the context)
class MockLinalg:
    @staticmethod
    def crossvector3(u, v):
        return np.cross(u, v)

    @staticmethod
    def normalisevector(v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def dotvector3(u, v):
        return np.dot(u, v)

    @staticmethod
    def inv3by3(matrix):
        return np.linalg.inv(matrix)

    @staticmethod
    def dot3by3mat3vec(matrix, vec):
        return np.dot(matrix, vec)

@pytest.fixture(scope="module", autouse=True)
def setup_linalg_module():
    with patch("mistreeplus.src.linalg", MockLinalg):
        yield

# Test usphererotate
def test_usphererotate_identity_rotation():
    phi, theta = np.pi / 4, np.pi / 4
    phi_start, theta_start = 0, 0
    phi_final, theta_final = 0, 0

    phi_new, theta_new = usphererotate(phi, theta, phi_start, theta_start, phi_final, theta_final)

    assert np.isclose(phi_new, phi), "Identity rotation should not alter phi"
    assert np.isclose(theta_new, theta), "Identity rotation should not alter theta"


def test_usphererotate_orthogonal_rotation():
    phi, theta = np.pi / 2, np.pi / 2
    phi_start, theta_start = 0, np.pi / 2
    phi_final, theta_final = np.pi / 2, np.pi / 2

    phi_new, theta_new = usphererotate(phi, theta, phi_start, theta_start, phi_final, theta_final)

    assert np.isclose(phi_new, np.pi), "Orthogonal rotation alters phi incorrectly"
    assert np.isclose(theta_new, np.pi / 2), "Orthogonal rotation alters theta incorrectly"


def test_usphererotate_edge_cases():
    # Check rotation at poles
    phi, theta = 0, 0
    phi_start, theta_start = 0, 0
    phi_final, theta_final = np.pi, 0

    phi_new, theta_new = usphererotate(phi, theta, phi_start, theta_start, phi_final, theta_final)

    assert np.isclose(phi_new, np.pi), "Rotation at poles alters phi incorrectly"
    assert np.isclose(theta_new, 0), "Rotation at poles alters theta incorrectly"


# Test randwalkusphere
def test_randwalkusphere_no_steps():
    steps = np.array([])
    prand = np.array([])
    phi0, theta0 = np.pi / 4, np.pi / 4

    phi, theta = randwalkusphere(steps, prand, phi0, theta0)

    assert len(phi) == 1, "Random walk with no steps should have one position"
    assert np.isclose(phi[0], phi0), "Initial phi should remain unchanged"
    assert np.isclose(theta[0], theta0), "Initial theta should remain unchanged"


def test_randwalkusphere_single_step():
    steps = np.array([np.pi / 4])
    prand = np.array([2.*np.pi / 4])
    phi0, theta0 = np.pi / 4, np.pi / 4

    phi, theta = randwalkusphere(steps, prand, phi0, theta0)

    assert len(phi) == 2, "Random walk with one step should have two positions"
    assert np.isclose(phi[0], phi0), "Initial phi should be preserved"
    assert np.isclose(theta[0], theta0), "Initial theta should be preserved"
    assert not np.isclose(phi[1], phi0), "Phi should update after one step"
    assert not np.isclose(theta[1], theta0), "Theta should update after one step"


def test_randwalkusphere_multiple_steps():
    steps = np.array([np.pi / 4, np.pi / 6])
    prand = np.array([np.pi / 4, np.pi / 6])
    phi0, theta0 = np.pi / 4, np.pi / 4

    phi, theta = randwalkusphere(steps, prand, phi0, theta0)

    assert len(phi) == 3, "Random walk with two steps should have three positions"
    assert np.isclose(phi[0], phi0), "Initial phi should be preserved"
    assert np.isclose(theta[0], theta0), "Initial theta should be preserved"


def test_randwalkusphere_edge_cases():
    # Walk crossing a pole
    steps = np.array([np.pi])
    prand = np.array([0])
    phi0, theta0 = 0, np.pi / 2

    phi, theta = randwalkusphere(steps, prand, phi0, theta0)

    assert len(phi) == 2, "Random walk with one step crossing a pole should have two positions"
    assert np.isclose(theta[-1], np.pi / 2), "Theta crossing a pole should remain consistent"
