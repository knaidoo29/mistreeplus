import pytest
import numpy as np
from mistreeplus.mst import find_branches, get_branch_weight, get_branch_end_index, get_branch_edge_count, get_branch_shape

# Sample data for testing
@pytest.fixture
def sample_data():
    edge_ind = np.array([[0, 1, 2], [1, 2, 3]])
    degree = np.array([1, 2, 2, 1])
    x = np.array([0.0, 1.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    return edge_ind, degree, x, y

# Test find_branches function
def test_find_branches(sample_data):
    edge_ind, degree, x, y = sample_data
    result, _ = find_branches(edge_ind, degree, x=x, y=y)
    expected_result = [[0, 1, 2]]  # Adjust according to what you expect
    np.testing.assert_array_equal(result, expected_result)

# Test get_branch_weight function
def test_get_branch_weight(sample_data):
    edge_ind, degree, x, y = sample_data
    weight = np.sqrt((x[edge_ind[0]]-x[edge_ind[1]])**2. + (y[edge_ind[0]]-y[edge_ind[1]])**2.)
    branch = [0, 1]  # Example branch for testing
    result = get_branch_weight(branch, weight)
    expected_result = 1.0  # Adjust according to expected result
    np.testing.assert_array_equal(result, expected_result)

# Test get_branch_end_index function
def test_get_branch_end_index(sample_data):
    edge_ind, degree, x, y = sample_data
    branch = [[0, 1, 2]]  # Example branch for testing
    edge_deg = np.array([degree[edge_ind[0]], degree[edge_ind[1]]])
    result = get_branch_end_index(edge_ind, edge_deg, branch)
    expected_result = [[0],[3]]  # Adjust as necessary
    assert all(result == expected_result)

# Test get_branch_edge_count function
def test_get_branch_edge_count(sample_data):
    edge_ind, degree, x, y = sample_data
    branch = [[0, 1, 2]]  # Example branch for testing
    result = get_branch_edge_count(branch)
    expected_result = [3]  # Adjust according to your expectation
    assert result == expected_result

# Test get_branch_shape function
def test_get_branch_shape(sample_data):
    edge_ind, degree, x, y = sample_data
    branch = [[0, 1, 2]]  # Example branch for testing
    edge_deg = np.array([degree[edge_ind[0]], degree[edge_ind[1]]])
    weight = np.sqrt((x[edge_ind[0]]-x[edge_ind[1]])**2. + (y[edge_ind[0]]-y[edge_ind[1]])**2.)
    branch_wei = get_branch_weight(branch, weight)
    result = get_branch_shape(edge_ind, edge_deg, branch, branch_wei, mode='2D', x=x, y=y)
    expected_result = [np.sqrt(5)/branch_wei[0]]  # Adjust expected values as needed
    assert all(result == expected_result)
