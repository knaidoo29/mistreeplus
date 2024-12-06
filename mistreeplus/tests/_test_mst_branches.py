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
    result = find_branches(edge_ind, degree, x=x, y=y)
    expected_result = [[0, 1, 2]]  # Adjust according to what you expect
    np.testing.assert_array_equal(result, expected_result)

# Test get_branch_weight function
def test_get_branch_weight(sample_data):
    edge_ind, degree, x, y = sample_data
    branch = [0, 1]  # Example branch for testing
    result = get_branch_weight(branch, edge_ind, x, y)
    expected_result = 2.0  # Adjust according to expected result
    assert result == expected_result

# Test get_branch_end_index function
def test_get_branch_end_index(sample_data):
    edge_ind, degree, x, y = sample_data
    branch = [0, 1]  # Example branch for testing
    result = get_branch_end_index(branch, edge_ind)
    expected_result = 2  # Adjust as necessary
    assert result == expected_result

# Test get_branch_edge_count function
def test_get_branch_edge_count(sample_data):
    edge_ind, degree, x, y = sample_data
    branch = [0, 1]  # Example branch for testing
    result = get_branch_edge_count(branch, edge_ind)
    expected_result = 2  # Adjust according to your expectation
    assert result == expected_result

# Test get_branch_shape function
def test_get_branch_shape(sample_data):
    edge_ind, degree, x, y = sample_data
    branch = [0, 1]  # Example branch for testing
    result = get_branch_shape(branch, edge_ind, x, y)
    expected_result = [(0.0, 0.0), (1.0, 0.0)]  # Adjust expected values as needed
    assert result == expected_result

# Test with edge cases
def test_empty_branch(sample_data):
    edge_ind, degree, x, y = sample_data
    branch = []
    result = get_branch_weight(branch, edge_ind, x, y)
    expected_result = 0.0  # Expected behavior when branch is empty
    assert result == expected_result

def test_invalid_data(sample_data):
    edge_ind, degree, x, y = sample_data
    branch = [0, 1, 5]  # Invalid index to test error handling
    with pytest.raises(IndexError):
        get_branch_weight(branch, edge_ind, x, y)
