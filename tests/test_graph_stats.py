import pytest
import numpy as np
from mistreeplus.graph import get_edge_index, get_stat_index, get_degree

# Mock src.getgraphdegree to enable testing without the full src module
def mock_getgraphdegree(i1, i2, nnodes):
    """
    A mock function to simulate the behavior of src.getgraphdegree.
    Computes the degree of nodes based on the edge indices.
    """
    degree = np.zeros(nnodes, dtype=int)
    for i in i1:
        degree[i] += 1
    for i in i2:
        degree[i] += 1
    return degree

# Replace the actual function in the src module with the mock
import mistreeplus.src as src
src.getgraphdegree = mock_getgraphdegree


def test_get_edge_index():
    """Test the get_edge_index function."""
    ind1 = np.array([0, 1, 2])
    ind2 = np.array([1, 2, 0])

    edge_ind = get_edge_index(ind1, ind2)

    # Expected edge index array
    expected_edge_ind = np.array([[0, 1, 2], [1, 2, 0]])

    assert np.array_equal(edge_ind, expected_edge_ind), "get_edge_index did not return the expected edge indices."


def test_get_stat_index():
    """Test the get_stat_index function."""
    edge_ind = np.array([[0, 1, 2], [1, 2, 0]])
    stat = np.array([10, 20, 30])

    stat_ind = get_stat_index(edge_ind, stat)

    # Expected statistics assigned to edge indices
    expected_stat_ind = np.array([[10, 20, 30], [20, 30, 10]])

    assert np.array_equal(stat_ind, expected_stat_ind), "get_stat_index did not assign the correct statistics to the edge indices."


def test_get_degree():
    """Test the get_degree function."""
    edge_ind = np.array([[0, 1, 2], [1, 2, 0]])
    nnodes = 3

    degree = get_degree(edge_ind, nnodes)

    # Expected degrees for nodes
    expected_degree = np.array([2, 2, 2])  # Each node is connected to two others

    assert np.array_equal(degree, expected_degree), "get_degree did not return the correct node degrees."
