import numpy as np
import pytest
from mistreeplus.src import getgraphdegree


def test_getgraphdegree_basic():
    # Example with a simple triangle graph
    i1 = np.array([0, 1, 2])
    i2 = np.array([1, 2, 0])
    nnodes = 3

    result = getgraphdegree(i1, i2, nnodes)

    # Expected degree for each node
    expected = np.array([2.0, 2.0, 2.0])  # Each node is connected to 2 others
    assert np.allclose(result, expected), "Degree computation for a triangle graph is incorrect"


def test_getgraphdegree_disconnected_nodes():
    # Graph with some disconnected nodes
    i1 = np.array([0, 1])
    i2 = np.array([1, 2])
    nnodes = 5

    result = getgraphdegree(i1, i2, nnodes)

    # Expected degree (nodes 3 and 4 are disconnected)
    expected = np.array([1.0, 2.0, 1.0, 0.0, 0.0])
    assert np.allclose(result, expected), "Degree computation for disconnected nodes is incorrect"


def test_getgraphdegree_single_edge():
    # Graph with only one edge
    i1 = np.array([0])
    i2 = np.array([1])
    nnodes = 2

    result = getgraphdegree(i1, i2, nnodes)

    # Expected degree
    expected = np.array([1.0, 1.0])
    assert np.allclose(result, expected), "Degree computation for a single edge is incorrect"


def test_getgraphdegree_no_edges():
    # Graph with no edges
    i1 = np.array([])
    i2 = np.array([])
    nnodes = 3

    result = getgraphdegree(i1, i2, nnodes)

    # Expected degree (all nodes are disconnected)
    expected = np.zeros(nnodes, dtype=np.float64)
    assert np.allclose(result, expected), "Degree computation for a graph with no edges is incorrect"


def test_getgraphdegree_large_graph():
    # Larger graph test case
    i1 = np.array([0, 1, 2, 3, 4, 5])
    i2 = np.array([1, 2, 3, 4, 5, 0])
    nnodes = 6

    result = getgraphdegree(i1, i2, nnodes)

    # Expected degree for a cycle graph
    expected = np.full(nnodes, 2.0)  # Each node in a cycle has 2 edges
    assert np.allclose(result, expected), "Degree computation for a large graph is incorrect"
