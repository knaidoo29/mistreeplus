import pytest
import numpy as np
from scipy.sparse import csr_matrix
from mistreeplus.graph import graph2data, data2graph

# Test for graph2data
def test_graph2data():
    # Create a sparse matrix representing a graph
    data = np.array([1.0, 2.0, 3.0])
    rows = np.array([0, 1, 2])
    cols = np.array([1, 2, 0])
    graph = csr_matrix((data, (rows, cols)), shape=(3, 3))

    # Expected output
    expected_ind1 = np.array([0, 1, 2])
    expected_ind2 = np.array([1, 2, 0])
    expected_weights = np.array([1.0, 2.0, 3.0])

    # Function call
    ind1, ind2, weights = graph2data(graph)

    # Assertions
    assert np.array_equal(ind1, expected_ind1)
    assert np.array_equal(ind2, expected_ind2)
    assert np.array_equal(weights, expected_weights)

# Test for data2graph
def test_data2graph():
    # Create test data
    ind1 = np.array([0, 1, 2])
    ind2 = np.array([1, 2, 0])
    weights = np.array([1.0, 2.0, 3.0])
    Nnodes = 3

    # Expected CSR matrix
    expected_graph = csr_matrix((weights, (ind1, ind2)), shape=(Nnodes, Nnodes))

    # Function call
    graph = data2graph(ind1, ind2, weights, Nnodes)

    # Assertions
    assert graph.shape == expected_graph.shape
    assert np.array_equal(graph.toarray(), expected_graph.toarray())
