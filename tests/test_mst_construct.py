import pytest
import numpy as np
from scipy.sparse import csr_matrix
from mistreeplus.mst import construct_mst

# Test case for a simple graph
def test_construct_mst_simple():
    # Create a simple graph with 4 nodes and 5 edges
    data = np.array([1, 1, 1, 1, 1, 1])
    rows = np.array([0, 0, 1, 1, 2, 3])
    cols = np.array([1, 2, 2, 3, 3, 0])
    graph = csr_matrix((data, (rows, cols)), shape=(4, 4))

    # Construct the MST
    mst_graph = construct_mst(graph)

    # Expected MST should have correct edges
    expected_data = np.array([1, 1, 1])
    expected_rows = np.array([0, 0, 1])
    expected_cols = np.array([1, 2, 3])
    expected_mst = csr_matrix((expected_data, (expected_rows, expected_cols)), shape=(4, 4))

    # Check if the MST matches the expected graph by comparing the data and indices
    assert (mst_graph.toarray().astype(np.int64) == expected_mst.toarray().astype(np.int64)).all(), "The MST does not match the expected graph"

# You might need to modify other test cases similarly to avoid data type issues.

# Test case for a graph with a single edge (minimum case)
def test_construct_mst_single_edge():
    data = np.array([1])
    rows = np.array([0])
    cols = np.array([1])
    graph = csr_matrix((data, (rows, cols)), shape=(2, 2))

    mst_graph = construct_mst(graph)

    expected_mst = csr_matrix((data, (rows, cols)), shape=(2, 2))

    assert (mst_graph != expected_mst).nnz == 0  # Check if the MST matches the expected graph

# Test case for a fully connected graph
def test_construct_mst_fully_connected():
    data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    rows = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3])
    cols = np.array([1, 2, 3, 0, 2, 3, 0, 1, 0])
    graph = csr_matrix((data, (rows, cols)), shape=(4, 4))

    mst_graph = construct_mst(graph)

    # Verify that the MST has 3 edges for a graph with 4 nodes (n-1 edges)
    assert mst_graph.nnz == 3  # Number of edges in a MST for 4 nodes should be 3

# Test for graph with self-loops
def test_construct_mst_self_loops():
    data = np.array([1, 1, 1, 1, 1, 1])
    rows = np.array([0, 0, 1, 1, 2, 3])
    cols = np.array([1, 2, 0, 2, 3, 3])
    graph = csr_matrix((data, (rows, cols)), shape=(4, 4))

    mst_graph = construct_mst(graph)

    # Check that the MST does not include self-loops
    assert np.array_equal(mst_graph.toarray(), [[0, 1, 1, 0],
                                                [0, 0, 0, 0],
                                                [0, 0, 0, 1],
                                                [0, 0, 0, 0]])

# Edge case: empty graph
def test_construct_mst_empty():
    graph = csr_matrix((0, 0))  # Empty graph with no nodes or edges

    mst_graph = construct_mst(graph)

    # The MST of an empty graph should also be empty
    assert mst_graph.shape == (0, 0)
    assert mst_graph.nnz == 0
