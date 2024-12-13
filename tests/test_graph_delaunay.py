import pytest
import numpy as np
from scipy.sparse import csr_matrix
from mistreeplus.graph import construct_del2D, construct_del3D

# Test for construct_delaunay2D
def test_construct_del2D():
    # Create 2D test points
    x = np.array([0, 1, 2, 3, 1, 2])
    y = np.array([0, 1, 0, 1, 2, 2])

    # Function call
    del_graph = construct_del2D(x, y)

    # Check that the output is a csr_matrix
    assert isinstance(del_graph, csr_matrix)

    # Check that the graph is not empty and has a proper structure
    assert del_graph.shape == (len(x), len(x))
    assert del_graph.nnz > 0  # Ensure that there are non-zero entries

# Test for construct_delaunay3D
def test_construct_del3D():
    # Create 3D test points
    x = np.array([0, 1, 2, 3, 1, 2])
    y = np.array([0, 1, 0, 1, 2, 2])
    z = np.array([0, 1, 2, 1, 0, 2])

    # Function call
    del_graph = construct_del3D(x, y, z)

    # Check that the output is a csr_matrix
    assert isinstance(del_graph, csr_matrix)

    # Check that the graph is not empty and has a proper structure
    assert del_graph.shape == (len(x), len(x))
    assert del_graph.nnz > 0  # Ensure that there are non-zero entries
