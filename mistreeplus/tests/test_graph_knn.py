import pytest
import numpy as np
from scipy.sparse import csr_matrix
from mistreeplus.graph import construct_knn2D, construct_knn3D


# Test for construct_knn2D
def test_construct_knn2D():
    # Create 2D test points
    x = np.array([0, 1, 2, 3, 1, 2])
    y = np.array([0, 1, 0, 1, 2, 2])
    k = 2  # Number of nearest neighbours

    # Function call
    knn_graph = construct_knn2D(x, y, k)

    # Check that the output is a csr_matrix
    assert isinstance(knn_graph, csr_matrix)

    # Check that the graph has the correct shape (should be len(x) x len(x))
    assert knn_graph.shape == (len(x), len(x))

    # Check that the graph has non-zero entries (i.e., edges exist)
    assert knn_graph.nnz > 0

    # Check that each row has exactly k non-zero entries for k-NN
    assert all(np.sum(knn_graph[i].toarray()) > 0 for i in range(len(x)))


# Test for construct_knn3D
def test_construct_knn3D():
    # Create 3D test points
    x = np.array([0, 1, 2, 3, 1, 2])
    y = np.array([0, 1, 0, 1, 2, 2])
    z = np.array([0, 1, 2, 1, 0, 2])
    k = 2  # Number of nearest neighbours

    # Function call
    knn_graph = construct_knn3D(x, y, z, k)

    # Check that the output is a csr_matrix
    assert isinstance(knn_graph, csr_matrix)

    # Check that the graph has the correct shape (should be len(x) x len(x))
    assert knn_graph.shape == (len(x), len(x))

    # Check that the graph has non-zero entries (i.e., edges exist)
    assert knn_graph.nnz > 0

    # Check that each row has exactly k non-zero entries for k-NN
    assert all(np.sum(knn_graph[i].toarray()) > 0 for i in range(len(x)))
