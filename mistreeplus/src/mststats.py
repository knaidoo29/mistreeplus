import numpy as np
from numba import njit


@njit
def getgraphdegree(
    i1: np.ndarray, i2: np.ndarray, nnodes: int
) -> np.ndarray:
    """
    Given the edge index, this function computes the degree of each node.

    Parameters
    ----------
    i1, i2 : array-like
        The index of the edges of a tree, where '1' and '2' refer to the ends of each edge.
    nnodes : int
        The total number of nodes used to construct the tree.

    Returns
    -------
    degree : ndarray
        The degree of each node, i.e., the number of edges connecting to each node.
    """
    degree = np.zeros(nnodes, dtype=np.float64)
    nedges = len(i1)
    for i in range(nedges):
        degree[i1[i]] += 1.0
        degree[i2[i]] += 1.0
    return degree
