import numpy as np
from numba import njit


@njit
def add2centrality(centrality: np.ndarray, idx1: int, idx2: int) -> np.ndarray:
    """
    Add to centrality array in place.

    Parameters
    ----------
    centrality : array
        The centrality of a node in a tree. This is initialised as an array of 
        ones and iteratively added to.
    idx1 : array
        Index to add.
    idx2 : array
        Index to be added.
    """
    for i in range(len(idx1)):
        centrality[idx1[i]] += centrality[idx2[i]]
    return centrality