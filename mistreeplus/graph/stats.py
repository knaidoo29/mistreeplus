import numpy as np

from .. import src


def get_edge_index(idx1: np.ndarray, idx2: np.ndarray) -> np.ndarray:
    """
    Combines edge indices into one array.

    Parameters
    ----------
    idx1, idx2 : array
        Graph edge node idxices.

    Returns
    -------
    edge_idx : 2darray
        Graph edge node indices.
    """
    edge_idx = np.array([idx1, idx2])
    return edge_idx


def get_stat_index(edge_idx: np.ndarray, stat: np.ndarray) -> np.ndarray:
    """
    Assigns statistics of the nodes to the edge indexes.

    Parameters
    ----------
    edge_idx : 2darray
        Graph edge node indices.
    stat : array
        A statistic at every node.

    Returns
    -------
    stat_idx : 2darray
        Edge index assigned statistics.
    """
    stat_idx = np.array([stat[edge_idx[0]], stat[edge_idx[1]]])
    return stat_idx


def get_degree(edge_idx: np.ndarray, Nnodes: int) -> np.ndarray:
    """
    Returns the degrees for the nodes.

    Parameters
    ----------
    edge_idx : 2darray
        Graph edge node indices.
    Nnodes : int
        Total number of nodes.

    Returns
    -------
    degree : array
        The degree of a node, i.e. the number of edges connecting to each node.
    """
    idx1, idx2 = edge_idx[0], edge_idx[1]
    degree = src.getgraphdegree(i1=idx1, i2=idx2, nnodes=Nnodes)
    return degree
