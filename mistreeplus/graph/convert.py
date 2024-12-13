import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple

from . import stats


def graph2data(graph: csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the index and data of a sparse csr_matrix

    Parameters
    ----------
    graph : csr_matrix
        A sparse matrix of the edges in a graph and corresponding node indexes.

    Returns
    -------
    edge_idx : array
        Graph edge node indices.
    weight : array
        Weights for each edge.
    """
    graph = graph.tocoo()
    weights = graph.data
    idx1 = graph.row
    idx2 = graph.col
    edge_idx = stats.get_edge_index(idx1, idx2)
    return edge_idx, weights


def data2graph(
    edge_idx: np.ndarray, weights: np.ndarray, Nnodes: int
) -> csr_matrix:
    """
    Returns the sparse matrix of a graph given the indices and data.

    Parameters
    ----------
    edge_idx : array
        Node indices of the graph.
    weights : array
        An array of the weights of each edge in the graph.
    Nnodes : int
        Total number of nodes.

    Returns
    -------
    graph : csr_matrix
        A sparse matrix of the edges in a graph and corresponding node indices.
    """
    graph = csr_matrix((weights, (edge_idx[0], edge_idx[1])), shape=(Nnodes, Nnodes))
    return graph
