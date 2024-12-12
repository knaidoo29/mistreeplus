import numpy as np


def get_edge_dict(
        id1: np.ndarray, 
        id2: np.ndarray, 
        wei: np.ndarray
    ) -> dict:
    """
    Returns the edge dictionary for easier weight finding.

    Parameters
    ----------
    id1, id2 : array
        Node indices for each side of a graph edge.
    wei : array
        Weight for each graph edge.
    
    Returns
    -------
    edge_dict : dict
        Edge dictionary to easily find weights.
    """
    edge_dict = {}
    for i in range(0, len(id1)):
        edge_dict[(id1[i], id2[i])] = wei[i]
        edge_dict[(id2[i], id1[i])] = wei[i]
    return edge_dict