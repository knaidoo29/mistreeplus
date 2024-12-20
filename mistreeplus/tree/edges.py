import numpy as np


def get_edge_dict(edge_idx: np.ndarray, wei: np.ndarray, directed: bool = False) -> dict:
    """
    Returns the edge dictionary for easier weight finding.

    Parameters
    ----------
    id1, id2 : array
        Node indices for each side of a graph edge.
    wei : array
        Weight for each graph edge.
    directed : bool, optional
        Is the input graph directed? If yes then weight directions and 
        signs are flipped when rotated.
    
    Returns
    -------
    edge_dict : dict
        Edge dictionary to easily find weights.
    """
    edge_dict = {}
    for i in range(0, len(edge_idx[0])):
        edge_dict[(edge_idx[0][i], edge_idx[1][i])] = wei[i]
        if directed:
            edge_dict[(edge_idx[1][i], edge_idx[0][i])] = -wei[i]
        else:
            edge_dict[(edge_idx[1][i], edge_idx[0][i])] = wei[i]
    return edge_dict
