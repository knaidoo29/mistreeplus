import numpy as np
from typing import Tuple, List


def get_adjacents(
    edge_idx: np.ndarray, wei: np.ndarray, Nnodes: int
) -> Tuple[List[int], List[float]]:
    """
    Returns the adjacency list (the neighbours to each node in a graph) from a graph given in array format.

    Parameters
    ----------
    id1, id2 : array
        Node indices for each side of a graph edge.
    wei : array
        Weight for each graph edge.
    Nnodes : int
        Number of nodes.

    Returns
    -------
    adjacents_idx : list
        List containing each adjacent node index in the graph or neighbours.
    adjacents_wei : list
        List containing each adjacent node weight in the graph or neighbours.
    """
    _adjacents_idx = [[] for i in range(0, Nnodes)]
    _adjacents_wei = [[] for i in range(0, Nnodes)]
    for i in range(0, len(edge_idx[0])):
        _adjacents_idx[edge_idx[0][i]].append(edge_idx[1][i])
        _adjacents_idx[edge_idx[1][i]].append(edge_idx[0][i])
        _adjacents_wei[edge_idx[0][i]].append(wei[i])
        _adjacents_wei[edge_idx[1][i]].append(wei[i])
    # filter out repeats
    adjacents_idx, adjacents_wei = [], []
    for i in range(0, Nnodes):
        uni, ind = np.unique(_adjacents_idx[i], return_index=True)
        adjacents_idx.append(uni.tolist())
        adjacents_wei.append(np.array(_adjacents_wei[i])[ind].tolist())
    return adjacents_idx, adjacents_wei


def smooth_stat_with_graph(adj_idx: List[int], stat: np.ndarray, iterations: int) -> np.ndarray:
    """
    Smooths a statistic iteratively by average across adjacent nodes.

    Parameters
    ----------
    adjacents_idx : list
        List containing each adjacent node index in the graph or neighbours.
    stat : array
        Statistic defined at each node point.
    iterations : int
        Number of iterations to apply the smoothing.

    Returns
    -------
    smoothed : array
        Smoothed statistics.
    """
    tosmooth = np.copy(stat)
    iterate = 0
    while iterate < iterations:
        smoothed = [(tosmooth[i] + np.sum(tosmooth[adj_idx[i]]))/(len(adj_idx[i]) + 1) for i in range(0, len(adj_idx))]
        tosmooth = np.copy(smoothed)
        iterate += 1
    smoothed = np.array(smoothed)
    return smoothed