import numpy as np
from typing import List, Optional

from .. import coords

def perc_from_root_by_N(
    adjacents_idx: List[int], Npoint: int, root: int, percpaths: Optional[List[int]] = None
) -> List[int]:
    """
    Finds the percolation paths for points N points away in the graph from the root node.

    Parameters
    ----------
    adjacents_idx : list
        List containing each adjacent node index in the graph or neighbours.
    Npoint : int
        N-point distance to the root node.
    root : int
        Root node.
    percpaths : list
        N-point percolation paths in the graph, for precaculated paths up to some Npoint smaller than the choosen Npoint.
    
    Returns
    -------
    percpaths : list
        N-point percolation paths in the graph.
    """
    if percpaths is None:
        percpaths = [[root, adjacents] for adjacents in adjacents_idx[root]]
        count = 2
    else:
        count = len(percpaths[0])
    while count < Npoint + 1:
        for i in range(0, len(percpaths)):
            neigh = adjacents_idx[percpaths[i][-1]]
            neigh = [x for x in neigh if x not in percpaths[i][:-1]]
            percpaths[i].append(neigh)
        _percpaths = []
        for i in range(0, len(percpaths)):
            for j in range(0, len(percpaths[i][-1])):
                _percpaths.append(
                    percpaths[i][:-1] + [percpaths[i][-1][j]]
                )
        percpaths = _percpaths
        count += 1
    return percpaths


def _structure_percpaths(percpaths: np.ndarray) -> List[int]:
    """
    Restructures percpaths to retain root nested loop format.
    
    Parameters
    ----------
    percpaths : array
        N-point percolation paths in the graph.
    """
    _percpaths = []
    root = 0
    __percpaths = []
    for percpath in percpaths:
        if percpath[0] == root:
            __percpaths.append(percpath.tolist())
        else:
            _percpaths.append(__percpaths)
            root += 1
            __percpaths = []
            __percpaths.append(percpath.tolist())
    return _percpaths


def perc_from_all_by_N(adjacents_idx: List[int], Npoint: int, percpaths: Optional[List[int]] = None) -> np.ndarray:
    """
    Finds the percolation paths for points N-points away in the graph from all nodes.

    Parameters
    ----------
    adjacents_idx : list
        List containing each adjacent node index in the graph or neighbours.
    Npoint : int
        N-point distance to the root node.

    Returns
    -------
    percpaths : array
        N-point percolation paths in the graph.
    """
    if percpaths is None:
        _percpaths = [
            perc_from_root_by_N(adjacents_idx, Npoint, root)
            for root in range(0, len(adjacents_idx))
        ]
    else:
        __percpaths = _structure_percpaths(percpaths)
        _percpaths = [
            perc_from_root_by_N(adjacents_idx, Npoint, root, percpaths=__percpaths[root])
            for root in range(0, len(__percpaths))
        ]
    percpaths = []
    for __percpaths in _percpaths:
        if len(__percpaths) > 0:
            percpaths.append(__percpaths)
    percpaths = np.concatenate(_percpaths)
    return percpaths


def percpath2weight(percpaths: np.ndarray, edge_dict: dict) -> np.ndarray:
    """
    Get path weight for adjacent paths in a graph.

    Parameters
    ----------
    percpaths : array
        N-point percolation paths in the graph.
    edge_dict : dict
        Edge dictionary to easily find weights.

    Returns
    -------
    pathweight : array
        Weight for each path.
    """
    pathweight = np.array(
        [
            np.sum(
                [
                    edge_dict[(percpaths[i][j], percpaths[i][j + 1])]
                    for j in range(0, len(percpaths[i]) - 1)
                ]
            )
            for i in range(0, len(percpaths))
        ]
    )
    return pathweight


def percpath2percends(percpaths: np.ndarray) -> np.ndarray:
    """
    Get path ends for adjacent paths in a graph.

    Parameters
    ----------
    adjacent_paths : array
        N-point paths in the graph.

    Returns
    -------
    pathends : array
        Nodes of the ends for each path.
    """
    percends = np.array([percpaths[:,0], percpaths[:,-1]])
    return percends


def percend_dist2D(x: np.ndarray, y: np.ndarray, percends: np.ndarray) -> np.ndarray:
    """
    Get the distance between path ends.

    Parameters
    ----------
    x : array
        X-coordinate of points.
    y : array
        Y-coordinate of points.
    pathends : array
        Nodes of the ends for each path.

    Returns
    -------
    pathend_dist : array
        Distance between pathends.
    """
    percend_dist = coords.dist2D(x[percends[0]], x[percends[1]], y[percends[0]], y[percends[1]])
    return percend_dist


def percend_dist3D(x: np.ndarray, y: np.ndarray, z: np.ndarray, percends: np.ndarray) -> np.ndarray:
    """
    Get the distance between path ends.

    Parameters
    ----------
    x : array
        X-coordinate of points.
    y : array
        Y-coordinate of points.
    z : array
        Z-coordinate of points.
    pathends : array
        Nodes of the ends for each path.

    Returns
    -------
    pathend_dist : array
        Distance between pathends.
    """
    percend_dist = coords.dist3D(x[percends[0]], x[percends[1]], y[percends[0]], y[percends[1]], z[percends[0]], z[percends[1]])
    return percend_dist