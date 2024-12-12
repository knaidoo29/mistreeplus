import numpy as np
from typing import List


def get_groups(adjacents_idx: List[int], Nnodes: int, root: int = 0):
    """
    Groups connecting parts of a graph as single entities.

    Parameters
    ----------
    adjacents_idx : list
        List containing each adjacent node idx in the graph or neighbours.
    Nnodes : int
        Number of nodes.
    root : int, optional
        The root of the tree, by default set to the first node.
    
    Returns
    -------
    groupid : array
        Group IDs.
    """

    Nvisited = 0
    visited = np.zeros(Nnodes)
    groupid = np.zeros(Nnodes)
    currentid = 1

    tovisitnext = []

    invoked_root = False

    while Nvisited != Nnodes:

        if invoked_root == False:
            Nvisited += 1
            visited[root] = 1.
            groupid[root] = currentid
            _adjacents_idx = np.unique(np.array(adjacents_idx[root]))
            _visited = visited[_adjacents_idx]
            cond = np.where(_visited != 1.)[0]
            tovisitnext = _adjacents_idx[cond]
            invoked_root = True

        elif len(tovisitnext) == 0:
            cond = np.where(visited == 0.)[0]
            _root = cond[0]
            currentid += 1
            Nvisited += 1
            visited[_root] = 1.
            groupid[_root] = currentid
            _adjacents_idx = np.unique(np.array(adjacents_idx[_root]))
            _visited = visited[_adjacents_idx]
            cond = np.where(_visited != 1.)[0]
            tovisitnext = _adjacents_idx[cond]
        
        while len(tovisitnext) > 0:
            Nvisited += len(tovisitnext)
            visited[tovisitnext] = 1.
            groupid[tovisitnext] = currentid
            _adjacents_idx = np.unique(np.concatenate([adjacents_idx[i] for i in tovisitnext]))
            _visited = visited[_adjacents_idx]
            cond = np.where(_visited != 1.)[0]
            tovisitnext = _adjacents_idx[cond]

    return groupid