import numpy as np
from typing import Tuple, List, Optional


def get_adjacents(id1: np.ndarray, id2: np.ndarray, wei: np.ndarray, Nnodes: int) -> Tuple[List[int], List[float]]:
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
        List containing each adjacent node idx in the graph or neighbours.
    adjacents_wei : list
        List containing each adjacent node weight in the graph or neighbours.
    """
    adjacents_idx = [[] for i in range(0, Nnodes)]
    adjacents_wei = [[] for i in range(0, Nnodes)]
    for i in range(0, len(id1)):
        adjacents_idx[id1[i]].append(id2[i])
        adjacents_idx[id2[i]].append(id1[i])
        adjacents_wei[id1[i]].append(wei[i])
        adjacents_wei[id2[i]].append(wei[i])
    return adjacents_idx, adjacents_wei


def adjacents2tree(
        adjacents_idx: List[int], 
        Nnodes: int, 
        adjacents_wei: Optional[List[float]] = None,
        root: int = 0
    ) -> dict:
    """
    Constructs a dictionary tree from a given graph and it's node adjacents.

    Parameters
    ----------
    adjacents_idx : list
        List containing each adjacent node idx in the graph or neighbours.
    Nnodes : int
        Number of nodes.
    adjacents_wei : list, optional
        List containing each adjacent node weight in the graph or neighbours.
    root : int, optional
        The root of the tree, by default set to the first node.
    
    Returns
    -------
    tree : dict
        Graph structured in a tree.
    """

    tree = {}
    visited = np.zeros(Nnodes)

    tree[root] = {'parent': None, 'children': adjacents_idx[root]}
    visited[root] = 1.

    visitparent = []
    visitchild = []

    visitparent.append(root)
    visitchild.append(adjacents_idx[root])

    while len(visitparent) != 0:

        _visitnextparent = []
        _visitnextchild = []

        for i in range(0, len(visitchild)):

            parent = visitparent[i]
            children = visitchild[i]
            
            for child in children:
            
                visited[child] = 1.
                _adjacents_idx = np.array(adjacents_idx[child])

                if adjacents_wei is None:
                    _adjacents_wei = None
                else:
                    _adjacents_wei = np.array(adjacents_wei[child])
                
                _visited = visited[_adjacents_idx]
                cond = np.where(_visited == 0.)[0]
            
                if len(cond) == 0:
                    tree[child] = {'parent': parent, 'children': None, 'weights': None}

                else:
                    if _adjacents_wei is None:
                        tree[child] = {'parent': parent, 'children': _adjacents_idx[cond].tolist(), 'weights': None}
                    else:
                        tree[child] = {'parent': parent, 'children': _adjacents_idx[cond].tolist(), 'weights': _adjacents_wei[cond].tolist()}
        
                    _visitnextparent.append(child)
                    _visitnextchild.append(_adjacents_idx[cond])
        
        visitparent = _visitnextparent
        visitchild = _visitnextchild
    
    return tree