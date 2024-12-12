import numpy as np
from typing import Tuple, List, Optional

from . import groups


def get_adjacents(
        id1: np.ndarray, 
        id2: np.ndarray, 
        wei: np.ndarray, 
        Nnodes: int
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
        List containing each adjacent node idx in the graph or neighbours.
    adjacents_wei : list
        List containing each adjacent node weight in the graph or neighbours.
    """
    _adjacents_idx = [[] for i in range(0, Nnodes)]
    _adjacents_wei = [[] for i in range(0, Nnodes)]
    for i in range(0, len(id1)):
        _adjacents_idx[id1[i]].append(id2[i])
        _adjacents_idx[id2[i]].append(id1[i])
        _adjacents_wei[id1[i]].append(wei[i])
        _adjacents_wei[id2[i]].append(wei[i])
    # filter out repeats
    adjacents_idx, adjacents_wei = [], []
    for i in range(0, Nnodes):
        uni, ind = np.unique(_adjacents_idx[i], return_index=True)
        adjacents_idx.append(uni.tolist())
        adjacents_wei.append(np.array(_adjacents_wei[i])[ind].tolist())
    return adjacents_idx, adjacents_wei


def adjacents2tree(
        adjacents_idx: List[int], 
        Nnodes: int, 
        root: int = 0,
        sanity: bool = True
    ) -> dict:
    """
    Constructs a dictionary tree from a given graph and it's node adjacents.

    Parameters
    ----------
    adjacents_idx : list
        List containing each adjacent node idx in the graph or neighbours.
    Nnodes : int
        Number of nodes.
    root : int, optional
        The root of the tree, by default set to the first node.
    sanity : bool, optional
        Tests whether the input graph is spanning. This should only be turned off 
        if you already know the input graph is spanning.
    
    Returns
    -------
    tree : dict
        Graph structured in a tree.
    """
    if sanity:
        groupid = groups.get_groups(adjacents_idx, Nnodes, root=root)
        assert len(np.unique(groupid)) == 1, "Graph is not spanning, since it produces more than one group."
    
    tree = {}
    visited = np.zeros(Nnodes)

    tree[root] = {
        'parent': None, 
        'children': adjacents_idx[root]
    }
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
                _visited = visited[_adjacents_idx]
                cond = np.where(_visited == 0.)[0]
            
                if len(cond) == 0:
                    tree[child] = {
                        'parent': parent, 'children': None
                    }

                else:
                    tree[child] = {
                        'parent': parent, 
                        'children': _adjacents_idx[cond].tolist()
                    }

                    _visitnextparent.append(child)
                    _visitnextchild.append(_adjacents_idx[cond])
        
        visitparent = _visitnextparent
        visitchild = _visitnextchild
    
    return tree


def findpath2root(id: int, tree: dict) -> list:
    """
    Finds the path along a tree from point 1 to 2 by finding the path to the root index and removing
    common edges.

    Parameters
    ----------
    id : array
        Node index.
    tree : dict
        Graph structured in a tree.
    
    Returns
    -------
    pathtoroot : list
        Path from the node to the tree root node.
    """
    _id = id
    path = [_id]
    while tree[_id]['parent'] is not None:
        _id = tree[_id]['parent']
        path.append(_id)
    return path


# def find_path(id1: int, id2: int, tree: dict) -> list:
#     """
#     Finds the path along a tree from point 1 to 2 by finding the path to the root index and removing
#     common edges.

#     Parameters
#     ----------
#     id1, id2 : array
#         Node indices for each side of a graph edge.
#     tree : dict
#         Graph structured in a tree.
    
#     Returns
#     -------
#     path1to2 : list
#         Path from index 1 to 2.
#     """
#     path1toroot = find_path2root(id1, tree)
#     path2toroot = find_path2root(id2, tree)
#     len1 = len(path1toroot)
#     len2 = len(path2toroot)
#     if len1 <= len2:
#         lenmax = len1
#     else:
#         lenmax = len2 
#     pathrootto1 = np.array(path1toroot)[::-1]
#     pathrootto2 = np.array(path2toroot)[::-1]
#     cond = np.where(pathrootto1[:lenmax] != pathrootto2[:lenmax])[0]
#     if len(cond) != 0:
#         splitnode = cond[0]-1
#     else:
#         splitnode = 0
#     path1to2 = pathrootto1[splitnode:][::-1].tolist() + pathrootto2[splitnode+1:].tolist()
#     return path1to2


def findpath(id1: int, id2: int, tree: dict) -> list:
    """
    Finds the unique path between two nodes in a tree.

    Parameters
    ----------
    id1, id2 : int
        Node indices.
    tree : dict
        Graph structured in a tree.
    
    Returns
    -------
    list
        Path from id1 to id2.
    """
    # Get paths to root for both nodes
    path1 = findpath2root(id1, tree)
    path2 = findpath2root(id2, tree)

    # Use a set for quick lookup of common ancestors
    path2_set = set(path2)

    # Find the lowest common ancestor
    lca = next(node for node in path1 if node in path2_set)

    # Split paths at the LCA
    path1_to_lca = path1[:path1.index(lca) + 1]
    path2_to_lca = path2[:path2.index(lca)][::-1]  # Reverse to go from LCA to id2

    # Combine paths
    return path1_to_lca + path2_to_lca


# def get_path_weight(path: list, edge_dict: dict) -> float:
#     """
#     Finds the total weight of an input path in a graph.

#     Parameters
#     ----------
#     path : list
#         Path between points in a graph.
#     edge_dict : dict
#         Edge dictionary to easily find weights.
    
#     Returns
#     -------
#     weight : float
#         Total weight of an input path.
#     """
#     weight = 0.
#     for i in range(0, len(path)-1):
#         weight += edge_dict[(path[i], path[i+1])]
#     return weight

def get_path_weight(path: list, edge_dict: dict) -> float:
    """
    Finds the total weight of an input path in a graph.

    Parameters
    ----------
    path : list
        Path between points in a graph.
    edge_dict : dict
        Edge dictionary to easily find weights.
    
    Returns
    -------
    weight : float
        Total weight of an input path.
    """
    # Use sum with generator for efficient computation
    return sum(edge_dict[(path[i], path[i+1])] for i in range(len(path) - 1))
