import numpy as np
from typing import Tuple, List, Optional

from . import groups
from .. import src


def get_adjacents(
        edge_idx: np.ndarray, 
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


def get_centrality(edge_idx: np.ndarray, Nnodes: int) -> np.ndarray:
    """
    Determines the nodes centrality in the graph.

    Parameters
    ----------
    edge_idx : 2darray
        Graph edge node indices.
    Nnodes : int
        Number of nodes.
    
    Returns
    -------
    centrality : array 
        The centrality of a node in a tree.
    """
     
    _id1 = np.copy(edge_idx[0])
    _id2 = np.copy(edge_idx[1])

    centrality = np.ones(Nnodes)

    while len(_id1) > 1:
        degree = src.getgraphdegree(i1=_id1, i2=_id2, nnodes=Nnodes)
        deg_id1, deg_id2 = degree[_id1], degree[_id2]
        cond = np.where((deg_id1 == 1.))[0]
        centrality = src.add2centrality(centrality, _id2[cond], _id1[cond])
        cond = np.where((deg_id2 == 1.))[0]
        centrality = src.add2centrality(centrality, _id1[cond], _id2[cond])
        cond = np.where((deg_id1 != 1.) & (deg_id2 != 1.))[0]
        _id1, _id2 = _id1[cond], _id2[cond]
    
    if len(_id1) == 1:
        if centrality[_id1[0]] >= centrality[_id2[0]]:
            centrality[_id1[0]] += centrality[_id2[0]]
        else:
            centrality[_id2[0]] += centrality[_id1[0]]

    return centrality


def get_spine(root: int, tree: dict, centrality: np.ndarray) -> list:
    """
    Descend a tree from a given root index along the main spine, i.e. the path with the largest nodes.

    Parameters
    ----------
    root : int
        Root index, for the very main spine this index is the central of the tree.
    tree : dict
        Graph structured in a tree.
    centrality : array 
        The centrality of a node in a tree.

    Returns
    -------
    spine : list
        Spine indices.
    """
    spine = [root]
    next2visit = spine[-1]
    while tree[next2visit]['children'] != None:
        if len(tree[next2visit]['children']) == 1:
            spine.append(tree[next2visit]['children'][0])
        else:
            nextidx = tree[next2visit]['children'][np.argmax(centrality[tree[next2visit]['children']])]
            spine.append(nextidx)
        next2visit = spine[-1]
    return spine


def get_spines(tree: dict, centrality: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Returns spines the spines of a graph tree structure. Ordered in spine hierarchy, 
    where the first spine is the backbone of the tree.

    Parameters
    ----------
    tree : dict
        Graph structured in a tree.
    centrality : array 
        The centrality of a node in a tree.

    Returns
    -------
    spines : list
        A list of spines.
    slevel : np.ndarray
        The spine level for each node.
    """
    Nnodes = len(centrality)
    mask = np.zeros(Nnodes)
    slevel = np.zeros(Nnodes)
    spines = []
    Nvisited = 0
    # Compute main spine
    # Find main branch from central
    _central = np.argmax(centrality)
    spine = get_spine(_central, tree, centrality)
    mask[np.array(spine)] = 1.
    # Find main branch going in the opposite direction along the path with the child 
    # node from the root node that has the second highest centrality
    cond = np.where(mask == 0.)[0]
    _central = cond[np.argmax(centrality[cond])]
    _spine = get_spine(_central, tree, centrality)
    mask[np.array(_spine)] = 1.
    # Merge two spines into one
    spine = np.array(spine)[::-1].tolist() + _spine
    spines.append(spine)
    Nvisited += len(spine)
    slevel[np.array(spine)] = 1
    # Compute sub spines
    _slevel = 2
    while Nvisited < Nnodes:
        cond = np.where(mask == 0.)[0]
        _central = cond[np.argmax(centrality[cond])]
        spine = get_spine(_central, tree, centrality)
        spines.append(spine)
        mask[np.array(spine)] = 1.
        slevel[np.array(spine)] = _slevel
        _slevel += 1
        Nvisited += len(spine)
    return spines, slevel