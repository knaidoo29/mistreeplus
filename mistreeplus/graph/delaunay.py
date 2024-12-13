import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import Delaunay as scDelaunay

from . import convert
from . import stats

from .. import coords
from .. import index


def construct_del2D(x: np.ndarray, y: np.ndarray) -> csr_matrix:
    """
    Constructs the Delaunay graph from 2D points.

    Parameters
    ----------
    x, y : array
        Cartesian coordinates.

    Return
    ------
    del_graph : csr_matrix
        Delaunay graph.
    """
    vert = coords.xy2vert(x, y)
    # construct Delaunay triangulation
    delaunay = scDelaunay(vert)
    tri = delaunay.simplices
    idx1 = np.concatenate([tri[:, 0], tri[:, 1], tri[:, 2]])
    idx2 = np.concatenate([tri[:, 1], tri[:, 2], tri[:, 0]])
    idx = index.cantor_pair(idx1, idx2)
    idx = np.sort(idx)
    idx = np.unique(idx)
    idx1, idx2 = index.uncantor_pair(idx)
    edge_idx = stats.get_edge_index(idx1, idx2)
    dist = coords.dist2D(x[idx1], x[idx2], y[idx1], y[idx2])
    del_graph = convert.data2graph(edge_idx, dist, len(x))
    return del_graph


def construct_del3D(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> csr_matrix:
    """
    Constructs the Delaunay graph from 3D points.

    Parameters
    ----------
    x, y, z : array
        Cartesian coordinates.

    Return
    ------
    del_graph : csr_matrix
        Delaunay graph.
    """
    vert = coords.xyz2vert(x, y, z)
    # construct Delaunay triangulation
    delaunay = scDelaunay(vert)
    tri = delaunay.simplices
    idx1 = np.concatenate(
        [tri[:, 0], tri[:, 0], tri[:, 0], tri[:, 1], tri[:, 1], tri[:, 2]]
    )
    idx2 = np.concatenate(
        [tri[:, 1], tri[:, 2], tri[:, 3], tri[:, 2], tri[:, 3], tri[:, 3]]
    )
    idx = index.cantor_pair(idx1, idx2)
    idx = np.sort(idx)
    idx = np.unique(idx)
    idx1, idx2 = index.uncantor_pair(idx)
    edge_idx = stats.get_edge_index(idx1, idx2)
    dist = coords.dist3D(x[idx1], x[idx2], y[idx1], y[idx2], z[idx1], z[idx2])
    del_graph = convert.data2graph(edge_idx, dist, len(x))
    return del_graph
