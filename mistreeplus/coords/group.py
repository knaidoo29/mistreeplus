import numpy as np
from typing import Optional, Union
from scipy.spatial import KDTree as scKDTree

from . import vertices


class KDTree2D:

    """Lightweight KDTree class in 2D"""

    def __init__(self):
        """Initialises the 2D KDTree class"""
        self.points = None
        self.KD = None


    def build_tree(self, x, y, boxsize=None):
        """Function for building the KDTree.

        Parameters
        ----------
        x : array
            X coordinates.
        y : array
            Y coordinates.
        boxsize : float, optional
            Periodic boundary boxsize.
        """
        self.points = vertices.xy2vert(x, y)
        self.KD = scKDTree(self.points, boxsize=boxsize)


    def nearest(self, x, y, k=1):
        """Returns the nearest index (and distance) of a point from the KDTree

        Parameters
        ----------
        x : array
            X coordinates.
        y : array
            Y coordinates.
        z : array
            Z coordinates.
        k : int
            Number of nearest points.
        
        Return
        ------
        nind : int
            Index of nearest point.
        ndist : float, optional
            Distance to nearest point.
        """
        points = vertices.xy2vert(x, y)
        ndist, nind = self.KD.query(points, k=k)
        return nind, ndist


    def clean(self):
        """Reinitilises the class."""
        self.__init__()


class KDTree3D:

    """Lightweight KDTree class in 3D"""

    def __init__(self):
        """Initialises the 3D KDTree class"""
        self.points = None
        self.KD = None


    def build_tree(self, x, y, z, boxsize=None):
        """Function for building the KDTree.

        Parameters
        ----------
        x : array
            X coordinates.
        y : array
            Y coordinates.
        z : array
            Z coordinates.
        boxsize : float, optional
            Periodic boundary boxsize.
        """
        self.points = vertices.xyz2vert(x, y, z)
        self.KD = scKDTree(self.points, boxsize=boxsize)


    def nearest(self, x, y, z, k=1):
        """Returns the nearest index (and distance) of a point from the KDTree

        Parameters
        ----------
        x : array
            X coordinates.
        y : array
            Y coordinates.
        z : array
            Z coordinates.
        k : int
            Number of nearest points.
        return_dist : bool, optional
            If True the distance to the nearest point will also be supplied.

        Return
        ------
        nind : int
            Index of nearest point.
        ndist : float, optional
            Distance to nearest point.
        """
        points = vertices.xyz2vert(x, y, z)
        ndist, nind = self.KD.query(points, k=k)
        return nind, ndist


    def clean(self):
        """Reinitilises the class."""
        self.__init__()


def PinchGroup2D(
    x: np.ndarray, 
    y: np.ndarray, 
    mindist: float, 
    w: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """
    Uses the hierarchical pinching algorithm to filter out small scales.

    Parameters
    ----------
    x, y : array
        2D coordinates.
    mindist : float
        Minimum distance between points.
    w : optional, array
        Weights for the points.
    
    Returns
    -------
    xg, yg : array
        2D coordinates of new group positions.
    wg : array
        Group weights.
    """
    xg, yg = np.copy(x), np.copy(y)
    if w is None:
        w = np.ones(len(x))
    wg = np.copy(w)
    run = True
    while run:
        mask = np.ones(len(xg))
        kd = KDTree2D()
        kd.build_tree(xg, yg)
        _idx, _dist = kd.nearest(xg, yg, k=2)
        idx = np.arange(len(_idx))
        dist = np.zeros(len(_dist))
        cond = np.where(_idx[:,0] != idx)[0]
        idx[cond] = _idx[cond,0]
        dist[cond] = _dist[cond,0]
        cond = np.where(_idx[:,1] != idx)[0]
        idx[cond] = _idx[cond,1]
        dist[cond] = _dist[cond,1]
        sidx = np.argsort(dist)
        newx = []
        newy = []
        neww = []
        for _id in sidx:
            if mask[_id] == 1. and mask[idx[_id]] == 1.:
                if dist[_id] <= mindist:
                    newx.append((wg[_id]*xg[_id] + wg[idx[_id]]*xg[idx[_id]])/(wg[_id] + wg[idx[_id]]))
                    newy.append((wg[_id]*yg[_id] + wg[idx[_id]]*yg[idx[_id]])/(wg[_id] + wg[idx[_id]]))
                    neww.append(wg[_id] + wg[idx[_id]])
                    mask[_id], mask[idx[_id]] = 0., 0.
        newx = np.array(newx)
        newy = np.array(newy)
        neww = np.array(neww)
        cond = np.where(mask == 1.)[0]
        xg = np.concatenate([xg[cond], newx])
        yg = np.concatenate([yg[cond], newy])
        wg = np.concatenate([wg[cond], neww])
        if len(cond) == len(xg):
            run = False
    return xg, yg, wg


def PinchGroup3D(
    x: np.ndarray, 
    y: np.ndarray, 
    z: np.ndarray, 
    mindist: float, 
    w: Optional[np.ndarray] = None
) -> Union[float, np.ndarray]:
    """
    Uses the hierarchical pinching algorithm to filter out small scales.

    Parameters
    ----------
    x, y, z : array
        3D coordinates.
    mindist : float
        Minimum distance between points.
    w : optional, array
        Weights for the points.
    
    Returns
    -------
    xg, yg, zg : array
        3D coordinates of new group positions.
    wg : array
        Group weights.
    """
    xg, yg, zg = np.copy(x), np.copy(y), np.copy(z)
    if w is None:
        w = np.ones(len(x))
    wg = np.copy(w)
    run = True
    while run:
        mask = np.ones(len(xg))
        kd = KDTree3D()
        kd.build_tree(xg, yg, zg)
        _idx, _dist = kd.nearest(xg, yg, zg, k=2)
        idx = np.arange(len(_idx))
        dist = np.zeros(len(_dist))
        cond = np.where(_idx[:,0] != idx)[0]
        idx[cond] = _idx[cond,0]
        dist[cond] = _dist[cond,0]
        cond = np.where(_idx[:,1] != idx)[0]
        idx[cond] = _idx[cond,1]
        dist[cond] = _dist[cond,1]
        sidx = np.argsort(dist)
        newx = []
        newy = []
        newz = []
        neww = []
        for _id in sidx:
            if mask[_id] == 1. and mask[idx[_id]] == 1.:
                if dist[_id] <= mindist:
                    newx.append((wg[_id]*xg[_id] + wg[idx[_id]]*xg[idx[_id]])/(wg[_id] + wg[idx[_id]]))
                    newy.append((wg[_id]*yg[_id] + wg[idx[_id]]*yg[idx[_id]])/(wg[_id] + wg[idx[_id]]))
                    newz.append((wg[_id]*zg[_id] + wg[idx[_id]]*zg[idx[_id]])/(wg[_id] + wg[idx[_id]]))
                    neww.append(wg[_id] + wg[idx[_id]])
                    mask[_id], mask[idx[_id]] = 0., 0.
        newx = np.array(newx)
        newy = np.array(newy)
        newz = np.array(newz)
        neww = np.array(neww)
        cond = np.where(mask == 1.)[0]
        xg = np.concatenate([xg[cond], newx])
        yg = np.concatenate([yg[cond], newy])
        zg = np.concatenate([zg[cond], newz])
        wg = np.concatenate([wg[cond], neww])
        if len(cond) == len(xg):
            run = False
    return xg, yg, zg, wg