import numpy as np
from typing import Union


def dist2D(
    x1 : Union[float, np.ndarray], x2 : Union[float, np.ndarray],
    y1 : Union[float, np.ndarray], y2 : Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
    """
    Determines distance between two sets of points.

    Parameters
    ----------
    x1 : float or array
        X-coordinate of point 1.
    x2 : float or array
        X-coordinate of point 2.
    y1 : float or array
        Y-coordinate of point 1.
    y2 : float or array
        Y-coordinate of point 2.

    Returns
    -------
    r : float or array
        Distance.
    """
    r = np.sqrt((x1-x2)**2. + (y1-y2)**2.)
    return r


def dist3D(
    x1 : Union[float, np.ndarray], x2 : Union[float, np.ndarray],
    y1 : Union[float, np.ndarray], y2 : Union[float, np.ndarray],
    z1 : Union[float, np.ndarray], z2 : Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
    """
    Determines distance between two sets of points in 3D.

    Parameters
    ----------
    x1 : float or array
        X-coordinate of point 1.
    x2 : float or array
        X-coordinate of point 2.
    y1 : float or array
        Y-coordinate of point 1.
    y2 : float or array
        Y-coordinate of point 2.
    z1 : float or array
        Z-coordinate of point 1.
    z2 : float or array
        Z-coordinate of point 2.

    Returns
    -------
    r : float or array
        Distance.
    """
    r = np.sqrt((x1-x2)**2. + (y1-y2)**2. + (z1-z2)**2.)
    return r
