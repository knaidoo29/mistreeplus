import numpy as np
from numba import njit
from typing import Tuple


@njit
def periodicboundary(x: float, boxsize: float) -> float:
    """
    Ensures particles remain within a periodic box.

    Parameters
    ----------
    x : float
        Position value.
    boxsize : float
        Box size.

    Returns
    -------
    x : float
        Adjusted position value.
    """
    while x < 0.0 or x > boxsize:
        if x < 0.0:
            x += boxsize
        elif x > boxsize:
            x -= boxsize
    return x


@njit
def randwalkcart2d(
    steps: np.ndarray,
    prand: np.ndarray,
    boxsize: float,
    x0: float,
    y0: float,
    useperiodic: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a random walk on a 2D grid.

    Parameters
    ----------
    steps : ndarray
        Array of step sizes.
    prand : ndarray
        Array of random angles.
    boxsize : float
        Box size.
    x0, y0 : float
        Starting x and y coordinates.
    useperiodic : int
        0 = does not enforce periodic boundary conditions.
        1 = enforces periodic boundary conditions.

    Returns
    -------
    x, y : ndarray
        Coordinates of the random walk simulation.
    """
    length = len(steps)
    x = np.zeros(length+1, dtype=np.float64)
    y = np.zeros(length+1, dtype=np.float64)

    x[0] = x0
    y[0] = y0

    xnow = x0
    ynow = y0

    for i in range(0, length):
        dx = steps[i] * np.cos(prand[i])
        dy = steps[i] * np.sin(prand[i])

        xnow += dx
        ynow += dy

        if useperiodic == 1:
            xnow = periodicboundary(xnow, boxsize)
            ynow = periodicboundary(ynow, boxsize)

        x[i+1] = xnow
        y[i+1] = ynow

    return x, y


@njit
def randwalkcart3d(
    steps: np.ndarray,
    prand: np.ndarray,
    trand: np.ndarray,
    boxsize: float,
    x0: float,
    y0: float,
    z0: float,
    useperiodic: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a random walk on a 3D grid.

    Parameters
    ----------
    steps : ndarray
        Array of step sizes.
    prand, trand : ndarray
        Random angles: phi (longitude) and theta (latitude).
    boxsize : float
        Box size.
    x0, y0, z0 : float
        Starting x, y, and z coordinates.
    useperiodic : int
        0 = does not enforce periodic boundary conditions.
        1 = enforces periodic boundary conditions.

    Returns
    -------
    x, y, z : ndarray
        Coordinates of the random walk simulation.
    """
    length = len(steps)
    x = np.zeros(length+1, dtype=np.float64)
    y = np.zeros(length+1, dtype=np.float64)
    z = np.zeros(length+1, dtype=np.float64)

    x[0] = x0
    y[0] = y0
    z[0] = z0

    xnow = x0
    ynow = y0
    znow = z0

    for i in range(0, length):
        dx = steps[i] * np.cos(prand[i]) * np.sin(trand[i])
        dy = steps[i] * np.sin(prand[i]) * np.sin(trand[i])
        dz = steps[i] * np.cos(trand[i])

        xnow += dx
        ynow += dy
        znow += dz

        if useperiodic == 1:
            xnow = periodicboundary(xnow, boxsize)
            ynow = periodicboundary(ynow, boxsize)
            znow = periodicboundary(znow, boxsize)

        x[i+1] = xnow
        y[i+1] = ynow
        z[i+1] = znow

    return x, y, z
