import numpy as np
from numba import njit


@njit
def periodicboundary(x : float, boxsize : float) -> float:
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
    steps : np.ndarray, prand : np.ndarray, boxsize : float, x0 : float, y0 : float,
    useperiodic : bool
    ) -> tuple(np.ndarray, np.ndarray):
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
    x = np.zeros(length, dtype=np.float64)
    y = np.zeros(length, dtype=np.float64)

    x[0] = x0
    y[0] = y0

    xnow = x0
    ynow = y0

    for i in range(1, length):
        dx = steps[i - 1] * np.cos(prand[i - 1])
        dy = steps[i - 1] * np.sin(prand[i - 1])

        xnow += dx
        ynow += dy

        if useperiodic == 1:
            xnow = periodicboundary(xnow, boxsize)
            ynow = periodicboundary(ynow, boxsize)

        x[i] = xnow
        y[i] = ynow

    return x, y

@njit
def randwalkcart3d(
    steps : np.ndarray, prand : np.ndarray, trand : np.ndarray,
    boxsize : float, x0 : float, y0 : float, z0 : float, useperiodic : bool
    ) -> tuple(np.ndarray, np.ndarray, np.ndarray):
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
    length = len(steps) + 1
    x = np.zeros(length, dtype=np.float64)
    y = np.zeros(length, dtype=np.float64)
    z = np.zeros(length, dtype=np.float64)

    x[0] = x0
    y[0] = y0
    z[0] = z0

    xnow = x0
    ynow = y0
    znow = z0

    for i in range(1, length):
        dx = steps[i - 1] * np.cos(prand[i - 1]) * np.sin(trand[i - 1])
        dy = steps[i - 1] * np.sin(prand[i - 1]) * np.sin(trand[i - 1])
        dz = steps[i - 1] * np.cos(trand[i - 1])

        xnow += dx
        ynow += dy
        znow += dz

        if useperiodic == 1:
            xnow = periodicboundary(xnow, boxsize)
            ynow = periodicboundary(ynow, boxsize)
            znow = periodicboundary(znow, boxsize)

        x[i] = xnow
        y[i] = ynow
        z[i] = znow

    return x, y, z
