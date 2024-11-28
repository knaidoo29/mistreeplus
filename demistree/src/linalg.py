import numpy as np
from numba import jit


@jit(nopython=True)
def dotvector3(a, b):
    """
    Calculates the dot product of two vectors of length 3.

    Parameters
    ----------
    a : array-like
        Input vector.
    b : array-like
        Input vector.

    Returns
    -------
    c : float
        Dot product of a and b.
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@jit(nopython=True)
def dot3by3mat3vec(a, b):
    """
    Calculates the dot product of a 3x3 matrix and a vector of length 3.

    Parameters
    ----------
    a : array-like, shape (3, 3)
        Input 3x3 matrix.
    b : array-like, shape (3,)
        Input vector of length 3.

    Returns
    -------
    c : ndarray
        Dot product result as a vector of length 3.
    """
    c = np.zeros(3)
    c[0] = a[0, 0] * b[0] + a[0, 1] * b[1] + a[0, 2] * b[2]
    c[1] = a[1, 0] * b[0] + a[1, 1] * b[1] + a[1, 2] * b[2]
    c[2] = a[2, 0] * b[0] + a[2, 1] * b[1] + a[2, 2] * b[2]
    return c

@jit(nopython=True)
def crossvector3(a, b):
    """
    Calculates the cross product of two vectors of length 3.

    Parameters
    ----------
    a : array-like
        Input vector.
    b : array-like
        Input vector.

    Returns
    -------
    c : ndarray
        Cross product result as a vector of length 3.
    """
    c = np.zeros(3)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c

@jit(nopython=True)
def normalisevector(vecin):
    """
    Normalises an input vector.

    Parameters
    ----------
    vecin : array-like
        Input vector.

    Returns
    -------
    vecout : ndarray
        Normalized vector.
    """
    mag = 0.0
    for i in range(len(vecin)):
        mag += vecin[i] ** 2
    mag = np.sqrt(mag)
    vecout = np.zeros(len(vecin))
    for i in range(len(vecin)):
        vecout[i] = vecin[i] / mag
    return vecout

@jit(nopython=True)
def inv3by3(m):
    """
    Inverts a 3x3 matrix.

    Parameters
    ----------
    m : array-like, shape (3, 3)
        Input 3x3 matrix.

    Returns
    -------
    invm : ndarray
        Inverse of the input matrix.
    """
    a, b, c = m[0, 0], m[0, 1], m[0, 2]
    d, e, f = m[1, 0], m[1, 1], m[1, 2]
    g, h, i = m[2, 0], m[2, 1], m[2, 2]

    aa = e * i - f * h
    bb = -(d * i - f * g)
    cc = d * h - e * g
    dd = -(b * i - c * h)
    ee = a * i - c * g
    ff = -(a * h - b * g)
    gg = b * f - c * e
    hh = -(a * f - c * d)
    ii = a * e - b * d

    detm = a * aa + b * bb + c * cc
    if detm == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")

    invm = np.zeros((3, 3))
    invm[0, 0] = aa / detm
    invm[0, 1] = dd / detm
    invm[0, 2] = gg / detm
    invm[1, 0] = bb / detm
    invm[1, 1] = ee / detm
    invm[1, 2] = hh / detm
    invm[2, 0] = cc / detm
    invm[2, 1] = ff / detm
    invm[2, 2] = ii / detm

    return invm
