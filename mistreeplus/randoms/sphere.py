import numpy as np
from typing import Tuple, List

from . import cart
from . import usphere

from .. import check


def sphere_r(
    size: int, rmin: float = 0.0, rmax: float = 1.0, units: str = "rads"
) -> np.ndarray:
    """
    Generates random radial values in spherical polar coordinates.

    Parameters
    ----------
    size : int
        Size of the output sample.
    rmin : float
        Minimum radial distance.
    rmax : float
        Maximum radial distance.
    units : str
        Angular units, either 'degs' for degrees or 'rads' for radians.

    Returns
    -------
    rrand : array
        Randoms radial values in spherical polar coordinates.
    """
    check.check_positive(rmin)
    check.check_positive(rmax)
    u = cart.cart1d(size)
    rrand = ((rmax**3.0 - rmin**3.0) * u + rmin**3.0) ** (1.0 / 3.0)
    return rrand


def sphere_phi(
    size: int, phimin: float = 0.0, phimax: float = 2.0 * np.pi, units: str = "rads"
) -> np.ndarray:
    """
    Generates random phi values in spherical polar coordinates.

    Parameters
    ----------
    size : int
        Size of the output sample.
    mins : list
        Minimum values in each axis.
    maxs : list
        Maximum values in each axis.
    units : str
        Angular units, either 'degs' for degrees or 'rads' for radians.

    Returns
    -------
    prand : array
        Randoms phi values in spherical polar coordinates.
    """
    prand = usphere.usphere_phi(size, phimin=phimin, phimax=phimax, units=units)
    return prand


def sphere_theta(
    size: int, thetamin: float = 0.0, thetamax: float = np.pi, units: str = "rads"
) -> np.ndarray:
    """
    Generates random theta values in spherical polar coordinates.

    Parameters
    ----------
    size : int
        Size of the output sample.
    thetamin : float
        Minimum theta angle.
    thetamax : float
        Maximum theta angle.
    units : str
        Angular units, either 'degs' for degrees or 'rads' for radians.

    Returns
    -------
    trand : array
        Randoms theta values in spherical polar coordinates.
    """
    trand = usphere.usphere_theta(
        size, thetamin=thetamin, thetamax=thetamax, units=units
    )
    return trand


def sphere_rphitheta(
    size: int,
    mins: List[float] = [0.0, 0.0, 0.0],
    maxs: List[float] = [1.0, 2.0 * np.pi, np.pi],
    units: str = "rads",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates randoms in spherical polar coordinates.

    Parameters
    ----------
    size : int
        Size of the output sample.
    mins : list
        Minimum values in each axis, i.e. mins=[rmin, phimin, thetamin].
    maxs : list
        Maximum values in each axis, i.e. maxs=[rmax, phimax, thetamax].
    units : str
        Angular units, either 'degs' for degrees or 'rads' for radians.

    Returns
    -------
    rrand, prand, trand : array
        Randoms in spherical polar coordinates.
    """
    rrand = sphere_r(size, rmin=mins[0], rmax=maxs[0])
    prand = sphere_phi(size, phimin=mins[1], phimax=maxs[1], units=units)
    trand = sphere_theta(size, thetamin=mins[2], thetamax=maxs[2], units=units)
    return rrand, prand, trand
