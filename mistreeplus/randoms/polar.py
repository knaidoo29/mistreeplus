import numpy as np
from typing import Tuple, List

from . import cart

from .. import check


def polar_r(size: int, rmin: float = 0.0, rmax: float = 1.0) -> np.ndarray:
    """
    Generates random radial values in a disc segment with inner radius rmin
    and outer radius rmax.

    Parameters
    ----------
    size : int
        Size of the output sample.
    rmin : float, optional
        Minimum radial distance.
    rmax : float, optional
        Maximum radial distance.

    Returns
    -------
    rrand : array
        Random radial values in polar coordinates.
    """
    check.check_positive(rmin)
    check.check_positive(rmax)
    u = cart.cart1d(size)
    rrand = np.sqrt((rmax**2.0 - rmin**2.0) * u + rmin**2.0)
    return rrand


def polar_phi(
    size: int, phimin: float = 0.0, phimax: float = 2.0 * np.pi, units: str = "rads"
) -> np.ndarray:
    """
    Generates random angles in polar coordinates.

    Parameters
    ----------
    size : int
        Size of the output sample.
    phimin : float, optional
        Minimum phi angle.
    phimax : float, optional
        Maximum phi angle.
    units : str, optional
        Angular units, either 'degs' for degrees or 'rads' for radians.

    Returns
    -------
    prand : array
        Random phis in polar coordinates.
    """
    check.check_angle_units(units)
    check.check_phi_in_range(phimin, units)
    check.check_phi_in_range(phimax, units)
    prand = cart.cart1d(size, xmin=phimin, xmax=phimax)
    return prand


def polar_rphi(
    size: int,
    mins: List[float] = [0.0, 0.0],
    maxs: List[float] = [1.0, 2.0 * np.pi],
    units: str = "rads",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates randoms in polar coordinates.

    Parameters
    ----------
    size : int
        Size of the output sample.
    mins : list
        Minimum values in each axis, i.e. [rmin, pmin].
    maxs : list
        Maximum values in each axis, i.e. [rmax, pmax].
    units : str, optional
        Angular units, either 'degs' for degrees or 'rads' for radians.

    Returns
    -------
    rrand, prand : array
        Randoms in polar coordinates.
    """
    rrand = polar_r(size, rmin=mins[0], rmax=maxs[0])
    prand = polar_phi(size, phimin=mins[1], phimax=maxs[1], units=units)
    return rrand, prand
