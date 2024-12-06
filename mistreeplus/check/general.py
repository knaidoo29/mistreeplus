import numpy as np
from typing import Union


def check_isscalar(x):
    """
    More general isscalar function to prevent 0 dimensional numpy arrays
    from being misidentified as arrays even though they are actually scalar
    variables.
    """
    if type(x).__module__ == np.__name__:
        if len(x.shape) == 0:
            return True
        else:
            return False
    else:
        return np.isscalar(x)


def check_length(array: np.ndarray, length: int) -> None:
    """Checks array is of the desired length."""
    if len(array) == length:
        pass
    else:
        raise AssertionError(
            "Length of array does not match expected length.", len(array)
        )


def check_positive(values: Union[float, np.ndarray]) -> None:
    """Checks values is positive."""
    if check_isscalar(values) == True:
        if values < 0:
            raise AssertionError("Values is negative.", values)
        else:
            pass
    else:
        cond = np.where(values < 0.0)[0]
        if len(cond) > 0:
            raise AssertionError("Some elements of the array are negative.")
        else:
            pass


def check_finite(values: Union[float, np.ndarray]) -> None:
    """Check values are finite."""
    if check_isscalar(values) == True:
        if np.isfinite(values) == False:
            raise AssertionError("Values are not finite.", values)
        else:
            pass
    else:
        cond = np.where(np.isfinite(values) == False)[0]
        if len(cond) > 0:
            raise AssertionError("Some elements of the array are not finite.")
        else:
            pass
