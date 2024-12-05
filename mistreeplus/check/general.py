import numpy as np
from typing import Union


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
    if isinstance(values, float) == True:
        if values < 0:
            raise AssertionError("Values is negative.", values)
        else:
            pass
    else:
        cond = np.where(array < 0.0)[0]
        if len(cond) > 0:
            raise AssertionError("Some elements of the array are negative.")
        else:
            pass


def check_finite(values: Union[float, np.ndarray]) -> None:
    """Check values are finite."""
    if isinstance(values) == True:
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
