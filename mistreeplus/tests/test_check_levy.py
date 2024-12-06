import pytest
from mistreeplus.check import check_levy_mode

def test_check_levy_mode():
    # Valid cases
    check_levy_mode("2D")
    check_levy_mode("3D")
    check_levy_mode("usphere")

    # Invalid cases
    with pytest.raises(AssertionError):
        check_levy_mode("1D")
    with pytest.raises(AssertionError):
        check_levy_mode("sphere")
    with pytest.raises(AssertionError):
        check_levy_mode("")
    with pytest.raises(AssertionError):
        check_levy_mode(None)  # Testing with None
    with pytest.raises(AssertionError):
        check_levy_mode("2d")  # Case sensitivity
