import sparse_numeric_table as snt
import pytest


def test_dtype_assertion_message():
    with pytest.raises(AssertionError) as excinfo:
        snt.validating.assert_dtypes_are_valid(
            dtypes={"A": [("a", "?")], "B": [("b", "<u8"), ("c", "<i4")]}
        )
    expec = (
        "Level 'A', column 'a' has dtype '?' "
        "which is not in "
        "(<u1, <u2, <u4, <u8, <i1, <i2, <i4, <i8, <f2, <f4, <f8)."
    )
    assert str(excinfo.value) == expec
