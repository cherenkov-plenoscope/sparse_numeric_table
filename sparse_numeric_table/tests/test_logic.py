import sparse_numeric_table as snt
import numpy as np


def _intersection_dtypes(list_of_dtypes):
    out = snt.logic.intersection(
        [np.array([1, 2, 3], dtype=dt) for dt in list_of_dtypes]
    )
    return out.dtype


def test_intersection():
    assert _intersection_dtypes([int, np.uint64]) == int
    assert _intersection_dtypes([np.uint64, np.uint64]) == np.uint64
    assert _intersection_dtypes([np.uint64, np.uint16]) == np.uint64
    assert _intersection_dtypes([np.uint16, np.uint16]) == np.uint16
    assert _intersection_dtypes([np.uint64, np.uint16, float]) == float
    assert _intersection_dtypes([int, np.uint16, float]) == float
