from ._sparse_numeric_table import SparseNumericTable
from . import validating

import numpy as np
import pandas


def dict_to_recarray(d):
    return pandas.DataFrame(d).to_records(index=False)


def assert_lists_have_same_items_regardless_of_order(keys_a, keys_b):
    """
    Asserts that two lists contain the same items, but order does not matter.
    """
    uni_keys = list(set(keys_a + keys_b))
    for key in uni_keys:
        assert key in keys_a and key in keys_b, f"Key: {key:s}"


def assert_tables_are_equal(table_a, table_b):
    assert_lists_have_same_items_regardless_of_order(
        list(table_a.keys()), list(table_b.keys())
    )
    for level_key in table_a:
        assert_lists_have_same_items_regardless_of_order(
            table_a[level_key].dtype.names, table_b[level_key].dtype.names
        )
        for column_key in table_a[level_key].dtype.names:
            assert (
                table_a[level_key].dtype[column_key]
                == table_b[level_key].dtype[column_key]
            )
            np.testing.assert_array_equal(
                actual=table_a[level_key][column_key],
                desired=table_b[level_key][column_key],
                err_msg=f"table[{level_key:s}][{column_key:s}]",
                verbose=True,
            )


def assert_dtypes_are_equal(a, b):
    """
    Parameters
    ----------
    a : dict of lists of tuples
        The dtypes of table 'a'.
    b : dict of lists of tuples
        The dtypes of table 'b'.
    """
    assert_lists_have_same_items_regardless_of_order(
        list(b.keys()), list(b.keys())
    )

    for lkey in a:
        assert (
            lkey in b
        ), f"Expected level key '{lkey:s}' from 'a' to be in 'b'."

        assert len(a[lkey]) == len(b[lkey]), (
            f"Expected level '{lkey:s}' to have same number of columns in both"
            "'a' and 'b'."
        )

        for i in range(len(a[lkey])):
            acol = a[lkey][i]
            bcol = b[lkey][i]

            assert acol[0] == bcol[0], f"Expected columns to have same keys."
            assert acol[1] == bcol[1], f"Expected columns to have same dtypes."


def make_example_table_dtypes(index_dtype=("uid", "<u8")):
    return {
        "elementary_school": [
            index_dtype,
            ("lunchpack_size", "<f8"),
            ("num_friends", "<i8"),
        ],
        "high_school": [
            index_dtype,
            ("time_spent_on_homework", "<f8"),
            ("num_best_friends", "<i8"),
        ],
        "university": [
            index_dtype,
            ("num_missed_classes", "<i8"),
            ("num_fellow_students", "<i8"),
        ],
    }


def make_example_table(prng, size, start_index=0, index_dtype=("uid", "<u8")):
    """
    Children start in elementary school. 10% progress to high school, and 10%
    of those progress to university.
    At each point in their career statistics are collected that can be put to
    columns, while every child is represented by a line.
    Unfortunately, a typical example of a sparse table.
    """
    idx = index_dtype[0]
    idx_dtype = index_dtype[1]

    example_table_dtypes = make_example_table_dtypes(index_dtype=index_dtype)

    t = SparseNumericTable(dtypes=example_table_dtypes, index_key=idx)
    t["elementary_school"].append_recarray(
        dict_to_recarray(
            {
                idx: start_index + np.arange(size).astype(idx_dtype),
                "lunchpack_size": prng.uniform(size=size).astype("<f8"),
                "num_friends": prng.uniform(low=0, high=5, size=size).astype(
                    "<i8"
                ),
            }
        )
    )
    high_school_size = size // 10
    t["high_school"].append_recarray(
        dict_to_recarray(
            {
                idx: prng.choice(
                    t["elementary_school"][idx],
                    size=high_school_size,
                    replace=False,
                ),
                "time_spent_on_homework": 100
                + 100 * prng.uniform(size=high_school_size).astype("<f8"),
                "num_best_friends": prng.uniform(
                    low=0, high=5, size=high_school_size
                ).astype("<i8"),
            }
        )
    )
    university_size = high_school_size // 10
    t["university"].append_recarray(
        dict_to_recarray(
            {
                idx: prng.choice(
                    t["high_school"][idx], size=university_size, replace=False
                ),
                "num_missed_classes": 100
                * prng.uniform(size=university_size).astype("<i8"),
                "num_fellow_students": prng.uniform(
                    low=0, high=5, size=university_size
                ).astype("<i8"),
            }
        )
    )
    validating.assert_dtypes_are_valid(dtypes=example_table_dtypes)
    assert_dtypes_are_equal(a=t.dtypes, b=example_table_dtypes)
    return t
