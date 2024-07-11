from .base import DTYPES
from .base import dict_to_recarray

import numpy as np


def _assert_same_keys(keys_a, keys_b):
    """
    Asserts that two lists contain the same items, but order does not matter.
    """
    uni_keys = list(set(keys_a + keys_b))
    for key in uni_keys:
        assert key in keys_a and key in keys_b, "Key: {:s}".format(key)


def assert_tables_are_equal(table_a, table_b):
    _assert_same_keys(list(table_a.keys()), list(table_b.keys()))
    for level_key in table_a:
        _assert_same_keys(
            table_a[level_key].dtype.names, table_b[level_key].dtype.names
        )
        for column_key in table_a[level_key].dtype.names:
            assert (
                table_a[level_key].dtype[column_key]
                == table_b[level_key].dtype[column_key]
            )
            np.testing.assert_array_equal(
                x=table_a[level_key][column_key],
                y=table_b[level_key][column_key],
                err_msg="table[{:s}][{:s}]".format(level_key, column_key),
                verbose=True,
            )


def assert_table_has_dtypes(table, dtypes):
    for level_key in dtypes:
        assert (
            level_key in table
        ), "Expected level '{:s}' in table, but it is not.".format(level_key)

        for column in dtypes[level_key]:
            column_key = column[0]
            column_dtype = column[1]
            assert column_key in table[level_key].dtype.names, (
                "Expected column '{:s}' in table's level '{:s}', "
                "but it is not.".format(column_key, level_key)
            )
            assert column_dtype == table[level_key].dtype[column_key], (
                "Expected table[{level_key:s}][{column_key:s}].dtype "
                "== {column_dtype:s}, "
                "but actually it is {actual_dtype:s}".format(
                    level_key=level_key,
                    column_key=column_key,
                    column_dtype=str(column_dtype),
                    actual_dtype=str(table[level_key].dtype[column_key]),
                )
            )


def _assert_no_whitespace(key):
    for char in key:
        assert not str.isspace(
            char
        ), "Key must not contain spaces, but key = '{:s}'".format(key)


def _assert_no_dot(key):
    assert "." not in key, "Key must not contain '.', but key = '{:s}'".format(
        key
    )


def _assert_no_directory_delimeter(key):
    assert "/" not in key, "Key must not contain '/', but key = '{:s}'".format(
        key
    )
    assert (
        "\\" not in key
    ), "Key must not contain '\\', but key = '{:s}'".format(key)


def _assert_key_is_valid(key):
    _assert_no_whitespace(key)
    _assert_no_dot(key)
    _assert_no_directory_delimeter(key)


def assert_dtypes_keys_are_valid(dtypes):
    for level_key in dtypes:
        _assert_key_is_valid(level_key)
        for column in dtypes[level_key]:
            column_key = column[0]
            column_dtype = column[1]
            _assert_key_is_valid(column_key)
            assert column_dtype in DTYPES, (
                "level: {:s}, column: {:s} has dtype: {:s} "
                "which is not a valid for sparse_numeric_table.".format(
                    level_key,
                    column_key,
                    str(column_dtype),
                )
            )


def make_example_table_dtypes(index_dtype=("idx", "<u8")):
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


def make_example_table(prng, size, start_index=0, index_dtype=("idx", "<u8")):
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

    t = {}
    t["elementary_school"] = dict_to_recarray(
        {
            idx: start_index + np.arange(size).astype(idx_dtype),
            "lunchpack_size": prng.uniform(size=size).astype("<f8"),
            "num_friends": prng.uniform(low=0, high=5, size=size).astype(
                "<i8"
            ),
        }
    )
    high_school_size = size // 10
    t["high_school"] = dict_to_recarray(
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
    university_size = high_school_size // 10
    t["university"] = dict_to_recarray(
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
    assert_dtypes_keys_are_valid(dtypes=example_table_dtypes)
    assert_table_has_dtypes(table=t, dtypes=example_table_dtypes)
    return t


def assert_dtypes_equal(a, b):
    _assert_same_keys(list(b.keys()), list(b.keys()))
    for level_key in a:
        alvl = a[level_key]
        blvl = b[level_key]
        for i in range(len(alvl)):
            a_column_name = alvl[i][0]
            b_column_name = blvl[i][0]
            assert a_column_name == b_column_name
            a_column_dtype = alvl[i][1]
            b_column_dtype = blvl[i][1]
            assert a_column_dtype == b_column_dtype
