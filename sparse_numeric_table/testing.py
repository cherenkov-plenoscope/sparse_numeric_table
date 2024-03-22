from .base import IDX
from .base import IDX_DTYPE
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


def assert_table_has_structure(table, structure):
    for level_key in structure:
        assert (
            level_key in table
        ), "Expected level '{:s}' in table, but it is not.".format(level_key)
        assert IDX in table[level_key].dtype.names, (
            "Expected table[{:s}] to have column '{:s}', "
            "but it has not.".format(level_key, IDX)
        )
        assert IDX_DTYPE == table[level_key].dtype[IDX], (
            "Expected table[{:s}][{:s}].dtype == {:s}"
            "but actually it is {:s}.".format(
                level_key,
                IDX,
                str(IDX_DTYPE),
                str(table[level_key].dtype[IDX]),
            )
        )
        for column_key in structure[level_key]:
            assert column_key in table[level_key].dtype.names, (
                "Expected column '{:s}' in table's level '{:s}', "
                "but it is not.".format(column_key, level_key)
            )
            expected_dtype = structure[level_key][column_key]["dtype"]
            assert expected_dtype == table[level_key].dtype[column_key], (
                "Expected table[{level_key:s}][{column_key:s}].dtype "
                "== {expected_dtype:s}, "
                "but actually it is {actual_dtype:s}".format(
                    level_key=level_key,
                    column_key=column_key,
                    expected_dtype=str(expected_dtype),
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


def assert_structure_keys_are_valid(structure):
    for level_key in structure:
        _assert_key_is_valid(level_key)
        for column_key in structure[level_key]:
            assert IDX != column_key
            _assert_key_is_valid(column_key)
            assert structure[level_key][column_key]["dtype"] in DTYPES, (
                "Structure[{:s}][{:s}]['dtype'] = {:s} "
                "is not in DTYPES".format(
                    level_key,
                    column_key,
                    str(structure[level_key][column_key]["dtype"]),
                )
            )


def make_example_table_structure():
    return {
        "elementary_school": {
            "lunchpack_size": {"dtype": "<f8"},
            "num_friends": {"dtype": "<i8"},
        },
        "high_school": {
            "time_spent_on_homework": {"dtype": "<f8"},
            "num_best_friends": {"dtype": "<i8"},
        },
        "university": {
            "num_missed_classes": {"dtype": "<i8"},
            "num_fellow_students": {"dtype": "<i8"},
        },
    }


def make_example_table(prng, size, start_index=0):
    """
    Children start in elementary school. 10% progress to high school, and 10%
    of those progress to university.
    At each point in their career statistics are collected that can be put to
    columns, while every child is represented by a line.
    Unfortunately, a typical example of a sparse table.
    """

    example_table_structure = make_example_table_structure()

    t = {}
    t["elementary_school"] = dict_to_recarray(
        {
            IDX: start_index + np.arange(size).astype(IDX_DTYPE),
            "lunchpack_size": prng.uniform(size=size).astype("<f8"),
            "num_friends": prng.uniform(low=0, high=5, size=size).astype(
                "<i8"
            ),
        }
    )
    high_school_size = size // 10
    t["high_school"] = dict_to_recarray(
        {
            IDX: prng.choice(
                t["elementary_school"][IDX],
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
            IDX: prng.choice(
                t["high_school"][IDX], size=university_size, replace=False
            ),
            "num_missed_classes": 100
            * prng.uniform(size=university_size).astype("<i8"),
            "num_fellow_students": prng.uniform(
                low=0, high=5, size=university_size
            ).astype("<i8"),
        }
    )
    assert_structure_keys_are_valid(structure=example_table_structure)
    assert_table_has_structure(table=t, structure=example_table_structure)
    return t
