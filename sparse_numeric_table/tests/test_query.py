import sparse_numeric_table as snt
import pytest
import numpy as np
import pandas as pd
import tempfile
import os


def make_example_table(size=1_000):
    prng = np.random.Generator(np.random.MT19937(seed=0))

    index_dtype = ("i", "<u8")
    dtypes = {
        "A": [index_dtype, ("a", "<i8"), ("b", "<i8")],
        "B": [index_dtype, ("c", "<i8"), ("d", "<i8")],
        "C": [index_dtype, ("e", "<i8")],
    }
    t = snt.SparseNumericTable(index_key=index_dtype[0], dtypes=dtypes)

    for i in range(size):
        t["A"].append(
            {
                "i": i,
                "a": prng.integers(low=0, high=1000),
                "b": prng.integers(low=100, high=200),
            }
        )
        if prng.uniform() > 0.5:
            t["B"].append(
                {
                    "i": i,
                    "c": prng.integers(low=-100, high=100),
                    "d": prng.integers(low=0, high=10),
                }
            )
            if prng.uniform() > 0.5:
                t["C"].append(
                    {
                        "i": i,
                        "e": prng.integers(low=-100, high=100),
                    }
                )
    t.shrink_to_fit()
    return t


def query_from_file(table, block_size=100, **query_kwargs):
    with tempfile.TemporaryDirectory(prefix="test_snt_") as tmp:
        path = os.path.join(tmp, "table.snt.zip")
        with snt.open(
            path, "w", dtypes_and_index_key_from=table, block_size=block_size
        ) as f:
            f.append_table(table)

        with snt.open(path, "r") as f:
            back = f.query(**query_kwargs)
    return back


def query_from_self(table, **query_kwargs):
    return table.query(**query_kwargs)


FILE_AND_SELF = [query_from_file, query_from_self]


def test_query_all():
    for query in FILE_AND_SELF:
        a = make_example_table()
        b = query(table=a)
        snt.testing.assert_tables_are_equal(a, b)


def test_query_level_columns():
    for query in FILE_AND_SELF:
        a = make_example_table()
        b = query(table=a, levels_and_columns={"A": ("i", "a"), "C": ("i",)})

        np.testing.assert_array_equal(a["A"]["i"], b["A"]["i"])
        np.testing.assert_array_equal(a["A"]["a"], b["A"]["a"])
        np.testing.assert_array_equal(a["C"]["i"], b["C"]["i"])

        assert "B" in a and "B" not in b
        assert "b" in a["A"].dtype.names and "b" not in b["A"].dtype.names
        assert "e" in a["C"].dtype.names and "e" not in b["C"].dtype.names


def test_query_indices():
    for query in FILE_AND_SELF:
        a = make_example_table()
        indices = np.arange(100)
        b = query(table=a, indices=indices)

        for level in b:
            assert b[level].shape[0] <= indices.shape[0]
            for item in b[level]:
                assert item["i"] in indices

        for level in a:
            a_level_mask = snt.logic.make_mask_of_right_in_left(
                left_indices=a[level]["i"],
                right_indices=indices,
            )
            a_level_part = a[level][a_level_mask]

            np.testing.assert_array_equal(a_level_part, b[level])
