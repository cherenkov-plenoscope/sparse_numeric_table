import sparse_numeric_table as snt
import numpy as np
import pandas as pd
import tempfile
import os


def test_dynamic():
    prng = np.random.Generator(np.random.MT19937(seed=0))
    rnd = prng.uniform

    # define what your table will look like
    # -------------------------------------
    dtypes = {
        "A": [("a", "<f8"), ("b", "<f8")],
        "B": [("c", "<f8"), ("d", "<f8")],
        "C": [("e", "<f8")],
    }

    table = snt.init(dtypes=dtypes)

    dtypes_back = snt.get_dtypes(table=table)
    snt.testing.assert_dtypes_equal(dtypes, dtypes_back)

    sizes = snt.get_sizes(table=table)
    for level_key in sizes:
        assert sizes[level_key] == 0


def test_to_static_inplace_false():
    table = snt.init(dtypes=snt.testing.make_example_table_dtypes())
    stable = snt.to_static(table=table, inplace=False)
    assert snt.is_static(stable)
    assert snt.is_dynamic(table)
    assert table is not stable


def test_to_static_inplace_true():
    table = snt.init(dtypes=snt.testing.make_example_table_dtypes())
    stable = snt.to_static(table=table, inplace=True)
    assert snt.is_static(stable)
    assert snt.is_static(table)
    assert table is stable


def test_to_dynamic_inplace_false():
    table = snt.init(dtypes=snt.testing.make_example_table_dtypes())
    table = snt.to_static(table=table, inplace=True)

    dtable = snt.to_dynamic(table=table, inplace=False)
    assert snt.is_dynamic(dtable)
    assert snt.is_static(table)
    assert table is not dtable


def test_to_dynamic_inplace_true():
    table = snt.init(dtypes=snt.testing.make_example_table_dtypes())
    table = snt.to_static(table=table, inplace=True)

    dtable = snt.to_dynamic(table=table, inplace=True)
    assert snt.is_dynamic(dtable)
    assert snt.is_dynamic(table)
    assert table is dtable
