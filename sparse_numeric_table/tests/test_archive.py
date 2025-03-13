import sparse_numeric_table as snt
import numpy as np
import tempfile
import pytest
import os


def test_write_read_full_table():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    my_table = snt.testing.make_example_table(prng=prng, size=100_100)
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "my_table.zip")

        with snt.open(
            path, "w", dtypes_and_index_key_from=my_table, block_size=10_000
        ) as f:
            f.append_table(my_table)

        with snt.open(path, "r") as f:
            my_table_back = f.query()

        snt.testing.assert_tables_are_equal(my_table, my_table_back)


def test_read_only_part_of_table():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    my_table = snt.testing.make_example_table(
        prng=prng, size=100_000, index_dtype=("uid", "<u8")
    )

    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "my_table.zip")

        with snt.open(
            path, "w", dtypes_and_index_key_from=my_table, block_size=10_000
        ) as f:
            f.append_table(my_table)

        with snt.open(path, "r") as f:
            partly_back = f.query(
                levels_and_columns={
                    "elementary_school": ["uid", "num_friends"]
                }
            )

        np.testing.assert_array_equal(
            my_table["elementary_school"]["num_friends"],
            partly_back["elementary_school"]["num_friends"],
        )
        np.testing.assert_array_equal(
            my_table["elementary_school"]["uid"],
            partly_back["elementary_school"]["uid"],
        )


def test_column_command():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    my_table = snt.testing.make_example_table(prng=prng, size=100_000)
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "my_table.zip")

        with snt.open(
            path, "w", dtypes_and_index_key_from=my_table, block_size=10_000
        ) as f:
            f.append_table(my_table)

        with snt.open(path, "r") as f:
            partly_back = f.query(
                levels_and_columns={"elementary_school": "__all__"}
            )

        with pytest.raises(KeyError):
            _ = f.query(levels_and_columns={"elementary_school": "foo"})

        np.testing.assert_array_equal(
            my_table["elementary_school"],
            partly_back["elementary_school"],
        )


def test_preserves_dtypes_without_writing_anything():
    table = snt.SparseNumericTable(
        index_key="uid",
        dtypes=snt.testing.make_example_table_dtypes(
            index_dtype=("uid", "<i4")
        ),
    )

    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "table.zip")

        with snt.open(path, "w", dtypes_and_index_key_from=table) as f:
            pass  # do not write anything.

        with snt.open(path, "r") as f:
            back = f.query()

        snt.testing.assert_dtypes_are_equal(table.dtypes, back.dtypes)


def test_preserves_dtypes_when_table_empty():
    table = snt.SparseNumericTable(
        index_key="uid",
        dtypes=snt.testing.make_example_table_dtypes(
            index_dtype=("uid", "<i4")
        ),
    )

    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "table.zip")

        with snt.open(path, "w", dtypes_and_index_key_from=table) as f:
            f.append_table(table)

        with snt.open(path, "r") as f:
            back = f.query()

        snt.testing.assert_dtypes_are_equal(table.dtypes, back.dtypes)
