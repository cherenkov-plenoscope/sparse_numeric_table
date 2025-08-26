import sparse_numeric_table as snt
import pytest
import numpy as np
import pandas as pd
import tempfile
import os


def test_repr():
    table = snt.SparseNumericTable(index_key="raletrale")
    _repr = table.__repr__()
    assert "SparseNumericTable(index_key='raletrale')" in _repr


def test_from_records():
    prng = np.random.Generator(np.random.MT19937(seed=0))
    rnd = prng.uniform

    # define what your table will look like
    # -------------------------------------
    index_dtype = ("i", "<u8")
    dtypes = {
        "A": [index_dtype, ("a", "<f8"), ("b", "<f8")],
        "B": [index_dtype, ("c", "<f8"), ("d", "<f8")],
        "C": [index_dtype, ("e", "<f8")],
    }

    # populate the table using records
    # --------------------------------
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        num_jobs = 100
        n = 5
        job_result_paths = []
        for j in range(num_jobs):
            # map the population of the sparse table onto many jobs
            # -----------------------------------------------------
            i = j * n
            db = snt.SparseNumericTable(
                index_key=index_dtype[0], dtypes=dtypes
            )

            db["A"].append({"i": i + 0, "a": rnd(), "b": rnd()})
            db["A"].append({"i": i + 1, "a": rnd(), "b": rnd()})
            db["A"].append({"i": i + 2, "a": rnd(), "b": rnd()})
            db["A"].append({"i": i + 3, "a": rnd(), "b": rnd()})
            db["A"].append({"i": i + 4, "a": rnd(), "b": rnd()})

            db["B"].append({"i": i + 0, "c": rnd(), "d": 5 * rnd()})
            db["B"].append({"i": i + 3, "c": rnd(), "d": 5 * rnd()})

            if rnd() > 0.9:
                db["C"].append({"i": i + 3, "e": -rnd()})

            path = os.path.join(tmp, "{:06d}.zip".format(j))
            job_result_paths.append(path)
            snt.testing.assert_dtypes_are_equal(a=db.dtypes, b=dtypes)
            with snt.open(path, "w", dtypes_and_index_key_from=db) as tout:
                tout.append_table(db)

        # reduce
        # ------
        full_path = os.path.join(tmp, "full.zip")
        snt.concatenate_files(
            input_paths=job_result_paths,
            output_path=full_path,
            dtypes=dtypes,
            index_key=index_dtype[0],
        )
        with snt.open(full_path, "r") as tin:
            full_table = tin.query()

    snt.testing.assert_dtypes_are_equal(a=full_table.dtypes, b=dtypes)


def test_write_read_full_table():
    prng = np.random.Generator(np.random.MT19937(seed=1337))
    table = snt.testing.make_example_table(prng=prng, size=1000 * 1000)

    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        zpath = os.path.join(tmp, "table.zip")
        with snt.open(zpath, "w", dtypes_and_index_key_from=table) as tout:
            tout.append_table(table)
        with snt.open(zpath, "r") as tin:
            zback = tin.query()
        snt.testing.assert_dtypes_are_equal(table.dtypes, zback.dtypes)
        snt.testing.assert_tables_are_equal(table, zback)


def test_write_read_empty_table():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    empty = snt.testing.make_example_table(prng=prng, size=0)
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        zpath = os.path.join(tmp, "empty.zip")
        with snt.open(zpath, "w", dtypes_and_index_key_from=empty) as tout:
            tout.append_table(empty)
        with snt.open(zpath, "r") as tin:
            zback = tin.query()
        snt.testing.assert_dtypes_are_equal(empty.dtypes, zback.dtypes)
        snt.testing.assert_tables_are_equal(empty, zback)


def test_merge_common():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    index_dtype = ("i", "<u8")
    my_table = snt.testing.make_example_table(
        prng=prng, size=1000 * 1000, index_dtype=index_dtype
    )

    common_indices = snt.logic.intersection(
        *[my_table[lvl]["i"] for lvl in my_table]
    )

    my_common_table = snt.logic.cut_table_on_indices(
        table=my_table,
        common_indices=common_indices,
    )
    my_sorted_common_table = snt.logic.sort_table_on_common_indices(
        table=my_common_table,
        common_indices=common_indices,
    )

    np.testing.assert_array_equal(
        my_sorted_common_table["elementary_school"]["i"],
        my_sorted_common_table["high_school"]["i"],
    )

    np.testing.assert_array_equal(
        my_sorted_common_table["elementary_school"]["i"],
        my_sorted_common_table["university"]["i"],
    )

    my_common_df = snt.logic.make_rectangular_DataFrame(
        table=my_sorted_common_table,
        index_key="i",
    )

    np.testing.assert_array_equal(
        my_sorted_common_table["elementary_school"]["i"],
        my_common_df["i"],
    )


def test_merge_across_all_levels_random_order_indices():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    size = 1000 * 1000
    index_dtype = ("i", "<u8")
    my_table = snt.testing.make_example_table(
        prng=prng, size=size, index_dtype=index_dtype
    )

    has_elementary_school = my_table["elementary_school"]["i"]
    has_high_school = my_table["high_school"]["i"]
    has_university = my_table["university"]["i"]
    has_big_lunchpack = my_table["elementary_school"]["i"][
        my_table["elementary_school"]["lunchpack_size"] > 0.5
    ]
    has_2best_friends = my_table["high_school"]["i"][
        my_table["high_school"]["num_best_friends"] >= 2
    ]

    cut_indices = np.intersect1d(has_elementary_school, has_high_school)
    cut_indices = np.intersect1d(cut_indices, has_university)
    cut_indices = np.intersect1d(cut_indices, has_big_lunchpack)
    cut_indices = np.intersect1d(cut_indices, has_2best_friends)
    np.random.shuffle(cut_indices)

    cut_table = snt.logic.cut_table_on_indices(
        table=my_table,
        common_indices=cut_indices,
        index_key="i",
        level_keys=["elementary_school", "high_school", "university"],
    )
    sorted_cut_table = snt.logic.sort_table_on_common_indices(
        table=cut_table,
        common_indices=cut_indices,
        index_key="i",
    )

    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"]["i"],
        sorted_cut_table["high_school"]["i"],
    )
    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"]["i"],
        sorted_cut_table["university"]["i"],
    )
    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"]["i"], cut_indices
    )


def test_merge_random_order_indices():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    index_dtype = ("uid", "<u8")
    size = 1000 * 1000
    my_table = snt.testing.make_example_table(
        prng=prng, size=size, index_dtype=index_dtype
    )

    has_elementary_school = my_table["elementary_school"]["uid"]
    has_high_school = my_table["high_school"]["uid"]
    has_big_lunchpack = my_table["elementary_school"]["uid"][
        my_table["elementary_school"]["lunchpack_size"] > 0.5
    ]
    has_2best_friends = my_table["high_school"]["uid"][
        my_table["high_school"]["num_best_friends"] >= 2
    ]

    cut_indices = np.intersect1d(has_elementary_school, has_high_school)
    cut_indices = np.intersect1d(cut_indices, has_big_lunchpack)
    cut_indices = np.intersect1d(cut_indices, has_2best_friends)
    np.random.shuffle(cut_indices)

    cut_table = snt.logic.cut_table_on_indices(
        table=my_table,
        common_indices=cut_indices,
        level_keys=["elementary_school", "high_school"],
        index_key="uid",
    )
    sorted_cut_table = snt.logic.sort_table_on_common_indices(
        table=cut_table,
        common_indices=cut_indices,
        index_key="uid",
    )

    assert "university" not in sorted_cut_table
    assert "elementary_school" in sorted_cut_table
    assert "high_school" in sorted_cut_table

    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"]["uid"],
        sorted_cut_table["high_school"]["uid"],
    )
    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"]["uid"], cut_indices
    )


def test_concatenate_several_tables():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    block_size = 10 * 1000
    num_blocks = 100

    index_dtype = ("uid", "<u8")

    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        paths = []
        for i in range(num_blocks):
            table_i = snt.testing.make_example_table(
                prng=prng,
                size=block_size,
                start_index=i * block_size,
                index_dtype=index_dtype,
            )
            table_i_dtypes = snt.testing.make_example_table_dtypes()
            paths.append(os.path.join(tmp, "{:06d}.zip".format(i)))
            snt.testing.assert_dtypes_are_equal(
                table_i.dtypes,
                table_i_dtypes,
            )
            with snt.open(
                paths[-1], "w", dtypes_and_index_key_from=table_i
            ) as f:
                f.append_table(table_i)

        output_path = os.path.join(tmp, "full.zip")
        snt.concatenate_files(
            input_paths=paths,
            output_path=output_path,
            dtypes=table_i_dtypes,
            index_key=index_dtype[0],
        )
        with snt.open(output_path, "r") as tin:
            full_table = tin.query()

    snt.testing.assert_dtypes_are_equal(full_table.dtypes, table_i_dtypes)

    assert (
        full_table["elementary_school"]["uid"].shape[0]
        == num_blocks * block_size
    )
    assert (
        len(set(full_table["elementary_school"]["uid"]))
        == num_blocks * block_size
    ), "The indices must be uniqe"
    assert (
        full_table["high_school"]["uid"].shape[0]
        == num_blocks * block_size // 10
    )
    assert (
        len(set(full_table["high_school"]["uid"]))
        == num_blocks * block_size // 10
    )
    assert (
        full_table["university"]["uid"].shape[0]
        == num_blocks * block_size // 100
    )
    assert (
        len(set(full_table["university"]["uid"]))
        == num_blocks * block_size // 100
    )


def test_concatenate_empty_list_of_paths():
    dtypes = snt.testing.make_example_table_dtypes(index_dtype=("uid", "<u8"))
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        output_path = os.path.join(tmp, "empty_table.tar")

        snt.concatenate_files(
            input_paths=[],
            output_path=output_path,
            dtypes=dtypes,
            index_key="uid",
        )

        with snt.open(output_path, "r") as tin:
            empty_table = tin.query()

    snt.testing.assert_dtypes_are_equal(dtypes, empty_table.dtypes)
    assert empty_table["elementary_school"]["uid"].shape[0] == 0


def test_only_index_in_level():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    dtypes = {
        "A": [("uid", "<u8"), ("height", "<i8")],
        "B": [("uid", "<u8")],
    }

    table = snt.SparseNumericTable(dtypes=dtypes, index_key="uid")

    for i in np.arange(10):
        table["A"].append({"uid": i, "height": 10})

    for i in prng.choice(table["A"]["uid"], 5):
        table["B"].append({"uid": i})

    snt.testing.assert_dtypes_are_equal(table.dtypes, dtypes)

    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "table_with_index_only_level.zip")
        with snt.open(path, "w", dtypes_and_index_key_from=table) as tout:
            tout.append_table(table)
        with snt.open(path, "r") as tin:
            table_back = tin.query()
        snt.testing.assert_dtypes_are_equal(table_back.dtypes, dtypes)
        snt.testing.assert_tables_are_equal(table, table_back)
