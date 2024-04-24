import sparse_numeric_table as snt
import numpy as np
import pandas as pd
import tempfile
import os


def test_from_records():
    prng = np.random.Generator(np.random.MT19937(seed=0))
    rnd = prng.uniform

    # define what your table will look like
    # -------------------------------------
    dtypes = {
        "A": [("a", "<f8"), ("b", "<f8")],
        "B": [("c", "<f8"), ("d", "<f8")],
        "C": [("e", "<f8")],
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
            table_records = {}

            table_records["A"] = []
            table_records["A"].append({snt.IDX: i + 0, "a": rnd(), "b": rnd()})
            table_records["A"].append({snt.IDX: i + 1, "a": rnd(), "b": rnd()})
            table_records["A"].append({snt.IDX: i + 2, "a": rnd(), "b": rnd()})
            table_records["A"].append({snt.IDX: i + 3, "a": rnd(), "b": rnd()})
            table_records["A"].append({snt.IDX: i + 4, "a": rnd(), "b": rnd()})

            table_records["B"] = []
            table_records["B"].append(
                {snt.IDX: i + 0, "c": rnd(), "d": 5 * rnd()}
            )
            table_records["B"].append(
                {snt.IDX: i + 3, "c": rnd(), "d": 5 * rnd()}
            )

            table_records["C"] = []
            if rnd() > 0.9:
                table_records["C"].append({snt.IDX: i + 3, "e": -rnd()})

            table = snt.table_of_records_to_sparse_numeric_table(
                table_records=table_records, dtypes=dtypes
            )

            path = os.path.join(tmp, "{:06d}.tar".format(j))
            job_result_paths.append(path)
            snt.write(path=path, table=table, dtypes=dtypes)

        # reduce
        # ------
        full_table = snt.concatenate_files(
            list_of_table_paths=job_result_paths, dtypes=dtypes
        )

    snt.testing.assert_table_has_dtypes(table=full_table, dtypes=dtypes)


def test_write_read_full_table():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    my_table = snt.testing.make_example_table(prng=prng, size=1000 * 1000)
    my_table_dtypes = snt.testing.make_example_table_dtypes()
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "my_table.tar")
        snt.write(path=path, table=my_table, dtypes=my_table_dtypes)
        my_table_back = snt.read(path=path, dtypes=my_table_dtypes)
        snt.testing.assert_tables_are_equal(my_table, my_table_back)

        # no dtypes
        path_nos = os.path.join(tmp, "my_table_no_dtypes.tar")
        snt.write(path=path_nos, table=my_table)
        my_table_back_nos = snt.read(path=path_nos)
        snt.testing.assert_tables_are_equal(my_table, my_table_back_nos)
        snt.testing.assert_table_has_dtypes(
            table=my_table_back_nos, dtypes=my_table_dtypes
        )


def test_write_read_empty_table():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    empty_table = snt.testing.make_example_table(prng=prng, size=0)
    empty_table_dtypes = snt.testing.make_example_table_dtypes()
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "my_empty_table.tar")
        snt.write(path=path, table=empty_table, dtypes=empty_table_dtypes)
        my_table_back = snt.read(path=path, dtypes=empty_table_dtypes)
        snt.testing.assert_tables_are_equal(empty_table, my_table_back)

        # no dtypes
        path_nos = os.path.join(tmp, "my_empty_table_no_dtypes.tar")
        snt.write(path=path_nos, table=empty_table)
        my_table_back_nos = snt.read(path=path_nos)
        snt.testing.assert_tables_are_equal(empty_table, my_table_back_nos)
        snt.testing.assert_table_has_dtypes(
            table=my_table_back_nos, dtypes=empty_table_dtypes
        )


def test_merge_common():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    my_table = snt.testing.make_example_table(prng=prng, size=1000 * 1000)

    common_indices = snt.intersection(
        [my_table[lvl][snt.IDX] for lvl in my_table]
    )

    my_common_table = snt.cut_table_on_indices(
        table=my_table, common_indices=common_indices
    )
    my_sorted_common_table = snt.sort_table_on_common_indices(
        table=my_common_table, common_indices=common_indices
    )

    np.testing.assert_array_equal(
        my_sorted_common_table["elementary_school"][snt.IDX],
        my_sorted_common_table["high_school"][snt.IDX],
    )

    np.testing.assert_array_equal(
        my_sorted_common_table["elementary_school"][snt.IDX],
        my_sorted_common_table["university"][snt.IDX],
    )

    my_common_df = snt.make_rectangular_DataFrame(table=my_sorted_common_table)

    np.testing.assert_array_equal(
        my_sorted_common_table["elementary_school"][snt.IDX],
        my_common_df[snt.IDX],
    )


def test_merge_across_all_levels_random_order_indices():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    size = 1000 * 1000
    my_table = snt.testing.make_example_table(prng=prng, size=size)

    has_elementary_school = my_table["elementary_school"][snt.IDX]
    has_high_school = my_table["high_school"][snt.IDX]
    has_university = my_table["university"][snt.IDX]
    has_big_lunchpack = my_table["elementary_school"][snt.IDX][
        my_table["elementary_school"]["lunchpack_size"] > 0.5
    ]
    has_2best_friends = my_table["high_school"][snt.IDX][
        my_table["high_school"]["num_best_friends"] >= 2
    ]

    cut_indices = np.intersect1d(has_elementary_school, has_high_school)
    cut_indices = np.intersect1d(cut_indices, has_university)
    cut_indices = np.intersect1d(cut_indices, has_big_lunchpack)
    cut_indices = np.intersect1d(cut_indices, has_2best_friends)
    np.random.shuffle(cut_indices)

    cut_table = snt.cut_table_on_indices(
        table=my_table,
        common_indices=cut_indices,
        level_keys=["elementary_school", "high_school", "university"],
    )
    sorted_cut_table = snt.sort_table_on_common_indices(
        table=cut_table, common_indices=cut_indices
    )

    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"][snt.IDX],
        sorted_cut_table["high_school"][snt.IDX],
    )
    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"][snt.IDX],
        sorted_cut_table["university"][snt.IDX],
    )
    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"][snt.IDX], cut_indices
    )


def test_merge_random_order_indices():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    size = 1000 * 1000
    my_table = snt.testing.make_example_table(prng=prng, size=size)

    has_elementary_school = my_table["elementary_school"][snt.IDX]
    has_high_school = my_table["high_school"][snt.IDX]
    has_big_lunchpack = my_table["elementary_school"][snt.IDX][
        my_table["elementary_school"]["lunchpack_size"] > 0.5
    ]
    has_2best_friends = my_table["high_school"][snt.IDX][
        my_table["high_school"]["num_best_friends"] >= 2
    ]

    cut_indices = np.intersect1d(has_elementary_school, has_high_school)
    cut_indices = np.intersect1d(cut_indices, has_big_lunchpack)
    cut_indices = np.intersect1d(cut_indices, has_2best_friends)
    np.random.shuffle(cut_indices)

    cut_table = snt.cut_table_on_indices(
        table=my_table,
        common_indices=cut_indices,
        level_keys=["elementary_school", "high_school"],
    )
    sorted_cut_table = snt.sort_table_on_common_indices(
        table=cut_table, common_indices=cut_indices
    )

    assert "university" not in sorted_cut_table
    assert "elementary_school" in sorted_cut_table
    assert "high_school" in sorted_cut_table

    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"][snt.IDX],
        sorted_cut_table["high_school"][snt.IDX],
    )
    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"][snt.IDX], cut_indices
    )


def test_concatenate_several_tables():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    block_size = 10 * 1000
    num_blocks = 100

    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        paths = []
        for i in range(num_blocks):
            table_i = snt.testing.make_example_table(
                prng=prng, size=block_size, start_index=i * block_size
            )
            table_i_dtypes = snt.testing.make_example_table_dtypes()
            paths.append(os.path.join(tmp, "{:06d}.tar".format(i)))
            snt.write(
                path=paths[-1],
                table=table_i,
                dtypes=table_i_dtypes,
            )
        output_path = os.path.join(tmp, "full.tar")
        full_table = snt.concatenate_files(
            list_of_table_paths=paths,
            dtypes=table_i_dtypes,
        )
    snt.testing.assert_table_has_dtypes(
        table=full_table, dtypes=table_i_dtypes
    )

    assert (
        full_table["elementary_school"][snt.IDX].shape[0]
        == num_blocks * block_size
    )
    assert (
        len(set(full_table["elementary_school"][snt.IDX]))
        == num_blocks * block_size
    ), "The indices must be uniqe"
    assert (
        full_table["high_school"][snt.IDX].shape[0]
        == num_blocks * block_size // 10
    )
    assert (
        len(set(full_table["high_school"][snt.IDX]))
        == num_blocks * block_size // 10
    )
    assert (
        full_table["university"][snt.IDX].shape[0]
        == num_blocks * block_size // 100
    )
    assert (
        len(set(full_table["university"][snt.IDX]))
        == num_blocks * block_size // 100
    )


def test_concatenate_empty_list_of_paths():
    dtypes = snt.testing.make_example_table_dtypes()
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        output_path = os.path.join(tmp, "empty_table.tar")
        empty_table = snt.concatenate_files(
            list_of_table_paths=[], dtypes=dtypes
        )
    assert empty_table["elementary_school"][snt.IDX].shape[0] == 0


def test_only_index_in_level():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    dtypes = {
        "A": [("height", "<i8")],
        "B": [],
    }

    table = {}
    table["A"] = snt.dict_to_recarray(
        {
            snt.IDX: np.arange(10).astype(snt.IDX_DTYPE),
            "height": np.ones(10, dtype="<i8"),
        }
    )
    table["B"] = snt.dict_to_recarray(
        {
            snt.IDX: prng.choice(table["A"][snt.IDX], 5),
        }
    )

    snt.testing.assert_table_has_dtypes(table=table, dtypes=dtypes)

    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "table_with_index_only_level.tar")
        snt.write(path=path, table=table, dtypes=dtypes)
        table_back = snt.read(path=path, dtypes=dtypes)
        snt.testing.assert_tables_are_equal(table, table_back)
