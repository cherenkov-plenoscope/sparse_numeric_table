import sparse_numeric_table as spt
import numpy as np
import pandas as pd
import tempfile
import os


def test_from_records():
    prng = np.random.Generator(np.random.MT19937(seed=0))
    rnd = prng.uniform

    # define what your table will look like
    # -------------------------------------
    structure = {
        "A": {
            "a": {"dtype": "<f8"},
            "b": {"dtype": "<f8"},
        },
        "B": {
            "c": {"dtype": "<f8"},
            "d": {"dtype": "<f8"},
        },
        "C": {
            "e": {"dtype": "<f8"},
        },
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
            table_records["A"].append({spt.IDX: i + 0, "a": rnd(), "b": rnd()})
            table_records["A"].append({spt.IDX: i + 1, "a": rnd(), "b": rnd()})
            table_records["A"].append({spt.IDX: i + 2, "a": rnd(), "b": rnd()})
            table_records["A"].append({spt.IDX: i + 3, "a": rnd(), "b": rnd()})
            table_records["A"].append({spt.IDX: i + 4, "a": rnd(), "b": rnd()})

            table_records["B"] = []
            table_records["B"].append(
                {spt.IDX: i + 0, "c": rnd(), "d": 5 * rnd()}
            )
            table_records["B"].append(
                {spt.IDX: i + 3, "c": rnd(), "d": 5 * rnd()}
            )

            table_records["C"] = []
            if rnd() > 0.9:
                table_records["C"].append({spt.IDX: i + 3, "e": -rnd()})

            table = spt.table_of_records_to_sparse_numeric_table(
                table_records=table_records, structure=structure
            )

            path = os.path.join(tmp, "{:06d}.tar".format(j))
            job_result_paths.append(path)
            spt.write(path=path, table=table, structure=structure)

        # reduce
        # ------
        full_table = spt.concatenate_files(
            list_of_table_paths=job_result_paths, structure=structure
        )

    spt.testing.assert_table_has_structure(
        table=full_table, structure=structure
    )


def test_write_read_full_table():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    my_table = spt.testing.make_example_table(prng=prng, size=1000 * 1000)
    my_table_structure = spt.testing.make_example_table_structure()
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "my_table.tar")
        spt.write(path=path, table=my_table, structure=my_table_structure)
        my_table_back = spt.read(path=path, structure=my_table_structure)
        spt.testing.assert_tables_are_equal(my_table, my_table_back)

        # no structure
        path_nos = os.path.join(tmp, "my_table_no_structure.tar")
        spt.write(path=path_nos, table=my_table)
        my_table_back_nos = spt.read(path=path_nos)
        spt.testing.assert_tables_are_equal(my_table, my_table_back_nos)
        spt.testing.assert_table_has_structure(
            table=my_table_back_nos, structure=my_table_structure
        )


def test_write_read_empty_table():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    empty_table = spt.testing.make_example_table(prng=prng, size=0)
    empty_table_structure = spt.testing.make_example_table_structure()
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "my_empty_table.tar")
        spt.write(
            path=path, table=empty_table, structure=empty_table_structure
        )
        my_table_back = spt.read(path=path, structure=empty_table_structure)
        spt.testing.assert_tables_are_equal(empty_table, my_table_back)

        # no structure
        path_nos = os.path.join(tmp, "my_empty_table_no_structure.tar")
        spt.write(path=path_nos, table=empty_table)
        my_table_back_nos = spt.read(path=path_nos)
        spt.testing.assert_tables_are_equal(empty_table, my_table_back_nos)
        spt.testing.assert_table_has_structure(
            table=my_table_back_nos, structure=empty_table_structure
        )


def test_merge_common():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    my_table = spt.testing.make_example_table(prng=prng, size=1000 * 1000)

    common_indices = spt.intersection(
        [my_table[lvl][spt.IDX] for lvl in my_table]
    )

    my_common_table = spt.cut_table_on_indices(
        table=my_table, common_indices=common_indices
    )
    my_sorted_common_table = spt.sort_table_on_common_indices(
        table=my_common_table, common_indices=common_indices
    )

    np.testing.assert_array_equal(
        my_sorted_common_table["elementary_school"][spt.IDX],
        my_sorted_common_table["high_school"][spt.IDX],
    )

    np.testing.assert_array_equal(
        my_sorted_common_table["elementary_school"][spt.IDX],
        my_sorted_common_table["university"][spt.IDX],
    )

    my_common_df = spt.make_rectangular_DataFrame(table=my_sorted_common_table)

    np.testing.assert_array_equal(
        my_sorted_common_table["elementary_school"][spt.IDX],
        my_common_df[spt.IDX],
    )


def test_merge_across_all_levels_random_order_indices():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    size = 1000 * 1000
    my_table = spt.testing.make_example_table(prng=prng, size=size)

    has_elementary_school = my_table["elementary_school"][spt.IDX]
    has_high_school = my_table["high_school"][spt.IDX]
    has_university = my_table["university"][spt.IDX]
    has_big_lunchpack = my_table["elementary_school"][spt.IDX][
        my_table["elementary_school"]["lunchpack_size"] > 0.5
    ]
    has_2best_friends = my_table["high_school"][spt.IDX][
        my_table["high_school"]["num_best_friends"] >= 2
    ]

    cut_indices = np.intersect1d(has_elementary_school, has_high_school)
    cut_indices = np.intersect1d(cut_indices, has_university)
    cut_indices = np.intersect1d(cut_indices, has_big_lunchpack)
    cut_indices = np.intersect1d(cut_indices, has_2best_friends)
    np.random.shuffle(cut_indices)

    cut_table = spt.cut_table_on_indices(
        table=my_table,
        common_indices=cut_indices,
        level_keys=["elementary_school", "high_school", "university"],
    )
    sorted_cut_table = spt.sort_table_on_common_indices(
        table=cut_table, common_indices=cut_indices
    )

    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"][spt.IDX],
        sorted_cut_table["high_school"][spt.IDX],
    )
    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"][spt.IDX],
        sorted_cut_table["university"][spt.IDX],
    )
    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"][spt.IDX], cut_indices
    )


def test_merge_random_order_indices():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    size = 1000 * 1000
    my_table = spt.testing.make_example_table(prng=prng, size=size)

    has_elementary_school = my_table["elementary_school"][spt.IDX]
    has_high_school = my_table["high_school"][spt.IDX]
    has_big_lunchpack = my_table["elementary_school"][spt.IDX][
        my_table["elementary_school"]["lunchpack_size"] > 0.5
    ]
    has_2best_friends = my_table["high_school"][spt.IDX][
        my_table["high_school"]["num_best_friends"] >= 2
    ]

    cut_indices = np.intersect1d(has_elementary_school, has_high_school)
    cut_indices = np.intersect1d(cut_indices, has_big_lunchpack)
    cut_indices = np.intersect1d(cut_indices, has_2best_friends)
    np.random.shuffle(cut_indices)

    cut_table = spt.cut_table_on_indices(
        table=my_table,
        common_indices=cut_indices,
        level_keys=["elementary_school", "high_school"],
    )
    sorted_cut_table = spt.sort_table_on_common_indices(
        table=cut_table, common_indices=cut_indices
    )

    assert "university" not in sorted_cut_table
    assert "elementary_school" in sorted_cut_table
    assert "high_school" in sorted_cut_table

    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"][spt.IDX],
        sorted_cut_table["high_school"][spt.IDX],
    )
    np.testing.assert_array_equal(
        sorted_cut_table["elementary_school"][spt.IDX], cut_indices
    )


def test_concatenate_several_tables():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    block_size = 10 * 1000
    num_blocks = 100

    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        paths = []
        for i in range(num_blocks):
            table_i = spt.testing.make_example_table(
                prng=prng, size=block_size, start_index=i * block_size
            )
            table_i_structure = spt.testing.make_example_table_structure()
            paths.append(os.path.join(tmp, "{:06d}.tar".format(i)))
            spt.write(
                path=paths[-1],
                table=table_i,
                structure=table_i_structure,
            )
        output_path = os.path.join(tmp, "full.tar")
        full_table = spt.concatenate_files(
            list_of_table_paths=paths,
            structure=table_i_structure,
        )
    spt.testing.assert_table_has_structure(
        table=full_table, structure=table_i_structure
    )

    assert (
        full_table["elementary_school"][spt.IDX].shape[0]
        == num_blocks * block_size
    )
    assert (
        len(set(full_table["elementary_school"][spt.IDX]))
        == num_blocks * block_size
    ), "The indices must be uniqe"
    assert (
        full_table["high_school"][spt.IDX].shape[0]
        == num_blocks * block_size // 10
    )
    assert (
        len(set(full_table["high_school"][spt.IDX]))
        == num_blocks * block_size // 10
    )
    assert (
        full_table["university"][spt.IDX].shape[0]
        == num_blocks * block_size // 100
    )
    assert (
        len(set(full_table["university"][spt.IDX]))
        == num_blocks * block_size // 100
    )


def test_concatenate_empty_list_of_paths():
    structure = spt.testing.make_example_table_structure()
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        output_path = os.path.join(tmp, "empty_table.tar")
        empty_table = spt.concatenate_files(
            list_of_table_paths=[], structure=structure
        )
    assert empty_table["elementary_school"][spt.IDX].shape[0] == 0


def test_only_index_in_level():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    structure = {
        "A": {"height": {"dtype": "<i8"}},
        "B": {},
    }

    table = {}
    table["A"] = spt.dict_to_recarray(
        {
            spt.IDX: np.arange(10).astype(spt.IDX_DTYPE),
            "height": np.ones(10, dtype="<i8"),
        }
    )
    table["B"] = spt.dict_to_recarray(
        {
            spt.IDX: prng.choice(table["A"][spt.IDX], 5),
        }
    )

    spt.testing.assert_table_has_structure(table=table, structure=structure)

    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "table_with_index_only_level.tar")
        spt.write(path=path, table=table, structure=structure)
        table_back = spt.read(path=path, structure=structure)
        spt.testing.assert_tables_are_equal(table, table_back)
