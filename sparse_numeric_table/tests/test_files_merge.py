import sparse_numeric_table
import numpy as np
import tempfile
import os


def test_blocks():
    cases = [
        {"x": [], "chunk_size": 1},
        {"x": [], "chunk_size": 100},
        {"x": np.arange(100), "chunk_size": 11},
        {"x": np.arange(1), "chunk_size": 1},
        {"x": np.arange(100), "chunk_size": 1},
        {"x": np.arange(100), "chunk_size": 100},
    ]

    for case in cases:
        chunk_size = case["chunk_size"]
        x = case["x"]

        blocks = sparse_numeric_table.files._split_into_chunks(
            x=x,
            chunk_size=chunk_size,
        )

        for block in blocks:
            assert len(block) <= chunk_size
            assert len(block) > 0

        y = []
        for block in blocks:
            for item in block:
                y.append(item)

        np.testing.assert_array_equal(desired=x, actual=y)


def test_merge():
    cases = []
    cases.append(
        {
            "seed": 2,
            "num_blocks": 1,
            "table_size": 1_000,
            "block_read_size": 133,
        }
    )
    cases.append(
        {
            "seed": 1,
            "num_blocks": 8,
            "table_size": 1_000,
            "block_read_size": 133,
        }
    )
    cases.append(
        {
            "seed": 1,
            "num_blocks": 3,
            "table_size": 10_000,
            "block_read_size": 133,
        }
    )

    for case in cases:
        with tempfile.TemporaryDirectory(prefix="snt_") as tmp_dir:
            prng = np.random.Generator(np.random.PCG64(case["seed"]))
            in_paths = []
            collect_as_we_go = sparse_numeric_table.testing.make_example_table(
                prng=prng,
                size=0,
                start_index=0,
            )

            for b in range(case["num_blocks"]):
                block_path = os.path.join(tmp_dir, f"{b:06d}.snt.zip")
                in_paths.append(block_path)
                block_table = sparse_numeric_table.testing.make_example_table(
                    prng=prng,
                    size=case["table_size"],
                    start_index=case["table_size"] * b,
                )

                collect_as_we_go.append(block_table)

                with sparse_numeric_table.open(
                    file=block_path,
                    mode="w",
                    dtypes_and_index_key_from=block_table,
                ) as tout:
                    tout.append_table(block_table)

            merge_path = os.path.join(tmp_dir, "merge.snt.zip")

            sparse_numeric_table.files.merge(
                out_path=merge_path,
                in_paths=in_paths,
                block_read_size=case["block_read_size"],
            )

            with sparse_numeric_table.open(merge_path, "r") as tin:
                back_from_merger = tin.query()

            sparse_numeric_table.testing.assert_tables_are_equal(
                collect_as_we_go,
                back_from_merger,
            )
