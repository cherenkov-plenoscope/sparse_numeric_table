from . import testing
from . import _file_io
import numpy as np


def merge(
    out_path,
    in_paths,
    sort_in_tables=False,
    block_read_size=262_144,
    open_file_function=None,
):
    if open_file_function is None:
        open_file_function = open
    assert len(in_paths) > 0

    with _file_io.open(file=in_paths[0], mode="r") as first:
        dtypes = first.dtypes
        index_key = first.index_key

    with _file_io.open(
        file=out_path, mode="w", dtypes=dtypes, index_key=index_key
    ) as out_table:
        for in_path in in_paths:
            with open_file_function(in_path, mode="rb") as fin, _file_io.open(
                file=fin, mode="r"
            ) as in_table:

                # read level by level
                for level_key in dtypes:
                    _index_table = in_table.query(
                        levels_and_columns={level_key: [index_key]},
                    )
                    level_indices_all = _index_table[level_key][index_key]
                    if sort_in_tables:
                        level_indices_all = sorted(level_indices_all)

                    # read in chunks of index
                    # this is potentially slow as it reads the level again and
                    # again but it avoids running out of memory
                    for level_indices_block in _split_into_chunks(
                        x=level_indices_all, chunk_size=block_read_size
                    ):
                        print(in_path, level_key)
                        out_table.append_table(
                            in_table.query(
                                levels_and_columns={level_key: "__all__"},
                                indices=level_indices_block,
                                sort=sort_in_tables,
                            )
                        )


def sort(in_path, out_path):
    merge(out_path=out_path, in_paths=[in_path], sort_in_tables=True)


def _split_into_chunks(x, chunk_size):
    assert chunk_size > 0
    num_blocks = len(x) // int(chunk_size)
    if num_blocks * chunk_size < len(x):
        num_blocks += 1
    blocks = []
    for b in range(num_blocks):
        start = b * chunk_size
        stop = min([start + chunk_size, len(x)])
        blocks.append(x[start:stop])
    return blocks
