from .base import IDX
from .base import IDX_DTYPE
from .base import make_mask_of_right_in_left

import json
import posixpath
import numpy as np
import sequential_tar
import dynamicsizerecarray


def write(
    table,
    path=None,
    fileobj=None,
    mode="w|",
    level_mode="wb|gz",
    level_block_size=2**25,
):
    assert mode.startswith("w|")
    assert level_mode.startswith("wb|")
    assert level_block_size > 0

    num_bytes_written = 0
    with sequential_tar.open(name=path, fileobj=fileobj, mode=mode) as tarf:
        num_bytes_written += tarf.write(
            name="dtype.json",
            payload=dumps_dtype(table=table),
            mode="wt",
        )

        for level_key in table:
            single_record_size = estimate_single_record_size(
                dtype=table[level_key].dtype,
            )
            num_records_per_block = int(
                np.ceil(level_block_size / single_record_size)
            )

            block_id = 0
            ifinal = len(table[level_key]) - 1
            istart = 0
            istop = 0

            while istop < ifinal:
                block_filename = posixpath.join(
                    level_key, "{:06d}.rec".format(block_id)
                )
                if "|gz" in level_mode:
                    block_filename += ".gz"

                istart = block_id * num_records_per_block
                istop = istart + num_records_per_block

                level_block = table[level_key][istart:istop]

                num_bytes_written += tarf.write(
                    name=block_filename,
                    payload=level_block.tobytes(),
                    mode=level_mode,
                )

                block_id += 1
    return num_bytes_written


def dumps_dtype(table):
    out = {}
    for level_key in table:
        out[level_key] = []
        outlevel = out[level_key]
        level = table[level_key]

        for column_key in level.dtype.names:
            if not column_key is IDX:
                column_dtype_key = level.dtype[column_key].str
                outlevel.append([column_key, column_dtype_key])

    return json.dumps(out, indent=4)


def estimate_single_record_size(dtype):
    dummy = np.core.records.recarray(shape=1, dtype=dtype)
    dummy_bytes = dummy.tobytes()
    return len(dummy_bytes)


def read(path=None, fileobj=None, mode="r|", levels=None, common_indices=None):
    assert mode.startswith("r|")
    dynamic_table = {}
    file_table_dtype = {}

    with sequential_tar.open(name=path, fileobj=fileobj, mode=mode) as tarf:
        item = tarf.next()
        filetext = item.read(mode="rt")
        assert item.name == "dtype.json"
        full_head = json.loads(filetext)

        if levels is None:
            head = full_head
        else:
            head = {}
            for level_key in levels:
                head[level_key] = full_head[level_key]

        for level_key in head:
            file_table_dtype[level_key] = add_idx_to_level_dtype(
                level_dtype=head[level_key]
            )
            dynamic_table[level_key] = dynamicsizerecarray.DynamicSizeRecarray(
                dtype=file_table_dtype[level_key]
            )

        for item in tarf:
            if str.endswith(item.name, ".gz"):
                buff = item.read(mode="rb|gz")
            else:
                buff = item.read(mode="rb")
            level_key, block_id_str = posixpath.split(item.name)
            if level_key in head:
                block_rec = np.frombuffer(
                    buff, dtype=file_table_dtype[level_key]
                )
                if common_indices is not None:
                    block_mask = make_mask_of_right_in_left(
                        left_indices=block_rec[IDX],
                        right_indices=common_indices,
                    )
                    block_rec = block_rec[block_mask]
                dynamic_table[level_key].append_recarray(block_rec)

    table = {}
    dtable_level_keys = list(dynamic_table.keys())
    for level_key in dtable_level_keys:
        dynrec = dynamic_table.pop(level_key)
        table[level_key] = dynrec.to_recarray()
    return table


def add_idx_to_level_dtype(level_dtype):
    full_dtype = [(IDX, IDX_DTYPE)]
    for column_key_dtype in level_dtype:
        full_dtype.append(tuple(column_key_dtype))
    return full_dtype
