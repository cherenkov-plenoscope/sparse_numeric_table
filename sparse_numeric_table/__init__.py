"""
sparse tables
=============

    Might look like this:

        level 1          level 2       level 3
          columns          columns       columns
     idx a b c d e f  idx g h i j k l  idx m n o p
     ___ _ _ _ _ _ _  ___ _ _ _ _ _ _
    |_0_|_|_|_|_|_|_||_0_|_|_|_|_|_|_|
    |_1_|_|_|_|_|_|_|
    |_2_|_|_|_|_|_|_| ___ _ _ _ _ _ _
    |_3_|_|_|_|_|_|_||_3_|_|_|_|_|_|_|
    |_4_|_|_|_|_|_|_||_4_|_|_|_|_|_|_| ___ _ _ _ _
    |_5_|_|_|_|_|_|_||_5_|_|_|_|_|_|_||_5_|_|_|_|_|
    |_6_|_|_|_|_|_|_|
    |_7_|_|_|_|_|_|_|
    |_8_|_|_|_|_|_|_| ___ _ _ _ _ _ _
    |_9_|_|_|_|_|_|_||_9_|_|_|_|_|_|_|
    |10_|_|_|_|_|_|_||10_|_|_|_|_|_|_|
    |11_|_|_|_|_|_|_| ___ _ _ _ _ _ _  ___ _ _ _ _
    |12_|_|_|_|_|_|_||12_|_|_|_|_|_|_||12_|_|_|_|_|
    |13_|_|_|_|_|_|_| ___ _ _ _ _ _ _
    |14_|_|_|_|_|_|_||14_|_|_|_|_|_|_|


    Can be represented in memory like this:

        level 1            level 2         level 3
          columns            columns         columns
     idx a b c d e f    idx g h i j k l    idx m n o p
     ___ _ _ _ _ _ _    ___ _ _ _ _ _ _    ___ _ _ _ _
    |_0_|_|_|_|_|_|_|  |_0_|_|_|_|_|_|_|  |_5_|_|_|_|_|
    |_1_|_|_|_|_|_|_|  |_3_|_|_|_|_|_|_|  |12_|_|_|_|_|
    |_2_|_|_|_|_|_|_|  |_4_|_|_|_|_|_|_|
    |_3_|_|_|_|_|_|_|  |_9_|_|_|_|_|_|_|
    |_4_|_|_|_|_|_|_|  |10_|_|_|_|_|_|_|
    |_6_|_|_|_|_|_|_|  |12_|_|_|_|_|_|_|
    |_7_|_|_|_|_|_|_|  |14_|_|_|_|_|_|_|
    |_8_|_|_|_|_|_|_|
    |_9_|_|_|_|_|_|_|
    |10_|_|_|_|_|_|_|
    |11_|_|_|_|_|_|_|
    |12_|_|_|_|_|_|_|
    |13_|_|_|_|_|_|_|
    |14_|_|_|_|_|_|_|

    Written to tape-archive

    table.tar
        |_ level_1/idx
        |_ level_1/column_a
        |_ level_1/column_b
        |_ level_1/column_c
        |_ level_1/column_d
        |_ level_1/column_e
        |_ level_1/column_f
        |_ level_2/idx
        |_ level_2/column_g
        |_ level_2/column_h
        |_ level_2/column_i
        |_ level_2/column_j
        |_ level_2/column_k
        |_ level_2/column_l
        |_ level_3/idx
        |_ level_3/column_m
        |_ level_3/column_n
        |_ level_3/column_o
        |_ level_3/column_p
"""
from .version import __version__
from . import tarstream
from . import testing
from .base import IDX
from .base import IDX_DTYPE
from .base import LEVEL_COLUMN_DELIMITER
from .base import FILEAME_TEMPLATE
from .base import DTYPES
from .base import make_mask_of_right_in_left
from .base import dict_to_recarray
from .base import add_idx_to_level_dtype

import pandas as pd
import numpy as np
import sequential_tar
import io
import shutil
import tempfile
import os
from dynamicsizerecarray import DynamicSizeRecarray


def init(dtypes):
    testing.assert_dtypes_keys_are_valid(dtypes=dtypes)
    table = {}
    for level_key in dtypes:
        full_level_dtype = add_idx_to_level_dtype(dtypes[level_key])
        table[level_key] = DynamicSizeRecarray(dtype=full_level_dtype)
    return table


def get_dtypes(table):
    out = {}
    for level_key in table:
        level_dtype = []
        for column_key in table[level_key].dtype.names:
            if column_key == IDX:
                continue
            column_dtype = table[level_key].dtype[column_key].descr[0][1]
            level_dtype.append((column_key, column_dtype))
        out[level_key] = level_dtype
    return out


def get_sizes(table):
    out = {}
    for level_key in table:
        out[level_key] = table[level_key].shape[0]
    return out


def get_modes(table):
    out = {}
    for level_key in table:
        if isinstance(table[level_key], DynamicSizeRecarray):
            out[level_key] = "dynamic"
        elif isinstance(table[level_key], np.recarray):
            out[level_key] = "static"
        else:
            raise ValueError("Expected DynamicSizeRecarray or np.recarray")
    return out


def is_dynamic(table):
    return _is_mode(table=table, mode="dynamic")


def is_static(table):
    return _is_mode(table=table, mode="static")


def _is_mode(table, mode):
    modes = get_modes(table)
    for level_key in modes:
        if modes[level_key] != mode:
            return False
    return True


def to_static(table, inplace=False):
    if inplace:
        out = table
    else:
        out = {}
    for level_key in table:
        if isinstance(table[level_key], DynamicSizeRecarray):
            out[level_key] = table[level_key].to_recarray()
    return out


def to_dynamic(table, inplace=False):
    if inplace:
        out = table
    else:
        out = {}
    for level_key in table:
        if isinstance(table[level_key], np.recarray):
            out[level_key] = DynamicSizeRecarray(recarray=table[level_key])
    return out


# logical operations
# ==================


def intersection(list_of_lists_of_indices):
    """
    Returns the common indices among the lists of indices.

    Example
    -------
    [4, 5, 6] = intersection([[1,2,3,4,5,6], [3,4,5,6,7,8], [4,5,6,7,8,9,10]])
    """
    inter = list_of_lists_of_indices[0]
    for i in range(len(list_of_lists_of_indices)):
        inter = np.intersect1d(inter, list_of_lists_of_indices[i])
    return inter


def cut_level_on_indices(level, indices, column_keys=None):
    """
    Returns a level (recarray) only containing the row-indices in 'indices'.

    Parameters
    ----------
    level : recarray
            A level in a sparse table.
    indices : list
            The row-indices to be written to the output-level.
    column_keys : list of strings (None)
            When specified, only these columns will be in the output-level.
    """
    if column_keys is None:
        column_keys = list(level.dtype.names)
    column_keys.append(IDX)
    _part = {}
    for column_key in column_keys:
        _part[column_key] = level[column_key]
    part_df = pd.DataFrame(_part)
    del _part
    common_df = pd.merge(
        part_df,
        pd.DataFrame(dict_to_recarray({IDX: indices})),
        on=IDX,
        how="inner",
    )
    del part_df
    return common_df.to_records(index=False)


def cut_table_on_indices(table, common_indices, level_keys=None):
    """
    Returns table but only with the rows listed in common_indices.

    Parameters
    ----------
    table : dict of recarrays.
            The sparse numeric table.
    common_indices : list of indices
            The row-indices to cut on. Only row-indices in this list will go
            in the output-table.
    level_keys : list of strings (None)
            When provided, the output-table will only contain these levels.
    """
    if level_keys is None:
        level_keys = list(table.keys())
    out = {}
    for level_key in level_keys:
        out[level_key] = cut_level_on_indices(
            level=table[level_key],
            indices=common_indices,
        )
    return out


def sort_table_on_common_indices(table, common_indices):
    """
    Returns a table with all row-indices ordered same as common_indices.

    table : dict of recarrays.
            The table. But must be rectangular, i.e. not sparse.
    common_indices : list of indices
            The row-indices to sort by.
    """
    common_order_args = np.argsort(common_indices)
    common_inv_order = np.zeros(shape=common_indices.shape, dtype=np.int64)
    common_inv_order[common_order_args] = np.arange(len(common_indices))
    del common_order_args

    out = {}
    for level_key in table:
        level = table[level_key]
        level_order_args = np.argsort(level[IDX])
        level_sorted = level[level_order_args]
        del level_order_args
        level_same_order_as_common = level_sorted[common_inv_order]
        out[level_key] = level_same_order_as_common
    return out


def cut_and_sort_table_on_indices(table, common_indices, level_keys=None):
    """
    Returns a table (rectangular, not sparse) containing only rows listed in
    common_indices and in this order.

    Parameters
    ----------
    table : dict of recarrays.
            The sparse table.
    common_indices : list of indices
            The row-indices to cut on and sort by.
    level_keys : list of strings (None)
            When specified, only this levels will be in the output-table.
    """
    out = cut_table_on_indices(
        table=table,
        common_indices=common_indices,
        level_keys=level_keys,
    )
    out = sort_table_on_common_indices(
        table=out, common_indices=common_indices
    )
    return out


def make_rectangular_DataFrame(table):
    """
    Returns a pandas.DataFrame made from a table.
    The table must already be rectangular, i.e. not sparse anymore.
    The row-indices among all levels in the table must have the same ordering.
    """
    out = {}
    for level_key in table:
        for column_key in table[level_key].dtype.names:
            if column_key == IDX:
                if IDX in out:
                    np.testing.assert_array_equal(
                        out[IDX], table[level_key][IDX]
                    )
                else:
                    out[IDX] = table[level_key][IDX]
            else:
                out[
                    "{:s}{:s}{:s}".format(
                        level_key, LEVEL_COLUMN_DELIMITER, column_key
                    )
                ] = table[level_key][column_key]
    return pd.DataFrame(out)


# input output
# ============


def write(path, table, dtypes=None):
    """
    Writes the table to path.

    parameters
    ----------
    path : string
            Path to be written to.

    table : dict of recarrays
            The sparse table.

    dtypes : dict (default: None)
            The dtypes of the table. If provided it is asserted that the
            table written has the provided dtypes.
    """

    if dtypes:
        testing.assert_table_has_dtypes(table=table, dtypes=dtypes)

    with sequential_tar.open(name=path + ".tmp", mode="w") as tar:
        for level_key in table:
            assert IDX in table[level_key].dtype.names
            for column_key in table[level_key].dtype.names:
                dtype_key = table[level_key].dtype[column_key].str
                tar.write(
                    name=FILEAME_TEMPLATE.format(
                        level_key, column_key, dtype_key
                    ),
                    payload=table[level_key][column_key].tobytes(),
                    mode="wb",
                )
    shutil.move(path + ".tmp", path)


def _split_level_column_dtype(path):
    level_key, column_key_and_dtype = str.split(path, LEVEL_COLUMN_DELIMITER)
    column_key, dtype_key = str.split(column_key_and_dtype, ".")
    return level_key, column_key, dtype_key


def read(path, dtypes=None):
    """
    Returns table which is read from path.

    parameters
    ----------
    path : string
            Path to tape-archive in filesystem

    dtypes : dict (default: None)
            The dtypes of the table. If provided it is asserted that the
            table read has the provided dtypes.
    """
    out = {}
    with sequential_tar.open(name=path, mode="r") as tar:
        for item in tar:
            level_key, column_key, dtype_key = _split_level_column_dtype(
                path=item.name
            )
            if column_key == IDX:
                assert dtype_key == IDX_DTYPE
            level_column_bytes = item.read(mode="rb")
            if level_key not in out:
                out[level_key] = {}
            out[level_key][column_key] = np.frombuffer(
                level_column_bytes, dtype=dtype_key
            )
    for level_key in out:
        out[level_key] = dict_to_recarray(out[level_key])

    if dtypes:
        testing.assert_table_has_dtypes(table=out, dtypes=dtypes)
    return out


# concatenate
# ===========

def concatenate_files(list_of_table_paths, dtypes):
    if len(list_of_table_paths) == 0:
        out = {}
        for level_key in dtypes:
            full_level_dtype = add_idx_to_level_dtype(dtypes[level_key])
            out[level_key] = np.frombuffer(b"", dtype=full_level_dtype)
        return out

    with tempfile.TemporaryDirectory(prefix="sparse_table_concatenate") as tmp:
        for part_table_path in list_of_table_paths:
            part_table = read(path=part_table_path, dtypes=dtypes)
            for level_key in dtypes:
                with open(os.path.join(tmp, level_key), "ab") as fa:
                    fa.write(part_table[level_key].tobytes())

        out = {}
        for level_key in dtypes:
            with open(os.path.join(tmp, level_key), "rb") as f:
                full_level_dtype = add_idx_to_level_dtype(dtypes[level_key])
                out[level_key] = np.frombuffer(
                    f.read(), dtype=full_level_dtype
                )

    return out


# from records
# ============


def _empty_recarray(dtypes, level_key):
    dd = {IDX: np.zeros(0, dtype=IDX_DTYPE)}
    for column_key in dtypes[level_key]:
        dd[column_key] = np.zeros(
            0, dtype=dtypes[level_key][column_key]["dtype"]
        )
    return dict_to_recarray(dd)


def records_to_recarray(level_records, level_key, dtypes):
    full_level_dtype = add_idx_to_level_dtype(dtypes[level_key])
    out = DynamicSizeRecarray(dtype=full_level_dtype)
    out.append_records(records=level_records)
    return out.to_recarray()


def table_of_records_to_sparse_numeric_table(table_records, dtypes):
    table = {}
    for level_key in table_records:
        table[level_key] = records_to_recarray(
            level_records=table_records[level_key],
            level_key=level_key,
            dtypes=dtypes,
        )
    return table


def get_column_as_dict_by_index(table, level_key, column_key):
    level = table[level_key]
    out = {}
    for ii in range(level.shape[0]):
        out[level[IDX][ii]] = level[column_key][ii]
    return out


# print
# =====


def list_size(table):
    out = io.StringIO()
    for level_key in table:
        out.write(
            "{: 9d} {: <30s}\n".format(table[level_key].shape[0], level_key)
        )
    out.seek(0)
    return out.read()
