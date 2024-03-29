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

import pandas as pd
import numpy as np
import tarfile
import io
import shutil
import tempfile
import os


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


def _append_tar(tarfout, name, payload_bytes):
    tarinfo = tarfile.TarInfo()
    tarinfo.name = name
    tarinfo.size = len(payload_bytes)
    with io.BytesIO() as fileobj:
        fileobj.write(payload_bytes)
        fileobj.seek(0)
        tarfout.addfile(tarinfo=tarinfo, fileobj=fileobj)


def write(path, table, structure=None):
    """
    Writes the table to path.

    parameters
    ----------
    path : string
            Path to be written to.

    table : dict of recarrays
            The sparse table.

    structure : dict (default: None)
            The structure of the table. If provided it is asserted that the
            table written has the provided structure.
    """

    if structure:
        testing.assert_table_has_structure(table=table, structure=structure)

    with tarfile.open(path + ".tmp", "w") as tarfout:
        for level_key in table:
            assert IDX in table[level_key].dtype.names
            for column_key in table[level_key].dtype.names:
                dtype_key = table[level_key].dtype[column_key].str
                _append_tar(
                    tarfout=tarfout,
                    name=FILEAME_TEMPLATE.format(
                        level_key, column_key, dtype_key
                    ),
                    payload_bytes=table[level_key][column_key].tobytes(),
                )
    shutil.move(path + ".tmp", path)


def _split_level_column_dtype(path):
    level_key, column_key_and_dtype = str.split(path, LEVEL_COLUMN_DELIMITER)
    column_key, dtype_key = str.split(column_key_and_dtype, ".")
    return level_key, column_key, dtype_key


def read(path, structure=None):
    """
    Returns table which is read from path.

    parameters
    ----------
    path : string
            Path to tape-archive in filesystem

    structure : dict (default: None)
            The structure of the table. If provided it is asserted that the
            table read has the provided structure.
    """
    out = {}
    with tarfile.open(path, "r") as tarfin:
        for tarinfo in tarfin:
            level_key, column_key, dtype_key = _split_level_column_dtype(
                path=tarinfo.name
            )
            if column_key == IDX:
                assert dtype_key == IDX_DTYPE
            level_column_bytes = tarfin.extractfile(tarinfo).read()
            if level_key not in out:
                out[level_key] = {}
            out[level_key][column_key] = np.frombuffer(
                level_column_bytes, dtype=dtype_key
            )
    for level_key in out:
        out[level_key] = dict_to_recarray(out[level_key])

    if structure:
        testing.assert_table_has_structure(table=out, structure=structure)
    return out


# concatenate
# ===========


def _make_tmp_paths(tmp, structure):
    tmp_paths = {}
    for level_key in structure:
        tmp_paths[level_key] = {}
        idx_fname = FILEAME_TEMPLATE.format(level_key, IDX, IDX_DTYPE)
        tmp_paths[level_key][IDX] = os.path.join(tmp, idx_fname)
        for column_key in structure[level_key]:
            col_dt = structure[level_key][column_key]["dtype"]
            col_fname = FILEAME_TEMPLATE.format(level_key, column_key, col_dt)
            tmp_paths[level_key][column_key] = os.path.join(tmp, col_fname)
    return tmp_paths


def concatenate_files(list_of_table_paths, structure):
    with tempfile.TemporaryDirectory(prefix="sparse_table_concatenate") as tmp:
        tmp_paths = _make_tmp_paths(tmp=tmp, structure=structure)
        for table_path in list_of_table_paths:
            _part_table = read(path=table_path, structure=structure)
            for level_key in tmp_paths:
                os.makedirs(os.path.join(tmp, level_key), exist_ok=True)
                for column_key in tmp_paths[level_key]:
                    with open(tmp_paths[level_key][column_key], "ab") as fa:
                        fa.write(_part_table[level_key][column_key].tobytes())
        out = {}
        for level_key in tmp_paths:
            out[level_key] = {}
            for column_key in tmp_paths[level_key]:
                if column_key == IDX:
                    dtype = IDX_DTYPE
                else:
                    dtype = structure[level_key][column_key]["dtype"]

                if os.path.exists(tmp_paths[level_key][column_key]):
                    out[level_key][column_key] = np.fromfile(
                        tmp_paths[level_key][column_key], dtype=dtype
                    )
                else:
                    out[level_key][column_key] = np.zeros(0, dtype=dtype)
    for level_key in out:
        out[level_key] = dict_to_recarray(out[level_key])
    return out


# from records
# ============


def _empty_recarray(structure, level_key):
    dd = {IDX: np.zeros(0, dtype=IDX_DTYPE)}
    for column_key in structure[level_key]:
        dd[column_key] = np.zeros(
            0, dtype=structure[level_key][column_key]["dtype"]
        )
    return dict_to_recarray(dd)


def records_to_recarray(level_records, level_key, structure):
    expected_keys = list(structure[level_key].keys()) + [IDX]
    if len(level_records) > 0:
        for record in level_records:
            testing._assert_same_keys(expected_keys, list(record.keys()))
        df = pd.DataFrame(level_records)
        dd = {IDX: df[IDX].values.astype(IDX_DTYPE)}
        for column_key in structure[level_key]:
            dd[column_key] = df[column_key].values.astype(
                structure[level_key][column_key]["dtype"]
            )
        return dict_to_recarray(dd)
    else:
        return _empty_recarray(structure=structure, level_key=level_key)


def table_of_records_to_sparse_numeric_table(table_records, structure):
    table = {}
    for level_key in table_records:
        table[level_key] = records_to_recarray(
            level_records=table_records[level_key],
            level_key=level_key,
            structure=structure,
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
