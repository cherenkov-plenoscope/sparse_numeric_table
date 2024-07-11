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

from . import archive
from . import testing
from . import logic
from .base import SparseNumericTable
from .base import dict_to_recarray

"""
from .base import LEVEL_COLUMN_DELIMITER
from .base import FILEAME_TEMPLATE
from .base import DTYPES
from .base import make_mask_of_right_in_left

"""

import pandas as pd
import numpy as np
import sequential_tar
import io
import shutil
import tempfile
import os
import copy
from dynamicsizerecarray import DynamicSizeRecarray


def append(table_a, table_b):
    """
    Appends table_b to table_a.
    """
    for level_key in table_a:
        table_a[level_key].append_recarray(table_b[level_key])
    return table_a


"""
def init(dtypes):
    testing.assert_dtypes_keys_are_valid(dtypes=dtypes)
    table = {}
    for level_key in dtypes:
        table[level_key] = DynamicSizeRecarray(dtype=dtypes[level_key])
    return table





def get_dtypes(table):
    out = {}
    for level_key in table:
        level_dtype = []
        for column_key in table[level_key].dtype.names:
            column_dtype = table[level_key].dtype[column_key].descr[0][1]
            level_dtype.append((column_key, column_dtype))
        out[level_key] = level_dtype
    return out


def get_sizes(table):
    out = {}
    for level_key in table:
        out[level_key] = table[level_key].shape[0]
    return out
"""

# input output
# ============

LEVEL_COLUMN_DELIMITER = "/"
FILEAME_TEMPLATE = "{:s}" + LEVEL_COLUMN_DELIMITER + "{:s}.{:s}"


def write(path, table):
    """
    Writes the table to path.

    parameters
    ----------
    path : string
            Path to be written to.

    table : dict of recarrays
            The sparse table.
    """
    with sequential_tar.open(name=path + ".tmp", mode="w") as tar:
        for level_key in table:
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


def read(path=None, fileobj=None):
    """
    Returns table which is read from path.

    parameters
    ----------
    path : string
            Path to tape-archive in filesystem
    """
    tmp = {}
    with sequential_tar.open(name=path, fileobj=fileobj, mode="r") as tar:
        for item in tar:
            level_key, column_key, dtype_key = _split_level_column_dtype(
                path=item.name
            )
            level_column_bytes = item.read(mode="rb")
            if level_key not in tmp:
                tmp[level_key] = {}
            tmp[level_key][column_key] = np.frombuffer(
                level_column_bytes, dtype=dtype_key
            )

    out = SparseNumericTable()
    for level_key in list(tmp.keys()):
        level_recarray = dict_to_recarray(tmp[level_key])
        out[level_key] = DynamicSizeRecarray(recarray=level_recarray)
        del tmp[level_key]

    return out


# concatenate
# ===========


def concatenate_files(list_of_table_paths, dtypes):
    if len(list_of_table_paths) == 0:
        out = {}
        for level_key in dtypes:
            out[level_key] = np.frombuffer(b"", dtype=dtypes[level_key])
        return out

    with tempfile.TemporaryDirectory(prefix="sparse_table_concatenate") as tmp:
        for part_table_path in list_of_table_paths:
            part_table = read(path=part_table_path)
            testing.assert_dtypes_are_equal(part_table.dtypes, dtypes)
            for level_key in dtypes:
                with open(os.path.join(tmp, level_key), "ab") as fa:
                    fa.write(part_table[level_key].tobytes())

        out = SparseNumericTable(dtypes=dtypes)
        for level_key in dtypes:
            with open(os.path.join(tmp, level_key), "rb") as f:
                out[level_key].append_recarray(
                    np.frombuffer(f.read(), dtype=dtypes[level_key])
                )

    return out
