from .base import SparseNumericTable
from .base import dict_to_recarray

import numpy as np
import sequential_tar
import shutil
from dynamicsizerecarray import DynamicSizeRecarray
import warnings


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
    warnings.warn(
        "The 'sparse_numeric_table.tar_format' is deprecated. "
        "Use 'sparse_numeric_table.open()' instead.",
        category=DeprecationWarning,
    )

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
