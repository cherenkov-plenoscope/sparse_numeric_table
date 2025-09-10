from ._sparse_numeric_table import SparseNumericTable
from . import logic

import copy
import pandas as pd
import numpy as np
from dynamicsizerecarray import DynamicSizeRecarray


def make_mask_of_right_in_left(left_indices, right_indices):
    """
    Returns a mask for left indices indicating wheter a right index is in it.

    Parameters
    ----------
    left_indices : list of indices

    right_indices : list of indices

    Example
    -------
    [0, 1, 0, 0] = make_mask_of_right_in_left([1,2,3,4], [0,2,9])
    """
    left_df = pd.DataFrame({"i": left_indices})
    right_df = pd.DataFrame({"i": right_indices})
    mask_df = pd.merge(left_df, right_df, on="i", how="left", indicator=True)
    indicator_df = mask_df["_merge"]
    mask = np.array(indicator_df == "both", dtype=bool)
    return mask


def _sub_table_dtypes(table_dtypes, levels_and_columns=None):
    if levels_and_columns is None:
        return table_dtypes

    sub_table_dtype = {}
    for level_key in levels_and_columns:
        sub_table_dtype[level_key] = _sub_level_dtypes(
            level_dtype=table_dtypes[level_key],
            column_keys=levels_and_columns[level_key],
        )
    return sub_table_dtype


def _sub_level_dtypes(level_dtype, column_keys=None):
    if column_keys is None:
        return level_dtype
    sub_dtype = []

    if isinstance(column_keys, str):
        if column_keys == "__all__":
            sub_dtype = level_dtype
        else:
            raise KeyError(
                "Expected column command to be in ['__all__']."
                f"But it is '{column_keys:s}'."
            )
    else:
        for column_key in column_keys:
            dt = None
            for _column_key, _column_dtype in level_dtype:
                if _column_key == column_key:
                    dt = (_column_key, _column_dtype)
            assert dt is not None
            sub_dtype.append(dt)

    return sub_dtype


def _get_simple_dtype_from_recarray(recarray):
    out = []
    for column_key in recarray.dtype.names:
        out.append((column_key, recarray.dtype[column_key].str))
    return out


def _query(
    handle,
    indices=None,
    levels_and_columns=None,
    sort=False,
):
    """
    Query levels and columns on either a SparseNumericTable or on
    archive.Reader.
    """
    if levels_and_columns is None:
        levels_and_columns = {}
        for level_key in handle.list_level_keys():
            levels_and_columns[level_key] = handle.list_column_keys(
                level_key=level_key
            )

    out = SparseNumericTable(index_key=copy.copy(handle._index_key))

    for level_key in levels_and_columns:
        out[level_key] = handle._get_level(
            level_key=level_key,
            column_keys=levels_and_columns[level_key],
            indices=indices,
        )

    if sort:
        assert indices is not None
        out._table = logic.sort_table_on_common_indices(
            table=out._table,
            common_indices=indices,
            inplace=True,
        )

    out.shrink_to_fit()
    return out
