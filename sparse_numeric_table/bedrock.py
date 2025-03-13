from ._sparse_numeric_table import SparseNumericTable
from . import logic

import copy
import numpy as np
from dynamicsizerecarray import DynamicSizeRecarray


def _intersection(handle, index, levels=None):
    if levels is None:
        levels = handle.list_level_keys()

    if len(levels) == 0:
        return []

    first_indices = handle._get_level_column(
        level_key=levels[0], column_key=index
    )
    inter = first_indices
    for ll in range(1, len(levels)):
        level_key = levels[ll]
        next_indices = handle._get_level_column(
            level_key=level_key, column_key=index
        )
        inter = np.intersect1d(inter, next_indices)
    return inter


def _sub_dtypes(dtypes, levels_and_columns=None):
    if levels_and_columns is None:
        return dtypes
    out = {}
    for lk in levels_and_columns:
        out[lk] = []

        if isinstance(levels_and_columns[lk], str):
            if levels_and_columns[lk] == "__all__":
                out[lk] = dtypes[lk]
            else:
                raise KeyError(
                    "Expected column command to be in ['__all__']."
                    f"But it is '{levels_and_columns[lk]:s}'."
                )
        else:
            for ck in levels_and_columns[lk]:
                dt = None
                for item in dtypes[lk]:
                    if item[0] == ck:
                        dt = (ck, item[1])
                assert dt is not None
                out[lk].append(dt)

    return out


def _query(
    handle,
    index=None,
    indices=None,
    levels_and_columns=None,
    align_indices=False,
):
    """
    Query levels and columns on either a SparseNumericTable or on
    archive.Reader.
    """
    sub_dtypes = _sub_dtypes(
        dtypes=handle.dtypes, levels_and_columns=levels_and_columns
    )

    out = SparseNumericTable(index_key=copy.copy(handle._index_key))
    for lk in sub_dtypes:
        if index is not None and indices is not None:
            mask = logic.make_mask_of_right_in_left(
                left_indices=handle._get_level_column(lk, index),
                right_indices=indices,
            )
        else:
            mask = np.ones(handle._get_len_level(lk), dtype=bool)

        level_shape = np.sum(mask)

        out[lk] = DynamicSizeRecarray(
            dtype=sub_dtypes[lk],
            shape=level_shape,
        )

        for ck, cdtype in sub_dtypes[lk]:
            out[lk][ck] = handle._get_level_column(lk, ck)[mask]

    if align_indices:
        assert index is not None and indices is not None
        out._table = logic.sort_table_on_common_indices(
            table=out._table,
            common_indices=indices,
            index_key=index,
        )

    out.shrink_to_fit()
    return out
