import pandas as pd
import numpy as np
import io
import copy

from dynamicsizerecarray import DynamicSizeRecarray

from . import validating
from . import logic


class SparseNumericTable:
    def __init__(self, dtypes=None):
        if dtypes is None:
            self._table = {}
        else:
            self._table = _init_tables_from_dtypes(dtypes=dtypes)

    def __setitem__(self, level_key, level_recarray):
        lk = level_key
        lr = level_recarray
        validating.assert_key_is_valid(lk)

        if isinstance(lr, DynamicSizeRecarray):
            lr = lr
        elif isinstance(lr, np.recarray):
            lr = DynamicSizeRecarray(recarray=lr)
        else:
            raise ValueError("Expected DynamicSizeRecarray or np.recarray")
        self._table[lk] = lr

    def __getitem__(self, level_key):
        return self._table[level_key]

    def __iter__(self):
        return self._table.__iter__()

    def keys(self):
        return self._table.keys()

    def list_level_keys(self):
        return list(self.keys())

    def list_column_keys(self, level_key):
        return list(self._table[level_key].keys())

    def _get_level_column(self, level_key, column_key):
        return self._table[level_key][column_key]

    def _get_len_level(self, level_key):
        return len(self._table[level_key])

    def shrink_to_fit(self):
        for lk in self._table:
            self._table[lk].shrink_to_fit()

    @property
    def dtypes(self):
        out = {}
        for lk in self._table:
            level_dtype = []
            for ck in self._table[lk].dtype.names:
                column_dtype = self._table[lk].dtype[ck].descr[0][1]
                level_dtype.append((ck, column_dtype))
            out[lk] = level_dtype
        return out

    @property
    def shapes(self):
        out = {}
        for lk in self._table:
            out[lk] = self._table[lk].shape
        return out

    def __repr__(self):
        o = f"{self.__class__.__name__:s}"
        o += f"()"
        return o

    def info(self):
        out = io.StringIO()
        out.write(self.__repr__())
        out.write("\n")
        for lk in self._table:
            pad_key = f"    {lk:s} "
            out.write(f"{pad_key:_<45s}[ {self._table[lk].shape[0]: 9_d} ]_\n")
            for ck in self._table[lk].dtype.names:
                cd = self._table[lk].dtype[ck].str
                out.write(f"        {ck:.<45s} {cd:s}\n")
        out.seek(0)
        return out.read()

    def append(self, other):
        """
        Append another table.

        Parameters
        ----------
        other : SparseNumericTable (or dict)
            The other table will be appended to this one (self).
        """
        for lk in other:
            self[lk].append_recarray(other[lk])

    def intersection(self, index, levels=None):
        return _intersection(handle=self, index=index, levels=levels)

    def cut(
        self,
        index=None,
        indices=None,
        levels_and_columns=None,
        align_indices=False,
    ):
        return _cut(
            handle=self,
            index=index,
            indices=indices,
            levels_and_columns=levels_and_columns,
            align_indices=align_indices,
        )


def _init_tables_from_dtypes(dtypes):
    validating.assert_dtypes_are_valid(dtypes=dtypes)
    tables = {}
    for level_key in dtypes:
        tables[level_key] = DynamicSizeRecarray(dtype=dtypes[level_key])
    return tables


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


def _cut(
    handle,
    index=None,
    indices=None,
    levels_and_columns=None,
    align_indices=False,
):
    sub_dtypes = _sub_dtypes(
        dtypes=handle.dtypes, levels_and_columns=levels_and_columns
    )

    out = SparseNumericTable()
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


def dict_to_recarray(d):
    return pd.DataFrame(d).to_records(index=False)
