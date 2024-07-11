import pandas as pd
import numpy as np
import io

from dynamicsizerecarray import DynamicSizeRecarray

from . import validating


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
            out.write(f"    {lk: <30s} ({self._table[lk].shape[0]: 9_d})\n")
            for ck in self._table[lk].dtype.names:
                out.write(f"        {ck:s}\n")
        out.seek(0)
        return out.read()


def _init_tables_from_dtypes(dtypes):
    validating.assert_dtypes_are_valid(dtypes=dtypes)
    tables = {}
    for level_key in dtypes:
        tables[level_key] = DynamicSizeRecarray(dtype=dtypes[level_key])
    return tables


def dict_to_recarray(d):
    return pd.DataFrame(d).to_records(index=False)
