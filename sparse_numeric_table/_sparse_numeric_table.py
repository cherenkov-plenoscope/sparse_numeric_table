import numpy as np
import io
import copy

from dynamicsizerecarray import DynamicSizeRecarray

from . import validating
from . import _base


class SparseNumericTable:
    """
    parameters
    ----------
    index_key : str
        Key of the index column which every level must have.
    dtypes : dict
        Dtypes of the individual levels.
        Example: {
            "level_a": [(index_key, "<i4", "column_x": "f4")],
            "level_b": [(index_key, "<i4", "column_y": "i1", "column_z": "i4")],
        }
    """

    def __init__(self, index_key, dtypes=None):
        self.set_index_key(index_key=index_key)

        if dtypes is None:
            self._table = {}
        else:
            validating.assert_all_levels_have_index_key(
                dtypes=dtypes, index_key=self.index_key
            )
            self._table = _init_tables_from_dtypes(dtypes=dtypes)

    def set_index_key(self, index_key):
        _index_key_str = str(index_key)
        validating.assert_key_is_valid(_index_key_str)
        self._index_key = copy.copy(_index_key_str)

    def __setitem__(self, level_key, level_recarray):
        lk = level_key
        lr = level_recarray
        validating.assert_key_is_valid(lk)

        if isinstance(lr, DynamicSizeRecarray):
            lr = lr
        elif isinstance(lr, np.recarray):
            lr = DynamicSizeRecarray(recarray=lr)
        else:
            raise ValueError(
                "Expected DynamicSizeRecarray or np.recarray, "
                f"but got '{repr(lr):s}'"
            )
        self._table[lk] = lr

        validating.assert_all_levels_have_index_key(
            dtypes=self.dtypes, index_key=self.index_key
        )

    def update(self, other):
        """
        Update a level in the table. If the level already exists it will be
        overwritten.
        """
        for level_key in other.keys():
            self[level_key] = other[level_key]

    def append(self, other):
        """
        Append the levels of another table to the levels of this table without
        overwriting the levels of this table. If a level in 'other' does not
        exist in 'self', the level will be created in 'self'.

        Parameters
        ----------
        other : SparseNumericTable
            Will be appended to 'self'.
        """
        for level_key in other.keys():
            _level_recarray = other[level_key].to_recarray()

            if level_key in self.keys():
                self[level_key].append(_level_recarray)
            else:
                self[level_key] = _level_recarray

    def __getitem__(self, level_key):
        return self._table[level_key]

    def __iter__(self):
        return self._table.__iter__()

    def keys(self):
        return self._table.keys()

    def list_level_keys(self):
        return list(self.keys())

    def list_column_keys(self, level_key):
        return list(self._table[level_key].dtype.names)

    def _get_level(self, level_key, column_keys, indices=None):
        out_dtype = _base._sub_level_dtypes(
            level_dtype=self.dtypes[level_key],
            column_keys=column_keys,
        )

        level_indices = self[level_key][self.index_key]
        if indices is not None:
            level_mask = _base.make_mask_of_right_in_left(
                left_indices=level_indices,
                right_indices=indices,
            )
        else:
            level_mask = np.ones(
                shape=level_indices.shape[0],
                dtype=bool,
            )

        out = DynamicSizeRecarray(shape=sum(level_mask), dtype=out_dtype)
        for column_key, _ in out_dtype:
            out[column_key] = self[level_key][column_key][level_mask]
        return out

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

    @property
    def index_key(self):
        return copy.copy(self._index_key)

    def __repr__(self):
        return self.info()

    def info(self):
        _head = f'{self.__class__.__name__:s}(index_key="{self.index_key:s}")'

        out = io.StringIO()
        out.write(_head)
        out.write("\n")
        for lk in self._table:
            out.write(
                f'    "{lk:s}", (len={self._table[lk].shape[0]:_d}) = [\n'
            )
            for ck in self._table[lk].dtype.names:
                cd = self._table[lk].dtype[ck].str
                out.write(f'        ("{ck:s}", "{cd:s}"),\n')
            out.write(f"    ]\n")
        out.seek(0)
        return out.read()

    def query(
        self,
        indices=None,
        levels_and_columns=None,
        sort=False,
    ):
        return _base._query(
            handle=self,
            indices=indices,
            levels_and_columns=levels_and_columns,
            sort=sort,
        )


def _init_tables_from_dtypes(dtypes):
    validating.assert_dtypes_are_valid(dtypes=dtypes)
    tables = {}
    for level_key in dtypes:
        tables[level_key] = DynamicSizeRecarray(dtype=dtypes[level_key])
    return tables
