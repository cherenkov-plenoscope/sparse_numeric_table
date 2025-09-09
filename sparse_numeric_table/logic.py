import pandas as pd
import numpy as np
from dynamicsizerecarray import DynamicSizeRecarray

from ._sparse_numeric_table import SparseNumericTable
from . import _base


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
    return _base.make_mask_of_right_in_left(
        left_indices=left_indices, right_indices=right_indices
    )


def intersection(*args):
    """
    Returns the logical intersection of indices among the '*args'.

    Parameters
    ----------
    *args : variable number of array like
        Lists of indices.

    Returns
    -------
    intersection : numpy.array(dtype=int)

    Example
    -------
    [4, 5, 6] = intersection([1,2,3,4,5,6], [3,4,5,6,7,8], [4,5,6,7,8,9,10])

    """
    num = len(args)

    if num == 0:
        return np.array([], dtype=int)
    else:
        out = _array_to_set(_asarray(args[0]))
        for i in range(1, num):
            out = out.intersection(_array_to_set(_asarray(args[i])))
        return np.asarray(list(out), dtype=int)


def difference(first, *others):
    """
    Returns the logical difference of indices in between 'first' and 'others'.

    Parameters
    ----------
    first : array like
        List of arrays to be subtracted from.
    *others : variable number of array like
        Lists being subtracted from 'first'.

    Returns
    -------
    difference : numpy.array(dtype=int)

    Example
    -------
    [5] = difference([1,2,3,4,5,6], [2,4,6], [1,2,3])
    """
    afirst = _asarray(first)
    sfirst = _array_to_set(afirst)
    sothers = _union_as_set(*others)
    diff = sfirst.difference(sothers)
    return np.asarray(list(diff), dtype=int)


def union(*args):
    """
    Returns the logical union of indices in '*args'.

    Parameters
    ----------
    *args : variable number of array like
        Lists of indices.

    Returns
    -------
    union : numpy.array(dtype=int)

    Example
    -------
    [1,2,3,4,5] = union([[1], [2], [3,4,5], [])
    """
    out = _union_as_set(*args)
    return np.asarray(list(out), dtype=int)


def _union_as_set(*args):
    num = len(args)
    if num == 0:
        return set()
    else:
        out = _array_to_set(_asarray(args[0]))
        for i in range(1, num):
            s = _array_to_set(_asarray(args[i]))
            out = set.union(out, s)
    return out


def _array_to_set(a):
    s = set(a)
    if len(s) < a.shape[0]:
        d = {}
        for i, v in enumerate(a):
            if v in d:
                d[v].append(i)
            else:
                d[v] = [i]
        report = ""
        for v in d:
            if len(d[v]) > 1:
                report += (
                    f"{{value={v:d}, indices=["
                    + str.join(",", [f"{i:d}" for i in d[v]])
                    + "]}, "
                )
        msg = (
            "Expected int/uint values in array to be unique. "
            f"But these values are not: {report:s}."
        )
        raise AssertionError(msg)
    return s


def _asarray(x):
    a = np.asarray(x)
    is_empty_anyhow = a.shape[0] == 0
    if not _is_int_uint_like_dtype(a.dtype) and not is_empty_anyhow:
        msg = f"Expected int/uint like dtype but got '{a.dtype.name:s}'."
        raise AssertionError(msg)
    return np.asarray(x, int)


def _is_int_uint_like_dtype(dtype):
    if dtype.name.startswith("int"):
        return True
    elif dtype.name.startswith("uint"):
        return True
    else:
        return False


def _use_index_key_of_table_if_None(table, index_key):
    if index_key is None:
        return table.index_key
    else:
        return index_key


def _use_level_keys_of_table_if_None(table, level_keys):
    if level_keys is None:
        return list(table.keys())
    else:
        return level_keys


def cut_on_common_indices(table, level_keys=None, index_key=None):
    """
    Parameters
    ----------
    table : dict of recarrays, or SparseNumericTable.
        The sparse numeric table.
    level_keys : list of strings (None)
        Cut on these levels.
    index_key : str (None)
        Key of the index column.
    """
    index_key = _use_index_key_of_table_if_None(table, index_key)
    level_keys = _use_level_keys_of_table_if_None(table, level_keys)

    list_of_lists_of_indices = []
    for level_key in level_keys:
        list_of_lists_of_indices.append(table[level_key][index_key])
    common_indices = intersection(list_of_lists_of_indices)

    return cut_and_sort_table_on_indices(
        table=table, common_indices=common_indices, level_keys=level_keys
    )


def cut_level_on_indices(level, indices, index_key, column_keys=None):
    """
    Returns a level (recarray) only containing the row-indices in 'indices'.

    Parameters
    ----------
    level : recarray
        A level in a sparse table.
    indices : list
        The row-indices to be written to the output-level.
    index_key : str
        Key of the index column.
    column_keys : list of strings (None)
        When specified, only these columns will be in the output-level.
    """
    if column_keys is None:
        column_keys = list(level.dtype.names)
    column_keys.append(index_key)
    _part = {}
    for column_key in column_keys:
        _part[column_key] = level[column_key]
    part_df = pd.DataFrame(_part)
    del _part
    common_df = pd.merge(
        part_df,
        pd.DataFrame({index_key: indices}),
        on=index_key,
        how="inner",
    )
    del part_df
    return DynamicSizeRecarray(recarray=common_df.to_records(index=False))


def cut_table_on_indices(
    table, common_indices, level_keys=None, index_key=None
):
    """
    Returns table but only with the rows listed in common_indices.

    Parameters
    ----------
    table : dict of recarrays, or SparseNumericTable.
        The sparse numeric table.
    common_indices : list of indices
        The row-indices to cut on. Only row-indices in this list will go in the
        output-table.
    level_keys : list of strings (None)
        When provided, the output-table will only contain these levels.
    index_key : str (None)
        Key of the index column.
    """
    common_indices = np.asarray(common_indices)
    index_key = _use_index_key_of_table_if_None(table, index_key)
    level_keys = _use_level_keys_of_table_if_None(table, level_keys)

    out = SparseNumericTable(index_key=index_key)
    for level_key in level_keys:
        out[level_key] = cut_level_on_indices(
            level=table[level_key],
            indices=common_indices,
            index_key=index_key,
        )
    return out


def sort_table_on_common_indices(table, common_indices, index_key=None):
    """
    Returns a table with all row-indices ordered same as common_indices.

    Parameters
    ----------
    table : dict of recarrays, or SparseNumericTable.
        The sparse numeric table, but must be rectangular, i.e. not sparse.
    common_indices : list of indices
        The row-indices to sort by.
    index_key : str (None)
        Key of the index column.
    """
    common_indices = np.asarray(common_indices)
    index_key = _use_index_key_of_table_if_None(table, index_key)

    common_order_args = np.argsort(common_indices)
    common_inv_order = np.zeros(shape=common_indices.shape, dtype=np.int64)
    common_inv_order[common_order_args] = np.arange(len(common_indices))
    del common_order_args

    out = SparseNumericTable(index_key=index_key)
    for level_key in table:
        level = table[level_key]
        level_order_args = np.argsort(level[index_key])
        level_sorted = level[level_order_args]
        del level_order_args
        level_same_order_as_common = level_sorted[common_inv_order]
        out[level_key] = level_same_order_as_common
    return out


def cut_and_sort_table_on_indices(
    table,
    common_indices,
    level_keys=None,
    index_key=None,
):
    """
    Returns a table (rectangular, not sparse) containing only rows listed in
    common_indices and in this order.

    Parameters
    ----------
    table : dict of recarrays, or SparseNumericTable.
        The sparse numeric table.
    common_indices : list of indices
        The row-indices to cut on and sort by.
    level_keys : list of strings (None)
        When specified, only this levels will be in the output-table.
    index_key : str (None)
        Key of the index column.
    """
    common_indices = np.asarray(common_indices)
    index_key = _use_index_key_of_table_if_None(table, index_key)

    out = cut_table_on_indices(
        table=table,
        common_indices=common_indices,
        index_key=index_key,
        level_keys=level_keys,
    )
    out = sort_table_on_common_indices(
        table=out,
        common_indices=common_indices,
        index_key=index_key,
    )
    return out


def make_rectangular_DataFrame(table, delimiter="/", index_key=None):
    """
    Returns a pandas.DataFrame made from a table.
    The table must already be rectangular, i.e. not sparse anymore.
    The row-indices among all levels in the table must have the same ordering.

    Parameters
    ----------
    table : dict of recarrays, or SparseNumericTable.
        The sparse numeric table.
    delimiter : str
        To join a level key with a column key.
    index_key : str (None)
        Key of the index column.
    """
    index_key = _use_index_key_of_table_if_None(table, index_key)

    out = {}
    for level_key in table:
        for column_key in table[level_key].dtype.names:
            if column_key == index_key:
                if index_key in out:
                    np.testing.assert_array_equal(
                        out[index_key], table[level_key][index_key]
                    )
                else:
                    out[index_key] = table[level_key][index_key]
            else:
                out[f"{level_key:s}{delimiter:s}{column_key:s}"] = table[
                    level_key
                ][column_key]
    return pd.DataFrame(out)
