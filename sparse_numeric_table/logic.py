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


def _cut_level_on_indices(level, indices, index_key, column_keys=None):
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
    mask = make_mask_of_right_in_left(
        left_indices=level[index_key],
        right_indices=indices,
    )
    out_dtype = _base._sub_level_dtypes(
        level_dtype=_base._get_simple_dtype_from_recarray(level),
        column_keys=column_keys,
    )
    out = DynamicSizeRecarray(shape=sum(mask), dtype=out_dtype)
    for ck, _ in out_dtype:
        out[ck] = level[ck][mask]
    return out


def cut_table_on_indices(table, common_indices):
    """
    Returns table but only with the rows listed in common_indices.

    Parameters
    ----------
    table : dict of recarrays, or SparseNumericTable.
        The sparse numeric table.
    common_indices : list of indices
        The row-indices to cut on. Only row-indices in this list will go in the
        output-table.
    """
    common_indices = np.asarray(common_indices)

    out = SparseNumericTable(index_key=table.index_key)
    for lk in table:
        out[lk] = _cut_level_on_indices(
            level=table[lk],
            indices=common_indices,
            index_key=table.index_key,
        )
    return out


def sort_table_on_common_indices(table, common_indices):
    """
    Returns a table with all row-indices ordered same as common_indices.

    Parameters
    ----------
    table : dict of recarrays, or SparseNumericTable.
        The sparse numeric table, but must be rectangular, i.e. not sparse.
    common_indices : list of indices
        The row-indices to sort by.
    """
    common_indices = np.asarray(common_indices)

    order = np.argsort(common_indices)
    inv_order = np.zeros(shape=common_indices.shape, dtype=int)
    inv_order[order] = np.arange(len(common_indices))
    del order

    out = SparseNumericTable(index_key=table.index_key)
    for lk in table:
        level = table[lk]
        level_order_args = np.argsort(level[table.index_key])
        level_sorted = level[level_order_args]
        del level_order_args
        level_same_order_as_common = level_sorted[inv_order]
        out[lk] = level_same_order_as_common
    return out


def cut_and_sort_table_on_indices(table, common_indices):
    """
    Returns a table (rectangular, not sparse) containing only rows listed in
    common_indices and in this order.

    Parameters
    ----------
    table : dict of recarrays, or SparseNumericTable.
        The sparse numeric table.
    common_indices : list of indices
        The row-indices to cut on and sort by.
    """
    common_indices = np.asarray(common_indices)

    out = cut_table_on_indices(
        table=table,
        common_indices=common_indices,
    )
    out = sort_table_on_common_indices(
        table=out,
        common_indices=common_indices,
    )
    return out


def make_rectangular_DataFrame(table, delimiter="/"):
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
    """
    ik = table.index_key

    out = {}
    for lk in table:
        for ck in table[lk].dtype.names:
            if ck == ik:
                if ik in out:
                    np.testing.assert_array_equal(out[ik], table[lk][ik])
                else:
                    out[ik] = table[lk][ik]
            else:
                out[f"{lk:s}{delimiter:s}{ck:s}"] = table[lk][ck]
    return pd.DataFrame(out)
