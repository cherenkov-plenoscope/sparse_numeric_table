import pandas as pd
import numpy as np


LEVEL_COLUMN_DELIMITER = "/"
FILEAME_TEMPLATE = "{:s}" + LEVEL_COLUMN_DELIMITER + "{:s}.{:s}"

DTYPES = [
    "<u1",
    "<u2",
    "<u4",
    "<u8",
    "<i1",
    "<i2",
    "<i4",
    "<i8",
    "<f2",
    "<f4",
    "<f8",
]


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


def dict_to_recarray(d):
    return pd.DataFrame(d).to_records(index=False)
