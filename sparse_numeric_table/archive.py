from .base import IDX
from .base import IDX_DTYPE

import zipfile
import numpy as np
import posixpath
import dynamicsizerecarray
import gzip
import copy


def open(file, mode="r", compress=False, block_size=1_000_000):
    if str.lower(mode) == "r":
        return Reader(file=file)
    elif str.lower(mode) == "w":
        return Writer(file=file, compress=compress, block_size=block_size)
    else:
        raise KeyError(
            f"Expected 'mode' to be in ['r', 'w']. But it is '{mode:s}'"
        )


class Writer:
    def __init__(self, file, compress=False, block_size=1_000_000):
        self.zipfile = zipfile.ZipFile(file=file, mode="w")
        self.gz = ".gz" if compress else ""
        self.block_size = int(block_size)
        assert self.block_size > 0

    def _write_level_block(self, level_block, level_key, block_id):
        level_block_path = posixpath.join(level_key, f"{block_id:06d}")

        for column_key in level_block.dtype.names:
            column_dtype_key = level_block.dtype[column_key].str
            path = posixpath.join(
                level_block_path,
                f"{column_key:s}.{column_dtype_key:s}{self.gz:s}",
            )
            with self.zipfile.open(path, mode="w") as fout:
                payload = level_block[column_key].tobytes()
                if self.gz:
                    payload = gzip.compress(payload)
                fout.write(payload)

    def write_level(self, level, level_key):
        block_steps = set(
            np.arange(start=0, stop=level.shape[0], step=self.block_size)
        )
        block_steps.add(level.shape[0])
        block_steps = list(block_steps)
        block_steps = np.array(block_steps)
        block_steps = sorted(block_steps)

        for block_id in range(len(block_steps) - 1):
            start = block_steps[block_id]
            stop = block_steps[block_id + 1]
            level_block = level[start:stop]

            self._write_level_block(
                level_block=level_block,
                level_key=level_key,
                block_id=block_id,
            )

    def write_table(self, table):
        for level_key in table:
            self.write_level(level=table[level_key], level_key=level_key)

    def close(self):
        self.zipfile.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"


class Reader:
    def __init__(self, file):
        self.zipfile = zipfile.ZipFile(file=file, mode="r")
        self.infolist = self.zipfile.infolist()

        self.info = {}
        for item in self.infolist:
            oo = split_filename(filename=item.filename)
            lk = oo["level_key"]
            ck = oo["column_key"]
            bk = oo["block_key"]
            if lk not in self.info:
                self.info[lk] = {}

            if ck not in self.info[lk]:
                self.info[lk][ck] = {}

            if bk not in self.info[lk][ck]:
                self.info[lk][ck][bk] = {
                    "filename": item.filename,
                    "compressed": oo["compressed"],
                    "dtype": oo["column_dtype_key"],
                }

        self.dtypes = {}
        for lk in self.list_level_keys():
            self.dtypes[lk] = []
            for ck in self.list_column_keys(lk):
                block_dtypes = set()
                for bk in self.info[lk][ck]:
                    block_dtype = self.info[lk][ck][bk]["dtype"]
                block_dtypes.add(block_dtype)
                assert len(block_dtypes) == 1
                entry = list(block_dtypes)[0]
                self.dtypes[lk].append((ck, entry))

    def list_level_keys(self):
        return list(self.info.keys())

    def list_column_keys(self, level_key):
        return list(self.info[level_key].keys())

    def read_column(self, level_key, column_key):
        dtype = None
        for item in self.dtypes[level_key]:
            if item[0] == column_key:
                dtype = [(column_key, item[1])]
        out = dynamicsizerecarray.DynamicSizeRecarray(dtype=dtype)

        for block_key in self.info[level_key][column_key]:
            filename = self.info[level_key][column_key][block_key]["filename"]
            with self.zipfile.open(filename, "r") as fin:
                payload = fin.read()
                if self.info[level_key][column_key][block_key]["compressed"]:
                    payload = gzip.decompress(payload)
                block = np.frombuffer(payload, dtype=dtype)
                out.append_recarray(block)
        return out.to_recarray()

    def _sub_dtypes(self, levels_and_columns=None):
        if levels_and_columns is None:
            return self.dtypes
        out = {}
        for lk in levels_and_columns:
            out[lk] = []

            if IDX not in levels_and_columns[lk]:
                levels_and_columns[lk].insert(0, IDX)

            for ck in levels_and_columns[lk]:
                dt = None
                for item in self.dtypes[lk]:
                    if item[0] == ck:
                        dt = (ck, item[1])
                assert dt is not None
                out[lk].append(dt)

        return out

    def read_table(self, levels_and_columns=None):
        sub_dtyes = self._sub_dtypes(levels_and_columns=levels_and_columns)

        out = {}
        for lk in sub_dtyes:
            level_dtype = copy.deepcopy(sub_dtyes[lk])
            first_dtype = level_dtype.pop(0)
            ck = first_dtype[0]
            first_column = self.read_column(level_key=lk, column_key=ck)
            out[lk] = np.core.records.recarray(
                shape=first_column.shape[0],
                dtype=sub_dtyes[lk],
            )
            out[lk][ck] = first_column

            for next_dtype in level_dtype:
                ck = next_dtype[0]
                next_column = self.read_column(level_key=lk, column_key=ck)
                out[lk][ck] = next_column

        return out

    def close(self):
        self.zipfile.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"


def split_filename(filename):
    filename, basename = posixpath.split(filename)
    out = {}
    basename, ext = posixpath.splitext(basename)
    if ext == ".gz":
        out["compressed"] = True
        out["column_key"], out["column_dtype_key"] = posixpath.splitext(
            basename
        )
    else:
        out["compressed"] = False
        out["column_key"] = basename
        out["column_dtype_key"] = ext

    out["column_dtype_key"] = str.replace(out["column_dtype_key"], ".", "")
    level_key, block_key = posixpath.split(filename)

    out["level_key"] = level_key
    out["block_key"] = block_key
    return out
