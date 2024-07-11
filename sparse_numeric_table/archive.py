import zipfile
import numpy as np
import posixpath
import dynamicsizerecarray
import gzip
import copy

from .base import SparseNumericTable


def open(file, mode="r", dtypes=None, compress=False, block_size=262_144):
    if str.lower(mode) == "r":
        return Reader(file=file)
    elif str.lower(mode) == "w":
        assert dtypes is not None, "mode='w' requires dtypes."
        return Writer(
            file=file, dtypes=dtypes, compress=compress, block_size=block_size
        )
    else:
        raise KeyError(
            f"Expected 'mode' to be in ['r', 'w']. But it is '{mode:s}'"
        )


class LevelBuffer:
    def __init__(
        self,
        zipfile,
        level_key,
        level_dtype,
        compress=False,
        block_size=100_000,
    ):
        self.zipfile = zipfile
        self.level_key = level_key
        self.level_dtype = level_dtype
        self.gz = ".gz" if compress else ""
        self.block_size = block_size
        assert self.block_size > 0
        self.block_id = 0
        self.level = np.core.records.recarray(
            shape=self.block_size,
            dtype=self.level_dtype,
        )
        self.size = 0

    def _append_level(self, level):
        assert level.shape[0] <= self.block_size

        new_size = self.size + level.shape[0]

        if new_size <= self.block_size:
            self.level[self.size : new_size] = level
            self.size = new_size
        else:
            part_size = self.block_size - self.size
            level_first_part = level[:part_size]
            self.level[self.size : self.block_size] = level_first_part
            self.size = self.block_size

            self.flush()

            level_second_part = level[part_size:]
            new_size = self.size + level_second_part.shape[0]
            self.level[self.size : new_size] = level_second_part
            self.size = new_size

    def append_level(self, level):
        block_steps = set(
            np.arange(start=0, stop=level.shape[0], step=self.block_size)
        )
        block_steps.add(level.shape[0])
        block_steps = list(block_steps)
        block_steps = np.array(block_steps)
        block_steps = sorted(block_steps)

        for i in range(len(block_steps) - 1):
            start = block_steps[i]
            stop = block_steps[i + 1]
            level_block = level[start:stop]
            self._append_level(level=level_block)

    def flush(self):
        if self.size == 0:
            return

        level_block_path = posixpath.join(
            self.level_key, f"{self.block_id:06d}"
        )

        for column_key in self.level.dtype.names:
            column_dtype_key = self.level.dtype[column_key].str
            path = posixpath.join(
                level_block_path,
                f"{column_key:s}.{column_dtype_key:s}{self.gz:s}",
            )
            with self.zipfile.open(path, mode="w") as fout:
                payload = self.level[column_key][: self.size].tobytes()
                if self.gz:
                    payload = gzip.compress(payload)
                fout.write(payload)

        self.block_id += 1
        self.size = 0


class Writer:
    def __init__(self, file, dtypes, compress, block_size):
        self.zipfile = zipfile.ZipFile(file=file, mode="w")
        self.compress = compress
        self.block_size = block_size
        self.dtypes = dtypes
        self.buffers = {}
        for lk in self.dtypes:
            self.buffers[lk] = LevelBuffer(
                zipfile=self.zipfile,
                level_key=lk,
                level_dtype=self.dtypes[lk],
                compress=self.compress,
                block_size=self.block_size,
            )

    def append_table(self, table):
        for lk in table:
            self.buffers[lk].append_level(level=table[lk])

    def close(self):
        for lk in self.buffers:
            self.buffers[lk].flush()
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

            if isinstance(levels_and_columns[lk], str):
                if levels_and_columns[lk] == "__all__":
                    out[lk] = self.dtypes[lk]
                else:
                    raise KeyError(
                        "Expected column command to be in ['__all__']."
                        f"But it is '{levels_and_columns[lk]:s}'."
                    )
            else:
                for ck in levels_and_columns[lk]:
                    dt = None
                    for item in self.dtypes[lk]:
                        if item[0] == ck:
                            dt = (ck, item[1])
                    assert dt is not None
                    out[lk].append(dt)

        return out

    def read_table(self, levels_and_columns=None):
        sub_dtypes = self._sub_dtypes(levels_and_columns=levels_and_columns)

        out = SparseNumericTable()
        for lk in sub_dtypes:
            level_dtype = copy.deepcopy(sub_dtypes[lk])
            first_dtype = level_dtype.pop(0)
            ck = first_dtype[0]
            first_column = self.read_column(level_key=lk, column_key=ck)

            _tmp_rec = np.core.records.recarray(
                shape=first_column.shape[0],
                dtype=sub_dtypes[lk],
            )

            out[lk] = dynamicsizerecarray.DynamicSizeRecarray(
                recarray=_tmp_rec
            )

            out[lk][ck] = first_column

            for next_dtype in level_dtype:
                ck = next_dtype[0]
                next_column = self.read_column(level_key=lk, column_key=ck)
                out[lk][ck] = next_column

        out.shrink_to_fit()
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
