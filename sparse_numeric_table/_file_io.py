import zipfile
import numpy as np
import posixpath
import dynamicsizerecarray
import gzip
import copy

from . import _base


def open(
    file,
    mode="r",
    dtypes_and_index_key_from=None,
    dtypes=None,
    index_key=None,
    compress=False,
    block_size=262_144,
):
    """
    Write or read a SparseNumericTable.
    When writing, the table can be appended peace by peace. No need to have the
    full table in memory.
    When reading, only certain columns of certain levels can be queried. No need
    to read the full table into memory at once.

    Parameters
    ----------
    file : file handle or path (str)
        See zipfile.Zipfile(file)?
    mode : str (default="r")
        Either "w"rite or "r"ead.
    dtypes_and_index_key_from : SparseNumericTable
        When mode="w" (writing), the 'dtypes' and 'index_key' must be known.
        Either from a 'SparseNumericTable' or from parameters 'dtypes' and
        'index_key'.
    dtypes : dict
        See parameter 'dtypes_and_index_key_from'.
    index_key : str
        See parameter 'dtypes_and_index_key_from'.
    compress : bool
        Compress internal blocks using gzip when True.
    block_size : int (default=262_144)
        The maximum size in bytes of a level block.
    """
    if str.lower(mode) == "r":
        return SparseNumericTableReader(file=file)
    elif str.lower(mode) == "w":
        dtypes, index_key = _get_dtypes_and_index_key(
            dtypes=dtypes,
            index_key=index_key,
            dtypes_and_index_key_from=dtypes_and_index_key_from,
        )
        return SparseNumericTableWriter(
            file=file,
            dtypes=dtypes,
            index_key=index_key,
            compress=compress,
            block_size=block_size,
        )
    else:
        raise KeyError(
            f"Expected 'mode' to be in ['r', 'w']. But it is '{mode:s}'"
        )


class SparseNumericTableLevelWriter:
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
        self.level = np.recarray(
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


class SparseNumericTableWriter:
    def __init__(self, file, dtypes, index_key, compress, block_size):
        self.zipfile = zipfile.ZipFile(file=file, mode="w")
        self.compress = compress
        self.block_size = block_size
        self.dtypes = dtypes
        self.index_key = index_key
        self.buffers = {}
        self.write_index_key()

        for lk in self.dtypes:
            self.buffers[lk] = SparseNumericTableLevelWriter(
                zipfile=self.zipfile,
                level_key=lk,
                level_dtype=self.dtypes[lk],
                compress=self.compress,
                block_size=self.block_size,
            )

    def write_index_key(self):
        path = "__index_key__.txt"
        with self.zipfile.open(path, mode="w") as fout:
            _index_key_bytes = self.index_key.encode()
            fout.write(_index_key_bytes)

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


class SparseNumericTableReader:
    def __init__(self, file):
        self.zipfile = zipfile.ZipFile(file=file, mode="r")
        self.infolist = self.zipfile.infolist()

        self.info = {}
        self._index_key = None

        for item in self.infolist:
            oo = _properties_from_filename(filename=item.filename)

            if oo["is_index_key"]:
                self._index_key = self.read_index_key(filename=item.filename)
            else:
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

    @property
    def index_key(self):
        return copy.copy(self._index_key)

    def list_level_keys(self):
        return list(self.info.keys())

    def list_column_keys(self, level_key):
        return list(self.info[level_key].keys())

    def _get_level_column(self, level_key, column_key):
        return self.read_column(level_key, column_key)

    def _get_len_level(self, level_key):
        out = 0
        first_column_key = list(self.info[level_key].keys())[0]
        first_column_in_level = self.read_column(level_key, first_column_key)
        return len(first_column_in_level)

    def read_index_key(self, filename):
        with self.zipfile.open(filename, "r") as fin:
            _index_key_bytes = fin.read()
            return _index_key_bytes.decode()

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

    def intersection(self, index, levels=None):
        return _base._intersection(handle=self, index=index, levels=levels)

    def query(
        self,
        index=None,
        indices=None,
        levels_and_columns=None,
        align_indices=False,
    ):
        return _base._query(
            handle=self,
            index=index,
            indices=indices,
            levels_and_columns=levels_and_columns,
            align_indices=align_indices,
        )

    def close(self):
        self.zipfile.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"


def _properties_from_filename(filename):
    out = {}
    out["is_index_key"] = False

    if filename == "__index_key__.txt":
        out["is_index_key"] = True
        return out

    filename, basename = posixpath.split(filename)

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


def concatenate_files(
    input_paths,
    output_path,
    dtypes=None,
    index_key=None,
    dtypes_and_index_key_from=None,
):
    dtypes, index_key = _get_dtypes_and_index_key(
        dtypes=dtypes,
        index_key=index_key,
        dtypes_and_index_key_from=dtypes_and_index_key_from,
    )

    with open(
        output_path, mode="w", dtypes=dtypes, index_key=index_key
    ) as tout:
        for input_path in input_paths:
            with open(input_path, mode="r") as tin:
                part = tin.query()
                tout.append_table(part)


def _get_dtypes_and_index_key(dtypes, index_key, dtypes_and_index_key_from):
    if dtypes_and_index_key_from is None:
        assert dtypes is not None, (
            "mode='w' requires 'dtypes' "
            "when 'dtypes_and_index_key_from' is not set."
        )
        assert index_key is not None, (
            "mode='w' requires 'index_key' "
            "when 'dtypes_and_index_key_from' is not set."
        )
        return dtypes, index_key
    else:
        assert dtypes is None, (
            "mode='w' with 'dtypes_and_index_key_from' "
            "set can not also have 'dtypes' set."
        )
        assert index_key is None, (
            "mode='w' with 'dtypes_and_index_key_from' "
            "set can not also have 'index_key' set."
        )
        return (
            dtypes_and_index_key_from.dtypes,
            dtypes_and_index_key_from.index_key,
        )
