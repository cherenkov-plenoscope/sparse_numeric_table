from .base import IDX
from .base import IDX_DTYPE
from .base import make_mask_of_right_in_left

import json
import posixpath
import gzip
import numpy as np
import tarfile
import io
import dynamicsizerecarray
import tempfile
import rename_after_writing


def write(f, snt, mode="w|", level_mode="wb|gz", level_block_size=2**25):
    assert mode.startswith("w|")
    assert level_mode.startswith("wb|")
    assert level_block_size > 0

    with tarfile.open(fileobj=f, mode=mode) as tarf:
        tarf_write(
            tarf=tarf,
            filename="dtype.json",
            payload=dumps_dtype(snt=snt),
            mode="wt",
        )

        for level_key in snt:
            single_record_size = estimate_single_record_size(
                dtype=snt[level_key].dtype,
            )
            num_records_per_block = int(
                np.ceil(level_block_size / single_record_size)
            )

            num_records_written = 0
            block_id = 0
            ifinal = len(snt[level_key]) - 1
            istart = 0
            istop = 0

            while istop < ifinal:
                block_filename = posixpath.join(
                    level_key, "{:06d}.rec".format(block_id)
                )
                if "|gz" in level_mode:
                    block_filename += ".gz"

                istart = block_id * num_records_per_block
                istop = istart + num_records_per_block

                level_block = snt[level_key][istart:istop]

                tarf_write(
                    tarf=tarf,
                    filename=block_filename,
                    payload=level_block.tobytes(),
                    mode=level_mode,
                )

                block_id += 1


def tarf_write(tarf, filename, payload, mode="wt"):
    if "b" in mode:
        payload_raw = payload
    elif "t" in mode:
        payload_raw = str.encode(payload)
    else:
        raise ValueError("mode must either contain 'b' or 't'.")

    if "|gz" in mode:
        assert str.endswith(filename, ".gz")
        payload_bytes = gzip.compress(payload_raw)
    else:
        payload_bytes = payload_raw

    with io.BytesIO() as buff:
        tarinfo = tarfile.TarInfo(filename)
        tarinfo.size = buff.write(payload_bytes)
        buff.seek(0)
        tarf.addfile(tarinfo, buff)


def dumps_dtype(snt):
    out = {}
    for level_key in snt:
        out[level_key] = []
        outlevel = out[level_key]
        level = snt[level_key]

        for column_key in level.dtype.names:
            if not column_key is IDX:
                column_dtype_key = level.dtype[column_key].str
                outlevel.append([column_key, column_dtype_key])

    return json.dumps(out, indent=4)


def estimate_single_record_size(dtype):
    dummy = np.core.records.recarray(shape=1, dtype=dtype)
    dummy_bytes = dummy.tobytes()
    return len(dummy_bytes)


def read(f, mode="r|", levels=None, indices=None):
    assert mode.startswith("r|")
    dynamic_table = {}
    file_table_dtype = {}

    with tarfile.open(fileobj=f, mode=mode) as tarf:
        tarinfo = tarf.next()
        filename, filetext = tarf_read(tarf=tarf, tarinfo=tarinfo, mode="rt")
        assert filename == "dtype.json"
        full_head = json.loads(filetext)

        if levels is None:
            head = full_head
        else:
            head = {}
            for level_key in levels:
                head[level_key] = full_head[level_key]

        for level_key in head:
            file_table_dtype[level_key] = add_idx_to_level_dtype(
                level_dtype=head[level_key]
            )
            dynamic_table[level_key] = dynamicsizerecarray.DynamicSizeRecarray(
                dtype=file_table_dtype[level_key]
            )

        while True:
            tarinfo = tarf.next()
            if tarinfo is None:
                break
            name, buff = tarf_read(tarf=tarf, tarinfo=tarinfo, mode="rb")
            level_key, block_id_str = posixpath.split(name)
            if level_key in head:
                block_rec = np.frombuffer(
                    buff, dtype=file_table_dtype[level_key]
                )
                if indices is not None:
                    block_mask = make_mask_of_right_in_left(
                        left_indices=block_rec[IDX],
                        right_indices=indices,
                    )
                    block_rec = block_rec[block_mask]
                dynamic_table[level_key].append_recarray(block_rec)

    snt = {}
    dsnt_level_keys = list(dynamic_table.keys())
    for level_key in dsnt_level_keys:
        dynrec = dynamic_table.pop(level_key)
        snt[level_key] = dynrec.to_recarray()
    return snt


def tarf_read(tarf, tarinfo, mode="rt"):
    payload_bytes = tarf.extractfile(tarinfo).read()
    _, filename_ext = posixpath.splitext(tarinfo.name)
    if filename_ext == ".gz":
        payload_raw = gzip.decompress(payload_bytes)
        filename, _ = posixpath.splitext(tarinfo.name)
    else:
        payload_raw = payload_bytes
        filename = tarinfo.name

    if "t" in mode:
        payload = bytes.decode(payload_raw)
    elif "b" in mode:
        payload = payload_raw
    else:
        raise ValueError("mode must either contain 'b' or 't'.")

    return filename, payload


def add_idx_to_level_dtype(level_dtype):
    full_dtype = [(IDX, IDX_DTYPE)]
    for column_key_dtype in level_dtype:
        full_dtype.append(tuple(column_key_dtype))
    return full_dtype


"""
def reacarray_has_dtype(recarray, dtype):
    pass



class FileWriter:
    def __init__(self, fileobj, mode, level_mode):
        assert mode.startswith("w|")
        assert level_mode.startswith("wb|")

        self.fileobj = fileobj
        self.mode = mode
        self.level_mode = level_mode
        self.tarf = tarfile.open(fileobj=self.fileobj, mode=self.mode)
        self.level_dtypes = None
        self.level_block_ids = {}

    def write_level_dtypes(self, level_dtypes):
        assert self.level_dtypes is None
        self.level_dtypes = level_dtypes
        tarf_write(
            tarf=self.tarf,
            filename="level_dtypes.json",
            payload=json.dumps(level_dtypes, indent=4),
            mode="wt",
        )
        for level_key in level_dtypes:
            self.level_block_ids[level_key] = 0

    def write_level_block(self, level_key, level_recarray):
        assert level_key in self.level_dtypes
        expected_level_dtype = add_idx_to_level_dtype(
            self.level_dtypes[level_key]
        )

        block_filename = posixpath.join(
            level_key, "{:06d}.rec".format(self.level_block_ids[level_key])
        )
        if "|gz" in self.level_mode:
            block_filename += ".gz"

        tarf_write(
            tarf=self.tarf,
            filename=block_filename,
            payload=level_recarray.tobytes(),
            mode=level_mode,
        )

        self.level_block_ids[level_key] += 1



    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        self.tarf.close()

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)




def concatenate(inpaths, outpath, work_dir=None):

    with tempfile.TemporaryDirectory(suffix=None, prefix=None, dir=work_dir) as tmp:
        tmp_path = os.path.join(tmp, "concatenate.tar")
        with rename_after_writing.open(tmp_path, "wb") as fout:




    rename_after_writing.move(tmp_path, outpath)
"""
