import sparse_numeric_table as spt
import numpy as np
import tempfile
import os


def test_write_sff_read_full_table():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    my_table = spt.testing.make_example_table(prng=prng, size=100 * 1000)
    with tempfile.TemporaryDirectory(prefix="test_sparse_table") as tmp:
        path = os.path.join(tmp, "my_table.tar")

        with open(path, "wb") as f:
            spt.tarstream.write(f=f, snt=my_table, level_block_size=int(1e5))

        with open(path, "rb") as f:
            my_table_back = spt.tarstream.read(f=f)

        spt.testing.assert_tables_are_equal(my_table, my_table_back)
