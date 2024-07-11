import sparse_numeric_table as snt
import numpy as np



def test_dtypes():
    prng = np.random.Generator(np.random.MT19937(seed=1337))
    table = snt.testing.make_example_table(prng=prng, size=1000)
    dtypes = table.dtypes
    for lk in table:
        assert dtypes[lk] == table[lk].dtype


def test_shapes():
    prng = np.random.Generator(np.random.MT19937(seed=1337))
    table = snt.testing.make_example_table(prng=prng, size=1000)
    shapes = table.shapes
    for lk in table:
        assert shapes[lk] == table[lk].shape


def test_info():
    prng = np.random.Generator(np.random.MT19937(seed=1337))
    table = snt.testing.make_example_table(prng=prng, size=1000)

    txt = table.info()

    assert txt

    for lk in table:
        assert lk in txt
        for ck in table[lk].dtype.names:
            assert ck in txt


def test_repr():
    prng = np.random.Generator(np.random.MT19937(seed=1337))
    table = snt.testing.make_example_table(prng=prng, size=1000)

    txt = repr(table)
    assert txt
    assert "SparseNumericTable" in txt

