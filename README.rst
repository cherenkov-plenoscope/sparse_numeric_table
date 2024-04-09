####################
Sparse-Numeric-Table
####################
|TestStatus| |PyPiStatus| |BlackStyle| |PackStyleBlack|


Query, write, and read sparse, numeric tables.


Restictions
===========
- Only numeric fields
- Index is unsigned integer

Pros
====
- Fast read / write with ``numpy`` binaries.
- Just a ``dict`` of ``numpy.recarray``'s. No classes. No stateful functions.
- Easy to explore files in the tapearchive ``.tar``.

Features
========
- Read from file / write to file.
- Create from 'records' (A list of dicts, each representing one row in the table)
- Query, cut, and merge on row-indices (columns can be omitted for speed)
- Concatenate files.


*****
Usage
*****


See ``./sparse_numeric_table/tests``.

1st) You create a ``dict`` representing the structure and ``dtype`` of your table.
Columns which only appear together are bundeled into a ``level`` . Each ``level`` has an index to merge and join with other ``level`` 's.


.. code-block:: python

    my_table_dtypes = {
        "A": [
            ("a", "<u8"),
            ("b", "<f8"),
            ("c", "<f4"),
        ],
        "B": [
            ("g", "<i8"),
        ],
        "C": [
            ("m", "<i2"),
            ("n", "<u8"),
        ],
    }


Here ``A`` , ``B`` , and ``C`` are the ``level`` -keys. ``a, ... , n`` are the column-keys.

2nd) You create/read/write the table.


.. code-block::

     A             B         C

     idx a b c     idx g     idx m n
     ___ _ _ _     ___ _
    |_0_|_|_|_|   |_0_|_|
    |_1_|_|_|_|
    |_2_|_|_|_|    ___ _
    |_3_|_|_|_|   |_3_|_|
    |_4_|_|_|_|   |_4_|_|    ___ _ _
    |_5_|_|_|_|   |_5_|_|   |_5_|_|_|
    |_6_|_|_|_|
    |_7_|_|_|_|
    |_8_|_|_|_|    ___ _
    |_9_|_|_|_|   |_9_|_|
    |10_|_|_|_|   |10_|_|
    |11_|_|_|_|    ___ _     ___ _ _
    |12_|_|_|_|   |12_|_|   |12_|_|_|
    |13_|_|_|_|    ___ _
    |14_|_|_|_|   |14_|_|


.. |TestStatus| image:: https://github.com/cherenkov-plenoscope/sparse_numeric_table/actions/workflows/test.yml/badge.svg?branch=main
    :target: https://github.com/cherenkov-plenoscope/sparse_numeric_table/actions/workflows/test.yml

.. |PyPiStatus| image:: https://img.shields.io/pypi/v/sparse_numeric_table_sebastian-achim-mueller
    :target: https://pypi.org/project/sparse_numeric_table_sebastian-achim-mueller

.. |BlackStyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. |PackStyleBlack| image:: https://img.shields.io/badge/pack%20style-black-000000.svg
    :target: https://github.com/cherenkov-plenoscope/black_pack
