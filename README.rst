####################
Sparse Numeric Table
####################
|TestStatus| |PyPiStatus| |BlackStyle| |BlackPackStyle| |MITLicenseBadge|

Query, write, read, and dynamically grow a sparse numeric table.
I do love ``pandas.DataFrame`` and I do love ``numpy.recarray``.
But when the table is sparse and still won't fit into your memory one needs
to combine the best of ``pandas``, ``numpy`` and ``zipfile`` to get the job done.
This is the Sparse Numeric Table.

Install
=======
.. code-block:: bash

    pip install sparse-numeric-table-sebastian-achim-mueller

Test
====
.. code-block:: bash

    pytest .




Fileformat
==========
Efficient write and read using binary blocks (``numpy`` dumps) in a ``zip`` file.
On read, you only need to read the columns and indices you need. No need to read the
entire file. Files can be explored with any ``zip`` file reader.


*****
Usage
*****

See ``./sparse_numeric_table/tests`` for examples.

1st) You create a ``dict`` representing the dtypes of your table.
Columns which only appear together are bundeled into a ``level`` .
Each ``level`` has an index to merge and join with other ``levels``.

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


Here ``A`` , ``B`` , and ``C`` are the ``level`` keys.
``a, ... , n`` are the column keys.

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

.. |BlackPackStyle| image:: https://img.shields.io/badge/pack%20style-black-000000.svg
    :target: https://github.com/cherenkov-plenoscope/black_pack

.. |MITLicenseBadge| image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
