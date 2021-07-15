[![Build Status](https://travis-ci.org/cherenkov-plenoscope/sparse_numeric_table.svg?branch=master)](https://travis-ci.org/cherenkov-plenoscope/sparse_numeric_table)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Sparse-Numeric-Table
====================

Query, write, and read sparse, numeric tables.

I love ```pandas.DataFrame``` and ```numpy.recarray```, but with large and sparse tables I run out of memory or struggle to represent empty integer fields with the float's ```NaN```.

Here I use a ```dict``` of ```numpy.recarray```s to represent large and sparse tables.
Writing into ```tarfile```s (```.tar```) preserves the table's hirachy and makes it easy to explore in the file-system. I use ```pandas.merge``` to query.

Restictions
-----------
- Only numeric fields
- Index is unsigned integer

Pros
----
- Fast read / write with ```numpy``` binaries (explicit endianness).
- Just a ```dict``` of ```numpy.recarray```s. No classes. No stateful functions.
- Easy to explore files in the tapearchive ```.tar```.

Features
--------
- Read from file / write to file.
- Create from 'records' (A list of dicts, each representing one row in the table)
- Query, cut, and merge on row-indices (columns can be omitted for speed)
- Concatenate files.

Usage
-----
See ```./sparse_numeric_table/tests```.

1st) You create a ```dict``` representing the structure and ```dtype``` of your table.
Columns which only appear together are bundeled into a ```level```. Each ```level``` has an index to merge and join with other ```level```s.

```python
my_table_structure = {
    "A": {
        "a": {"dtype": "<u8"},
        "b": {"dtype": "<f8"},
        "c": {"dtype": "<f4"},
    },
    "B": {
        "g": {"dtype": "<i8"},
    },
    "C": {
        "m": {"dtype": "<i2"},
        "n": {"dtype": "<u8", "comment": "Some comment related to 'n'."},
    },
}
```
Here ```A```, ```B```, and ```C``` are the ```level```-keys. ```a, ... , n``` are the column-keys.
You can add comments for yourself, but ```sparse_numeric_table``` will ignore these.

2nd) You create/read/write the table.


```
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
```

