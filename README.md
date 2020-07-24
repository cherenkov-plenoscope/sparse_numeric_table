[![Build Status](https://travis-ci.org/cherenkov-plenoscope/sparse_numeric_table.svg?branch=master)](https://travis-ci.org/cherenkov-plenoscope/sparse_numeric_table)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Sparse-Numeric-Table
====================

Query, write, and read sparse, numeric tables.

I often use ```pandas.DataFrame``` and ```numpy.recarray``` , but for sparse tables it does not work for me.
Here, I represent sparse numeric tables as a ```dict``` full of ```numpy.recarray```s.
Writing and reading is done with ```tarfile```s so that the sparse table's hirachy is represented in the tapearchives file-system.
The queries are done internally using the powerful ```pandas.merge```.

Restictions
-----------
- Only numerical fields
- Index must be unsigned integer
- Column-names must not have ```;``` character in it.

Pro
---
- Fast input/output with ```numpy``` binaries. 
- no custom ```class```, just a combination of ```dict``` and ```numpy.recarray```.
- Easy to explore hirachy and structure in output-files due to file-system in tapearchive.

Issues
------
- only supports queries common in my workflow
- Unneccessary strong restrictions on column-names

Usage
-----
See also ```./sparse_numeric_table/tests```.

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

