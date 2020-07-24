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
See also ```./sparse_numeric_tables/tests```.

1st) You create a ```dict``` representing the structure and ```dtype``` of your table.

2nd) You create/read/write the table.


```
        level 1          level 2       level 2
          columns          columns       columns
     idx a b c d e f  idx g h i j k l  idx m n o p
     ___ _ _ _ _ _ _  ___ _ _ _ _ _ _
    |_0_|_|_|_|_|_|_||_0_|_|_|_|_|_|_|
    |_1_|_|_|_|_|_|_|
    |_2_|_|_|_|_|_|_| ___ _ _ _ _ _ _
    |_3_|_|_|_|_|_|_||_3_|_|_|_|_|_|_|
    |_4_|_|_|_|_|_|_||_4_|_|_|_|_|_|_| ___ _ _ _ _
    |_5_|_|_|_|_|_|_||_5_|_|_|_|_|_|_||_5_|_|_|_|_|
    |_6_|_|_|_|_|_|_|
    |_7_|_|_|_|_|_|_|
    |_8_|_|_|_|_|_|_| ___ _ _ _ _ _ _
    |_9_|_|_|_|_|_|_||_9_|_|_|_|_|_|_|
    |10_|_|_|_|_|_|_||10_|_|_|_|_|_|_|
    |11_|_|_|_|_|_|_| ___ _ _ _ _ _ _  ___ _ _ _ _
    |12_|_|_|_|_|_|_||12_|_|_|_|_|_|_||12_|_|_|_|_|
    |13_|_|_|_|_|_|_| ___ _ _ _ _ _ _
    |14_|_|_|_|_|_|_||14_|_|_|_|_|_|_|
```

