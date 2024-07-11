DTYPES = [
    "<u1",
    "<u2",
    "<u4",
    "<u8",
    "<i1",
    "<i2",
    "<i4",
    "<i8",
    "<f2",
    "<f4",
    "<f8",
]


def _assert_no_whitespace(key):
    for char in key:
        assert not str.isspace(
            char
        ), "Key must not contain spaces, but key = '{:s}'".format(key)


def _assert_no_dot(key):
    assert "." not in key, "Key must not contain '.', but key = '{:s}'".format(
        key
    )


def _assert_no_directory_delimeter(key):
    assert "/" not in key, "Key must not contain '/', but key = '{:s}'".format(
        key
    )
    assert (
        "\\" not in key
    ), "Key must not contain '\\', but key = '{:s}'".format(key)


def assert_key_is_valid(key):
    _assert_no_whitespace(key)
    _assert_no_dot(key)
    _assert_no_directory_delimeter(key)


def assert_dtypes_are_valid(dtypes):
    for level_key in dtypes:
        assert_key_is_valid(level_key)
        for column in dtypes[level_key]:
            column_key = column[0]
            column_dtype = column[1]
            assert_key_is_valid(column_key)
            assert column_dtype in DTYPES, (
                "level: {:s}, column: {:s} has dtype: {:s} "
                "which is not a valid for sparse_numeric_table.".format(
                    level_key,
                    column_key,
                    str(column_dtype),
                )
            )
