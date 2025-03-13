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


def is_column_key_in_dtypes(dtypes, column_key):
    has_index_key = {}
    for level_key in dtypes:
        has_index_key[level_key] = False
        for this_column_key, _ in dtypes[level_key]:
            if this_column_key == column_key:
                has_index_key[level_key] = True
    return has_index_key


def assert_all_levels_have_index_key(dtypes, index_key):
    has_index_key = is_column_key_in_dtypes(
        dtypes=dtypes, column_key=index_key
    )
    for level_key in has_index_key:
        if not has_index_key[level_key]:
            assert False, (
                f"Expected level '{level_key:s}' to "
                f"have 'index_key': '{self.index_key:s}'."
            )


def _assert_starts_not_with_dunderscore(key):
    assert not str.startswith(
        key, "__"
    ), f"Key must not start with double underscoe '__', but key = '{key:s}'."


def _assert_no_whitespace(key):
    for char in key:
        assert not str.isspace(
            char
        ), f"Key must not contain spaces, but key = '{key:s}'."


def _assert_no_dot(key):
    assert "." not in key, f"Key must not contain '.', but key = '{key:s}'."


def _assert_no_directory_delimeter(key):
    assert "/" not in key, f"Key must not contain '/', but key = '{key:s}'."
    assert "\\" not in key, f"Key must not contain '\\', but key = '{key:s}'"


def assert_key_is_valid(key):
    _assert_starts_not_with_dunderscore(key)
    _assert_no_whitespace(key)
    _assert_no_dot(key)
    _assert_no_directory_delimeter(key)


def assert_dtypes_are_valid(dtypes):
    for level_key in dtypes:
        assert_key_is_valid(level_key)
        for column_key, column_dtype in dtypes[level_key]:
            assert_key_is_valid(column_key)
            assert column_dtype in DTYPES, (
                f"Level: {level_key:s}, column: {column_key:s} "
                f"has dtype: {column_dtype:s} which is not a valid "
                "for sparse_numeric_table."
            )
