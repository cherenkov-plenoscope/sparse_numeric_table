def records_to_recarray(level_records, level_key, dtypes):
    out = DynamicSizeRecarray(dtype=dtypes[level_key])
    out.append_records(records=level_records)
    return out.to_recarray()


def table_of_records_to_sparse_numeric_table(table_records, dtypes):
    table = {}
    for level_key in table_records:
        table[level_key] = records_to_recarray(
            level_records=table_records[level_key],
            level_key=level_key,
            dtypes=dtypes,
        )
    return table


def get_column_as_dict_by_index(table, level_key, column_key):
    level = table[level_key]
    out = {}
    for ii in range(level.shape[0]):
        out[level[IDX][ii]] = level[column_key][ii]
    return out
