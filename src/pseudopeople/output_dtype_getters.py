import polars as pl


def output_dtype_getter_always_string(_: pl.DataType) -> pl.DataType:
    return pl.String()
