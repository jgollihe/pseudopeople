import polars as pl


class DtypeNames:
    """Container of expected dtype names"""

    CATEGORICAL = "category"
    DATETIME = "datetime"
    OBJECT = "string"
    INT = "int64"


POLARS_DTYPES: dict[str, pl.DataType] = {
    DtypeNames.CATEGORICAL: pl.Categorical(),
    DtypeNames.DATETIME: pl.Datetime("ns"),
    DtypeNames.OBJECT: pl.String(),
    DtypeNames.INT: pl.Int64(),
}


def to_polars_dtype(dtype_name: str) -> pl.DataType:
    return POLARS_DTYPES[dtype_name]
