import operator
from pathlib import Path
from typing import Any, List, Tuple

import polars as pl

from pseudopeople.exceptions import DataSourceError

# The index in our simulated population files is never meaningful, and pandas
# saves it as a column when writing parquet.
PANDAS_INDEX_COLUMN = "__index_level_0__"

_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def filters_to_expr(user_filters: List[Tuple[str, str, Any]]) -> pl.Expr:
    """Converts (column, operator, value) filters into a single Polars predicate."""
    predicates = [
        _OPERATORS[op](pl.col(column), pl.lit(value)) for column, op, value in user_filters
    ]
    return pl.all_horizontal(predicates) if predicates else pl.lit(True)


def scan_standard_dataset(
    data_path: Path,
    user_filters: List[Tuple],
) -> pl.LazyFrame:
    if data_path.suffix != ".parquet":
        raise DataSourceError(
            f"Source path must be a .parquet file. Provided {data_path.suffix}"
        )

    data = pl.scan_parquet(data_path)
    if PANDAS_INDEX_COLUMN in data.collect_schema().names():
        data = data.drop(PANDAS_INDEX_COLUMN)
    if user_filters:
        # Pushed down into the parquet scan, so only matching row groups are read
        data = data.filter(filters_to_expr(user_filters))
    return data


def load_standard_dataset(
    data_path: Path,
    user_filters: List[Tuple],
) -> pl.DataFrame:
    return scan_standard_dataset(data_path, user_filters).collect()
