import sys
from functools import cache

import polars as pl
import yaml
from loguru import logger

from pseudopeople.constants import metadata, paths
from pseudopeople.constants.noise_type_metadata import INT_TO_STRING_COLUMNS

TOKEN = "token"
OPTIONS = "options"


def configure_logging_to_terminal(verbose: bool = False):
    logger.remove()  # Clear default configuration
    add_logging_sink(sys.stdout, verbose, colorize=True)


def add_logging_sink(sink, verbose, colorize=False, serialize=False):
    message_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
        "- <level>{message}</level>"
    )
    if verbose:
        logger.add(
            sink, colorize=colorize, level="DEBUG", format=message_format, serialize=serialize
        )
    else:
        logger.add(
            sink, colorize=colorize, level="INFO", format=message_format, serialize=serialize
        )


def get_state_abbreviation(state: str) -> str:
    """
    Get the two letter abbreviation of a state in the US.

    :param state: A string representation of the state.
    :return: A string of length 2
    """
    state = state.upper()
    if state in metadata.US_STATE_ABBRV_MAP.values():
        return state
    try:
        return metadata.US_STATE_ABBRV_MAP[state]
    except KeyError:
        raise ValueError(f"Unexpected state input: '{state}'") from None


######################
# String conversions #
######################


def to_string_expr(
    column_name: str, dtype: pl.DataType, integer_valued: bool | None = None
) -> pl.Expr:
    """Expression converting a column to strings, keeping nulls as nulls.

    Integer-valued columns (by default, those in INT_TO_STRING_COLUMNS) never
    get a trailing ``.0``.
    """
    if integer_valued is None:
        integer_valued = column_name in INT_TO_STRING_COLUMNS
    column = pl.col(column_name)
    if dtype == pl.String:
        if integer_valued:
            return column.str.split(".").list.first()
        return column
    if integer_valued and dtype.is_numeric():
        return column.cast(pl.Int64).cast(pl.String)
    return column.cast(pl.String)


def to_string(series: pl.Series) -> pl.Series:
    return series.to_frame().select(to_string_expr(series.name, series.dtype)).to_series()


def count_number_of_tokens_per_string(tokens: list[str], strings: pl.Series) -> pl.Series:
    """
    For each string, the total number of (possibly overlapping) occurrences of
    each of ``tokens`` in it.
    """
    strings = strings.cast(pl.String)
    unique_strings = strings.unique().drop_nulls()
    counts = {
        s: sum(_count_occurrences(s, str(token)) for token in tokens) for s in unique_strings
    }
    return strings.replace_strict(counts, default=0, return_dtype=pl.Int64)


# https://stackoverflow.com/a/2970542/
def _count_occurrences(string, sub):
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count


#################
# Lookup tables #
#################


def _token_options_table(pairs: pl.DataFrame) -> pl.DataFrame:
    """(token, option) pairs -> one row per token with the list of its options."""
    return (
        pairs.group_by(TOKEN, maintain_order=True)
        .agg(pl.col(OPTIONS))
        .with_columns(n_options=pl.col(OPTIONS).list.len().cast(pl.Int64))
    )


@cache
def load_ocr_errors() -> pl.DataFrame:
    pairs = pl.read_csv(
        paths.OCR_ERRORS_DATA,
        skip_rows=2,
        has_header=False,
        new_columns=[TOKEN, OPTIONS],
        schema_overrides={TOKEN: pl.String, OPTIONS: pl.String},
    )
    return _token_options_table(pairs)


@cache
def load_phonetic_errors() -> pl.DataFrame:
    pairs = pl.read_csv(
        paths.PHONETIC_ERRORS_DATA,
        skip_rows=2,
        has_header=False,
        new_columns=["where", TOKEN, OPTIONS, "pre", "post", "pattern", "start"],
        infer_schema=False,
    ).select(TOKEN, pl.col(OPTIONS).str.replace_all("@", "", literal=True))
    return _token_options_table(pairs)


@cache
def load_qwerty_errors_data() -> pl.DataFrame:
    with open(paths.QWERTY_ERRORS) as f:
        qwerty_errors = yaml.safe_load(f)
    return pl.DataFrame(
        {
            TOKEN: [str(k) for k in qwerty_errors],
            OPTIONS: [[str(o) for o in v] for v in qwerty_errors.values()],
        },
        schema={TOKEN: pl.String, OPTIONS: pl.List(pl.String)},
    ).with_columns(n_options=pl.col(OPTIONS).list.len().cast(pl.Int64))
