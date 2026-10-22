from functools import cache

import polars as pl

from pseudopeople.constants import paths
from pseudopeople.constants.noise_type_metadata import COPY_HOUSEHOLD_MEMBER_COLS
from pseudopeople.utilities import OPTIONS, TOKEN


def scale_choose_wrong_option(data: pl.DataFrame, column_name: str) -> float:
    """
    Function to scale noising for choose_wrong_option to adjust for the possibility
    of noising with the original values.
    """

    # Get possible noise values
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = get_options_for_column(column_name)

    # Scale to adjust for possibility of noising with original value
    noise_scaling_value = 1 / (1 - 1 / len(options))

    return noise_scaling_value


def scale_nicknames(data: pl.DataFrame, column_name: str) -> float:
    # Constant calculated by number of names with nicknames / number of names used in PRL name mapping
    nicknames = load_nicknames_data()
    have_nickname, not_null = data.select(
        pl.col(column_name).is_in(nicknames[TOKEN].implode()).sum().alias("have_nickname"),
        pl.col(column_name).is_not_null().sum().alias("not_null"),
    ).row(0)
    if have_nickname == 0:
        return 0.0
    return not_null / have_nickname


def scale_copy_from_household_member(data: pl.DataFrame, column_name: str) -> float:
    original = pl.col(column_name)
    copy = pl.col(COPY_HOUSEHOLD_MEMBER_COLS[column_name])
    original_not_missing = original.is_not_null() & (original != "")
    eligible = copy.is_not_null() & (copy != "") & original_not_missing
    n_eligible, n_original = data.select(
        eligible.sum().alias("eligible"), original_not_missing.sum().alias("original")
    ).row(0)
    if n_eligible == 0:
        return 0.0
    return n_original / n_eligible


####################
# Helper functions #
####################


@cache
def load_nicknames_data() -> pl.DataFrame:
    """One row per nickname-eligible name with the list of its nicknames (title case)."""
    nicknames = pl.read_csv(paths.NICKNAMES_DATA, infer_schema=False)
    nickname_columns = [c for c in nicknames.columns if c != "name"]
    return (
        nicknames.select(
            pl.col("name").str.to_titlecase().alias(TOKEN),
            pl.concat_list(pl.col(nickname_columns).str.to_titlecase())
            .list.drop_nulls()
            .alias(OPTIONS),
        )
        .with_columns(n_options=pl.col(OPTIONS).list.len().cast(pl.Int64))
        .filter(pl.col("n_options") > 0)
    )


def get_options_for_column(column_name: str) -> pl.Series:
    """
    For a column that has a set list of options, returns that set of options as
    a Series.
    Should only be passed a column that has options (i.e. should not be a free-form
    string column such as first name, or a numeric column), or it will error.
    """
    from pseudopeople.schema_entities import COLUMNS

    selection_type = {
        COLUMNS.employer_state.name: COLUMNS.state.name,
        COLUMNS.mailing_state.name: COLUMNS.state.name,
    }.get(column_name, column_name)

    return load_incorrect_select_options()[selection_type].drop_nulls()


@cache
def load_incorrect_select_options() -> pl.DataFrame:
    return pl.read_csv(paths.INCORRECT_SELECT_NOISE_OPTIONS_DATA, infer_schema=False)
