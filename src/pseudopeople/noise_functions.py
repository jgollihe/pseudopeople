import polars as pl
from vivarium.config_tree import ConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.constants import data_values
from pseudopeople.constants.metadata import DatasetNames
from pseudopeople.constants.noise_type_metadata import (
    COPY_HOUSEHOLD_MEMBER_COLS,
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
    HOUSING_TYPE_GUARDIAN_DUPLICATION_RELATONSHIP_MAP,
)
from pseudopeople.data.fake_names import fake_first_names, fake_last_names
from pseudopeople.noise_scaling import get_options_for_column, load_nicknames_data
from pseudopeople.randomness import ROW_ID, Randomness
from pseudopeople.utilities import (
    OPTIONS,
    TOKEN,
    load_ocr_errors,
    load_phonetic_errors,
    load_qwerty_errors_data,
)

# Scratch column names used while exploding strings into characters
_ROW = "__row"
_POS = "__pos"
_CHAR = "__char"
_LEN = "__len"
_TOKEN_LENGTH = "__token_length"
_REPLACEMENT = "__replacement"


#############
# Row noise #
#############


def omit_rows(
    dataset_name: str,
    dataset_data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
) -> pl.DataFrame:
    """
    Function that omits rows from a dataset and returns only the remaining rows.
    :param dataset_name: Dataset object being noised
    :param dataset_data: DataFrame of one of the dataset types used in Pseudopeople
    :param configuration: ConfigTree object containing noise level values
    :param randomness: Randomness source to make random selection for noise
    :return: DataFrame with rows from the original dataframe removed
    """
    noise_level = configuration[Keys.ROW_PROBABILITY]
    omitted = randomness.uniform(f"{dataset_name}_omit_choice") < noise_level
    return dataset_data.filter(~omitted)


def _census_omission_noise_levels(
    base_probability: float = data_values.DO_NOT_RESPOND_BASE_PROBABILITY,
) -> pl.Expr:
    """
    Expression for do_not_respond probabilities based on demography of age,
    race/ethnicity, and sex.
    """
    race_effect = (
        pl.col("race_ethnicity")
        .cast(pl.String)
        .replace_strict(
            data_values.DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_RACE,
            default=0.0,
            return_dtype=pl.Float64,
        )
    )
    age = pl.col("age").cast(pl.Int64)
    sex = pl.col("sex").cast(pl.String)
    edges = data_values.DO_NOT_RESPOND_AGE_BIN_EDGES
    sex_age_effect = None
    for sex_value, effects in data_values.DO_NOT_RESPOND_ADDITIVE_PROBABILITY_BY_SEX_AGE.items():
        for lower, upper, effect in zip(edges[:-1], edges[1:], effects):
            in_bin = (sex == sex_value) & (age >= lower) & (age < upper)
            sex_age_effect = (
                pl.when(in_bin).then(effect)
                if sex_age_effect is None
                else sex_age_effect.when(in_bin).then(effect)
            )
    sex_age_effect = sex_age_effect.otherwise(0.0)
    return (pl.lit(base_probability) + race_effect + sex_age_effect).clip(0.0, 1.0)


def _get_census_omission_noise_levels(
    population: pl.DataFrame,
    base_probability: float = data_values.DO_NOT_RESPOND_BASE_PROBABILITY,
) -> pl.Series:
    return population.select(
        _census_omission_noise_levels(base_probability).alias("noise_level")
    ).to_series()


def apply_do_not_respond(
    dataset_name: str,
    dataset_data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
) -> pl.DataFrame:
    """
    Applies targeted omission based on demographic model for census and surveys.

    :param dataset_name: Dataset object name being noised
    :param dataset_data: DataFrame of one of the form types used in Pseudopeople
    :param configuration: ConfigTree object containing noise level values
    :param randomness: Randomness source to make random selection for noise
    :return: DataFrame with rows from the original dataframe removed
    """
    required_columns = ("age", "race_ethnicity", "sex")
    missing_columns = [col for col in required_columns if col not in dataset_data.columns]
    if len(missing_columns):
        raise ValueError(
            f"Dataset {dataset_name} is missing required columns: {missing_columns}"
        )

    # do_not_respond noise_levels are based on census
    noise_levels = _census_omission_noise_levels()

    # Apply an overall non-response rate of 27.6% for Current Population Survey (CPS)
    if dataset_name == DatasetNames.CPS:
        noise_levels = noise_levels + 0.276

    # Apply user-configured noise level
    configured_noise_level = configuration[Keys.ROW_PROBABILITY]
    default_noise_level = data_values.DEFAULT_DO_NOT_RESPOND_ROW_PROBABILITY[dataset_name]
    noise_levels = noise_levels * (configured_noise_level / default_noise_level)

    # Account for ACS and CPS oversampling
    if dataset_name in [DatasetNames.ACS, DatasetNames.CPS]:
        noise_levels = 0.5 + noise_levels / 2

    omitted = randomness.uniform(f"do_not_respond_{dataset_name}") < noise_levels
    return dataset_data.filter(~omitted)


def duplicate_with_guardian(
    dataset_name: str,
    dataset_data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
) -> pl.DataFrame:
    """
    Function that duplicates rows of a dataset. Rows that are duplicated fall into one of three groups of
    dependents that are typically duplicated in administrative data where a dependent lives in a different
    location than their guardian and they show up in the data at both locations. For simplicity, rows will
    be only duplicated once for a maximum of two rows per dependent. When a row is duplicated, one row will
    have the dependent's correct address and the other will have the guardian's address.
    :param dataset_name: Name of the dataset being noised
    :param dataset_data: DataFrame that will be noised
    :param configuration: ConfigTree object containing noise level values. Dict with three key groups for duplication
    :param randomness: Randomness source to make random selection for noise
    :return: DataFrame with rows from the original dataframe duplicated along with the original
    dataframe itself.
    """
    age = pl.col("age").cast(pl.Int64)
    has_guardian = pl.col("guardian_1").is_not_null()
    dependent_groups = {
        Keys.ROW_PROBABILITY_IN_HOUSEHOLDS_UNDER_18: (age < 18)
        & (pl.col("housing_type") == "Household")
        & has_guardian,
        Keys.ROW_PROBABILITY_IN_COLLEGE_GROUP_QUARTERS_UNDER_24: (age < 24)
        & (pl.col("housing_type") == "College")
        & has_guardian,
    }
    # The guardian's year always equals the dependent's (it is a join key),
    # so it is never copied.
    copied_columns = [c for c in GUARDIAN_DUPLICATION_ADDRESS_COLUMNS if c != "year"]
    guardian_addresses = dataset_data.select("simulant_id", *GUARDIAN_DUPLICATION_ADDRESS_COLUMNS)

    def _with_guardian_addresses(dependents: pl.DataFrame) -> pl.DataFrame:
        # Join twice to check if either guardian lives at a separate location from the dependent.
        for guardian in ["guardian_1", "guardian_2"]:
            dependents = dependents.join(
                guardian_addresses.select(
                    pl.col("simulant_id").alias(guardian),
                    "year",
                    *[pl.col(c).alias(f"{guardian}_{c}") for c in copied_columns],
                ),
                on=[guardian, "year"],
                how="left",
                maintain_order="left",
            )
        return dependents

    duplicates = []
    for group, is_in_group in dependent_groups.items():
        group_data = _with_guardian_addresses(dataset_data.filter(is_in_group))
        lives_apart = {
            guardian: pl.col(f"{guardian}_household_id").is_not_null()
            & pl.col("household_id").ne_missing(pl.col(f"{guardian}_household_id"))
            for guardian in ["guardian_1", "guardian_2"]
        }
        # Choose which guardian to copy when dependent lives in different address from both guardians
        choose_guardian_1 = (
            randomness.uniform(
                f"{dataset_name}_duplicate_with_guardian_{group}_guardian_choice"
            )
            < 0.5
        )
        copy_guardian_1 = (lives_apart["guardian_1"] & ~lives_apart["guardian_2"]) | (
            lives_apart["guardian_1"] & lives_apart["guardian_2"] & choose_guardian_1
        )
        is_duplicated = (lives_apart["guardian_1"] | lives_apart["guardian_2"]) & (
            randomness.uniform(f"{dataset_name}_duplicate_with_guardian_{group}")
            < configuration[group]
        )
        group_duplicates = group_data.filter(is_duplicated).with_columns(
            pl.when(copy_guardian_1)
            .then(pl.col(f"guardian_1_{c}"))
            .otherwise(pl.col(f"guardian_2_{c}"))
            .alias(c)
            for c in copied_columns
        )
        duplicates.append(group_duplicates.select(dataset_data.columns))

    duplicates = pl.concat(duplicates)
    if duplicates.is_empty():
        return dataset_data

    # Update relationship to reference person for duplicated simulants based on housing type
    first_new_row_id = dataset_data[ROW_ID].max() + 1
    duplicates = duplicates.with_columns(
        pl.col("housing_type")
        .cast(pl.String)
        .replace_strict(HOUSING_TYPE_GUARDIAN_DUPLICATION_RELATONSHIP_MAP, default=None)
        .cast(dataset_data.schema["relationship_to_reference_person"])
        .alias("relationship_to_reference_person"),
        pl.int_range(
            first_new_row_id,
            first_new_row_id + duplicates.height,
            dtype=dataset_data.schema[ROW_ID],
        ).alias(ROW_ID),
    )
    return pl.concat([dataset_data, duplicates])


################
# Column noise #
################


def _column_noise_result(data: pl.DataFrame, expr: pl.Expr, column_name: str) -> pl.Series:
    return data.select(expr.alias(column_name)).to_series()


def choose_wrong_options(
    data: pl.DataFrame,
    _: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """
    Function that takes a categorical series and applies noise so some values has been replace with other options from
    a list.

    :param data: DataFrame containing the rows selected for noise
    :param _: ConfigTree with rate at which to blank the data in column.
    :param randomness: Randomness source.
    :param column_name: String for column that will be noised, will be the key for randomness
    :returns: Series where data has been noised with other values from a list of possibilities
    """
    # todo: Update with exclusive resampling when vectorized_choice is improved
    options = get_options_for_column(column_name)
    return _column_noise_result(
        data,
        randomness.from_options(f"{column_name}_incorrect_select_choice", options),
        column_name,
    )


def copy_from_household_member(
    data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """
    :param data: DataFrame containing the rows selected for noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness: Randomness source.
    :param column_name: String for column that will be noised
    :returns: Series of the household member's values for the column
    """
    return data[COPY_HOUSEHOLD_MEMBER_COLS[column_name]].alias(column_name)


def swap_months_and_days(
    data: pl.DataFrame,
    _: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """
    Function that swaps month and day of dates.

    :param data: DataFrame containing the rows selected for noise
    :param _: ConfigTree object containing noise level values
    :param randomness: Randomness source
    :param column_name: String for column that will be noised
    :return: Noised Series where some dates have month and day swapped.
    """
    from pseudopeople.schema_entities import DATASETS, DATEFORMATS

    date_format = DATASETS.get_dataset(dataset_name).date_format

    column = pl.col(column_name)
    if date_format == DATEFORMATS.YYYYMMDD:  # YYYYMMDD
        year, month, day = column.str.slice(0, 4), column.str.slice(4, 2), column.str.slice(6)
        noised = pl.concat_str(year, day, month)
    elif date_format == DATEFORMATS.MM_DD_YYYY:  # MM/DD/YYYY
        year, month, day = column.str.slice(6), column.str.slice(0, 3), column.str.slice(3, 3)
        noised = pl.concat_str(day, month, year)
    elif date_format == DATEFORMATS.MMDDYYYY:  # MMDDYYYY
        year, month, day = column.str.slice(4), column.str.slice(0, 2), column.str.slice(2, 2)
        noised = pl.concat_str(day, month, year)
    else:
        raise ValueError(f"Invalid date format in {dataset_name}.")

    return _column_noise_result(data, noised, column_name)


def write_wrong_zipcode_digits(
    data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """
    Function that noises a 5 digit zipcode

    :param data: DataFrame containing the rows selected for noise
    :param configuration: Config tree object at column node.
    :param randomness: Randomness source
    :param column_name: String for column that will be noised
    :return: Series of noised zipcodes
    """
    column = pl.col(column_name)
    if data.select((column.str.len_chars() != 5).any()).item():
        # TODO: This is a BAD error message. It should never appear and if it
        #   does, the user shouldn't be checking the simulated population data.
        raise ValueError(
            "Zipcode data contains zipcodes that are not 5 digits long. Please check simulated population data."
        )

    # todo: Update when vectorized choice is improved
    # Scale up noise levels to adjust for inclusive sampling with all numbers
    scaleup_factor = 1 / (1 - (1 / 10))
    key = f"{column_name}_write_wrong_zipcode_digits"
    digits = []
    for i, probability in enumerate(configuration[Keys.ZIPCODE_DIGIT_PROBABILITIES]):
        replace = randomness.uniform(f"{key}_{i}") < scaleup_factor * probability
        random_digit = randomness.choice_index(f"{key}_{i}_digit", 10).cast(pl.String)
        digits.append(pl.when(replace).then(random_digit).otherwise(column.str.slice(i, 1)))
    return _column_noise_result(data, pl.concat_str(digits), column_name)


def misreport_ages(
    data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """Function to mis-write ages based on perturbation parameters included in
    the config file.

    :param data: DataFrame containing the rows selected for noise
    :param configuration: Config tree object at column node.
    :param randomness: Randomness source
    :param column_name: String for column that will be noised
    :return: Series with some values noised from the original
    """
    possible_perturbations = configuration[Keys.POSSIBLE_AGE_DIFFERENCES].to_dict()
    perturbation = randomness.weighted_choice(
        f"{column_name}_miswrite_ages",
        [int(p) for p in possible_perturbations],
        list(possible_perturbations.values()),
    )
    age = pl.col(column_name).cast(pl.Int64)
    # Reflect negative values to positive
    new_age = (age + perturbation).abs()
    # If new age == original age, subtract 1
    new_age = pl.when(new_age == age).then(new_age - 1).otherwise(new_age)
    return _column_noise_result(data, new_age, column_name)


def _explode_characters(data: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """One row per character: (row position, row id, character position, character).

    Row positions are taken from an existing ``_ROW`` column if there is one.
    """
    row = pl.col(_ROW) if _ROW in data.columns else pl.int_range(pl.len(), dtype=pl.UInt32)
    return (
        data.select(
            row.alias(_ROW),
            ROW_ID,
            pl.col(column_name).str.split("").alias(_CHAR),
        )
        .explode(_CHAR)
        .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).over(_ROW).alias(_POS))
    )


def _implode_characters(exploded: pl.DataFrame, column_name: str) -> pl.Series:
    return (
        exploded.group_by(_ROW, maintain_order=True)
        .agg(pl.col(_CHAR).str.join(""))
        .sort(_ROW)[_CHAR]
        .alias(column_name)
    )


def write_wrong_digits(
    data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """
    Function that noises numeric characters in a series.

    :param data: DataFrame containing the rows selected for noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness: Randomness source.
    :param column_name: String for column that will be noised

    returns: Series with some numeric values experiencing noise.
    """
    # This is a fix to not replacing the original token for noise options
    token_noise_level = configuration[Keys.TOKEN_PROBABILITY] / 0.9
    key = f"{column_name}_write_wrong_digits"
    char = pl.col(_CHAR)
    replace = char.str.contains(r"^[0-9]$") & (
        randomness.uniform(key, _POS) < token_noise_level
    )
    random_digit = randomness.choice_index(f"{key}_digit", 10, _POS).cast(pl.String)
    exploded = _explode_characters(data, column_name).with_columns(
        pl.when(replace).then(random_digit).otherwise(char).alias(_CHAR)
    )
    return _implode_characters(exploded, column_name)


def use_nicknames(
    data: pl.DataFrame,
    _: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """
    Function that replaces a name with a choice of potential nicknames.

    :param data: DataFrame containing the rows selected for noise
    :param _: ConfigTree with rate at which to blank the data in column.
    :param randomness: Randomness source.
    :param column_name: String for column that will be noised
    :return: Series of nicknames replacing original names
    """
    nickname = pl.col(OPTIONS).list.get(
        randomness.choice_index(f"{column_name}_use_nicknames", pl.col("n_options"))
    )
    return (
        data.select(ROW_ID, column_name)
        .join(
            load_nicknames_data(),
            left_on=column_name,
            right_on=TOKEN,
            how="left",
            maintain_order="left",
        )
        .select(
            pl.when(pl.col(OPTIONS).is_not_null())
            .then(nickname)
            .otherwise(pl.col(column_name))
            .alias(column_name)
        )
        .to_series()
    )


FAKE_NAMES = {
    "first_name": fake_first_names,
    "middle_name": fake_first_names,
    "last_name": fake_last_names,
    "spouse_first_name": fake_first_names,
    "spouse_last_name": fake_last_names,
    "dependent_1_first_name": fake_first_names,
    "dependent_1_last_name": fake_last_names,
    "dependent_2_first_name": fake_first_names,
    "dependent_2_last_name": fake_last_names,
    "dependent_3_first_name": fake_first_names,
    "dependent_3_last_name": fake_last_names,
    "dependent_4_first_name": fake_first_names,
    "dependent_4_last_name": fake_last_names,
}


def use_fake_names(
    data: pl.DataFrame,
    _: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """

    :param data: DataFrame containing the rows selected for noise
    :param _: ConfigTree with rate at which to blank the data in column.
    :param randomness: Randomness source.
    :param column_name: String for column that will be noised
    :return: Series of fake names
    """
    return _column_noise_result(
        data,
        randomness.from_options(f"{column_name}_fake_names", FAKE_NAMES[column_name]),
        column_name,
    )


def make_phonetic_errors(
    data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """

    :param data: DataFrame containing the rows selected for noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness: Randomness source.
    :param column_name: String for column that will be noised
    :return: Series of noised data
    """
    return _corrupt_tokens(
        load_phonetic_errors(),
        data,
        column_name,
        configuration[Keys.TOKEN_PROBABILITY],
        randomness,
        addl_key=f"{column_name}_make_phonetic_errors",
    )


def leave_blanks(
    data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """
    Function that takes a column and blanks out all values.

    :param data: DataFrame containing the rows selected for noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness: Randomness source.
    :param column_name: String for column that will be noised
    """
    return pl.repeat(
        None, data.height, dtype=data.schema[column_name], eager=True
    ).alias(column_name)


def make_typos(
    data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """Function that applies noise to the string values
    representative of keyboard mistyping.

    :param data: DataFrame containing the rows selected for noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness: Randomness source.
    :param column_name: String for column that will be noised
    :returns: Series of column with noised data
    """
    token_noise_level = configuration[Keys.TOKEN_PROBABILITY]
    # TODO: remove this hard-coding
    include_token_probability_level = 0.1
    key = f"{column_name}_make_typos"

    char = pl.col(_CHAR)
    replace = pl.col(OPTIONS).is_not_null() & (
        randomness.uniform(key, _POS) < token_noise_level
    )
    keep_original = replace & (
        randomness.uniform(f"{key}_keep_original", _POS) < include_token_probability_level
    )
    replacement = pl.col(OPTIONS).list.get(
        randomness.choice_index(f"{key}_replacement", pl.col("n_options"), _POS)
    )
    exploded = (
        _explode_characters(data, column_name)
        .join(
            load_qwerty_errors_data(),
            left_on=_CHAR,
            right_on=TOKEN,
            how="left",
            maintain_order="left",
        )
        .with_columns(
            pl.when(keep_original)
            .then(replacement + char)
            .when(replace)
            .then(replacement)
            .otherwise(char)
            .alias(_CHAR)
        )
    )
    return _implode_characters(exploded, column_name)


def make_ocr_errors(
    data: pl.DataFrame,
    configuration: ConfigTree,
    randomness: Randomness,
    dataset_name: str,
    column_name: str,
) -> pl.Series:
    """
    :param data: DataFrame containing the rows selected for noise
    :param configuration: ConfigTree with rate at which to blank the data in column.
    :param randomness: Randomness source.
    :param column_name: String for column that will be noised
    :return: Series of noised data
    """
    return _corrupt_tokens(
        load_ocr_errors(),
        data,
        column_name,
        configuration[Keys.TOKEN_PROBABILITY],
        randomness,
        addl_key=f"{column_name}_make_ocr_errors",
    )


def _corrupt_tokens(
    errors: pl.DataFrame,
    data: pl.DataFrame,
    column_name: str,
    token_probability: float,
    randomness: Randomness,
    addl_key: str,
) -> pl.Series:
    """
    Performs token-level corruption on a string column when the tokens to corrupt
    (and the tokens they get corrupted to) can be more than one character long.
    A token does not need to be the same length as a token it gets corrupted to.

    `errors` has one row per corruptible token (`token`), with the list of its
    possible corruptions (`options`) and how many there are (`n_options`).

    Strings are scanned left to right. At each position, corruptible tokens
    starting there are considered longest first, and each is corrupted with
    `token_probability`; the first one chosen is corrupted, and scanning
    resumes after the end of that token. This is computed without a scan:
    first every (position, token length) is decided independently, then the
    left-to-right "skip past corrupted tokens" rule is resolved exactly in a
    few vectorized passes.
    """
    rows = data.select(
        pl.int_range(pl.len(), dtype=pl.UInt32).alias(_ROW),
        ROW_ID,
        pl.col(column_name),
        pl.col(column_name).str.len_chars().cast(pl.Int64).alias(_LEN),
    )
    positions = (
        rows.filter(pl.col(_LEN) > 0)
        .with_columns(pl.int_ranges(0, pl.col(_LEN)).alias(_POS))
        .explode(_POS)
    )
    errors = errors.with_columns(pl.col(TOKEN).str.len_chars().cast(pl.Int64).alias(_TOKEN_LENGTH))

    candidates = []
    for token_length in sorted(errors[_TOKEN_LENGTH].unique()):
        candidates.append(
            positions.filter(pl.col(_POS) + token_length <= pl.col(_LEN))
            .with_columns(
                pl.col(column_name).str.slice(pl.col(_POS), token_length).alias(TOKEN),
                pl.lit(token_length, dtype=pl.Int64).alias(_TOKEN_LENGTH),
            )
            .join(errors, on=[TOKEN, _TOKEN_LENGTH], how="inner")
            .filter(
                randomness.uniform(addl_key, _POS, _TOKEN_LENGTH) < token_probability
            )
        )
    candidates = pl.concat(candidates)
    if candidates.is_empty():
        return data[column_name]

    # Longer tokens to-be-corrupted take precedence over shorter ones at the same position.
    # Only draw which corruption to use when there are multiple options.
    corruption_index = (
        pl.when(pl.col("n_options") > 1)
        .then(
            randomness.choice_index(
                f"{addl_key}_choice", pl.col("n_options"), _POS, _TOKEN_LENGTH
            )
        )
        .otherwise(0)
    )
    candidates = (
        candidates.filter(
            pl.col(_TOKEN_LENGTH) == pl.col(_TOKEN_LENGTH).max().over(_ROW, _POS)
        )
        .select(
            _ROW,
            _POS,
            (pl.col(_POS) + pl.col(_TOKEN_LENGTH)).alias("__end"),
            pl.col(OPTIONS).list.get(corruption_index).alias(_REPLACEMENT),
        )
        .sort(_ROW, _POS)
    )
    corrupted = _resolve_left_to_right(candidates)

    # Rebuild only the strings that have at least one corruption.
    corrupted_rows = rows.join(corrupted.select(_ROW).unique(), on=_ROW, how="semi")
    pieces = (
        _explode_characters(corrupted_rows, column_name)
        .join(corrupted, on=[_ROW, _POS], how="left", maintain_order="left")
        .with_columns(pl.col("__end").forward_fill().over(_ROW).alias("__covered_until"))
        .with_columns(
            pl.when(pl.col(_REPLACEMENT).is_not_null())
            .then(pl.col(_REPLACEMENT))
            .when(pl.col(_POS) < pl.col("__covered_until"))
            .then(pl.lit(""))
            .otherwise(pl.col(_CHAR))
            .alias(_CHAR)
        )
    )
    rebuilt = pieces.group_by(_ROW, maintain_order=True).agg(
        pl.col(_CHAR).str.join("").alias("__rebuilt")
    )
    return (
        rows.join(rebuilt, on=_ROW, how="left", maintain_order="left")
        .select(pl.coalesce("__rebuilt", column_name).alias(column_name))
        .to_series()
    )


def _resolve_left_to_right(candidates: pl.DataFrame) -> pl.DataFrame:
    """
    Given candidate corruptions (row, start position, end position), sorted by
    row and position with at most one per position, return those a left-to-right
    scan keeps: a candidate is kept unless it starts inside an earlier kept one.

    Each pass keeps every undecided candidate not overlapped by any earlier
    remaining candidate, then drops everything overlapped by a kept one. The
    earliest undecided candidate in each row is always decided, and overlap
    chains are short, so this converges in a few passes.
    """
    kept = []
    remaining = candidates
    previous_end = pl.col("__end").cum_max().shift(1).over(_ROW)
    while not remaining.is_empty():
        remaining = remaining.with_columns(
            (previous_end.is_null() | (pl.col(_POS) >= previous_end)).alias("__keep")
        )
        newly_kept = remaining.filter("__keep")
        kept.append(newly_kept.drop("__keep"))
        kept_end = (
            pl.when(pl.col("__keep"))
            .then(pl.col("__end"))
            .cum_max()
            .forward_fill()
            .shift(1)
            .over(_ROW)
        )
        remaining = (
            remaining.with_columns(kept_end.alias("__kept_end"))
            .filter(
                ~pl.col("__keep")
                & (pl.col("__kept_end").is_null() | (pl.col(_POS) >= pl.col("__kept_end")))
            )
            .drop("__keep", "__kept_end")
        )
    return pl.concat(kept).sort(_ROW, _POS)
