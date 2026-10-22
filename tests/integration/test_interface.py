import math
from functools import partial

import polars as pl
import pytest
from vivarium.config_tree import ConfigTree

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.dtypes import to_polars_dtype
from pseudopeople.interface import (
    generate_american_community_survey,
    generate_current_population_survey,
    generate_decennial_census,
    generate_social_security,
    generate_taxes_1040,
    generate_taxes_w2_and_1099,
    generate_women_infants_and_children,
)
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.schema_entities import COLUMNS, DATASETS, Column
from pseudopeople.utilities import (
    TOKEN,
    count_number_of_tokens_per_string,
    load_ocr_errors,
    load_phonetic_errors,
    load_qwerty_errors_data,
)
from tests.conftest import FuzzyChecker
from tests.integration.conftest import (
    CELL_PROBABILITY,
    IDX_COLS,
    SEED,
    STATE,
    _format_like_output,
    _get_common_datasets,
    _load_sample_data,
)

DATASET_GENERATION_FUNCS = {
    DATASETS.census.name: generate_decennial_census,
    DATASETS.acs.name: generate_american_community_survey,
    DATASETS.cps.name: generate_current_population_survey,
    DATASETS.ssa.name: generate_social_security,
    DATASETS.tax_w2_1099.name: generate_taxes_w2_and_1099,
    DATASETS.wic.name: generate_women_infants_and_children,
    DATASETS.tax_1040.name: generate_taxes_1040,
}
ALL_DATASET_NAMES = list(DATASET_GENERATION_FUNCS)
DATASETS_WITH_STATE = [n for n in ALL_DATASET_NAMES if n != DATASETS.ssa.name]

TOKENS_PER_STRING_MAPPER = {
    NOISE_TYPES.make_ocr_errors.name: partial(
        count_number_of_tokens_per_string, load_ocr_errors()[TOKEN].to_list()
    ),
    NOISE_TYPES.make_phonetic_errors.name: partial(
        count_number_of_tokens_per_string, load_phonetic_errors()[TOKEN].to_list()
    ),
    NOISE_TYPES.write_wrong_digits.name: lambda x: x.cast(pl.String).str.count_matches(r"\d"),
    NOISE_TYPES.make_typos.name: partial(
        count_number_of_tokens_per_string, load_qwerty_errors_data()[TOKEN].to_list()
    ),
}


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_generate_dataset_from_multiple_shards(
    dataset_name: str,
    config,
    request,
    split_sample_data_dir,
    mocker,
    fuzzy_checker: FuzzyChecker,
):
    """Tests that the amount of noising is approximately the same whether we
    noise a single sample dataset or we concatenate and noise multiple datasets
    """
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    generation_function = DATASET_GENERATION_FUNCS.get(dataset_name)
    data = _load_sample_data(dataset_name, request)
    noised_sample = request.getfixturevalue(f"noised_sample_data_{dataset_name}")

    noised_dataset = generation_function(
        seed=SEED,
        year=None,
        source=split_sample_data_dir,
        config=config,
    )

    # Check same order of magnitude of rows was removed -- we don't know the
    # full data size (we would need unnoised data for that), so we just check
    # for similar lengths
    assert 0.9 <= (len(noised_dataset) / len(noised_sample)) <= 1.1
    # Check that columns are identical
    assert noised_dataset.columns == noised_sample.columns

    # Check that each columns level of noising are similar
    check_noised, check_original = _get_common_datasets(dataset_name, data, noised_dataset)

    config = get_configuration(config)
    for col_name in check_noised.columns:
        col = COLUMNS.get_column(col_name)
        if col.noise_types:
            noise_level, to_compare = _get_column_noise_level(col, check_noised, check_original)
            _validate_column_noise_level(
                dataset_name=dataset_name,
                check_data=check_original.filter(to_compare),
                noise_level=noise_level,
                col=col,
                config=config,
                fuzzy_name="test_generate_dataset_from_sample_and_source_dataset",
                validator=fuzzy_checker,
            )


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_seed_behavior(dataset_name: str, config, request):
    """Tests seed behavior"""
    generation_function = DATASET_GENERATION_FUNCS.get(dataset_name)
    data = _format_like_output(_load_sample_data(dataset_name, request), dataset_name)
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")

    # Generate new (non-fixture) noised datasets with the same seed and a different
    # seed as the fixture
    noised_data_same_seed = generation_function(seed=SEED, year=None, config=config)
    noised_data_different_seed = generation_function(seed=SEED + 1, year=None, config=config)

    assert not data.equals(noised_data)
    assert noised_data.equals(noised_data_same_seed)
    assert not noised_data.equals(noised_data_different_seed)


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_column_dtypes(dataset_name: str, request):
    """Tests that column dtypes are as expected"""
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    for col_name in noised_data.columns:
        col = COLUMNS.get_column(col_name)
        assert noised_data.schema[col.name] == to_polars_dtype(col.dtype_name), col.name


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_column_noising(dataset_name: str, config, request, fuzzy_checker: FuzzyChecker):
    """Tests that columns are noised as expected"""
    data = _load_sample_data(dataset_name, request)
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    check_noised, check_original = _get_common_datasets(dataset_name, data, noised_data)

    config = get_configuration(config)
    for col_name in check_noised.columns:
        col = COLUMNS.get_column(col_name)
        noise_level, to_compare = _get_column_noise_level(col, check_noised, check_original)
        if col.noise_types:
            assert noise_level > 0
            _validate_column_noise_level(
                dataset_name=dataset_name,
                check_data=check_original.filter(to_compare),
                noise_level=noise_level,
                col=col,
                config=config,
                fuzzy_name="test_column_noising",
                validator=fuzzy_checker,
            )
        else:  # No noising - should be identical
            assert noise_level == 0


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_row_noising_omit_row_or_do_not_respond(dataset_name: str, config, request):
    """Tests that omit_row and do_not_respond row noising are being applied"""
    idx_cols = IDX_COLS.get(dataset_name)
    data = _format_like_output(_load_sample_data(dataset_name, request), dataset_name)
    noised_data = request.getfixturevalue(f"noised_sample_data_{dataset_name}")
    row_noise_config = get_configuration(config)[dataset_name][Keys.ROW_NOISE]
    noise_type = [
        n
        for n in row_noise_config
        if n in [NOISE_TYPES.omit_row.name, NOISE_TYPES.do_not_respond.name]
    ]
    if dataset_name in [DATASETS.census.name, DATASETS.acs.name, DATASETS.cps.name]:
        # Census and household surveys have do_not_respond and omit_row.
        # For all other datasets they are mutually exclusive
        assert len(noise_type) == 2
    else:
        assert len(noise_type) < 2
    added = noised_data.join(data, on=idx_cols, how="anti")
    removed = data.join(noised_data, on=idx_cols, how="anti")
    assert added.is_empty()
    if not noise_type:  # Check that there are no missing rows
        assert removed.is_empty()
    else:  # Check that there are some omissions
        # TODO: assert levels are as expected
        assert not removed.is_empty()


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_generate_dataset_with_year(dataset_name: str, request):
    year = 2030  # not default 2020
    generation_function = DATASET_GENERATION_FUNCS.get(dataset_name)
    data = _load_sample_data(dataset_name, request)
    # Generate a new (non-fixture) noised dataset for a single year
    noised_data = generation_function(year=year)
    assert not data.equals(noised_data)


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        DATASETS.tax_1040.name,
    ],
)
def test_dataset_filter_by_year(mocker, dataset_name: str):
    """Mock the noising function so that it returns the date column of interest
    with the original (unnoised) values to ensure filtering is happening
    """
    year = 2030  # not default 2020
    # Generate a new (non-fixture) noised dataset for a single year but mocked such
    # that no noise actually happens (otherwise the years would get noised and
    # we couldn't tell if the filter was working properly)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = DATASET_GENERATION_FUNCS[dataset_name](year=year)
    dataset = DATASETS.get_dataset(dataset_name)
    assert (noised_data[dataset.date_column_name] == year).all()


def _years(noised_data: pl.DataFrame, dataset) -> pl.Series:
    column = noised_data[dataset.date_column_name]
    if column.dtype.is_temporal():
        return column.dt.year()
    return column.str.to_date(dataset.date_format).dt.year()


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.acs.name,
        DATASETS.cps.name,
        DATASETS.ssa.name,
    ],
)
def test_dataset_filter_by_year_with_full_dates(mocker, dataset_name: str):
    """Mock the noising function so that it returns the date column of interest
    with the original (unnoised) values to ensure filtering is happening
    """
    year = 2030  # not default 2020
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = DATASET_GENERATION_FUNCS[dataset_name](year=year)
    dataset = DATASETS.get_dataset(dataset_name)

    years = _years(noised_data, dataset)
    if dataset == DATASETS.ssa:
        assert (years <= year).all()
    else:
        assert (years == year).all()


@pytest.mark.parametrize("dataset_name", DATASETS_WITH_STATE)
def test_generate_dataset_with_state_filtered(
    dataset_name: str, split_sample_data_dir_state_edit, mocker
):
    """Test that values returned by dataset generators are only for the specified state"""
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    dataset = DATASETS.get_dataset(dataset_name)
    # Skip noising (noising can incorrect select another state)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = DATASET_GENERATION_FUNCS[dataset_name](
        source=split_sample_data_dir_state_edit, state=STATE
    )
    assert (noised_data[dataset.state_column_name] == STATE).all()


@pytest.mark.parametrize("dataset_name", DATASETS_WITH_STATE)
def test_generate_dataset_with_state_unfiltered(
    dataset_name: str, split_sample_data_dir_state_edit, mocker
):
    # Important note: Currently the way this test is working is we have a fixture where we have
    # edited the sample data so half of it has a state to filter to. However, when we split the
    # sample data and do this, all the 2020 data (the year we default to for all generate_xxx functions)
    # results in the 2020 data being only in one of the files. In practice, this is how we want
    # the functionality of these functions to work but we should consider updating fixtures/tests
    # in the future. - albrja
    """Test that values returned by dataset generators are for all locations if state unspecified"""
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    dataset = DATASETS.get_dataset(dataset_name)
    # Skip noising (noising can incorrect select another state)
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = DATASET_GENERATION_FUNCS[dataset_name](
        source=split_sample_data_dir_state_edit
    )
    assert noised_data[dataset.state_column_name].n_unique() > 1


@pytest.mark.parametrize(
    "dataset_name",
    [
        DATASETS.census.name,
        DATASETS.tax_w2_1099.name,
        DATASETS.wic.name,
        DATASETS.tax_1040.name,
    ],
)
def test_dataset_filter_by_state_and_year(
    mocker, split_sample_data_dir_state_edit, dataset_name: str
):
    """Test that dataset generation works with state and year filters in conjunction"""
    year = 2030  # not default 2020
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = DATASET_GENERATION_FUNCS[dataset_name](
        source=split_sample_data_dir_state_edit,
        year=year,
        state=STATE,
    )
    dataset = DATASETS.get_dataset(dataset_name)
    assert (noised_data[dataset.date_column_name] == year).all()
    assert (noised_data[dataset.state_column_name] == STATE).all()


@pytest.mark.parametrize("dataset_name", [DATASETS.acs.name, DATASETS.cps.name])
def test_dataset_filter_by_state_and_year_with_full_dates(
    mocker, split_sample_data_dir_state_edit, dataset_name: str
):
    """Test that dataset generation works with state and year filters in conjunction"""
    year = 2030  # not default 2020
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    mocker.patch("pseudopeople.interface.noise_dataset", side_effect=_mock_noise_dataset)
    noised_data = DATASET_GENERATION_FUNCS[dataset_name](
        source=split_sample_data_dir_state_edit,
        year=year,
        state=STATE,
    )
    dataset = DATASETS.get_dataset(dataset_name)
    assert (_years(noised_data, dataset) == year).all()
    assert (noised_data[dataset.state_column_name] == STATE).all()


@pytest.mark.parametrize("dataset_name", DATASETS_WITH_STATE)
def test_generate_dataset_with_bad_state(
    dataset_name: str, split_sample_data_dir_state_edit, mocker
):
    """Test that bad state values result in informative ValueErrors"""
    bad_state = "Silly State That Doesn't Exist"
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    with pytest.raises(ValueError, match=bad_state.upper()):
        DATASET_GENERATION_FUNCS[dataset_name](
            source=split_sample_data_dir_state_edit,
            state=bad_state,
        )


@pytest.mark.parametrize("dataset_name", DATASETS_WITH_STATE)
def test_generate_dataset_with_bad_year(dataset_name: str, split_sample_data_dir, mocker):
    """Test that a ValueError is raised both for a bad year and a year that has no data"""
    bad_year = 0
    no_data_year = 2000
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    with pytest.raises(ValueError):
        generation_function(source=split_sample_data_dir, year=bad_year)
    with pytest.raises(ValueError):
        generation_function(source=split_sample_data_dir, year=no_data_year)


@pytest.mark.parametrize("dataset_name", ALL_DATASET_NAMES)
def test_output_path_writes_shards(
    dataset_name: str, config, split_sample_data_dir, tmp_path, mocker
):
    """Writing to output_path gives the same data as generating in memory, one file per shard"""
    mocker.patch("pseudopeople.interface.validate_source_compatibility")
    generation_function = DATASET_GENERATION_FUNCS[dataset_name]
    in_memory = generation_function(
        seed=SEED, year=None, config=config, source=split_sample_data_dir
    )
    output_path = tmp_path / "out"
    written = generation_function(
        seed=SEED,
        year=None,
        config=config,
        source=split_sample_data_dir,
        output_path=output_path,
    )
    assert isinstance(written, pl.LazyFrame)
    assert len(list(output_path.glob("*.parquet"))) == 2
    assert written.collect().equals(in_memory)


####################
# HELPER FUNCTIONS #
####################


def _mock_noise_dataset(
    dataset,
    dataset_data: pl.DataFrame,
    configuration,
    seed: int,
    *args,
    **kwargs,
):
    """Mock noise_dataset that just returns unnoised data"""
    return dataset_data


def _validate_column_noise_level(
    dataset_name: str,
    check_data: pl.DataFrame,
    noise_level: int,
    col: Column,
    config: ConfigTree,
    fuzzy_name: str,
    validator: FuzzyChecker,
):
    """
    This helper function iterates through all column noise types for a particular column
    and calculates the expected noise level for each. It then accumulates the expected
    noise level as we layer more noise types on top of each other.
    """
    tmp_config = config[dataset_name][Keys.COLUMN_NOISE][col.name]
    includes_token_noising = [
        c
        for c in tmp_config
        for k in [Keys.TOKEN_PROBABILITY, Keys.ZIPCODE_DIGIT_PROBABILITIES]
        if k in tmp_config[c].keys()
    ]
    values = check_data[col.name]

    # Calculate expected noise (target proportion for fuzzy checker)
    not_noised = 1
    for col_noise_type in col.noise_types:
        if col_noise_type.name not in includes_token_noising:
            not_noised = not_noised * (1 - CELL_PROBABILITY)
        else:
            token_probability_key = {
                NOISE_TYPES.write_wrong_zipcode_digits.name: Keys.ZIPCODE_DIGIT_PROBABILITIES,
            }.get(col_noise_type.name, Keys.TOKEN_PROBABILITY)
            token_probability = tmp_config[col_noise_type.name][token_probability_key]

            # Calculate probability no token is noised
            if col_noise_type.name == NOISE_TYPES.write_wrong_zipcode_digits.name:
                # Calculate write wrong zipcode average digits probability any token is noise
                avg_probability_any_token_noised = 1 - math.prod(
                    [1 - p for p in token_probability]
                )
            else:
                # Get number of tokens per string to calculate expected proportion
                tokens_per_string_getter = TOKENS_PER_STRING_MAPPER.get(
                    col_noise_type.name, lambda x: x.cast(pl.String).str.len_chars()
                )
                tokens_per_string = tokens_per_string_getter(values).cast(pl.Float64)
                avg_probability_any_token_noised = (
                    1 - (1 - token_probability) ** tokens_per_string
                ).mean()

            # This is accumulating not_noised over all noise types
            not_noised = not_noised * (1 - avg_probability_any_token_noised * CELL_PROBABILITY)

    validator.fuzzy_assert_proportion(
        name=fuzzy_name,
        observed_numerator=noise_level,
        observed_denominator=len(values),
        target_proportion=1 - not_noised,
        name_additional=f"{dataset_name}_{col.name}_{col_noise_type.name}",
    )


def _get_column_noise_level(
    column: Column,
    noised_data: pl.DataFrame,
    unnoised_data: pl.DataFrame,
) -> tuple[int, pl.Series]:
    """Number of originally non-missing values that changed, and which rows those were."""
    original = unnoised_data[column.name]
    noised = noised_data[column.name]

    # Check that originally missing data remained missing
    originally_missing = original.is_null()
    assert noised.filter(originally_missing).is_null().all()

    # Check for noising where applicable
    to_compare = ~originally_missing
    noise_level = original.filter(to_compare).ne_missing(noised.filter(to_compare)).sum()
    return noise_level, to_compare
