import math

import polars as pl
import pytest

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.constants.noise_type_metadata import (
    GUARDIAN_DUPLICATION_ADDRESS_COLUMNS,
)
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.noise_functions import _get_census_omission_noise_levels
from pseudopeople.randomness import ROW_ID, Randomness
from pseudopeople.schema_entities import DATASETS
from tests.conftest import FuzzyChecker

RANDOMNESS = Randomness("test_row_noise_0")


@pytest.fixture()
def dummy_data():
    num_simulants = 1_000_000
    numbers_list = list(range(10))
    word_list = ["word_1", "word_2", "word_3", "word_4", "word5"]
    return pl.DataFrame(
        {
            "numbers": numbers_list * (num_simulants // len(numbers_list)),
            "words": word_list * (num_simulants // len(word_list)),
        }
    ).with_row_index(ROW_ID)


def test_omit_row(dummy_data, fuzzy_checker: FuzzyChecker):
    config = get_configuration()[DATASETS.tax_w2_1099.name][Keys.ROW_NOISE][
        NOISE_TYPES.omit_row.name
    ]
    dataset_name_1 = "dummy_dataset_name"
    noised_data1 = NOISE_TYPES.omit_row(dataset_name_1, dummy_data, config, RANDOMNESS)

    expected_noise_1 = config[Keys.ROW_PROBABILITY]
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_omit_row",
        observed_numerator=len(noised_data1),
        observed_denominator=len(dummy_data),
        target_proportion=1 - expected_noise_1,
    )
    assert noised_data1.schema == dummy_data.schema


def test_do_not_respond(mocker, dummy_data, fuzzy_checker: FuzzyChecker):
    config = get_configuration()[DATASETS.census.name][Keys.ROW_NOISE][
        NOISE_TYPES.do_not_respond.name
    ]
    mocker.patch(
        "pseudopeople.noise_functions._census_omission_noise_levels",
        side_effect=(lambda *_: pl.lit(config[Keys.ROW_PROBABILITY])),
    )
    dataset_name_1 = DATASETS.census.name
    dataset_name_2 = DATASETS.acs.name
    my_dummy_data = dummy_data.with_columns(
        age=pl.lit(27), sex=pl.lit("Female"), race_ethnicity=pl.lit("Vulcan")
    )
    noised_data1 = NOISE_TYPES.do_not_respond(
        dataset_name_1, my_dummy_data, config, RANDOMNESS
    )
    noised_data2 = NOISE_TYPES.do_not_respond(
        dataset_name_2, my_dummy_data, config, RANDOMNESS
    )

    # Test that noising affects expected proportion with expected types
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_do_not_respond",
        observed_numerator=len(my_dummy_data) - len(noised_data1),
        observed_denominator=len(my_dummy_data),
        target_proportion=config[Keys.ROW_PROBABILITY],
        name_additional="noised_data1",
    )
    assert noised_data1.schema == my_dummy_data.schema

    # Check ACS data is scaled properly due to oversampling
    expected_noise = 0.5 + config[Keys.ROW_PROBABILITY] / 2
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_do_not_respond",
        observed_numerator=len(my_dummy_data) - len(noised_data2),
        observed_denominator=len(my_dummy_data),
        target_proportion=expected_noise,
        name_additional="noised_data2",
    )
    assert noised_data2.schema == my_dummy_data.schema
    assert len(noised_data1) != len(noised_data2)


@pytest.mark.parametrize(
    "age, race_ethnicity, sex, expected_level",
    [
        (3, "White", "Female", 0.0091),
        (35, "Black", "Male", 0.0611),
        (55, "Asian", "Female", 0),
    ],
)
def test__get_census_omission_noise_levels(age, race_ethnicity, sex, expected_level):
    """Test helper function for do_not_respond noising based on demography of age, race/ethnicity, and sex"""
    pop = pl.DataFrame(
        {"age": [age] * 10, "race_ethnicity": [race_ethnicity] * 10, "sex": [sex] * 10}
    )
    result = _get_census_omission_noise_levels(pop)
    assert all(math.isclose(r, expected_level, rel_tol=0.0001) for r in result)


def test_do_not_respond_missing_columns(dummy_data):
    """Test do_not_respond only applies to expected datasets."""
    config = get_configuration()[DATASETS.census.name][Keys.ROW_NOISE][
        NOISE_TYPES.do_not_respond.name
    ]
    with pytest.raises(ValueError, match="missing required columns"):
        _ = NOISE_TYPES.do_not_respond("silly_dataset", dummy_data, config, RANDOMNESS)


def test_guardian_duplication():
    # We are going to make a small dataframe and update the configuration to noise 100% of the
    # available rows. We will then check that the correct rows were copied with the correct
    # information.
    dummy_data = pl.DataFrame(
        {
            "simulant_id": [str(i) for i in list(range(10))],
            "household_id": [str(h) for h in [0, 0, 0, 1, 2, 3, 4, 4, 5, 6]],
            "housing_type": [
                "Household",
                "Household",
                "Household",
                "College",
                "Military",
                "Household",
                "Household",
                "Household",
                "Household",
                "Household",
            ],
            "age": [10, 10, 10, 10, 10, 17, 50, 50, 50, 50],
            "guardian_1": ["8", "7", "5", "9", "7", "6", None, None, None, None],
            "guardian_2": ["9", None, "8", None, None, None, None, None, None, None],
            "relationship_to_reference_person": ["Fake value"] * 10,
            "year": [2019] * 10,
            "street_name": [
                "Street 0",
                "Street 0",
                "Street 1",
                "Street 1",
                "Street 1",
                "Street 2",
                "Street 3",
                "Street 3",
                "Street 4",
                "Street 5",
            ],
            "street_number": [str(n) for n in [0, 0, 1, 1, 1, 2, 3, 3, 4, 5]],
            "unit_number": [str(n) for n in [0, 0, 1, 1, 1, 2, 3, 3, 4, 5]],
            "city": [
                "City 0",
                "City 0",
                "City 1",
                "City 1",
                "City 1",
                "City 2",
                "City 3",
                "City 3",
                "City 4",
                "City 5",
            ],
            "state": [
                "State 0",
                "State 0",
                "State 1",
                "State 1",
                "State 1",
                "State 2",
                "State 3",
                "State 3",
                "State 4",
                "State 5",
            ],
            "zipcode": [
                "00000",
                "00000",
                "11111",
                "11111",
                "11111",
                "22222",
                "33333",
                "33333",
                "44444",
                "55555",
            ],
        }
    ).with_row_index(ROW_ID)
    # Noise 100% of rows
    overrides = {
        key: 1.0
        for key in get_configuration()[DATASETS.census.name][Keys.ROW_NOISE][
            NOISE_TYPES.duplicate_with_guardian.name
        ]
    }
    dataset_name_1 = DATASETS.census.name
    noised = NOISE_TYPES.duplicate_with_guardian(
        dataset_name_1, dummy_data, overrides, RANDOMNESS
    )

    # We know the following since every dependent is duplicated:
    #  - Simulant ids 0, 1, 2, 3, and-5 will all be duplicated
    #  - Simulant ids 5-9 are guardians. The only overlap is simulant id 5,
    #    who is both a dependent and a guardian
    #  - Simulant id 0 and 3 have two guardians, 8 and 9.
    assert noised.schema == dummy_data.schema
    duplicated = noised.filter(pl.col("simulant_id").is_duplicated()).filter(
        pl.col(ROW_ID) >= len(dummy_data)
    )
    assert len(noised) == len(dummy_data) + len(duplicated)
    assert set(duplicated["simulant_id"]) == {"0", "1", "2", "3", "5"}
    # Only duplicate a dependent one time
    assert noised["simulant_id"].value_counts()["count"].max() == 2
    # Duplicates get new, unique row ids
    assert noised[ROW_ID].is_unique().all()

    # Check address information is copied in new rows
    guardians = dummy_data.filter(pl.col("simulant_id").is_in(dummy_data["guardian_1"].implode()))
    for dependent in duplicated.iter_rows(named=True):
        guardian_ids = [g for g in (dependent["guardian_1"], dependent["guardian_2"]) if g]
        candidate_guardians = guardians.filter(pl.col("simulant_id").is_in(guardian_ids))
        assert any(
            all(dependent[c] == guardian[c] for c in GUARDIAN_DUPLICATION_ADDRESS_COLUMNS)
            for guardian in candidate_guardians.iter_rows(named=True)
        )
