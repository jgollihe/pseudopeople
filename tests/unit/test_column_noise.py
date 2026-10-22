import math

import polars as pl
import pytest

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.constants.noise_type_metadata import COPY_HOUSEHOLD_MEMBER_COLS
from pseudopeople.data.fake_names import fake_first_names, fake_last_names
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.noise_scaling import load_nicknames_data
from pseudopeople.randomness import ROW_ID, Randomness
from pseudopeople.schema_entities import DATASETS
from pseudopeople.utilities import (
    TOKEN,
    load_ocr_errors,
    load_phonetic_errors,
    load_qwerty_errors_data,
)
from tests.conftest import FuzzyChecker

RANDOMNESS0 = Randomness("test_column_noise_0")
RANDOMNESS1 = Randomness("test_column_noise_1")
CHARACTERS_LIST = [
    "A",
    "test123",
    "oF0cr",
    "erR0r5",
    "For456",
    "QUality",
    "contro1",
    "In789",
    "Pseud0peop12E",
]
PHONETIC_STRESS_TEST_LIST = [
    "~",  # No tokens -- note that the tilde character is never noised (from or to) by OCR or phonetic
    "oo",  # Single token
    "~~oo~~",  # Single token, with buffer
    "oooo",  # Only overlapping tokens
    "~~~oooo~~ooo~~~",  # Only overlapping tokens, with buffer
    "ach",  # Overlapping tokens whose corruptions overlap with themselves
    "~~~ach~ach~~~",  # Overlapping tokens whose corruptions overlap with themselves, with buffer
]
PHONETIC_STRESS_TEST_PATHWAYS = {
    # Tuples of (token noised, token not noised)
    "~": {"~": (0, 0)},
    "oo": {
        "oo": (0, 1),
        "u": (1, 0),
    },
    "~~oo~~": {
        "~~oo~~": (0, 1),
        "~~u~~": (1, 0),
    },
    "oooo": {
        "oooo": (0, 3),  # Nothing noised
        "uoo": (1, 1),  # Only first noised
        "ouo": (1, 1),  # Only second noised
        "oou": (1, 2),  # Only third noised
        "uu": (2, 0),  # First and third noised
    },
    "~~~oooo~~ooo~~~": {
        # In each of these additions, the first number represents the number
        # in the first set of o's and the second number represents the number
        # in the second set.
        "~~~oooo~~ooo~~~": (0 + 0, 3 + 2),  # Nothing noised; nothing noised
        "~~~uoo~~ooo~~~": (1 + 0, 1 + 2),  # Only first noised; nothing noised
        "~~~ouo~~ooo~~~": (1 + 0, 1 + 2),  # Only second noised; nothing noised
        "~~~oou~~ooo~~~": (1 + 0, 2 + 2),  # Only third noised; nothing noised
        "~~~uu~~ooo~~~": (2 + 0, 0 + 2),  # First and third noised; nothing noised
        "~~~oooo~~uo~~~": (0 + 1, 3 + 0),  # Nothing noised; only first noised
        "~~~uoo~~uo~~~": (1 + 1, 1 + 0),  # Only first noised; only first noised
        "~~~ouo~~uo~~~": (1 + 1, 1 + 0),  # Only second noised; only first noised
        "~~~oou~~uo~~~": (1 + 1, 2 + 0),  # Only third noised; only first noised
        "~~~uu~~uo~~~": (2 + 1, 0 + 0),  # First and third noised; only first noised
        "~~~oooo~~ou~~~": (0 + 1, 3 + 1),  # Nothing noised; only second noised
        "~~~uoo~~ou~~~": (1 + 1, 1 + 1),  # Only first noised; only second noised
        "~~~ouo~~ou~~~": (1 + 1, 1 + 1),  # Only second noised; only second noised
        "~~~oou~~ou~~~": (1 + 1, 2 + 1),  # Only third noised; only second noised
        "~~~uu~~ou~~~": (2 + 1, 0 + 1),  # First and third noised; only second noised
    },
    # Only enumerate pathways for non-buffer version
    "ach": {
        "ach": (0, 4),  # ach, ch, c, h
        "k": (1, 0),
        "ax": (1, 1),
        "akh": (1, 3),
        "ac": (1, 3),
        "ak": (2, 2),
    },
}
OCR_STRESS_TEST_LIST = [
    "~",  # No tokens
    "YDUu",  # Single-character tokens
    "~~~YD~Uu~~~",  # Single-character tokens, with buffer
    "LII-IIJ",  # Non-overlapping multi-character tokens
    "~~~LI~~I-I~IJ~~~",  # Non-overlapping multi-character tokens, with buffer
    "LI-I",  # Overlapping tokens
    "~~~LI-I~LI-I~~~",  # Overlapping tokens, with buffer
]
OCR_STRESS_TEST_PATHWAYS = {
    # Tuples of (token noised, token not noised)
    "~": {"~": (0, 0)},
    "YDUu": {
        "YDUu": (0, 4),
        "VDUu": (1, 3),
        "YOUu": (1, 3),
        "YDVu": (1, 3),
        "YDUv": (1, 3),
        "VOUu": (2, 2),
        "VDVu": (2, 2),
        "VDUv": (2, 2),
        "YOVu": (2, 2),
        "YOUv": (2, 2),
        "YDVv": (2, 2),
        "YOVv": (3, 1),
        "VDVv": (3, 1),
        "VOUv": (3, 1),
        "VOVu": (3, 1),
        "VOVv": (4, 0),
    },
    "~~~YD~Uu~~~": {
        "~~~YD~Uu~~~": (0, 4),
        "~~~VD~Uu~~~": (1, 3),
        "~~~YO~Uu~~~": (1, 3),
        "~~~YD~Vu~~~": (1, 3),
        "~~~YD~Uv~~~": (1, 3),
        "~~~VO~Uu~~~": (2, 2),
        "~~~VD~Vu~~~": (2, 2),
        "~~~VD~Uv~~~": (2, 2),
        "~~~YO~Vu~~~": (2, 2),
        "~~~YO~Uv~~~": (2, 2),
        "~~~YD~Vv~~~": (2, 2),
        "~~~YO~Vv~~~": (3, 1),
        "~~~VD~Vv~~~": (3, 1),
        "~~~VO~Uv~~~": (3, 1),
        "~~~VO~Vu~~~": (3, 1),
        "~~~VO~Vv~~~": (4, 0),
    },
    "LII-IIJ": {
        "LII-IIJ": (0, 3),  # LI, I-I, IJ
        "UI-IIJ": (1, 2),
        "LIHIJ": (1, 2),
        "LII-IU": (1, 2),
        "UHIJ": (2, 1),
        "UI-IU": (2, 1),
        "LIHU": (2, 1),
        "UHU": (3, 0),
    },
    "LI-I": {
        "LI-I": (0, 2),  # LI, I-I
        "U-I": (1, 0),
        "LH": (1, 1),
    },
}
INTEGERS_LIST = [12345, 67890, 54321, 918273, 987654]
FIRST_NAMES = ["Abigail", "Catherine", "Bill", "Fake name"]
LAST_NAMES = ["Johnson", "Smith", "Gates", "Lastname"]
STRING_LIST = [
    "fo1",
    "fo2",
    "fo3",
    "Unit 1A",
    "1234",
    "12/31/2020",
    "a1b2c3",
    "100000.00",
    "123-45-6789",
]
NUM_SIMULANTS = 1_000_000


def _cycle(values: list, n: int = NUM_SIMULANTS) -> list:
    return (values * (n // len(values) + 1))[:n]


@pytest.fixture(scope="module")
def dummy_dataset() -> pl.DataFrame:
    """Data in the form column noise sees it: strings, with nulls for missing values."""
    maximum_age = 120
    max_integer = max(INTEGERS_LIST)
    numbers = pl.Series(_cycle([str(x) for x in INTEGERS_LIST]))
    is_missing = pl.int_range(NUM_SIMULANTS, eager=True) % 3 == 0
    return (
        pl.DataFrame(
            {
                "numbers": numbers,
                "characters": _cycle(CHARACTERS_LIST + [None]),
                "state": _cycle(["CA", "WA", "FL", "OR", "CO", "TX", "NY", "VA", "AZ", "''"]),
                "age": (numbers.cast(pl.Int64) / max_integer * (maximum_age + 1))
                .round()
                .cast(pl.Int64)
                .cast(pl.String),
                "street_number": _cycle(STRING_LIST + [None]),
                "zipcode": _cycle(["12345", "98765", "02468", "13579", None]),
                "first_name": _cycle(FIRST_NAMES + [None]),
                "last_name": _cycle(LAST_NAMES + [None]),
                "event_date": _cycle(
                    ["01/25/1990", "05/30/1995", "10/01/2000", "12/31/2010", None]
                ),
                "date_of_birth": _cycle(
                    ["01/31/1950", "05/01/1990", "10/01/2000", "12/31/2010", None]
                ),
                "copy_age": _cycle(
                    ["10", "20", "30", "40", "50", "60", "70", "80", None, "100"]
                ),
                "phonetic_stress_test": _cycle(PHONETIC_STRESS_TEST_LIST + [None]),
                "ocr_stress_test": _cycle(OCR_STRESS_TEST_LIST + [None]),
            }
        )
        .with_columns(pl.when(~is_missing).then(pl.col("numbers", "age")).name.keep())
        .with_row_index(ROW_ID)
    )


def _noise(noise_type, data, config, randomness, dataset_name, column) -> pl.Series:
    """Apply a column noise type and return the noised column."""
    return noise_type(data, config, randomness, dataset_name, column)[column]


def _single_column_data(column: str, values: list) -> pl.DataFrame:
    return pl.DataFrame({column: values}).with_row_index(ROW_ID)


def test_leave_blank(dummy_dataset, fuzzy_checker: FuzzyChecker):
    config = get_configuration(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "zipcode": {
                        NOISE_TYPES.leave_blank.name: {
                            Keys.CELL_PROBABILITY: 0.25,
                        },
                    },
                },
            },
        }
    )[DATASETS.census.name][Keys.COLUMN_NOISE]["zipcode"][NOISE_TYPES.leave_blank.name]

    data = dummy_dataset.select(ROW_ID, "numbers")
    noised_data = _noise(
        NOISE_TYPES.leave_blank, data, config, RANDOMNESS0, "dataset", "numbers"
    )
    original = data["numbers"]

    # Calculate newly missing data, ie data that didn't come in as already missing
    orig_non_missing = original.is_not_null()
    newly_missing = orig_non_missing & noised_data.is_null()

    # Check for expected noise level
    expected_noise = config[Keys.CELL_PROBABILITY]
    fuzzy_checker.fuzzy_assert_proportion(
        name="leave_blank",
        observed_numerator=newly_missing.sum(),
        observed_denominator=orig_non_missing.sum(),
        target_proportion=expected_noise,
    )

    # Check that un-noised values are unchanged
    not_noised = noised_data.is_not_null()
    assert (original.filter(not_noised) == noised_data.filter(not_noised)).all()


def test_choose_wrong_option(dummy_dataset, fuzzy_checker: FuzzyChecker):
    config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE]["state"][
        NOISE_TYPES.choose_wrong_option.name
    ]
    data = dummy_dataset.select(ROW_ID, "state")
    noised_data = _noise(
        NOISE_TYPES.choose_wrong_option, data, config, RANDOMNESS0, "dataset", "state"
    )
    original = data["state"]
    # Check for expected noise level
    expected_noise = config[Keys.CELL_PROBABILITY]
    # todo: Update when choose_wrong_options uses exclusive resampling
    actual_noise = noised_data.ne_missing(original).sum()
    fuzzy_checker.fuzzy_assert_proportion(
        name="choose_wrong_option",
        observed_numerator=actual_noise,
        observed_denominator=original.is_not_null().sum(),
        target_proportion=expected_noise,
    )
    assert (original.is_null() == noised_data.is_null()).all()


def test_generate_copy_from_household_member(dummy_dataset, fuzzy_checker: FuzzyChecker):
    config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE]["age"][
        NOISE_TYPES.copy_from_household_member.name
    ]
    data = dummy_dataset.select(ROW_ID, "age", "copy_age")
    noised_data = _noise(
        NOISE_TYPES.copy_from_household_member, data, config, RANDOMNESS0, "dataset", "age"
    )

    # Check for expected noise level
    expected_noise = config[Keys.CELL_PROBABILITY]
    original = data["age"]
    eligible = original.is_not_null()
    was_noised = noised_data.ne_missing(original)
    fuzzy_checker.fuzzy_assert_proportion(
        name="generate_copy_from_household_member",
        observed_numerator=(was_noised & eligible).sum(),
        observed_denominator=eligible.sum(),
        target_proportion=expected_noise,
    )

    # Noised values should be the same as the copy column
    copy_column = data[COPY_HOUSEHOLD_MEMBER_COLS["age"]]
    assert (copy_column.filter(was_noised) == noised_data.filter(was_noised)).all()
    assert original.filter(~was_noised).equals(noised_data.filter(~was_noised))


def test_swap_months_and_days(dummy_dataset, fuzzy_checker: FuzzyChecker):
    for col in ["event_date", "date_of_birth"]:
        data = dummy_dataset.select(ROW_ID, col)
        if col == "event_date":
            config = get_configuration()[DATASETS.ssa.name][Keys.COLUMN_NOISE][col][
                NOISE_TYPES.swap_month_and_day.name
            ]
            config.update(
                {
                    DATASETS.ssa.name: {
                        Keys.COLUMN_NOISE: {
                            col: {
                                NOISE_TYPES.swap_month_and_day.name: {
                                    Keys.CELL_PROBABILITY: 0.25,
                                },
                            },
                        },
                    },
                }
            )
        else:
            config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE][col][
                NOISE_TYPES.swap_month_and_day.name
            ]
        expected_noise = config[Keys.CELL_PROBABILITY]
        noised_data = _noise(
            NOISE_TYPES.swap_month_and_day,
            data,
            config,
            RANDOMNESS0,
            DATASETS.census.name,
            col,
        )

        # Confirm missing data remains missing
        original = data[col]
        orig_missing = original.is_null()
        assert noised_data.filter(orig_missing).is_null().all()

        original, noised_data = (
            original.filter(~orig_missing),
            noised_data.filter(~orig_missing),
        )
        assert (original.str.slice(6) == noised_data.str.slice(6)).all()
        fuzzy_checker.fuzzy_assert_proportion(
            name="swap_months_and_days",
            observed_numerator=(original != noised_data).sum(),
            observed_denominator=len(original),
            target_proportion=expected_noise,
        )


def test_write_wrong_zipcode_digits(dummy_dataset, fuzzy_checker: FuzzyChecker):
    dummy_digit_probabilities = [0.3, 0.3, 0.4, 0.5, 0.5]
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "zipcode": {
                        NOISE_TYPES.write_wrong_zipcode_digits.name: {
                            Keys.CELL_PROBABILITY: 0.5,
                            Keys.ZIPCODE_DIGIT_PROBABILITIES: dummy_digit_probabilities,
                        },
                    },
                },
            },
        }
    )
    config = config[DATASETS.census.name][Keys.COLUMN_NOISE]["zipcode"][
        NOISE_TYPES.write_wrong_zipcode_digits.name
    ]

    # Get configuration values for each piece of 5 digit zipcode
    cell_probability = config[Keys.CELL_PROBABILITY]
    token_probability = config[Keys.ZIPCODE_DIGIT_PROBABILITIES]
    data = dummy_dataset.select(ROW_ID, "zipcode")
    noised_data = _noise(
        NOISE_TYPES.write_wrong_zipcode_digits,
        data,
        config,
        RANDOMNESS0,
        "dataset",
        "zipcode",
    )

    # Confirm missing data remains missing
    original = data["zipcode"]
    orig_missing = original.is_null()
    assert noised_data.filter(orig_missing).is_null().all()
    original, noised_data = original.filter(~orig_missing), noised_data.filter(~orig_missing)
    # Check noise for each digits position matches expected noise
    for i in range(5):
        digit_prob = config["digit_probabilities"][i]
        actual_noise = (original.str.slice(i, 1) != noised_data.str.slice(i, 1)).sum()
        fuzzy_checker.fuzzy_assert_proportion(
            name="write_wrong_zipcode_digits",
            observed_numerator=actual_noise,
            observed_denominator=len(original),
            target_proportion=digit_prob * cell_probability,
        )

    # Validate whole column at once
    avg_probability_any_token_noised = 1 - math.prod([1 - p for p in token_probability])
    probability_not_noised = 1 - avg_probability_any_token_noised * cell_probability
    fuzzy_checker.fuzzy_assert_proportion(
        name="write_wrong_zipcode_digits_series",
        observed_numerator=(original != noised_data).sum(),
        observed_denominator=len(original),
        target_proportion=1 - probability_not_noised,
    )


def test_miswrite_ages_default_config(dummy_dataset, fuzzy_checker: FuzzyChecker):
    """Test that miswritten ages are appropriately handled, including
    no perturbation probabilities defaults to uniform distribution,
    perturbation probabilities"""
    config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE]["age"][
        NOISE_TYPES.misreport_age.name
    ]
    data = dummy_dataset.select(ROW_ID, "age")
    noised_data = _noise(
        NOISE_TYPES.misreport_age, data, config, RANDOMNESS0, "dataset", "age"
    )
    original = data["age"]

    # Check for expected noise level
    not_missing = original.is_not_null()
    expected_noise = config[Keys.CELL_PROBABILITY]
    # NOTE: the expected noise calculated above does not account for the fact that
    # if a perturbed age ends up being the same as the original age, then 1 is subtracted.
    fuzzy_checker.fuzzy_assert_proportion(
        name="misreport_age",
        observed_numerator=(
            noised_data.filter(not_missing) != original.filter(not_missing)
        ).sum(),
        observed_denominator=not_missing.sum(),
        target_proportion=expected_noise,
    )

    # Check that missing data remains missing
    assert (original.is_null() == noised_data.is_null()).all()

    # Check that there are no negative ages generated
    assert noised_data.drop_nulls().cast(pl.Int64).min() >= 0


def _misreport_age_config(perturbations) -> dict:
    return get_configuration(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.misreport_age.name: {
                            Keys.CELL_PROBABILITY: 1,
                            Keys.POSSIBLE_AGE_DIFFERENCES: perturbations,
                        },
                    },
                },
            },
        },
    )[DATASETS.census.name][Keys.COLUMN_NOISE]["age"][NOISE_TYPES.misreport_age.name]


def test_miswrite_ages_uniform_probabilities(fuzzy_checker: FuzzyChecker):
    """Test that a list of perturbations passed in results in uniform probabilities"""
    num_rows = 100_000
    original_age = 25
    perturbations = [-2, -1, 1]
    config = _misreport_age_config(perturbations)

    data = _single_column_data("age", [str(original_age)] * num_rows)
    noised_data = _noise(
        NOISE_TYPES.misreport_age, data, config, RANDOMNESS0, "dataset", "age"
    )
    expected_noise = 1 / len(perturbations)
    for perturbation in perturbations:
        actual_noise = (noised_data.cast(pl.Int64) - original_age == perturbation).sum()
        fuzzy_checker.fuzzy_assert_proportion(
            name="misreport_age_uniform_probabilities",
            observed_numerator=actual_noise,
            observed_denominator=num_rows,
            target_proportion=expected_noise,
        )


def test_miswrite_ages_provided_probabilities(fuzzy_checker: FuzzyChecker):
    """Test that provided age perturation probabilites are handled"""
    num_rows = 100_000
    original_age = 25
    perturbations = {-1: 0.1, 1: 0.9}
    config = _misreport_age_config(perturbations)

    data = _single_column_data("age", [str(original_age)] * num_rows)
    noised_data = _noise(
        NOISE_TYPES.misreport_age, data, config, RANDOMNESS0, "dataset", "age"
    )
    for perturbation, expected_noise in perturbations.items():
        actual_noise = (noised_data.cast(pl.Int64) - original_age == perturbation).sum()
        fuzzy_checker.fuzzy_assert_proportion(
            name="misreport_age_provided_probabilities",
            observed_numerator=actual_noise,
            observed_denominator=num_rows,
            target_proportion=expected_noise,
        )


def test_miswrite_ages_handles_perturbation_to_same_age():
    """Tests an edge case. It's possible that after an age is perturbed it ends
    up being the original age. In that case, subtract 1. eg, an age of 1 that is
    perturbed -2 becomes -1. But we cannot have negative so we flip the sign to +1.
    But that's the same as the original age and so should become 1-1=0.
    """
    config = _misreport_age_config([-2])  # This will cause -1 which will be flipped to +1
    data = _single_column_data("age", ["1"] * 100)
    noised_data = _noise(
        NOISE_TYPES.misreport_age, data, config, RANDOMNESS0, "dataset", "age"
    )
    assert (noised_data == "0").all()


def test_miswrite_ages_flips_negative_to_positive():
    """Test that any ages perturbed to <0 are reflected to positive values"""
    config = _misreport_age_config([-7])  # This will cause -4 and should flip to +4
    data = _single_column_data("age", ["3"] * 100)
    noised_data = _noise(
        NOISE_TYPES.misreport_age, data, config, RANDOMNESS0, "dataset", "age"
    )
    assert (noised_data == "4").all()


def _write_wrong_digits_config() -> dict:
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "street_number": {
                        NOISE_TYPES.write_wrong_digits.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                            Keys.TOKEN_PROBABILITY: 0.5,
                        },
                    },
                },
            },
        }
    )
    return config[DATASETS.census.name][Keys.COLUMN_NOISE]["street_number"][
        NOISE_TYPES.write_wrong_digits.name
    ]


def test_write_wrong_digits_robust(dummy_dataset, fuzzy_checker: FuzzyChecker):
    """
    Validates that only numeric characters are noised in a series at a provided noise level.
    """
    config = _write_wrong_digits_config()
    p_row_noise = config[Keys.CELL_PROBABILITY]
    p_token_noise = config[Keys.TOKEN_PROBABILITY]
    data = dummy_dataset.select(ROW_ID, "street_number")
    # Note: I changed this column from string_series to street number. It has several string formats
    # containing both numeric and alphabetically string characters.
    noised_data = _noise(
        NOISE_TYPES.write_wrong_digits, data, config, RANDOMNESS0, "dataset", "street_number"
    )
    original = data["street_number"]
    expected_noise = p_row_noise * p_token_noise

    # Check missing values haven't changed
    assert (original.is_null() == noised_data.is_null()).all()
    # Assert string length doesn't change after noising
    assert original.str.len_chars().equals(noised_data.str.len_chars())

    # For each example string, the character positions that are digits (and so can be noised)
    examples = {
        "fo1": [2],
        "fo2": [2],
        "fo3": [2],
        "1234": [0, 1, 2, 3],
        "a1b2c3": [1, 3, 5],
        "Unit 1A": [5],
        "100000.00": [0, 1, 2, 3, 4, 5, 7, 8],
        "12/31/2020": [0, 1, 3, 4, 6, 7, 8, 9],
        "123-45-6789": [0, 1, 2, 4, 5, 7, 8, 9, 10],
    }
    for example, digit_positions in examples.items():
        is_example = original == example
        example_original = original.filter(is_example)
        example_noised = noised_data.filter(is_example)
        for i in range(len(example)):
            original_char = example_original.str.slice(i, 1)
            noised_char = example_noised.str.slice(i, 1)
            if i in digit_positions:
                fuzzy_checker.fuzzy_assert_proportion(
                    name=f"write_wrong_digits_robust_{example}",
                    observed_numerator=(original_char != noised_char).sum(),
                    observed_denominator=len(example_original),
                    target_proportion=expected_noise,
                )
                assert noised_char.str.contains(r"^[0-9]$").all()
            else:
                assert (original_char == noised_char).all()


def test_write_wrong_digits(dummy_dataset, fuzzy_checker: FuzzyChecker):
    # This is a quicker (less robust) version of the test above.
    # It only checks that numeric characters are noised at the correct level as
    # a sanity check our noise is of the right magnitude
    config = _write_wrong_digits_config()
    expected_cell_noise = config[Keys.CELL_PROBABILITY]
    expected_token_noise = config[Keys.TOKEN_PROBABILITY]
    data = dummy_dataset.select(ROW_ID, "street_number")
    noised_data = _noise(
        NOISE_TYPES.write_wrong_digits, data, config, RANDOMNESS0, "dataset", "street_number"
    )

    # Validate we do not change any missing data
    original = data["street_number"]
    missing = original.is_null()
    assert noised_data.filter(missing).is_null().all()

    # Check expected noise level
    digits_per_string = [sum(c.isdigit() for c in s) for s in STRING_LIST]
    avg_probability_any_token_noised = sum(
        1 - (1 - expected_token_noise) ** n for n in digits_per_string
    ) / len(digits_per_string)
    fuzzy_checker.fuzzy_assert_proportion(
        name="write_wrong_digits",
        observed_numerator=(original.filter(~missing) != noised_data.filter(~missing)).sum(),
        observed_denominator=(~missing).sum(),
        target_proportion=expected_cell_noise * avg_probability_any_token_noised,
    )


def test_use_nickname(dummy_dataset, fuzzy_checker: FuzzyChecker):
    config = get_configuration()[DATASETS.census.name][Keys.COLUMN_NOISE]["first_name"][
        NOISE_TYPES.use_nickname.name
    ]
    expected_noise = config[Keys.CELL_PROBABILITY]
    data = dummy_dataset.select(ROW_ID, "first_name")
    noised_data = _noise(
        NOISE_TYPES.use_nickname, data, config, RANDOMNESS0, "dataset", "first_name"
    )
    original = data["first_name"]

    # Validate missing stays missing
    orig_missing = original.is_null()
    assert noised_data.filter(orig_missing).is_null().all()
    # Validate noise level
    fuzzy_checker.fuzzy_assert_proportion(
        name="use_nickname",
        observed_numerator=(
            noised_data.filter(~orig_missing) != original.filter(~orig_missing)
        ).sum(),
        observed_denominator=(~orig_missing).sum(),
        target_proportion=expected_noise,
    )

    # Validation for nicknames
    nicknames = {
        row[TOKEN]: row["options"] for row in load_nicknames_data().iter_rows(named=True)
    }
    for real_name in original.drop_nulls().unique():
        is_name = original == real_name
        noised_name = noised_data.filter(is_name)
        if real_name not in nicknames:
            # Validates names that are not nickname eligible do not not get noised
            assert (noised_name == real_name).all()
        else:
            # Verify options chosen are valid nicknames for original names that were noised
            assert set(noised_name.drop_nulls()).issubset(
                set(nicknames[real_name] + [real_name])
            )
            # Validate we choose the nicknames for each name randomly (equally)
            chosen_nickname_counts = noised_name.filter(
                noised_name != real_name
            ).value_counts()
            total = chosen_nickname_counts["count"].sum()
            for nickname, count in chosen_nickname_counts.iter_rows():
                fuzzy_checker.fuzzy_assert_proportion(
                    name="use_nickname_proportion",
                    observed_numerator=count,
                    observed_denominator=total,
                    target_proportion=1 / len(nicknames[real_name]),
                    name_additional=f"for nickname: {nickname} of real name: {real_name}",
                )


def test_use_fake_name(dummy_dataset, fuzzy_checker: FuzzyChecker):
    """
    Function to test that fake names are noised and replace raw values at a configured percentage
    """
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "first_name": {
                        NOISE_TYPES.use_fake_name.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                        },
                    },
                    "last_name": {
                        NOISE_TYPES.use_fake_name.name: {Keys.CELL_PROBABILITY: 0.5}
                    },
                },
            },
        }
    )

    for column, fake_names in [
        ("first_name", fake_first_names),
        ("last_name", fake_last_names),
    ]:
        column_config = config[DATASETS.census.name][Keys.COLUMN_NOISE][column][
            NOISE_TYPES.use_fake_name.name
        ]
        data = dummy_dataset.select(ROW_ID, column)
        noised = _noise(
            NOISE_TYPES.use_fake_name, data, column_config, RANDOMNESS0, "dataset", column
        )
        original = data[column]

        # Check missing are unchanged
        orig_missing = original.is_null()
        assert noised.filter(orig_missing).is_null().all()
        # Check noised values
        changed = noised.ne_missing(original)
        fuzzy_checker.fuzzy_assert_proportion(
            name=f"use_fake_name_{column}",
            observed_numerator=changed.sum(),
            observed_denominator=(~orig_missing).sum(),
            target_proportion=column_config[Keys.CELL_PROBABILITY],
        )
        assert noised.filter(changed).is_in(fake_names).all()


def _number_of_tokens_per_string(tokens, strings) -> list[int]:
    """For each of strings, the total number of (overlapping) occurrences of tokens in it."""
    return [sum(_occurrences(str(s), str(token)) for token in tokens) for s in strings]


# https://stackoverflow.com/a/2970542/
def _occurrences(string, sub):
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count


def _average_probability_any_token_noised(tokens, strings, token_probability) -> float:
    tokens_per_string = _number_of_tokens_per_string(tokens, strings)
    return sum(1 - (1 - token_probability) ** n for n in tokens_per_string) / len(
        tokens_per_string
    )


EQUALLY_LIKELY_VALUES = {
    "first_name": FIRST_NAMES,
    "last_name": LAST_NAMES,
    "characters": CHARACTERS_LIST,
    "numbers": [str(i) for i in INTEGERS_LIST],
    "phonetic_stress_test": PHONETIC_STRESS_TEST_LIST,
    "ocr_stress_test": OCR_STRESS_TEST_LIST,
}


def _token_noise_config(column: str, noise_type, cell_probability, token_probability):
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    column: {
                        noise_type.name: {
                            Keys.CELL_PROBABILITY: cell_probability,
                            Keys.TOKEN_PROBABILITY: token_probability,
                        },
                    }
                },
            },
        }
    )
    return config[DATASETS.census.name][Keys.COLUMN_NOISE][column][noise_type.name]


@pytest.mark.parametrize(
    "column",
    [
        "first_name",
        "last_name",
        "numbers",
        "characters",
        "phonetic_stress_test",
    ],
)
def test_generate_phonetic_errors(dummy_dataset, column, fuzzy_checker: FuzzyChecker):
    data = dummy_dataset.select(ROW_ID, column)
    config = _token_noise_config(column, NOISE_TYPES.make_phonetic_errors, 0.1, 0.5)
    noised_data = _noise(
        NOISE_TYPES.make_phonetic_errors, data, config, RANDOMNESS0, "dataset", column
    )
    original = data[column]

    # Validate we do not change any missing data
    missing = original.is_null()
    assert noised_data.filter(missing).is_null().all()

    # Check expected noise level
    avg_probability_any_token_noised = _average_probability_any_token_noised(
        load_phonetic_errors()[TOKEN],
        EQUALLY_LIKELY_VALUES[column],
        config[Keys.TOKEN_PROBABILITY],
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="generate_phonetic_errors",
        observed_numerator=(original.filter(~missing) != noised_data.filter(~missing)).sum(),
        observed_denominator=(~missing).sum(),
        target_proportion=config[Keys.CELL_PROBABILITY] * avg_probability_any_token_noised,
    )


def _check_pathways(string, pathways, noise_type, fuzzy_checker, name):
    num_rows = 100_000
    cell_probability = 0.9
    token_probability = 0.3
    config = {
        Keys.CELL_PROBABILITY: cell_probability,
        Keys.TOKEN_PROBABILITY: token_probability,
    }
    data = _single_column_data("column", [string] * num_rows)
    noised_data = _noise(noise_type, data, config, RANDOMNESS0, "dataset", "column")

    unexpected = set(noised_data) - set(pathways.keys())
    assert not unexpected, f"Unexpected results for {string}: {unexpected}"
    for result, (num_noised, num_not_noised) in pathways.items():
        pathway_probability = (
            token_probability**num_noised * (1 - token_probability) ** num_not_noised
        )
        probability = pathway_probability * cell_probability
        if result == string:
            # If cell is not selected, we end up with this result
            probability += 1 - cell_probability

        fuzzy_checker.fuzzy_assert_proportion(
            name=name,
            observed_numerator=(noised_data == result).sum(),
            observed_denominator=num_rows,
            target_proportion=probability,
            name_additional=f"result {result} from string {string}",
        )


@pytest.mark.parametrize(
    "pair",
    PHONETIC_STRESS_TEST_PATHWAYS.items(),
)
def test_phonetic_error_values(pair, fuzzy_checker: FuzzyChecker):
    string, pathways = pair
    _check_pathways(
        string,
        pathways,
        NOISE_TYPES.make_phonetic_errors,
        fuzzy_checker,
        "test_phonetic_error_values",
    )


@pytest.mark.parametrize(
    "column",
    [
        "numbers",
        "characters",
        "first_name",
        "last_name",
        "ocr_stress_test",
    ],
)
def test_generate_ocr_errors(dummy_dataset, column, fuzzy_checker: FuzzyChecker):
    config = _token_noise_config(column, NOISE_TYPES.make_ocr_errors, 0.1, 0.1)
    data = dummy_dataset.select(ROW_ID, column)
    noised_data = _noise(
        NOISE_TYPES.make_ocr_errors, data, config, RANDOMNESS0, "dataset", column
    )
    original = data[column]

    # Validate we do not change any missing data
    missing = original.is_null()
    assert noised_data.filter(missing).is_null().all()

    # Check expected noise level. Not all tokens can be noised for OCR errors.
    avg_probability_any_token_noised = _average_probability_any_token_noised(
        load_ocr_errors()[TOKEN],
        EQUALLY_LIKELY_VALUES[column],
        config[Keys.TOKEN_PROBABILITY],
    )
    fuzzy_checker.fuzzy_assert_proportion(
        name="generate_ocr_errors",
        observed_numerator=(original.filter(~missing) != noised_data.filter(~missing)).sum(),
        observed_denominator=(~missing).sum(),
        target_proportion=config[Keys.CELL_PROBABILITY] * avg_probability_any_token_noised,
    )


@pytest.mark.parametrize(
    "pair",
    OCR_STRESS_TEST_PATHWAYS.items(),
)
def test_ocr_replacement_values(pair, fuzzy_checker: FuzzyChecker):
    string, pathways = pair
    _check_pathways(
        string,
        pathways,
        NOISE_TYPES.make_ocr_errors,
        fuzzy_checker,
        "test_ocr_replacement_values",
    )


@pytest.mark.parametrize(
    "column",
    [
        "numbers",
        "characters",
    ],
)
def test_make_typos(dummy_dataset, column, fuzzy_checker: FuzzyChecker):
    config = _token_noise_config(column, NOISE_TYPES.make_typos, 0.1, 0.1)
    data = dummy_dataset.select(ROW_ID, column)
    noised_data = _noise(NOISE_TYPES.make_typos, data, config, RANDOMNESS0, "dataset", column)
    original = data[column]

    not_missing = original.is_not_null()
    check_original = original.filter(not_missing)
    check_noised = noised_data.filter(not_missing)

    # Check for expected noise level
    cell_probability = config[Keys.CELL_PROBABILITY]
    token_probability = config[Keys.TOKEN_PROBABILITY]
    tokens_per_string = _number_of_tokens_per_string(
        load_qwerty_errors_data()[TOKEN], EQUALLY_LIKELY_VALUES[column]
    )
    avg_probability_any_token_noised = sum(
        1 - (1 - token_probability) ** n for n in tokens_per_string
    ) / len(tokens_per_string)
    fuzzy_checker.fuzzy_assert_proportion(
        name="make_typos",
        observed_numerator=(check_noised != check_original).sum(),
        observed_denominator=len(check_original),
        target_proportion=cell_probability * avg_probability_any_token_noised,
    )

    # Check for expected string growth due to keeping original noised token
    noised_lengths = check_noised.str.len_chars()
    original_lengths = check_original.str.len_chars()
    assert (noised_lengths >= original_lengths).all()
    # TODO: remove this hard-coding
    p_include_original_token = 0.1
    p_token_does_not_increase_string_length = 1 - token_probability * p_include_original_token
    p_strings_increase_length = sum(
        1 - p_token_does_not_increase_string_length**n for n in tokens_per_string
    ) / len(tokens_per_string)
    fuzzy_checker.fuzzy_assert_proportion(
        name="make_typos_string_length",
        observed_numerator=(noised_lengths != original_lengths).sum(),
        observed_denominator=len(check_original),
        target_proportion=cell_probability * p_strings_increase_length,
    )

    # Check that we did not touch the missing data
    assert noised_data.filter(~not_missing).is_null().all()


@pytest.mark.parametrize(
    "noise_type, data_col, dataset, dataset_col",
    [
        (NOISE_TYPES.leave_blank, "numbers", "decennial_census", "zipcode"),
        (NOISE_TYPES.choose_wrong_option, "state", "decennial_census", "state"),
        (NOISE_TYPES.copy_from_household_member, "age", "decennial_census", "age"),
        (NOISE_TYPES.swap_month_and_day, "event_date", "social_security", "event_date"),
        (NOISE_TYPES.write_wrong_zipcode_digits, "zipcode", "decennial_census", "zipcode"),
        (NOISE_TYPES.misreport_age, "age", "decennial_census", "age"),
        (
            NOISE_TYPES.write_wrong_digits,
            "street_number",
            "decennial_census",
            "street_number",
        ),
        (NOISE_TYPES.use_nickname, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.use_fake_name, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.use_fake_name, "last_name", "decennial_census", "last_name"),
        (NOISE_TYPES.make_phonetic_errors, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.make_ocr_errors, "first_name", "decennial_census", "first_name"),
        (NOISE_TYPES.make_typos, "numbers", "decennial_census", "zipcode"),
        (NOISE_TYPES.make_typos, "characters", "decennial_census", "street_name"),
    ],
)
def test_seeds_behave_as_expected(noise_type, data_col, dataset, dataset_col, dummy_dataset):
    """Tests that different seeds produce different results and the same seed
    produces the same results
    """
    noise = noise_type.name
    config = get_configuration()[dataset][Keys.COLUMN_NOISE][dataset_col][noise]
    if noise == NOISE_TYPES.copy_from_household_member.name:
        data = dummy_dataset.select(ROW_ID, data_col, COPY_HOUSEHOLD_MEMBER_COLS[data_col])
    else:
        data = dummy_dataset.select(ROW_ID, data_col)

    noised_data = _noise(noise_type, data, config, RANDOMNESS0, dataset, data_col)
    noised_data_same_seed = _noise(noise_type, data, config, RANDOMNESS0, dataset, data_col)
    noised_data_different_seed = _noise(
        noise_type, data, config, RANDOMNESS1, dataset, data_col
    )
    original = data[data_col]

    assert noised_data.ne_missing(original).any()
    assert noised_data.equals(noised_data_same_seed)
    assert noised_data.ne_missing(noised_data_different_seed).any()

    # Check that different seeds noise different cells
    assert (
        noised_data.ne_missing(original) != noised_data_different_seed.ne_missing(original)
    ).any()


def test_age_write_wrong_digits(dummy_dataset, fuzzy_checker: FuzzyChecker):
    # Tests write wrong digits is now applied to age column - albrja(10/23/23)
    config = get_configuration()
    config.update(
        {
            DATASETS.census.name: {
                Keys.COLUMN_NOISE: {
                    "age": {
                        NOISE_TYPES.write_wrong_digits.name: {
                            Keys.CELL_PROBABILITY: 0.4,
                            Keys.TOKEN_PROBABILITY: 0.5,
                        },
                    },
                },
            },
        }
    )
    config = config[DATASETS.census.name][Keys.COLUMN_NOISE]["age"][
        NOISE_TYPES.write_wrong_digits.name
    ]
    data = dummy_dataset.select(ROW_ID, "age")
    noised_data = _noise(
        NOISE_TYPES.write_wrong_digits, data, config, RANDOMNESS0, "dataset", "age"
    )
    original = data["age"]
    missing = original.is_null()
    check_original = original.filter(~missing)
    check_noised = noised_data.filter(~missing)
    token_probability = config[Keys.TOKEN_PROBABILITY]
    tokens_per_string = check_original.str.len_chars()
    avg_probability_any_token_noised = (
        1 - (1 - token_probability) ** tokens_per_string.cast(pl.Float64)
    ).mean()
    fuzzy_checker.fuzzy_assert_proportion(
        name="write_wrong_digits_age",
        observed_numerator=(check_original != check_noised).sum(),
        observed_denominator=len(check_original),
        target_proportion=config[Keys.CELL_PROBABILITY] * avg_probability_any_token_noised,
    )
