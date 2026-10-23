import polars as pl
import pytest

from pseudopeople.noise_functions import _corrupt_tokens
from pseudopeople.randomness import ROW_ID, Randomness
from pseudopeople.utilities import OPTIONS, TOKEN, to_string
from tests.conftest import FuzzyChecker

RANDOMNESS0 = Randomness("test_utils_0")
CORRUPT_TOKENS_TEST_CASES = {
    # Possible tokens to noise: abc, ab, a, c
    # Tuples of (token noised, token not noised)
    # Example: "abc" can be noised to "heybsee". This means "a" becomes "hey"
    # and "c" becomes "see" which means 2 tokens are noised ("a" and "c") and 2
    # are not ("abc" and "ab").
    # If a overlapping token is noised, we proceed with assuming the shorter
    # token didn't exist.
    "abc": {
        "abc": (0, 4),
        "absee": (1, 3),
        "tyc": (1, 2),
        "heybsee": (2, 2),
        "heybc": (1, 3),
        "tysee": (2, 1),
        "def": (1, 0),
    },
    "abzc": {
        # No "abc" token and z is not a token we noise
        "abzc": (0, 3),
        "abzsee": (1, 2),
        "tyzc": (1, 1),
        "heybzsee": (2, 1),
        "tyzsee": (2, 0),
        "heybzc": (1, 2),
    },
    "jkl": {
        # No tokens - to test nothing is noised
        "jkl": (0, 0),
    },
}


def _errors_table(errors: dict[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame({TOKEN: list(errors), OPTIONS: list(errors.values())}).with_columns(
        n_options=pl.col(OPTIONS).list.len().cast(pl.Int64)
    )


def _rows(n: int) -> pl.DataFrame:
    return pl.select(pl.int_range(n, dtype=pl.UInt32).alias(ROW_ID))


def test_to_string_integer_valued():
    # Integer-valued columns (age, wages, and po box) return integer strings, keeping nulls.
    floats = pl.Series("age", [None, 1.0, 2.0, 3.01, 4.0, None])
    assert to_string(floats).to_list() == [None, "1", "2", "3", "4", None]
    strings = pl.Series("wages", [None, "1", "2.0", "5.055"])
    assert to_string(strings).to_list() == [None, "1", "2", "5"]
    other = pl.Series("zipcode", [1.5, None])
    assert to_string(other).to_list() == ["1.5", None]


@pytest.mark.parametrize("pair", CORRUPT_TOKENS_TEST_CASES.items())
def test__corrupt_tokens(pair, fuzzy_checker: FuzzyChecker):
    """
    Unit test for _corrupt_tokens. We want to test that the noise level is correct, that the
    correct tokens are noised, and that the correct behavior happens (meaning the longer tokens
    are noised first) in a string).
    """
    fake_errors = _errors_table({"abc": ["def"], "ab": ["ty"], "a": ["hey"], "c": ["see"]})
    string, pathways = pair

    num_rows = 100_000
    data = _rows(num_rows).with_columns(column=pl.lit(string))
    token_probability = 0.4

    noised = _corrupt_tokens(
        errors=fake_errors,
        data=data,
        column_name="column",
        token_probability=token_probability,
        randomness=RANDOMNESS0,
        addl_key="test__corrupt_tokens",
    )

    # Assert our noised data is one of our possible strings. This also checks that
    # the tokens are noised in the correct order (longer tokens first) because if
    # a longer token overlaps a shorter token and the longer token is noised, we
    # do not count the shorter token as a possiblility of being noised.
    assert noised.is_in(list(pathways.keys())).all()
    for result, (num_noised, num_not_noised) in pathways.items():
        pathway_probability = (
            token_probability**num_noised * (1 - token_probability) ** num_not_noised
        )
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_phonetic_error_values",
            observed_numerator=(noised == result).sum(),
            observed_denominator=num_rows,
            target_proportion=pathway_probability,
            name_additional=f"result {result} from string {string}",
        )

    # Assert token noise level is correct.
    tokens_per_string = {"abc": 4, "abzc": 3, "jkl": 0}[string]
    any_token_noised = 1 - (1 - token_probability) ** tokens_per_string
    fuzzy_checker.fuzzy_assert_proportion(
        name="generate_phonetic_errors",
        observed_numerator=(noised != string).sum(),
        observed_denominator=num_rows,
        target_proportion=any_token_noised,
    )


def test__corrupt_tokens_multiple_options(fuzzy_checker: FuzzyChecker):
    """
    Tests that multiple options can be chosen for a token
    """
    fake_errors = _errors_table({"abc": ["def", "ghi", "jkl"]})
    num_rows = 100_000
    data = _rows(num_rows).with_columns(column=pl.lit("abc"))
    token_probability = 0.4

    noised = _corrupt_tokens(
        errors=fake_errors,
        data=data,
        column_name="column",
        token_probability=token_probability,
        randomness=RANDOMNESS0,
        addl_key="test__corrupt_tokens",
    )
    strings = ["abc", "def", "ghi", "jkl"]
    assert noised.is_in(strings).all()

    for string in strings:
        proportion = 1 - token_probability if string == "abc" else token_probability / 3
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_multiple_options__corruprt_tokens",
            observed_numerator=(noised == string).sum(),
            observed_denominator=num_rows,
            target_proportion=proportion,
            name_additional=f"result {string}",
        )


def test_uniform(fuzzy_checker: FuzzyChecker):
    """Tests that rows selected by a uniform draw match the expected noise level"""
    num_rows = 100_000
    noise_level = 0.62
    draws = _rows(num_rows).select(RANDOMNESS0.uniform("test_uniform")).to_series()
    assert draws.min() >= 0 and draws.max() < 1
    fuzzy_checker.fuzzy_assert_proportion(
        name="test_uniform",
        observed_numerator=(draws < noise_level).sum(),
        observed_denominator=num_rows,
        target_proportion=noise_level,
    )


def test_uniform_is_deterministic_and_keyed():
    data = _rows(10_000)
    draws = data.select(RANDOMNESS0.uniform("a")).to_series()
    assert draws.equals(data.select(RANDOMNESS0.uniform("a")).to_series())
    # Draws for a row depend only on that row, not on which other rows are present
    assert draws.tail(100).equals(data.tail(100).select(RANDOMNESS0.uniform("a")).to_series())
    # Different keys and seeds give different draws
    assert not draws.equals(data.select(RANDOMNESS0.uniform("b")).to_series())
    assert not draws.equals(data.select(Randomness("other").uniform("a")).to_series())


def test_weighted_choice(fuzzy_checker: FuzzyChecker):
    options = ["Supersonics", "Mariners", "Kraken"]
    num_choices = 50_000
    choice_weights = [0.40, 0.35, 0.25]
    picks = (
        _rows(num_choices)
        .select(
            RANDOMNESS0.weighted_choice("best_seattle_sports_team", options, choice_weights)
        )
        .to_series()
    )

    assert picks.is_in(options).all()
    # Test weights validate
    for option, weight in zip(options, choice_weights):
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_weighted_choice",
            observed_numerator=(picks == option).sum(),
            observed_denominator=num_choices,
            target_proportion=weight,
            name_additional=f"option {option}",
        )


def test_from_options(fuzzy_checker: FuzzyChecker):
    options = ["Supersonics", "Mariners", "Kraken"]
    num_choices = 50_000
    picks = _rows(num_choices).select(RANDOMNESS0.from_options("teams", options)).to_series()
    for option in options:
        fuzzy_checker.fuzzy_assert_proportion(
            name="test_from_options",
            observed_numerator=(picks == option).sum(),
            observed_denominator=num_choices,
            target_proportion=1 / len(options),
            name_additional=f"option {option}",
        )


def test_choice_index_per_row(fuzzy_checker: FuzzyChecker):
    """Choosing from a per-row list of options, as for nicknames"""
    options = {
        "Basketball": ["Supersonics", "Nuggets", "Trail Blazers"],
        "Baseball": ["Mariners", "Dodgers"],
        "Football": ["Ducks", "Steelers", "Cougars"],
    }
    sports = list(options) * 1000
    data = pl.DataFrame({"sport": sports}).with_row_index(ROW_ID)
    choices = (
        data.with_columns(
            teams=pl.col("sport").replace_strict(options, return_dtype=pl.List(pl.String))
        )
        .select(
            pl.col("teams").list.get(
                RANDOMNESS0.choice_index("2D_best_sports_team", pl.col("teams").list.len())
            )
        )
        .to_series()
    )
    for sport, teams in options.items():
        is_sport = data["sport"] == sport
        for team in teams:
            fuzzy_checker.fuzzy_assert_proportion(
                name="test_choice_index_per_row",
                observed_numerator=(choices.filter(is_sport) == team).sum(),
                observed_denominator=is_sport.sum(),
                target_proportion=1 / len(teams),
                name_additional=f"team {team} for sport {sport}",
            )
