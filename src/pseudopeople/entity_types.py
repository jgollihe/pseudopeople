from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl
from loguru import logger
from vivarium.config_tree import ConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.randomness import ROW_ID, Randomness


@dataclass
class NoiseType(ABC):
    name: str
    noise_function: Callable
    probability: float | None = 0.0
    additional_parameters: dict[str, Any] = None

    @abstractmethod
    def probability_key(self) -> str:
        pass


@dataclass
class RowNoiseType(NoiseType):
    """
    Defines a type of noise that can be applied to a row.

    The name is the name of the particular noise type (e.g. "omit_row" or
    "duplicate_row").

    The noise function takes as input the dataset name, a DataFrame, the
    configuration value for this RowNoise operation, and a Randomness source.
    It applies the noising operation to the entire DataFrame and returns the
    modified DataFrame.
    """

    @property
    def probability_key(self) -> str:
        return Keys.ROW_PROBABILITY

    def __call__(
        self,
        dataset_name: str,
        dataset_data: pl.DataFrame,
        configuration: ConfigTree,
        randomness: Randomness,
    ) -> pl.DataFrame:
        return self.noise_function(dataset_name, dataset_data, configuration, randomness)


@dataclass
class ColumnNoiseType(NoiseType):
    """
    Defines a type of noise that can be applied to a column.

    The name is the name of the particular noise type (e.g. use_nickname" or
    "make_phonetic_errors").

    The noise function takes as input a DataFrame holding only the rows selected
    for noise (the noised column, any additional columns it needs, and the row
    id), the ConfigTree for this ColumnNoise operation, a Randomness source, the
    dataset name, and the column name. It returns a Series of noised values,
    one per selected row, in the same order.

    Calling the noise type selects which cells to noise, applies the noise
    function to them, and returns the full DataFrame with the noised column
    updated.
    """

    probability: float | None = 0.01
    noise_level_scaling_function: Callable[[pl.DataFrame, str], float] = lambda x, y: 1.0
    additional_column_getter: Callable[[str], list[str]] = lambda column_name: []
    output_dtype_getter: Callable[[pl.DataType], pl.DataType] = lambda dtype: dtype

    @property
    def probability_key(self) -> str:
        return Keys.CELL_PROBABILITY

    def get_noise_level(
        self, data: pl.DataFrame, configuration: ConfigTree, column_name: str
    ) -> float:
        noise_level = configuration[
            Keys.CELL_PROBABILITY
        ] * self.noise_level_scaling_function(data, column_name)
        # Certain columns have their noise level scaled so we must check to make sure the noise level is within the
        # allowed range between 0 and 1 for probabilities
        return min(noise_level, 1.0)

    def select_rows(
        self,
        data: pl.DataFrame,
        noise_level: float,
        randomness: Randomness,
        column_name: str,
    ) -> pl.Series:
        """Positions of the rows to noise: non-missing in every required column, then sampled."""
        required_columns = [column_name] + self.additional_column_getter(column_name)
        eligible = pl.all_horizontal([pl.col(c).is_not_null() for c in required_columns])
        selected = eligible & (randomness.uniform(f"{self.name}_{column_name}") < noise_level)
        return data.select(pl.arg_where(selected)).to_series()

    def __call__(
        self,
        data: pl.DataFrame,
        configuration: ConfigTree,
        randomness: Randomness,
        dataset_name: str,
        column_name: str,
    ) -> pl.DataFrame:
        if data[column_name].null_count() == data.height:
            return data
        noise_level = self.get_noise_level(data, configuration, column_name)
        positions = self.select_rows(data, noise_level, randomness, column_name)
        if positions.is_empty():
            logger.debug(
                f"No cells chosen to noise for noise function {self.name} on column {column_name}. "
                "This is likely due to a combination of the configuration noise levels and the simulated population data."
            )
            return data

        required_columns = [column_name] + self.additional_column_getter(column_name)
        to_noise = data.select(ROW_ID, *required_columns)[positions]
        noised = self.noise_function(
            to_noise, configuration, randomness, dataset_name, column_name
        )

        output_dtype = self.output_dtype_getter(data.schema[column_name])
        original = data[column_name]
        if original.dtype != output_dtype:
            original = original.cast(output_dtype)
        updated = original.scatter(positions, noised.cast(output_dtype))
        return data.with_columns(updated)
