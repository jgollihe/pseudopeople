"""
===================
   Noise Dataset
===================

A function that will noise data based on a user's configuration.

This function is the core of Pseudopeople, which takes in a Polars DataFrame of
dataset data and will add noise to both rows and columns based on the
configured values provided by the user. Missing values are always nulls, so
whether a cell is eligible for noise is read directly from the data.
"""

from typing import Any

import polars as pl
from tqdm import tqdm
from vivarium.config_tree import ConfigTree

from pseudopeople.configuration import Keys
from pseudopeople.entity_types import ColumnNoiseType, RowNoiseType
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.randomness import ROW_ID, Randomness
from pseudopeople.schema_entities import Dataset


def noise_dataset(
    dataset: Dataset,
    dataset_data: pl.DataFrame,
    configuration: ConfigTree,
    seed: Any,
    progress_bar: bool = True,
) -> pl.DataFrame:
    """
    Adds noise to the input dataset data. Noise functions are executed in the order
    defined by :py:const: `.NOISE_TYPES`. Row noise functions are applied to the
    whole DataFrame. Column noise functions will be applied to each column that
    is pertinent to it.

    Noise levels are determined by the noise_config.
    :param dataset:
        Dataset needing to be noised
    :param dataset_data:
        Simulated population data which needs to be noised. Gets a row id column
        if it does not already have one.
    :param configuration:
        Object to configure noise levels
    :param seed:
        Seed for controlling randomness
    :return:
        Noised dataset data
    """
    randomness = Randomness(f"{dataset.name}_{seed}")
    if ROW_ID not in dataset_data.columns:
        dataset_data = dataset_data.with_row_index(ROW_ID)

    noise_configuration = configuration[dataset.name]

    if progress_bar:
        noise_type_iterator = tqdm(
            NOISE_TYPES, desc="Applying noise", unit="type", leave=False
        )
    else:
        noise_type_iterator = NOISE_TYPES

    for noise_type in noise_type_iterator:
        if isinstance(noise_type, RowNoiseType):
            if (
                Keys.ROW_NOISE in noise_configuration
                and noise_type.name in noise_configuration.row_noise
            ):
                dataset_data = noise_type(
                    dataset.name,
                    dataset_data,
                    noise_configuration[Keys.ROW_NOISE][noise_type.name],
                    randomness,
                )

        elif isinstance(noise_type, ColumnNoiseType):
            if Keys.COLUMN_NOISE in noise_configuration:
                columns_to_noise = [
                    col
                    for col in noise_configuration.column_noise
                    if col in dataset_data.columns
                    and noise_type.name in noise_configuration.column_noise[col]
                ]
                for column in columns_to_noise:
                    dataset_data = noise_type(
                        dataset_data,
                        noise_configuration.column_noise[column][noise_type.name],
                        randomness,
                        dataset.name,
                        column,
                    )
        else:
            raise TypeError(
                f"Invalid noise type. Allowed types are {RowNoiseType} and "
                f"{ColumnNoiseType}. Provided {type(noise_type)}."
            )

    return dataset_data
