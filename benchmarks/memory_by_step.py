"""Report process memory after each noising step on one shard.

Usage:
    uv run python benchmarks/memory_by_step.py PATH/TO/SHARD.parquet --dataset taxes_w2_and_1099
"""

import argparse
import gc
from pathlib import Path

import psutil

from pseudopeople.configuration import Keys, get_configuration
from pseudopeople.entity_types import RowNoiseType
from pseudopeople.interface import _clean_input_data, _reformat_dates_for_noising
from pseudopeople.loader import scan_standard_dataset
from pseudopeople.noise_entities import NOISE_TYPES
from pseudopeople.randomness import ROW_ID, Randomness
from pseudopeople.schema_entities import DATASETS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard", type=Path)
    parser.add_argument("--dataset", default="taxes_w2_and_1099")
    args = parser.parse_args()

    process = psutil.Process()

    def report(step: str, data=None) -> None:
        gc.collect()
        live = f"  live data {data.estimated_size('gb'):.2f} GB" if data is not None else ""
        print(f"{step:28s} RSS {process.memory_info().rss / 2**30:.2f} GiB{live}")

    dataset = DATASETS.get_dataset(args.dataset)
    configuration = get_configuration()[dataset.name]
    report("start")
    data = scan_standard_dataset(args.shard, [])
    data = _clean_input_data(_reformat_dates_for_noising(data, dataset), dataset)
    data = data.collect().with_row_index(ROW_ID)
    report("loaded and prepped", data)

    randomness = Randomness("memory")
    for noise_type in NOISE_TYPES:
        if isinstance(noise_type, RowNoiseType):
            row_noise = configuration.get(Keys.ROW_NOISE, {})
            if noise_type.name in row_noise:
                data = noise_type(dataset.name, data, row_noise[noise_type.name], randomness)
        else:
            column_noise = configuration[Keys.COLUMN_NOISE]
            for column in column_noise:
                if column in data.columns and noise_type.name in column_noise[column]:
                    data = noise_type(
                        data,
                        column_noise[column][noise_type.name],
                        randomness,
                        dataset.name,
                        column,
                    )
        report(noise_type.name, data)


if __name__ == "__main__":
    main()
