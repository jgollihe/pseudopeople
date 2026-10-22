"""Build a large synthetic simulated population by replicating the bundled sample data.

Each replicate gets its own id prefix (the sample ids look like ``0_123``), so
simulants, households and guardian links stay internally consistent but distinct
across replicates. Output mimics the real data layout: one directory per dataset
containing several dictionary-encoded parquet shards, plus a ``CHANGELOG.rst``
that passes pseudopeople's data-version check.

Usage:
    uv run python benchmarks/make_synthetic_population.py OUT_DIR --replicates 100 --shards 10
"""

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SAMPLE_ROOT = Path(__file__).parents[1] / "src" / "pseudopeople" / "data" / "sample_datasets"
ID_COLUMN_MARKERS = ("simulant_id", "household_id", "guardian_1", "guardian_2")


def _is_id_column(name: str) -> bool:
    return any(marker in name for marker in ID_COLUMN_MARKERS)


def _replicate(table: pa.Table, replicate: int) -> pa.Table:
    columns = []
    for name in table.column_names:
        column = table[name]
        if _is_id_column(name):
            was_dictionary = pa.types.is_dictionary(column.type)
            as_string = column.cast(pa.string()) if was_dictionary else column
            column = pc.replace_substring_regex(as_string, r"^\d+_", f"{replicate}_")
            if was_dictionary:
                column = pc.dictionary_encode(column)
        columns.append(column)
    return pa.table(columns, names=table.column_names)


def build(out_dir: Path, replicates: int, shards: int, datasets: list[str] | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "CHANGELOG.rst").write_text("**1.4.2 - 01/01/2024**\n\n - Synthetic benchmark data\n")
    dataset_dirs = sorted(p for p in SAMPLE_ROOT.iterdir() if p.is_dir())
    for dataset_dir in dataset_dirs:
        if datasets and dataset_dir.name not in datasets:
            continue
        source = pq.read_table(next(dataset_dir.glob("*.parquet")))
        if "__index_level_0__" in source.column_names:
            source = source.drop_columns(["__index_level_0__"])
        target = out_dir / dataset_dir.name
        if target.exists():
            shutil.rmtree(target)
        target.mkdir()
        per_shard = max(1, replicates // shards)
        replicate = 0
        for shard in range(shards):
            parts = []
            for _ in range(per_shard):
                parts.append(_replicate(source, replicate))
                replicate += 1
            shard_table = pa.concat_tables(parts).unify_dictionaries().combine_chunks()
            pq.write_table(shard_table, target / f"{dataset_dir.name}_{shard}.parquet")
        print(f"{dataset_dir.name}: {replicate * source.num_rows:,} rows in {shards} shards")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    build(args.out_dir, args.replicates, args.shards, args.datasets)
