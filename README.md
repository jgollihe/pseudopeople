# Pseudopeople (Polars fork)

pseudopeople generates realistic simulated data about a fictional United States
population, for testing entity resolution (record linkage) methods and other
data science algorithms at scale.

This is a **fork** of [ihmeuw/pseudopeople](https://github.com/ihmeuw/pseudopeople).
The main difference is a **[Polars](https://pola.rs/)** backend: datasets are
returned as Polars DataFrames (or LazyFrames when writing to disk), shards can
be noised in parallel, and large runs can stream parquet out without holding the
full result in memory.

**Requires Python 3.11 or newer.**

## Install

This fork is not published on PyPI. Clone and install with
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jgollihe/pseudopeople.git
cd pseudopeople
uv sync
```

Optional pandas support (for `.to_pandas()` on Polars results):

```bash
uv sync --extra pandas
```

Run tests with:

```bash
uv run pytest
```

## Usage notes

- Default return type is a `polars.DataFrame`.
- Pass `output_path` to write one parquet file per shard and get back a
  `polars.LazyFrame` (useful for full-US sized runs).
- Pass `n_workers` to control how many shards are noised at once (default: up
  to 4). More workers are faster but use more memory; results do not depend on
  the worker count.
- Seeds are deterministic for a given Polars minor version (`polars>=2.0,<2.1`),
  but outputs differ from the pandas-era upstream package.

## Documentation

Upstream docs: https://pseudopeople.readthedocs.io/en/latest/

Some APIs (notably the Polars return types, `output_path`, and `n_workers`) are
specific to this fork and may not match the published docs yet.
