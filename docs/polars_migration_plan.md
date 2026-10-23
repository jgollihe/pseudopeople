# Pseudopeople: Polars + uv migration

Status: implemented on the `polars` branch (personal fork).

## Decisions

- Polars replaces pandas everywhere in `src/`. pandas is only an optional extra
  (`pseudopeople[pandas]`) for users who want `.to_pandas()`.
- The Dask engine and the `engine` parameter are gone.
- uv manages the project: `pyproject.toml` (hatchling + hatch-vcs) and `uv.lock`.
  `setup.py`, `tox.ini`, `Jenkinsfile`, `MANIFEST.in`, `python_versions.json`, and the
  `vivarium_build_utils` Makefiles were removed. CI is a GitHub Actions workflow
  using `astral-sh/setup-uv`.
- Reproducibility is statistical: a seed is deterministic for a given Polars
  version (pinned to `>=2.0,<2.1`), but outputs differ from pandas-era pseudopeople.

## Architecture

### Data conventions

- No index. Rows are identified by a `__pseudopeople_row_id` column assigned
  per shard (duplicates from `duplicate_with_guardian` get new ids).
- Missing values are always `null`. `""` is normalized to `null` at load, so a
  cell is eligible for noise iff it is not null; the old full-frame
  `missingness` DataFrame is gone.
- Categorical inputs are cast to `pl.String` for noising; output dtypes are
  cast from the schema at the end (`DtypeNames` -> `dtypes.to_polars_dtype`).
- Integer-valued string columns (`age`, `wages`, `mailing_address_po_box`, and
  the copy columns that feed them) never get a trailing `.0`.

### Randomness (`randomness.py`)

`Randomness(seed).uniform(key, *by)` returns a Polars expression: a uniform
[0, 1) draw computed by hashing the row id (and optionally integer `by`
columns such as a character position) with a seed derived from
`(shard seed, key)`. Draws are independent across keys, depend only on the
row they belong to, and run fully vectorized. Helpers: `choice_index`,
`weighted_choice`, `from_options`. Vivarium's `RandomnessStream` (and the
`vivarium-engine` dependency) is no longer used.

Hashes are chained (`hash(hash(row) ^ pos)`) rather than hashing a struct,
because `pl.struct(...).hash()` panics on multi-chunk data in Polars 2.0's
streaming engine.

### Noise

Every noise type is pure Polars; there is no numpy in `src/`.

- Row noise: `omit_rows` and `apply_do_not_respond` are filters on a hashed
  draw (do-not-respond probabilities come from an integer-keyed
  `replace_strict` lookup on (sex, age)). `duplicate_with_guardian` is two
  left joins plus a concat.
- Column noise (`ColumnNoiseType.__call__`): select eligible rows with
  `uniform < noise_level`, run the noise function on just those rows, and
  `scatter` the results back.
- Character-level noise (typos, wrong digits) explodes the selected strings
  into one row per character, edits characters with expressions/joins, and
  re-joins them.
- OCR and phonetic errors (`_corrupt_tokens`) handle multi-character tokens
  with a left-to-right "skip past a corrupted token" rule. Candidates are found
  in one lazy query (positions x token lengths, joined to the error table, then
  longest-token-wins per position). The sequential skip rule is resolved
  exactly by `_resolve_left_to_right` in a few vectorized passes. The
  pathway-distribution tests confirm it matches the original algorithm.

### Execution and output

- `load_standard_dataset` uses `pl.scan_parquet` with year/state filters
  pushed into the scan.
- Shards are noised independently on a thread pool (`n_workers`, default up
  to 4). Polars releases the GIL. Results are reassembled in shard order, so
  output doesn't depend on `n_workers`.
- `output_path` writes one parquet file per shard as each one finishes and
  returns a `pl.LazyFrame` over them, so full-US runs never hold the whole
  dataset in memory.

## Benchmarks

Synthetic population: the sample data replicated 100x with distinct ids
(`benchmarks/make_synthetic_population.py`), 10 shards per dataset. Windows,
20 logical CPUs. Default configuration, `year=None`.

| Dataset | Rows out | pandas (before) | Polars, 1 worker | Polars, default (4) |
| --- | --- | --- | --- | --- |
| Decennial census | 3.3M | 83.0 s, 3.2 GiB | 13.9 s, 2.1 GiB | 5.9 s, 2.9 GiB |
| Social Security | 2.4M | 29.6 s, 1.6 GiB | - | 2.2 s, 1.2 GiB |
| W2 & 1099 | 24M | 682 s, 20.7 GiB | 69 s, 4.8 GiB (with `output_path`) | 39 s, 19.6 GiB |

Reproduce with `make bench-data` and
`uv run python benchmarks/run_benchmark.py benchmarks/data/pop100 --dataset <name>`.

### Memory

Live data for a 2.4M-row W2 shard is ~0.4 GB, but each worker peaks at
~3-4.5 GiB of process memory. That comes from transient copies during noising
and from the allocator holding on to freed memory
(`benchmarks/memory_by_step.py` reports it step by step). Peak memory therefore
scales with `n_workers`:

- In memory: the full result plus the working set of `n_workers` shards.
- With `output_path`: roughly `n_workers` x the per-shard peak. For W2 that's
  4.8 GiB with 1 worker, 9.0 GiB with 2, and 15.7 GiB with 4.

On a memory-constrained machine, use `output_path` with `n_workers=1` or `2`.

## Behavior differences from pandas pseudopeople

- Outputs for a given seed differ (new randomness), but noise rates match;
  the unit and integration fuzzy-checker tests are unchanged in what they assert.
- Return type is `polars.DataFrame` (or `polars.LazyFrame` with `output_path`).
- "Nan" is now a valid nickname (pandas' CSV reader used to turn it into a missing value).
- Rows with missing race, sex, or age get the base do-not-respond probability
  instead of never being omitted.

## Possible follow-ups

- Batch all columns of a character-level noise type into one query to amortize
  per-call overhead (most of the remaining per-shard time is many small queries).
- Try `pl.Enum` for fixed-option output columns.
- Run against a single real US shard to tune `n_workers` and memory.
