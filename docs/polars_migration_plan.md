# Pseudopeople: Polars + uv migration plan

Status: proposal. Target: personal fork (free to break API and drop IHME CI).

## Goals and decisions

- Replace pandas with Polars as the dataframe library everywhere in `src/`.
- Remove the Dask engine entirely.
- Manage the project with uv (`pyproject.toml` + `uv.lock`), dropping `setup.py`,
  `tox.ini`, `Jenkinsfile`, and the `vivarium_build_utils` Makefiles.
- `generate_*` functions return a `polars.DataFrame`, and can optionally write
  noised output straight to parquet (one file per input shard) so full-US runs
  never need to fit in memory.
- Reproducibility requirement is **statistical only**: a given seed must be
  deterministic within the new version, and noise rates must match the configured
  levels, but outputs need not match current pseudopeople bit-for-bit. This lets us
  drop Vivarium's `RandomnessStream`.
- Development and benchmarking use the bundled sample data plus a synthetic
  upscaled copy of it. The full US population is not used unless a specific problem
  requires it (copying a single US shard off the external drive would be enough).

## Current architecture (what we are changing)

- `interface._generate_dataset` has two paths:
  - pandas: loops over shard files serially, noises each, `pd.concat`s everything.
  - Dask: `dd.read_parquet` + `map_partitions` over the same per-shard logic.
- `loader.load_standard_dataset` reads parquet via pyarrow into pandas and resets
  the index.
- `_clean_input_data` converts every dictionary-encoded (categorical) string column
  to Python-object strings and replaces `""` with NaN. `_reformat_dates_for_noising`
  formats dates as strings. `_coerce_dtypes` coerces output dtypes.
- `noise.noise_dataset` applies `NOISE_TYPES` in order and maintains a full-frame
  boolean `missingness` DataFrame, realigned with `index.intersection/difference`
  after every row-noise step.
- `entity_types.ColumnNoiseType.__call__` selects `to_noise_idx` (a pandas Index),
  calls the noise function on `data.loc[to_noise_idx]`, and writes back with `.loc`.
- Randomness flows through Vivarium's `RandomnessStream` (hashes index values) and
  through numpy generators seeded by `vivarium...randomness.get_hash`.
- The pandas index is never meaningful (it's reset to a RangeIndex per shard), so
  all index-based logic can become positional.

Pandas is imported in: `interface.py`, `loader.py`, `noise.py`, `noise_functions.py`,
`noise_scaling.py`, `entity_types.py`, `utilities.py`, `output_dtype_getters.py`,
`configuration/generator.py`, `configuration/validator.py`,
`constants/data_values.py`.

## Target architecture

### Data conventions

| Concern | Convention |
| --- | --- |
| Row identity | Positional only; no index. Selections are numpy `int64` position arrays or boolean masks. |
| Missing values | Polars `null` only. `""` is normalized to `null` at load time. |
| Free-text string columns | `pl.String` |
| Columns with a fixed option set (sex, state, race_ethnicity, ...) | Kept as `pl.Categorical` during noising when untouched; cast to `pl.String` only when a string-producing noise function needs them. Output dtype defined per column in the schema. |
| Dates | Formatted to strings once with `dt.strftime` per dataset date format. |
| Integers that become strings (`age`, `wages`, `mailing_address_po_box`) | `cast(pl.Int64).cast(pl.String)`, which removes the `26.0` problem that `to_string_as_integer` works around. |

Because missingness is always `null`, `col.is_null()` (a validity-bitmap check, nearly
free) replaces the tracked `missingness` DataFrame entirely. One caveat to handle:
noise functions must never emit `""` (phonetic errors could in theory collapse a
string). We normalize `""` to `null` after string-producing noise. This means such a
cell now counts as missing for later noise types, which is a tiny behavior change.

### Randomness (`src/pseudopeople/randomness.py`, new)

- Vendor a stable hash (`hashlib`-based, equivalent to Vivarium's `get_hash`).
- `make_rng(seed, *keys) -> np.random.Generator`. Every noise decision derives its
  own generator from `(seed, dataset, noise type, column, purpose)`, so ordering
  changes in one noise type don't shift randomness in another.
- Helpers replacing the current ones in `utilities.py`:
  - `select_rows(n, probability, rng, eligible_mask=None) -> np.ndarray` replaces
    `get_index_to_noise` (it keeps the binomial-then-sample fast path for scalar
    probabilities < 0.2, and handles per-row probabilities with
    `rng.random(n) < p`).
  - `vectorized_choice(options, n, rng, weights=None)`, the same algorithm minus
    `RandomnessStream`.
  - `two_d_array_choice(values, options_table, rng)`, a Polars join on the
    options table, then a uniform pick among the non-null options.
- `vivarium-engine` is no longer needed as a dependency. `vivarium-config-tree`
  (pure Python `ConfigTree`) stays.

### Noise contracts

```python
# Row noise: returns the new frame (filtered and/or with appended rows)
RowNoiseFn = Callable[[str, pl.DataFrame, ConfigTree, int | str], pl.DataFrame]

# Column noise: receives only the selected rows (plus any additional columns);
# returns a Series of the same length with the noised values for `column_name`.
ColumnNoiseFn = Callable[[pl.DataFrame, ConfigTree, int | str, str, str], pl.Series]
```

`ColumnNoiseType.__call__` becomes:

1. eligible = all required columns non-null;
2. scale the noise level, then `positions = select_rows(...)`;
3. `noised = fn(df[positions, required_cols], ...)`;
4. cast the column to the output dtype, then `df[column].scatter(positions, noised)`.

`noise.noise_dataset` loses all index/missingness bookkeeping and becomes a
loop over `NOISE_TYPES` that threads the frame through.

### Execution and output

- `load_shard(path, filters) -> pl.DataFrame`: `pl.scan_parquet(path).filter(...)
  .collect()`. Filters are built as Polars expressions, so year/state pushdown
  reads only the needed row groups. Drop `__index_level_0__`.
- Shards are processed independently (they already are: the per-shard seed is
  `f"{seed}_{shard_index}"` and `duplicate_with_guardian` only looks within a shard).
- New parameters on every `generate_*` function:
  - `output_path: str | Path | None = None`. When set, each noised shard is
    written to `output_path/<dataset>_<shard>.parquet` as soon as it finishes, and
    the function returns `None` (or a `pl.LazyFrame` scanning the outputs; see
    open questions).
  - `n_workers: int = 1`. When >1, shards run in a `ProcessPoolExecutor`
    (spawn context, which is required on Windows; worker functions must be top-level
    and picklable). Set `POLARS_MAX_THREADS` per worker to avoid oversubscription.
- In-memory path: concatenate shard results with `pl.concat(how="vertical")`.
  Categorical columns must share a string cache (`pl.StringCache()`) or be cast
  to `pl.Enum`/`pl.String` before concat.

## Phased plan

Each phase ends with green tests and a commit.

### Phase 0: Baseline and benchmark harness (0.5 to 1 day)

- Create a uv environment for the *current* code and confirm the existing test
  suite passes (`uv run pytest`, plus `--runslow` for integration).
- `benchmarks/make_synthetic_population.py`: replicate the sample
  `decennial_census` (and `taxes_w2_and_1099`, `social_security`) N times with
  offset `simulant_id`/`household_id`/guardian ids, written as multiple
  dictionary-encoded parquet shards matching the real layout (target about 5M to 20M
  rows across about 10 shards). Include a `CHANGELOG.rst` with version 1.4.2 so
  `validate_source_compatibility` passes.
- `benchmarks/run_benchmark.py`: wall time, peak RSS, and a `cProfile`/`py-spy`
  profile for `generate_decennial_census` on the synthetic data.
- Record the baseline in `benchmarks/RESULTS.md`. **Decision point:** confirm where
  the time goes (expected: object-string conversion, `.replace`, `to_string`,
  index set operations, final concat) before investing in Phase 3+.

### Phase 1: uv and packaging (0.5 to 1 day)

- New PEP 621 `pyproject.toml`:
  - build backend: `hatchling` + `hatch-vcs` (or keep setuptools + `setuptools_scm`)
    writing `src/pseudopeople/_version.py`;
  - `requires-python = ">=3.11"` (consider adding 3.13);
  - runtime deps: `polars`, `numpy`, `pyarrow` (only if still needed for
    `to_pandas`/metadata), `pyyaml`, `loguru`, `tqdm`, `packaging`,
    `vivarium-config-tree`. Explicit pins replace the `vivarium_dependencies`
    meta-package. Drop `vivarium-build-utils` (it is currently a *runtime* dep).
  - `[dependency-groups]`: `test` (pytest, pytest-mock, scipy for `FuzzyChecker`,
    pytest-vivarium if still needed), `lint` (ruff), `docs` (sphinx stack).
  - optional extra `pandas` for users who want `.to_pandas()`.
- Move black/isort config to ruff (`line-length = 94`).
- Delete `setup.py`, `python_versions.json`, `tox.ini`, `Jenkinsfile`, and the
  `vivarium_build_utils` `Makefile`; replace with a small `Makefile`/`justfile`
  wrapping `uv sync`, `uv run pytest`, `uv run ruff`.
- Replace `.github/workflows/deploy.yml` (or add `ci.yml`) using
  `astral-sh/setup-uv`. Update `.readthedocs.yaml` to install via uv.
- Commit `uv.lock`.

### Phase 2: Remove Dask (0.5 day)

- `interface.py`: delete the Dask branch and the `engine` parameter and docstrings on all
  7 `generate_*` functions.
- `utilities.py`: delete `Engine`, `PANDAS_ENGINE`, `DASK_ENGINE`,
  `get_engine_from_string`, and the `DataFrame` union.
- `loader.py`: delete the Dask path and the `engine`/`is_file` parameters.
- Tests: remove the `engine` parametrization (about 30 sites in
  `tests/integration/test_interface.py`).
- Docs: remove Dask content from `docs/source/simulated_populations/index.rst` and
  `docs/source/conf.py` intersphinx.
- This is done on pandas *before* the Polars port so the port has one code path.

### Phase 3: Core infrastructure on Polars (2 to 3 days)

- `randomness.py` (new, see above) with unit tests for determinism and rates.
- `dtypes.py`: map `DtypeNames` to Polars dtypes; update `schema_entities.Column`
  so each column declares its Polars output dtype.
- `loader.py`: `load_shard` with Polars filter expressions; drop the index column.
  The `user_filters` tuples become expressions: `pl.col(c) == v`.
- `interface.py` prep: `_clean_input_data` (empty string to null),
  `_reformat_dates_for_noising` (`dt.strftime`), `_coerce_dtypes`
  (`cast` from the schema), all as single `with_columns` calls.
- `entity_types.py`: new contracts and `ColumnNoiseType.__call__` (select, noise
  the subset, scatter back).
- `noise.py`: simplified loop, no missingness frame.
- Lookup tables (`load_nicknames_data`, `load_ocr_errors`, `load_phonetic_errors`,
  `load_qwerty_errors_data`, `load_incorrect_select_options`): load with Polars or
  plain Python and cache as numpy arrays and dicts in the shape the noise functions need.
- `configuration/validator.py`: port `metadata_proportions.csv` handling to Polars;
  replace `pd.isnull`/`pd.Timestamp` uses. `constants/data_values.py`: replace
  `pd.Interval` bins and `pd.Series` with plain lists of `(lower, upper, value)` and
  use `pl.cut` or `pl.when` chains.

### Phase 4: Port noise functions (3 to 5 days)

Row noise:

| Function | Approach |
| --- | --- |
| `omit_rows` | `select_rows` and then `df.filter(~mask)` |
| `apply_do_not_respond` | Build a probability column with expressions (race map via `replace_strict`, age-by-sex bins via `pl.cut`/`when`), clip to [0, 1], `rng.random(n) < p`, filter |
| `duplicate_with_guardian` | Two left joins of dependents to guardian address columns on (`guardian_k`, `year`); choose a guardian with masks; append the selected duplicates with `pl.concat`. This is mostly expression code and should get much faster |

Column noise (functions only see the selected subset, typically 1 to 10% of rows):

| Function | Approach |
| --- | --- |
| `leave_blanks` | `pl.repeat(None, n)` cast to output dtype |
| `choose_wrong_options` | `vectorized_choice` over the option list |
| `copy_from_household_member` | return the `copy_*` column |
| `swap_months_and_days` | `str.slice` + `pl.concat_str` per date format |
| `misreport_ages` | numpy perturbation on the int column |
| `use_fake_names` | `vectorized_choice` |
| `use_nicknames` | `two_d_array_choice` via join with the nickname table |
| `write_wrong_zipcode_digits` | keep the numpy `U5` char-array trick on `to_numpy()` |
| `write_wrong_digits` | keep the numpy char-array approach; replace `pd.DataFrame.isin` with a lookup-table check (`np.isin` on code points, or `np.frombuffer` to `uint32` and compare to a range) |
| `make_typos` | keep the numpy approach; replace the pandas merge with a precomputed `(char, option_index) -> replacement` array lookup |
| `make_ocr_errors`, `make_phonetic_errors` (`_corrupt_tokens`) | keep the algorithm; swap `pd.Series.isin` and `.loc` lookups for dict/array lookups. Optional later: evaluate a pure-Polars `str.replace_many`-style rewrite if profiling shows it matters |

`noise_scaling.py`: `scale_nicknames` and `scale_copy_from_household_member` become
simple Polars aggregations; `scale_choose_wrong_option` is unchanged.

### Phase 5: Interface, output, parallelism (1 to 2 days)

- `_generate_dataset` rewritten around `load_shard` → prep → `noise_dataset` →
  coerce, with `output_path` sinking and an `n_workers` process pool.
- Progress: tqdm over shards (and noise types when single-worker).
- Return types: `pl.DataFrame` for in-memory runs. When `output_path` is given, the return
  value depends on the open question below.
- Keep error behavior: no shards, all shards empty after filters, incompatible data
  version.

### Phase 6: Tests (3 to 5 days)

- Port `tests/conftest.py` fixtures to Polars. `FuzzyChecker` is library-agnostic
  (counts and proportions) and stays.
- `tests/unit/test_column_noise.py` (largest, about 70 index-based assertions): rewrite
  assertions in positional terms. Rate checks carry over directly.
- `tests/unit/test_row_noise.py`, `test_noise_form.py`, `test_utils.py`,
  `test_interface.py`, `test_configuration.py`: port.
- `tests/integration/*`: port; add tests for `output_path` and `n_workers > 1`.
- New: determinism test (same seed gives identical output, and a different shard order
  or worker count gives identical output).

### Phase 7: Benchmark, tune, document (1 day)

- Re-run Phase 0 benchmarks and compare wall time and peak memory.
- Tune: shard-level parallelism vs `POLARS_MAX_THREADS`, Categorical vs String for
  big columns, row-group size of output files.
- Update `README.rst`, `docs/` (installation via uv, new return type, `output_path`,
  `n_workers`), `CHANGELOG.rst`.

Rough total: about 2 to 3 weeks of focused work, dominated by Phases 4 and 6.

## Risks and mitigations

- **Speedup depends on where time actually goes.** Phase 0 measures it first. The
  character-level noise stays numpy and only runs on the selected cells, so it likely
  isn't the bottleneck, but this is verified rather than assumed.
- **Polars streaming won't apply to most noise.** Per-cell random noise and
  character-level corruption are numpy code, not lazy expressions. The scalable
  design is per-shard in-memory processing plus parallel workers plus per-shard
  parquet sinks, not one giant streaming plan.
- **Categorical concat across shards** needs a shared string cache or a cast; covered in Phase 5.
- **Windows multiprocessing** (spawn) needs picklable top-level workers and
  guarded entry points; covered by an integration test.
- **Numpy `U` arrays for long strings** allocate `n * max_len * 4` bytes; this is fine
  on selected subsets, but avoid applying to full columns.
- **Subtle semantic drift** (for example, `""` produced by noise becomes null, or integer
  columns becoming nullable after `leave_blank`, which Polars handles natively without the float
  upcast): document it in the changelog and cover it with tests.

## Open questions

1. With `output_path` set, should `generate_*` return `None`, the list of written
   files, or a `pl.LazyFrame` over them?
2. Add Python 3.13 support while we're changing packaging?
3. Keep an optional `.to_pandas()` convenience/extra, or Polars only?
4. Output file layout: one parquet per input shard (simple, parallel-friendly), or
   a hive-partitioned directory by `state`/`year`?
