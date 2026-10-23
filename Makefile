.PHONY: install test test-all lint format docs build bench-data bench

install:
	uv sync

test:
	uv run pytest tests/unit

test-all:
	uv run pytest --runslow tests

lint:
	uv run ruff check src tests benchmarks
	uv run ruff format --check src tests benchmarks

format:
	uv run ruff check --fix src tests benchmarks
	uv run ruff format src tests benchmarks

docs:
	uv run --group docs sphinx-build -b html docs/source docs/build/html

build:
	uv build

bench-data:
	uv run python benchmarks/make_synthetic_population.py benchmarks/data/pop100 --replicates 100 --shards 10

bench:
	uv run python benchmarks/run_benchmark.py benchmarks/data/pop100 --dataset decennial_census
