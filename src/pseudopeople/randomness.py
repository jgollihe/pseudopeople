"""
Deterministic, vectorized randomness built on Polars hashing.

Every random draw is a pure function of ``(seed, key, row id, ...)``: a row's
draw for a given purpose never depends on which other rows are present or on
the order operations run in. Draws for different keys are independent.

``pl.Expr.hash`` is only stable within a single Polars version, so a seed
reproduces the same output only on the same Polars version.
"""

import hashlib
from dataclasses import dataclass

import polars as pl

ROW_ID = "__pseudopeople_row_id"

_MANTISSA_BITS = 53
_UNIFORM_SCALE = 2.0**-_MANTISSA_BITS
_DISCARDED_BITS_DIVISOR = 2 ** (64 - _MANTISSA_BITS)


def get_hash(key: str) -> int:
    """Stable 63-bit integer hash of a string (independent of the Polars version)."""
    return int.from_bytes(hashlib.sha256(key.encode("utf8")).digest()[:8], "little") >> 1


def _to_unit_interval(hashed: pl.Expr) -> pl.Expr:
    # Keep the top 53 bits so the result is exactly representable and strictly < 1.
    return (hashed // _DISCARDED_BITS_DIVISOR).cast(pl.Float64) * _UNIFORM_SCALE


@dataclass(frozen=True)
class Randomness:
    """Source of keyed random draws for one dataset shard."""

    seed: str

    def key_seed(self, key: str) -> int:
        return get_hash(f"{self.seed}_{key}")

    def uniform(self, key: str, *by: pl.Expr | str) -> pl.Expr:
        """Uniform [0, 1) draw per row, keyed on the row id plus any ``by`` expressions."""
        columns = [pl.col(ROW_ID), *(pl.col(b) if isinstance(b, str) else b for b in by)]
        hashed_input = columns[0] if len(columns) == 1 else pl.struct(columns)
        return _to_unit_interval(hashed_input.hash(self.key_seed(key)))

    def choice_index(self, key: str, n_options: pl.Expr | int, *by: pl.Expr | str) -> pl.Expr:
        """Uniform integer in [0, n_options) per row."""
        if isinstance(n_options, int):
            n_options = pl.lit(n_options)
        return (self.uniform(key, *by) * n_options).floor().cast(pl.Int64)

    def weighted_choice(
        self, key: str, options: list, weights: list[float], *by: pl.Expr | str
    ) -> pl.Expr:
        """Choose among literal ``options`` with the given (unnormalized) ``weights``."""
        total = float(sum(weights))
        draw = self.uniform(key, *by)
        cumulative = 0.0
        expr = None
        for option, weight in zip(options[:-1], weights[:-1]):
            cumulative += weight / total
            branch = draw < cumulative
            expr = pl.when(branch).then(pl.lit(option)) if expr is None else expr.when(branch).then(pl.lit(option))
        last = pl.lit(options[-1])
        return last if expr is None else expr.otherwise(last)

    def from_options(self, key: str, options: list | pl.Series, *by: pl.Expr | str) -> pl.Expr:
        """Uniformly choose one of ``options`` per row."""
        options = options if isinstance(options, pl.Series) else pl.Series(options)
        return pl.lit(options).gather(self.choice_index(key, len(options), *by))
