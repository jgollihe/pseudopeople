"""Time (and optionally profile) pseudopeople dataset generation.

Usage:
    uv run python benchmarks/run_benchmark.py SOURCE_DIR --dataset decennial_census [--profile out.prof]
"""

import argparse
import cProfile
import pstats
import threading
import time
from pathlib import Path

import psutil

import pseudopeople as psp

GENERATORS = {
    "decennial_census": psp.generate_decennial_census,
    "american_community_survey": psp.generate_american_community_survey,
    "current_population_survey": psp.generate_current_population_survey,
    "taxes_w2_and_1099": psp.generate_taxes_w2_and_1099,
    "women_infants_and_children": psp.generate_women_infants_and_children,
    "social_security": psp.generate_social_security,
    "taxes_1040": psp.generate_taxes_1040,
}


class PeakMemory:
    def __init__(self, interval: float = 0.05) -> None:
        self.process = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def __enter__(self) -> "PeakMemory":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--dataset", default="decennial_census", choices=GENERATORS)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--extra", nargs="*", default=[], help="key=value kwargs")
    args = parser.parse_args()

    kwargs = {"source": args.source, "year": args.year}
    for item in args.extra:
        key, value = item.split("=", 1)
        kwargs[key] = value

    generator = GENERATORS[args.dataset]
    profiler = cProfile.Profile() if args.profile else None
    with PeakMemory() as memory:
        start = time.perf_counter()
        if profiler:
            profiler.enable()
        result = generator(**kwargs)
        if profiler:
            profiler.disable()
        elapsed = time.perf_counter() - start

    rows = len(result) if result is not None else "n/a (written to disk)"
    print(f"{args.dataset}: {elapsed:.1f}s, peak RSS {memory.peak / 2**30:.2f} GiB, rows {rows}")
    if profiler:
        profiler.dump_stats(args.profile)
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(35)


if __name__ == "__main__":
    main()
