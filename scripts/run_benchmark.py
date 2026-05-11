from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.runner import DEFAULT_WORKLOADS, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run everyday LLM workload benchmarks.")
    parser.add_argument("--config", required=True, help="Path to a YAML/JSON benchmark config.")
    parser.add_argument("--out", default=None, help="Output directory for metrics and responses.")
    parser.add_argument(
        "--workloads",
        nargs="*",
        default=None,
        choices=DEFAULT_WORKLOADS,
        help="Subset of workloads to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per workload.")
    parser.add_argument("--repeats", type=int, default=None, help="Override repeat count.")
    args = parser.parse_args()

    config_path = Path(args.config)
    out_dir = Path(args.out) if args.out else Path("results") / config_path.stem
    metrics_path = run_benchmark(
        config_path=config_path,
        out_dir=out_dir,
        workloads=args.workloads,
        limit=args.limit,
        repeats=args.repeats,
    )
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
