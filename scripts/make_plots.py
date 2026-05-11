from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.plots import make_all_plots


def main() -> None:
    parser = argparse.ArgumentParser(description="Create poster-ready plots from metrics.csv.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.csv.")
    parser.add_argument("--out", default=None, help="Output directory for figures.")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    out_dir = Path(args.out) if args.out else metrics_path.parent / "figures"
    created = make_all_plots(metrics_path, out_dir)
    print("Created:")
    for path in created:
        print(f"  {path}")


if __name__ == "__main__":
    main()
