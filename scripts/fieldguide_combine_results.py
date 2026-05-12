from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine multiple field guide result folders.")
    parser.add_argument("--inputs", required=True, help="Comma-separated result directories.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    inputs = [Path(item.strip()) for item in args.inputs.split(",") if item.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)

    metrics = []
    concurrency = []
    for src in inputs:
        if (src / "metrics.csv").exists():
            metrics.append(pd.read_csv(src / "metrics.csv"))
        if (src / "concurrency_metrics.csv").exists():
            concurrency.append(pd.read_csv(src / "concurrency_metrics.csv"))
        trace = src / "traces" / "power_samples.jsonl"
        if trace.exists():
            with trace.open("r", encoding="utf-8") as handle, (out / "traces" / "power_samples.jsonl").open("a", encoding="utf-8") as target:
                shutil.copyfileobj(handle, target)
        ctrace = src / "concurrency_power_samples.jsonl"
        if ctrace.exists():
            shutil.copy(ctrace, out / "concurrency_power_samples.jsonl")
        for name in ("hardware.json", "model_registry.csv"):
            if (src / name).exists() and not (out / name).exists():
                shutil.copy(src / name, out / name)

    if not metrics:
        raise SystemExit("No metrics.csv files found in inputs")
    pd.concat(metrics, ignore_index=True).drop_duplicates("run_id").to_csv(out / "metrics.csv", index=False)
    if concurrency:
        pd.concat(concurrency, ignore_index=True).drop_duplicates("run_id").to_csv(out / "concurrency_metrics.csv", index=False)


if __name__ == "__main__":
    main()
