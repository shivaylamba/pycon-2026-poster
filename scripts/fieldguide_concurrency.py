from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fieldguide.energy import EnergyTrace
from fieldguide.docker_model_runner_client import DockerModelRunnerClient
from fieldguide.registry import by_id


PROMPT = """You are a concise Python infrastructure assistant.
Explain in 4 bullets when a Python team should deploy a quantized local LLM
instead of calling a hosted API. Mention latency, privacy, concurrency, and cost."""


HARDWARE = {
    "a100_gpu": {"label": "A100 GPU", "options": {}, "cpu_tdp_watts": 120, "gpu_indices": [0]},
    "l40s_gpu": {"label": "L40S GPU", "options": {}, "cpu_tdp_watts": 120, "gpu_indices": [0]},
    "datacenter_gpu": {"label": "Datacenter GPU", "options": {}, "cpu_tdp_watts": 120, "gpu_indices": [0]},
    "consumer_gpu": {"label": "Consumer GPU", "options": {}, "cpu_tdp_watts": 75, "gpu_indices": [0]},
    "cpu": {"label": "CPU only", "options": {"num_gpu": 0}, "cpu_tdp_watts": 120, "gpu_indices": [0]},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small Docker Model Runner concurrency scaling benchmark.")
    parser.add_argument("--model", default="codellama-7b-q4", help="Model id or Docker Model Runner tag from fieldguide registry.")
    parser.add_argument("--hardware", default="a100_gpu,cpu")
    parser.add_argument("--levels", default="1,2,4", help="Comma-separated concurrent request counts.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="http://localhost:12434/v1")
    parser.add_argument("--max-tokens", type=int, default=96)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = DockerModelRunnerClient(args.base_url, timeout_s=1200)
    spec = by_id(args.model)
    rows: List[Dict[str, Any]] = []

    for hw_name in [x.strip() for x in args.hardware.split(",") if x.strip()]:
        profile = HARDWARE[hw_name]
        for concurrency in [int(x) for x in args.levels.split(",") if x.strip()]:
            run_id = f"{int(time.time()*1000)}_{spec.id}_{hw_name}_c{concurrency}"
            started = time.perf_counter()
            with EnergyTrace(interval_s=0.1, gpu_indices=profile["gpu_indices"], cpu_tdp_watts=profile["cpu_tdp_watts"]) as energy:
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [
                        pool.submit(client.generate, spec.docker_model_runner, PROMPT, profile["options"], args.max_tokens)
                        for _ in range(concurrency)
                    ]
                    outputs = []
                    for future in as_completed(futures):
                        outputs.append(future.result())
            wall = time.perf_counter() - started
            summary = energy.summary()
            output_tokens = sum(int(item.get("eval_count") or 0) for item in outputs)
            input_tokens = sum(int(item.get("prompt_eval_count") or 0) for item in outputs)
            rows.append({
                "run_id": run_id,
                "model_id": spec.id,
                "model_label": spec.label,
                "docker_model_runner_model": spec.docker_model_runner,
                "hardware_profile": hw_name,
                "hardware_label": profile["label"],
                "concurrency": concurrency,
                "wall_s": wall,
                "requests": concurrency,
                "requests_per_sec": concurrency / wall if wall > 0 else 0.0,
                "tokens_per_sec": output_tokens / wall if wall > 0 else 0.0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_joule": output_tokens / summary.total_energy_j if summary.total_energy_j else 0.0,
                **summary.as_dict(),
            })
            energy.write_samples(out / "concurrency_power_samples.jsonl", run_id)
            pd.DataFrame(rows).to_csv(out / "concurrency_metrics.csv", index=False)
            print(json.dumps(rows[-1], default=str))
        client.unload(spec.docker_model_runner)


if __name__ == "__main__":
    main()
