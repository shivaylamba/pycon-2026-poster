from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fieldguide.registry import EMBEDDING_MODELS, specs_for


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Docker Model Runner models for the field guide experiments.")
    parser.add_argument("--experiment", default="all", choices=["all", "A", "B", "C", "D", "E"])
    parser.add_argument("--include-embeddings", action="store_true")
    args = parser.parse_args()
    specs = specs_for(args.experiment, include_embeddings=args.include_embeddings or args.experiment in ("all", "C", "E"))
    seen = set()
    for spec in specs:
        if spec.docker_model_runner in seen:
            continue
        seen.add(spec.docker_model_runner)
        print(f"==> docker model pull {spec.docker_model_runner}")
        proc = subprocess.run(["docker", "model", "pull", spec.docker_model_runner])
        if proc.returncode != 0:
            print(f"FAILED: {spec.docker_model_runner}", file=sys.stderr)


if __name__ == "__main__":
    main()
