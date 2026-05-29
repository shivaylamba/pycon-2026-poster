from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

import requests


class DockerModelRunnerClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_s: int = 900) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None, max_tokens: int = 128) -> Dict[str, Any]:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.0, "num_predict": max_tokens, "seed": 42, **(options or {})},
        }
        started = time.perf_counter()
        response = requests.post(f"{self.base_url}/api/generate", json=body, timeout=self.timeout_s)
        wall_s = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        data["request_wall_s"] = wall_s
        return data

    def embed(self, model: str, inputs: Iterable[str], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = {"model": model, "input": list(inputs), "keep_alive": "10m", "options": options or {}}
        started = time.perf_counter()
        response = requests.post(f"{self.base_url}/api/embed", json=body, timeout=self.timeout_s)
        wall_s = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        data["request_wall_s"] = wall_s
        return data

    def show(self, model: str) -> Dict[str, Any]:
        response = requests.post(f"{self.base_url}/api/show", json={"model": model}, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def unload(self, model: str) -> None:
        try:
            requests.post(f"{self.base_url}/api/generate", json={"model": model, "prompt": "", "keep_alive": 0}, timeout=60)
        except Exception:
            pass


def latency_parts_docker_model_runner(data: Dict[str, Any], wall_s: float) -> Dict[str, float]:
    total = data.get("total_duration", 0) / 1e9 if data.get("total_duration") else wall_s
    load = data.get("load_duration", 0) / 1e9
    prefill = data.get("prompt_eval_duration", 0) / 1e9
    decode = data.get("eval_duration", 0) / 1e9
    post = max(0.0, wall_s - total)
    network = max(0.0, total - load - prefill - decode)
    return {
        "network_s": network,
        "load_s": load,
        "tokenization_s": prefill,
        "inference_s": decode,
        "postprocess_s": post,
        "total_latency_s": wall_s,
    }
