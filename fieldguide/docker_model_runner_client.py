from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


class DockerModelRunnerClient:
    """OpenAI-compatible client for Docker Model Runner (DMR).

    DMR does not have its own Python SDK. It exposes an OpenAI-compatible API,
    so we use the standard ``openai`` package to interact with models running
    locally via Docker Desktop.  Default endpoint: ``http://localhost:12434/v1``.
    """

    def __init__(self, base_url: str = "http://localhost:12434/v1", timeout_s: int = 900, api_key: str = "docker") -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

    def generate(self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None, max_tokens: int = 128) -> Dict[str, Any]:
        options = options or {}
        messages = [{"role": "user", "content": prompt}]
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=options.get("temperature", 0.0),
            max_tokens=max_tokens,
            seed=options.get("seed", 42),
        )
        wall_s = time.perf_counter() - started
        usage = response.usage
        text = response.choices[0].message.content or ""
        data = {
            "response": text,
            "prompt_eval_count": int(getattr(usage, "prompt_tokens", 0) or 0),
            "eval_count": int(getattr(usage, "completion_tokens", 0) or 0),
            "request_wall_s": wall_s,
        }
        return data

    def embed(self, model: str, inputs: Iterable[str], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        input_list = list(inputs)
        started = time.perf_counter()
        response = self.client.embeddings.create(model=model, input=input_list)
        wall_s = time.perf_counter() - started
        usage = response.usage
        data = {
            "embeddings": [item.embedding for item in response.data],
            "prompt_eval_count": int(getattr(usage, "prompt_tokens", 0) or 0),
            "request_wall_s": wall_s,
        }
        return data

    def show(self, model: str) -> Dict[str, Any]:
        # DMR does not have a model info endpoint like Ollama's /api/show.
        # Return a minimal stub for compatibility.
        return {"name": model, "backend": "docker_model_runner"}

    def unload(self, model: str) -> None:
        # Docker Model Runner manages model lifecycle automatically;
        # explicit unload is not needed.
        pass


def latency_parts_docker_model_runner(data: Dict[str, Any], wall_s: float) -> Dict[str, float]:
    """Build latency breakdown from a DMR response.

    Because DMR uses the OpenAI-compatible API, server-side timing buckets
    (prompt-eval, decode, load) are not available.  We attribute all latency
    to the wall-clock request time.
    """
    return {
        "network_s": wall_s,
        "load_s": 0.0,
        "tokenization_s": 0.0,
        "inference_s": 0.0,
        "postprocess_s": 0.0,
        "total_latency_s": wall_s,
    }
