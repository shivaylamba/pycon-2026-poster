from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import requests

from .base import EmbeddingResult, GenerateResult


class DockerModelRunnerClient:
    """Small HTTP adapter around Docker Model Runner's local API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.base_url = str(config.get("base_url", "http://localhost:11434")).rstrip("/")
        self.timeout_s = float(config.get("timeout_s", 600))

    def stop(self, model: str) -> None:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except Exception:
            # Stopping is best-effort; a model may not exist for a given adapter.
            return

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        }
        if system:
            payload["system"] = system

        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout_s,
        )
        wall_s = time.perf_counter() - started
        response.raise_for_status()
        raw = response.json()

        prompt_eval_s = _ns_to_s(raw.get("prompt_eval_duration", 0))
        decode_s = _ns_to_s(raw.get("eval_duration", 0))
        backend_total_s = _ns_to_s(raw.get("total_duration", 0))
        load_s = _ns_to_s(raw.get("load_duration", 0))
        backend_overhead_s = max(0.0, backend_total_s - prompt_eval_s - decode_s)
        # Docker Model Runner exposes prompt evaluation and token generation, but not a pure
        # tokenizer timer. For this field guide we name prompt evaluation as the
        # tokenization/prefill bucket and keep the raw value for transparency.
        timings = {
            "request_wall_s": wall_s,
            "network_s": max(0.0, wall_s - backend_total_s),
            "tokenization_s": prompt_eval_s,
            "inference_s": decode_s + backend_overhead_s,
            "backend_overhead_s": backend_overhead_s,
            "decode_s": decode_s,
            "load_s": load_s,
        }
        return GenerateResult(
            text=str(raw.get("response", "")),
            input_tokens=int(raw.get("prompt_eval_count", 0) or 0),
            output_tokens=int(raw.get("eval_count", 0) or 0),
            timings=timings,
            raw=raw,
        )

    def embed(
        self,
        model: str,
        texts: Sequence[str],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        payload = {"model": model, "input": list(texts), "options": options or {}}
        started = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/api/embed",
            json=payload,
            timeout=self.timeout_s,
        )
        if response.status_code == 404:
            return self._embed_legacy(model, texts, options=options, started=started)
        wall_s = time.perf_counter() - started
        response.raise_for_status()
        raw = response.json()
        vectors = raw.get("embeddings", [])
        if vectors and isinstance(vectors[0], (int, float)):
            vectors = [vectors]
        backend_total_s = _ns_to_s(raw.get("total_duration", 0))
        prompt_eval_s = _ns_to_s(raw.get("prompt_eval_duration", 0))
        timings = {
            "request_wall_s": wall_s,
            "network_s": max(0.0, wall_s - backend_total_s) if backend_total_s else wall_s,
            "tokenization_s": prompt_eval_s,
            "inference_s": max(0.0, backend_total_s - prompt_eval_s),
            "backend_overhead_s": 0.0,
            "decode_s": 0.0,
        }
        return EmbeddingResult(
            vectors=vectors,
            input_tokens=int(raw.get("prompt_eval_count", 0) or 0),
            timings=timings,
            raw=raw,
        )

    def _embed_legacy(
        self,
        model: str,
        texts: Sequence[str],
        *,
        options: Optional[Dict[str, Any]] = None,
        started: Optional[float] = None,
    ) -> EmbeddingResult:
        vectors: List[List[float]] = []
        raw_items: List[Dict[str, Any]] = []
        token_count = 0
        if started is None:
            started = time.perf_counter()
        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text, "options": options or {}},
                timeout=self.timeout_s,
            )
            response.raise_for_status()
            raw = response.json()
            raw_items.append(raw)
            vectors.append(raw.get("embedding", []))
            token_count += _rough_token_count(text)
        wall_s = time.perf_counter() - started
        return EmbeddingResult(
            vectors=vectors,
            input_tokens=token_count,
            timings={
                "request_wall_s": wall_s,
                "network_s": wall_s,
                "tokenization_s": 0.0,
                "inference_s": 0.0,
                "backend_overhead_s": 0.0,
                "decode_s": 0.0,
            },
            raw={"legacy": True, "items": raw_items},
        )


def _ns_to_s(value: Any) -> float:
    try:
        return float(value) / 1e9
    except (TypeError, ValueError):
        return 0.0


def _rough_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.25))
