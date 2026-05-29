from __future__ import annotations

import hashlib
import math
import random
import time
from typing import Any, Dict, List, Optional, Sequence

from .base import EmbeddingResult, GenerateResult


class MockClient:
    """Deterministic local stand-in used for notebook smoke tests.

    It lets the full benchmark and plotting pipeline run on laptops that do not
    have Docker Model Runner, GPUs, API keys, or model weights installed.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.seed = int(self.config.get("seed", 42))

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        options = options or {}
        model_scale = _model_scale(model)
        hardware_speedup = float(options.get("mock_speedup", 1.0))
        prompt_tokens = _estimate_tokens(prompt) + _estimate_tokens(system or "")
        output_tokens = min(
            int(options.get("num_predict", 128)),
            max(12, int(18 + math.sqrt(max(prompt_tokens, 1)) * model_scale)),
        )
        prompt_tps = 900.0 * hardware_speedup / model_scale
        decode_tps = 85.0 * hardware_speedup / model_scale
        network_s = 0.015 + 0.002 * model_scale
        tokenization_s = prompt_tokens / prompt_tps
        inference_s = output_tokens / decode_tps
        jitter = _stable_unit_interval(f"{self.seed}:{model}:{prompt}") * 0.015
        total_s = network_s + tokenization_s + inference_s + jitter
        time.sleep(min(total_s, float(options.get("mock_max_sleep_s", 0.05))))

        return GenerateResult(
            text=self._fake_text(prompt, output_tokens),
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            timings={
                "request_wall_s": total_s,
                "network_s": network_s,
                "tokenization_s": tokenization_s,
                "inference_s": inference_s + jitter,
                "backend_overhead_s": 0.0,
                "decode_s": inference_s,
            },
            raw={"mock": True, "model_scale": model_scale},
        )

    def embed(
        self,
        model: str,
        texts: Sequence[str],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        options = options or {}
        model_scale = _model_scale(model)
        hardware_speedup = float(options.get("mock_speedup", 1.0))
        input_tokens = sum(_estimate_tokens(text) for text in texts)
        tokenization_s = input_tokens / (1500.0 * hardware_speedup / model_scale)
        inference_s = max(0.002, len(texts) * 0.006 * model_scale / hardware_speedup)
        network_s = 0.012 + 0.001 * model_scale
        total_s = network_s + tokenization_s + inference_s
        time.sleep(min(total_s, float(options.get("mock_max_sleep_s", 0.05))))
        vectors = [_stable_vector(f"{model}:{text}", dims=64) for text in texts]
        return EmbeddingResult(
            vectors=vectors,
            input_tokens=input_tokens,
            timings={
                "request_wall_s": total_s,
                "network_s": network_s,
                "tokenization_s": tokenization_s,
                "inference_s": inference_s,
                "backend_overhead_s": 0.0,
                "decode_s": 0.0,
            },
            raw={"mock": True, "model_scale": model_scale},
        )

    def _fake_text(self, prompt: str, output_tokens: int) -> str:
        lowered = prompt.lower()
        if "choose one label" in lowered or "label:" in lowered:
            labels = ["bug", "how-to", "concept", "performance", "security"]
            return labels[int(_stable_unit_interval(prompt) * len(labels)) % len(labels)]
        if "summarize" in lowered:
            return "Summary: the workload is measurable with latency, token, cost, and energy signals."
        return "This mock response stands in for an LLM answer with reproducible timing metadata."


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.25))


def _model_scale(model: str) -> float:
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()
    return 1.0 + (int(digest[:2], 16) / 255.0) * 2.5


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _stable_vector(text: str, dims: int) -> List[float]:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    values = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]

