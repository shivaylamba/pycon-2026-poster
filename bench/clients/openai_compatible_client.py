from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Sequence

from .base import EmbeddingResult, GenerateResult


class OpenAICompatibleClient:
    """Adapter for OpenAI-compatible chat and embedding APIs.

    Hosted APIs usually do not expose server-side tokenization or inference
    timers. Rows produced by this adapter therefore use wall time as the
    visible request bucket and include a timing note in ``raw``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional 'openai' package to use this provider.") from exc
        api_key = config.get("api_key") or os.getenv(str(config.get("api_key_env", "OPENAI_API_KEY")))
        base_url = config.get("base_url")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.timeout_s = float(config.get("timeout_s", 600))

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        options = options or {}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=options.get("temperature", 0.1),
            max_tokens=options.get("max_tokens", options.get("num_predict")),
        )
        wall_s = time.perf_counter() - started
        usage = response.usage
        text = response.choices[0].message.content or ""
        return GenerateResult(
            text=text,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            timings={
                "request_wall_s": wall_s,
                "network_s": wall_s,
                "tokenization_s": 0.0,
                "inference_s": 0.0,
                "backend_overhead_s": 0.0,
                "decode_s": 0.0,
            },
            raw={"timing_note": "Hosted API does not expose server-side timing buckets."},
        )

    def embed(
        self,
        model: str,
        texts: Sequence[str],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        started = time.perf_counter()
        response = self.client.embeddings.create(model=model, input=list(texts))
        wall_s = time.perf_counter() - started
        usage = response.usage
        return EmbeddingResult(
            vectors=[item.embedding for item in response.data],
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            timings={
                "request_wall_s": wall_s,
                "network_s": wall_s,
                "tokenization_s": 0.0,
                "inference_s": 0.0,
                "backend_overhead_s": 0.0,
                "decode_s": 0.0,
            },
            raw={"timing_note": "Hosted API does not expose server-side timing buckets."},
        )

