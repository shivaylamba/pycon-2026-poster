from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


@dataclass
class GenerateResult:
    text: str
    input_tokens: int
    output_tokens: int
    timings: Dict[str, float]
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class EmbeddingResult:
    vectors: List[List[float]]
    input_tokens: int
    timings: Dict[str, float]
    raw: Dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    def generate(
        self,
        model: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> GenerateResult:
        ...

    def embed(
        self,
        model: str,
        texts: Sequence[str],
        *,
        options: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        ...

