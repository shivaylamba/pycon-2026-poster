from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    docker_model_runner: str
    family: str
    architecture: str
    params_b: float
    quantization: str
    precision_bits: int
    specialization: str
    experiments: tuple[str, ...]
    enabled: bool = True
    notes: str = ""

    def as_dict(self) -> Dict[str, object]:
        row = asdict(self)
        row["experiments"] = ",".join(self.experiments)
        return row


MODELS: List[ModelSpec] = [
    # Experiment A: same architecture, different model sizes.
    ModelSpec("gemma-2b-q4", "Gemma 2B Q4", "gemma:2b-instruct-q4_K_M", "Gemma", "Gemma", 2.0, "q4", 4, "general", ("A",)),
    ModelSpec("gemma-7b-q4", "Gemma 7B Q4", "gemma:7b-instruct-q4_K_M", "Gemma", "Gemma", 7.0, "q4", 4, "general", ("A", "B", "C")),
    ModelSpec("phi3-3.8b-q4", "Phi-3 3.8B Q4", "phi3:3.8b-mini-4k-instruct-q4_K_M", "Phi3", "Phi3", 3.8, "q4", 4, "general", ("A",)),
    ModelSpec("phi3-14b-q4", "Phi-3 14B Q4", "phi3:14b-medium-4k-instruct-q4_K_M", "Phi3", "Phi3", 14.0, "q4", 4, "general", ("A", "B")),
    ModelSpec("granite-code-3b-q4", "Granite Code 3B Q4", "granite-code:3b", "Granite", "Granite", 3.0, "q4", 4, "code", ("A",)),
    ModelSpec("granite-code-8b-q4", "Granite Code 8B Q4", "granite-code:8b", "Granite", "Granite", 8.0, "q4", 4, "code", ("A",)),
    ModelSpec("granite-code-20b-q4", "Granite Code 20B Q4", "granite-code:20b", "Granite", "Granite", 20.0, "q4", 4, "code", ("A",)),
    ModelSpec("codellama-7b-q4", "CodeLlama 7B Q4", "codellama:7b-code-q4_K_M", "CodeLlama", "LLaMA", 7.0, "q4", 4, "code", ("A", "B", "C", "D")),
    ModelSpec("codellama-13b-q4", "CodeLlama 13B Q4", "codellama:13b-code-q4_K_M", "CodeLlama", "LLaMA", 13.0, "q4", 4, "code", ("A",)),

    # Experiment B: same model, different quantization.
    ModelSpec("gemma-7b-q8", "Gemma 7B Q8", "gemma:7b-instruct-q8_0", "Gemma", "Gemma", 7.0, "q8", 8, "general", ("B",)),
    ModelSpec("gemma-7b-fp16", "Gemma 7B FP16", "gemma:7b-instruct-fp16", "Gemma", "Gemma", 7.0, "fp16", 16, "general", ("B",)),
    ModelSpec("phi3-14b-q8", "Phi-3 14B Q8", "phi3:14b-medium-4k-instruct-q8_0", "Phi3", "Phi3", 14.0, "q8", 8, "general", ("B",)),
    ModelSpec("phi3-14b-fp16", "Phi-3 14B FP16", "phi3:14b-medium-4k-instruct-fp16", "Phi3", "Phi3", 14.0, "fp16", 16, "general", ("B",)),
    ModelSpec("codellama-7b-q8", "CodeLlama 7B Q8", "codellama:7b-code-q8_0", "CodeLlama", "LLaMA", 7.0, "q8", 8, "code", ("B",)),
    ModelSpec("codellama-7b-fp16", "CodeLlama 7B FP16", "codellama:7b-code-fp16", "CodeLlama", "LLaMA", 7.0, "fp16", 16, "code", ("B",)),

    # Experiment C: similar size, different architectures.
    ModelSpec("mistral-7b-q4", "Mistral 7B Q4", "mistral:7b-instruct-v0.3-q4_K_M", "Mistral", "Mistral", 7.0, "q4", 4, "general", ("C",)),
    ModelSpec("deepseek-coder-6.7b-q4", "DeepSeek Coder 6.7B Q4", "deepseek-coder:6.7b-instruct-q4_K_M", "DeepSeek Coder", "DeepSeek", 6.7, "q4", 4, "code", ("C",)),
]

EMBEDDING_MODELS: List[ModelSpec] = [
    ModelSpec("nomic-embed-text", "Nomic Embed Text", "nomic-embed-text:latest", "Nomic", "Embedding", 0.0, "q4", 4, "embedding", ("C", "E")),
    ModelSpec("mxbai-embed-large", "mxbai Embed Large", "mxbai-embed-large:latest", "mixedbread", "Embedding", 0.0, "q4", 4, "embedding", ("C", "E")),
]


def specs_for(experiment: Optional[str] = None, include_embeddings: bool = False) -> List[ModelSpec]:
    specs: Iterable[ModelSpec] = MODELS + (EMBEDDING_MODELS if include_embeddings else [])
    if not experiment or experiment.lower() == "all":
        return [spec for spec in specs if spec.enabled]
    key = experiment.upper()
    return [spec for spec in specs if spec.enabled and key in spec.experiments]


def by_id(model_id: str) -> ModelSpec:
    for spec in MODELS + EMBEDDING_MODELS:
        if spec.id == model_id or spec.docker_model_runner == model_id:
            return spec
    raise KeyError(model_id)


def registry_rows() -> List[Dict[str, object]]:
    return [spec.as_dict() for spec in MODELS + EMBEDDING_MODELS]
