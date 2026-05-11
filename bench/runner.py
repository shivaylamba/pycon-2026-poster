from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .clients.mock_client import MockClient
from .clients.ollama_client import OllamaClient
from .clients.openai_compatible_client import OpenAICompatibleClient
from .config import deep_merge, load_config
from .costs import cost_columns
from .energy import EnergySampler, hardware_snapshot
from .workloads import (
    load_embedding_texts,
    load_rag_corpus,
    load_rag_queries,
    load_workload_items,
    postprocess_generation,
    rank_by_cosine,
)

GENERATION_WORKLOADS = {"chat", "summarization", "classification"}
EMBEDDING_WORKLOADS = {"batch_embeddings", "rag_rank"}
DEFAULT_WORKLOADS = ["chat", "summarization", "classification", "rag_rank", "batch_embeddings"]


def run_benchmark(
    *,
    config_path: str | Path,
    out_dir: str | Path,
    workloads: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    repeats: Optional[int] = None,
) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "responses.jsonl").write_text("", encoding="utf-8")
    shutil.copy2(config_path, out_path / config_path.name)
    (out_path / "hardware.json").write_text(
        json.dumps(hardware_snapshot(), indent=2),
        encoding="utf-8",
    )

    selected_workloads = list(workloads or config.get("workloads", DEFAULT_WORKLOADS))
    selected_workloads = [name for name in selected_workloads if name in GENERATION_WORKLOADS | EMBEDDING_WORKLOADS]
    run_repeats = int(repeats or config.get("repeats", 1))
    data_dir = (config_path.parent / config.get("data_dir", "../data")).resolve()
    run_id = config.get("run_id") or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    rows: List[Dict[str, Any]] = []
    clients: Dict[str, Any] = {}

    for profile_name, profile in (config.get("hardware_profiles") or {}).items():
        if not profile.get("enabled", True):
            continue
        _stop_configured_models(config, clients)
        for workload in selected_workloads:
            if workload in GENERATION_WORKLOADS:
                rows.extend(
                    _run_generation_workload(
                        config=config,
                        clients=clients,
                        run_id=run_id,
                        profile_name=profile_name,
                        profile=profile,
                        workload=workload,
                        data_dir=data_dir,
                        out_path=out_path,
                        repeats=run_repeats,
                        limit=limit,
                    )
                )
            elif workload == "batch_embeddings":
                rows.extend(
                    _run_batch_embeddings(
                        config=config,
                        clients=clients,
                        run_id=run_id,
                        profile_name=profile_name,
                        profile=profile,
                        data_dir=data_dir,
                        repeats=run_repeats,
                        limit=limit,
                    )
                )
            elif workload == "rag_rank":
                rows.extend(
                    _run_rag_rank(
                        config=config,
                        clients=clients,
                        run_id=run_id,
                        profile_name=profile_name,
                        profile=profile,
                        data_dir=data_dir,
                        repeats=run_repeats,
                        limit=limit,
                    )
                )

    metrics_path = out_path / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    return metrics_path


def _run_generation_workload(
    *,
    config: Dict[str, Any],
    clients: Dict[str, Any],
    run_id: str,
    profile_name: str,
    profile: Dict[str, Any],
    workload: str,
    data_dir: Path,
    out_path: Path,
    repeats: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    items = load_workload_items(workload, data_dir, limit=limit)
    for model in _enabled(config.get("generation_models", [])):
        client = _client_for(model["provider"], config, clients)
        for repeat in range(repeats):
            for item in items:
                options = _options_for(config, profile, model)
                with EnergySampler(**_energy_kwargs(profile)) as sampler:
                    result = client.generate(
                        model["name"],
                        item["prompt"],
                        system=item.get("system"),
                        options=options,
                    )
                    post_started = time.perf_counter()
                    parsed = postprocess_generation(workload, result.text, item)
                    postprocess_s = time.perf_counter() - post_started
                energy = sampler.result()
                row = _base_row(
                    run_id=run_id,
                    profile_name=profile_name,
                    profile=profile,
                    workload=workload,
                    item_id=item["id"],
                    repeat=repeat,
                    model=model,
                    result=result,
                    energy=energy,
                    postprocess_s=postprocess_s,
                    is_embedding=False,
                    config=config,
                )
                row.update(parsed)
                rows.append(row)
                _append_response(out_path / "responses.jsonl", row, result.text, item)
    return rows


def _run_batch_embeddings(
    *,
    config: Dict[str, Any],
    clients: Dict[str, Any],
    run_id: str,
    profile_name: str,
    profile: Dict[str, Any],
    data_dir: Path,
    repeats: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    workload_cfg = (config.get("workload_options") or {}).get("batch_embeddings", {})
    batch_size = int(workload_cfg.get("batch_size", 8))
    items = load_embedding_texts(data_dir, limit=limit)
    for model in _enabled(config.get("embedding_models", [])):
        client = _client_for(model["provider"], config, clients)
        for repeat in range(repeats):
            for batch_index, batch in enumerate(_batches(items, batch_size)):
                options = _options_for(config, profile, model)
                texts = [item["text"] for item in batch]
                with EnergySampler(**_energy_kwargs(profile)) as sampler:
                    result = client.embed(model["name"], texts, options=options)
                    post_started = time.perf_counter()
                    vector_dim = len(result.vectors[0]) if result.vectors else 0
                    postprocess_s = time.perf_counter() - post_started
                energy = sampler.result()
                row = _base_embedding_row(
                    run_id=run_id,
                    profile_name=profile_name,
                    profile=profile,
                    workload="batch_embeddings",
                    item_id=f"batch_{batch_index}",
                    repeat=repeat,
                    model=model,
                    result=result,
                    energy=energy,
                    postprocess_s=postprocess_s,
                    config=config,
                )
                row.update({"batch_size": len(batch), "vector_dim": vector_dim})
                rows.append(row)
    return rows


def _run_rag_rank(
    *,
    config: Dict[str, Any],
    clients: Dict[str, Any],
    run_id: str,
    profile_name: str,
    profile: Dict[str, Any],
    data_dir: Path,
    repeats: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    workload_cfg = (config.get("workload_options") or {}).get("rag_rank", {})
    top_k = int(workload_cfg.get("top_k", 3))
    corpus = load_rag_corpus(data_dir, limit=None)
    queries = load_rag_queries(data_dir, limit=limit)
    corpus_texts = [item["text"] for item in corpus]
    for model in _enabled(config.get("embedding_models", [])):
        client = _client_for(model["provider"], config, clients)
        options = _options_for(config, profile, model)
        corpus_result = client.embed(model["name"], corpus_texts, options=options)
        corpus_vectors = corpus_result.vectors
        for repeat in range(repeats):
            for query in queries:
                with EnergySampler(**_energy_kwargs(profile)) as sampler:
                    result = client.embed(model["name"], [query["query"]], options=options)
                    post_started = time.perf_counter()
                    ranked_indices = rank_by_cosine(result.vectors[0], corpus_vectors, top_k=top_k)
                    retrieved_ids = [corpus[index]["id"] for index in ranked_indices]
                    postprocess_s = time.perf_counter() - post_started
                energy = sampler.result()
                row = _base_embedding_row(
                    run_id=run_id,
                    profile_name=profile_name,
                    profile=profile,
                    workload="rag_rank",
                    item_id=query["id"],
                    repeat=repeat,
                    model=model,
                    result=result,
                    energy=energy,
                    postprocess_s=postprocess_s,
                    config=config,
                )
                row.update(
                    {
                        "query": query["query"],
                        "expected_doc_id": query.get("expected_doc_id", ""),
                        "retrieved_ids": "|".join(retrieved_ids),
                        "retrieved_expected": query.get("expected_doc_id", "") in retrieved_ids,
                    }
                )
                rows.append(row)
    return rows


def _base_row(
    *,
    run_id: str,
    profile_name: str,
    profile: Dict[str, Any],
    workload: str,
    item_id: str,
    repeat: int,
    model: Dict[str, Any],
    result: Any,
    energy: Any,
    postprocess_s: float,
    is_embedding: bool,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    timings = dict(result.timings)
    request_wall_s = float(timings.get("request_wall_s", 0.0) or 0.0)
    input_tokens = int(getattr(result, "input_tokens", 0) or 0)
    output_tokens = int(getattr(result, "output_tokens", 0) or 0)
    total_tokens = int(getattr(result, "total_tokens", input_tokens + output_tokens) or 0)
    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "workload": workload,
        "item_id": item_id,
        "repeat": repeat,
        "provider": model["provider"],
        "model": model["name"],
        "model_label": model.get("label", model["name"]),
        "model_family": model.get("family", ""),
        "model_size": model.get("size", ""),
        "hardware_profile": profile_name,
        "hardware_label": profile.get("label", profile_name),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "request_wall_s": request_wall_s,
        "network_s": float(timings.get("network_s", 0.0) or 0.0),
        "tokenization_s": float(timings.get("tokenization_s", 0.0) or 0.0),
        "inference_s": float(timings.get("inference_s", 0.0) or 0.0),
        "postprocess_s": postprocess_s,
        "decode_s": float(timings.get("decode_s", 0.0) or 0.0),
        "backend_overhead_s": float(timings.get("backend_overhead_s", 0.0) or 0.0),
        "total_latency_s": request_wall_s + postprocess_s,
    }
    row.update(energy.as_dict())
    row.update(
        cost_columns(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            total_energy_j=row.get("total_energy_j"),
            pricing=model.get("pricing", {}),
            global_costs=config.get("costs", {}),
            is_embedding=is_embedding,
        )
    )
    if row["total_tokens"]:
        row["energy_j_per_1k_tokens"] = (row.get("total_energy_j") or 0.0) / row["total_tokens"] * 1000.0
    else:
        row["energy_j_per_1k_tokens"] = 0.0
    return row


def _base_embedding_row(**kwargs: Any) -> Dict[str, Any]:
    return _base_row(is_embedding=True, **kwargs)


def _client_for(provider_name: str, config: Dict[str, Any], clients: Dict[str, Any]) -> Any:
    if provider_name in clients:
        return clients[provider_name]
    provider_cfg = (config.get("providers") or {}).get(provider_name)
    if not provider_cfg:
        raise ValueError(f"Provider '{provider_name}' is not configured.")
    provider_type = provider_cfg.get("type", provider_name)
    if provider_type == "mock":
        client = MockClient(provider_cfg)
    elif provider_type == "ollama":
        client = OllamaClient(provider_cfg)
    elif provider_type in {"openai", "openai_compatible"}:
        client = OpenAICompatibleClient(provider_cfg)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
    clients[provider_name] = client
    return client


def _stop_configured_models(config: Dict[str, Any], clients: Dict[str, Any]) -> None:
    for model in list(_enabled(config.get("generation_models", []))) + list(_enabled(config.get("embedding_models", []))):
        client = _client_for(model["provider"], config, clients)
        stop = getattr(client, "stop", None)
        if callable(stop):
            stop(model["name"])


def _options_for(config: Dict[str, Any], profile: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    provider_name = model["provider"]
    provider_type = (config.get("providers") or {}).get(provider_name, {}).get("type", provider_name)
    profile_provider_options = (profile.get("provider_options") or {}).get(provider_type, {})
    return deep_merge(config.get("default_options", {}), model.get("options", {}), profile_provider_options)


def _energy_kwargs(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "interval_s": float(profile.get("energy_interval_s", 0.1)),
        "gpu_indices": profile.get("gpu_indices"),
        "cpu_tdp_watts": profile.get("cpu_tdp_watts"),
    }


def _append_response(path: Path, row: Dict[str, Any], text: str, item: Dict[str, Any]) -> None:
    record = {
        "run_id": row["run_id"],
        "workload": row["workload"],
        "item_id": row["item_id"],
        "model": row["model"],
        "hardware_profile": row["hardware_profile"],
        "repeat": row["repeat"],
        "response": text,
        "expected": item.get("expected", ""),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _enabled(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in items if item.get("enabled", True)]


def _batches(items: List[Any], size: int) -> Iterable[List[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]
