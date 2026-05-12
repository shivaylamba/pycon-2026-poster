from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .energy import EnergyTrace, hardware_snapshot
from .ollama_client import OllamaClient, latency_parts_ollama
from .registry import EMBEDDING_MODELS, ModelSpec, registry_rows, specs_for
from .workloads import embedding_items, evaluate_embedding_search, evaluate_generation, generation_workloads


HARDWARE_PROFILES: Dict[str, Dict[str, Any]] = {
    "a100_gpu": {"label": "A100 GPU", "ollama_options": {}, "cpu_tdp_watts": 120, "gpu_indices": [0]},
    "l40s_gpu": {"label": "L40S GPU", "ollama_options": {}, "cpu_tdp_watts": 120, "gpu_indices": [0]},
    "datacenter_gpu": {"label": "Datacenter GPU", "ollama_options": {}, "cpu_tdp_watts": 120, "gpu_indices": [0]},
    "consumer_gpu": {"label": "Consumer GPU", "ollama_options": {}, "cpu_tdp_watts": 75, "gpu_indices": [0]},
    "cpu": {"label": "CPU only", "ollama_options": {"num_gpu": 0}, "cpu_tdp_watts": 120, "gpu_indices": [0]},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run systems field guide LLM benchmarks.")
    parser.add_argument("--experiment", default="all", choices=["all", "A", "B", "C", "D", "E"])
    parser.add_argument("--hardware", default="a100_gpu,cpu", help="Comma-separated hardware profiles.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=2, help="Items per workload.")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--models", default="", help="Optional comma-separated model ids or Ollama tags.")
    parser.add_argument("--workloads", default="", help="Optional comma-separated workload names.")
    parser.add_argument("--strict", action="store_true", help="Stop on the first failed model/workload instead of recording an error row.")
    args = parser.parse_args()
    run_benchmarks(args)


def run_benchmarks(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)
    client = OllamaClient(args.base_url)
    include_embeddings = args.experiment in ("all", "C", "E")
    model_specs = filter_specs(specs_for(args.experiment, include_embeddings=False), args.models)
    embedding_specs = filter_specs(EMBEDDING_MODELS if include_embeddings else [], args.models)
    hardware_names = [item.strip() for item in args.hardware.split(",") if item.strip()]
    workload_names = {item.strip() for item in args.workloads.split(",") if item.strip()}
    generation_items = [
        item for item in generation_workloads()
        if workload_index(item) < args.limit and (not workload_names or item.workload in workload_names)
    ]
    embedding_workload_enabled = not workload_names or "embedding_search" in workload_names

    (out / "hardware.json").write_text(json.dumps(hardware_snapshot(), indent=2), encoding="utf-8")
    pd.DataFrame(registry_rows()).to_csv(out / "model_registry.csv", index=False)
    rows: List[Dict[str, Any]] = []
    response_path = out / "responses.jsonl"

    for spec in model_specs:
        if not ensure_model_available(spec, out, args.skip_pull, args.strict):
            continue
        for hw_name in hardware_names:
            profile = HARDWARE_PROFILES[hw_name]
            for repeat in range(args.repeats):
                for item in generation_items:
                    row = safely_run(
                        lambda: run_generation_item(client, spec, item, hw_name, profile, repeat, out, args.save_traces),
                        spec, item.workload, item.item_id, hw_name, profile, repeat, out, args.strict,
                    )
                    rows.append(row)
                    append_jsonl(response_path, {"run_id": row["run_id"], "response": row.get("response", ""), "item_id": item.item_id})
                    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            client.unload(spec.ollama)

    if not embedding_workload_enabled:
        metrics = pd.DataFrame(rows)
        metrics.to_csv(out / "metrics.csv", index=False)
        summarize(metrics).to_csv(out / "summary.csv", index=False)
        return

    for spec in embedding_specs:
        if not ensure_model_available(spec, out, args.skip_pull, args.strict):
            continue
        for hw_name in hardware_names:
            profile = HARDWARE_PROFILES[hw_name]
            for repeat in range(args.repeats):
                for item in embedding_items()[: args.limit]:
                    row = safely_run(
                        lambda: run_embedding_item(client, spec, item, hw_name, profile, repeat, out, args.save_traces),
                        spec, item.workload, item.item_id, hw_name, profile, repeat, out, args.strict,
                    )
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            client.unload(spec.ollama)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    summarize(metrics).to_csv(out / "summary.csv", index=False)


def workload_index(item: Any) -> int:
    try:
        suffix = item.item_id.rsplit("_", 1)[-1]
        return int(suffix)
    except Exception:
        order = {
            "code_fib": 0, "code_dedupe": 1, "code_slug": 2,
            "sum_energy": 0, "sum_quant": 1, "sum_ops": 2,
            "chat_choose_model": 0, "chat_gpu_cpu": 1, "chat_tokens_joule": 2,
            "rag_quant": 0, "rag_latency": 1, "rag_gpu": 2,
        }
        return order.get(item.item_id, 0)


def filter_specs(specs: Sequence[ModelSpec], requested: str) -> List[ModelSpec]:
    wanted = {item.strip() for item in requested.split(",") if item.strip()}
    if not wanted:
        return list(specs)
    return [spec for spec in specs if spec.id in wanted or spec.ollama in wanted or spec.family in wanted]


def ensure_model_available(spec: ModelSpec, out: Path, skip_pull: bool, strict: bool) -> bool:
    try:
        if skip_pull:
            proc = subprocess.run(["ollama", "show", spec.ollama], capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"{spec.ollama} is not available locally. Pull it first or rerun without --skip-pull.")
        else:
            pull_model(spec.ollama, out)
        return True
    except Exception as exc:
        append_jsonl(out / "skipped_models.jsonl", {"model_id": spec.id, "ollama_model": spec.ollama, "error": str(exc)})
        if strict:
            raise
        print(f"[skip] {spec.label}: {exc}")
        return False


def safely_run(fn: Any, spec: ModelSpec, workload: str, item_id: str, hw_name: str, profile: Dict[str, Any], repeat: int, out: Path, strict: bool) -> Dict[str, Any]:
    try:
        row = fn()
        row["status"] = "ok"
        row["error"] = ""
        return row
    except Exception as exc:
        if strict:
            raise
        run_id = f"{int(time.time()*1000)}_{spec.id}_{hw_name}_{item_id}_r{repeat}_error"
        row = {
            **base_row(spec, workload, item_id, hw_name, profile, repeat, run_id),
            "status": "error",
            "error": str(exc),
            "network_s": 0.0,
            "load_s": 0.0,
            "tokenization_s": 0.0,
            "inference_s": 0.0,
            "postprocess_s": 0.0,
            "total_latency_s": 0.0,
            "cpu_energy_j": 0.0,
            "gpu_energy_j": 0.0,
            "total_energy_j": 0.0,
            "gpu_power_avg_w": 0.0,
            "gpu_power_peak_w": 0.0,
            "gpu_mem_peak_mb": 0.0,
            "score": 0.0,
            "is_correct": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "tokens_per_sec": 0.0,
            "tokens_per_joule": 0.0,
            "cost_per_request_usd": 0.0,
            "response": "",
        }
        append_jsonl(out / "errors.jsonl", row)
        print(f"[error] {spec.label} {hw_name} {workload}/{item_id}: {exc}")
        return row


def run_generation_item(client: OllamaClient, spec: ModelSpec, item: Any, hw_name: str, profile: Dict[str, Any], repeat: int, out: Path, save_traces: bool) -> Dict[str, Any]:
    run_id = f"{int(time.time()*1000)}_{spec.id}_{hw_name}_{item.item_id}_r{repeat}"
    options = profile["ollama_options"]
    with EnergyTrace(interval_s=0.1, gpu_indices=profile.get("gpu_indices"), cpu_tdp_watts=profile.get("cpu_tdp_watts")) as energy:
        data = client.generate(spec.ollama, item.prompt, options=options, max_tokens=item.max_tokens)
    summary = energy.summary()
    if save_traces:
        energy.write_samples(out / "traces" / "power_samples.jsonl", run_id)
    response = data.get("response", "")
    eval_row = evaluate_generation(item, response)
    parts = latency_parts_ollama(data, data["request_wall_s"])
    input_tokens = int(data.get("prompt_eval_count") or 0)
    output_tokens = int(data.get("eval_count") or 0)
    total_tokens = input_tokens + output_tokens
    tokens_sec = output_tokens / parts["inference_s"] if parts["inference_s"] > 0 else 0.0
    energy_j = summary.total_energy_j or 0.0
    return {
        **base_row(spec, item.workload, item.item_id, hw_name, profile, repeat, run_id),
        **parts,
        **summary.as_dict(),
        **eval_row,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_sec,
        "tokens_per_joule": output_tokens / energy_j if energy_j > 0 else 0.0,
        "cost_per_request_usd": estimate_cost(summary.elapsed_s, summary.total_energy_j, hw_name),
        "response": response,
    }


def run_embedding_item(client: OllamaClient, spec: ModelSpec, item: Any, hw_name: str, profile: Dict[str, Any], repeat: int, out: Path, save_traces: bool) -> Dict[str, Any]:
    run_id = f"{int(time.time()*1000)}_{spec.id}_{hw_name}_{item.item_id}_r{repeat}"
    docs = item.metadata["corpus"]
    inputs = [item.prompt] + [text for _, text in docs]
    with EnergyTrace(interval_s=0.1, gpu_indices=profile.get("gpu_indices"), cpu_tdp_watts=profile.get("cpu_tdp_watts")) as energy:
        data = client.embed(spec.ollama, inputs, options=profile["ollama_options"])
    summary = energy.summary()
    if save_traces:
        energy.write_samples(out / "traces" / "power_samples.jsonl", run_id)
    embeddings = data.get("embeddings", [])
    query_vec = embeddings[0] if embeddings else []
    doc_vectors = {doc_id: vec for (doc_id, _), vec in zip(docs, embeddings[1:])}
    eval_row = evaluate_embedding_search(item, query_vec, doc_vectors)
    wall = data["request_wall_s"]
    total_tokens = int(data.get("prompt_eval_count") or 0)
    energy_j = summary.total_energy_j or 0.0
    return {
        **base_row(spec, item.workload, item.item_id, hw_name, profile, repeat, run_id),
        "network_s": 0.0,
        "load_s": 0.0,
        "tokenization_s": 0.0,
        "inference_s": wall,
        "postprocess_s": 0.0,
        "total_latency_s": wall,
        **summary.as_dict(),
        **eval_row,
        "input_tokens": total_tokens,
        "output_tokens": 0,
        "total_tokens": total_tokens,
        "tokens_per_sec": len(inputs) / wall if wall > 0 else 0.0,
        "tokens_per_joule": total_tokens / energy_j if energy_j > 0 else 0.0,
        "cost_per_request_usd": estimate_cost(summary.elapsed_s, summary.total_energy_j, hw_name),
        "response": "",
    }


def base_row(spec: ModelSpec, workload: str, item_id: str, hw_name: str, profile: Dict[str, Any], repeat: int, run_id: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp_unix": time.time(),
        "experiment_tags": ",".join(spec.experiments),
        "model_id": spec.id,
        "model_label": spec.label,
        "ollama_model": spec.ollama,
        "family": spec.family,
        "architecture": spec.architecture,
        "params_b": spec.params_b,
        "quantization": spec.quantization,
        "precision_bits": spec.precision_bits,
        "specialization": spec.specialization,
        "hardware_profile": hw_name,
        "hardware_label": profile["label"],
        "workload": workload,
        "item_id": item_id,
        "repeat": repeat,
    }


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    group_cols = ["model_id", "model_label", "family", "architecture", "params_b", "quantization", "hardware_profile", "workload"]
    numeric = [
        "score", "is_correct", "total_latency_s", "tokens_per_sec", "tokens_per_joule", "gpu_energy_j", "cpu_energy_j",
        "total_energy_j", "gpu_power_avg_w", "gpu_power_peak_w", "gpu_mem_peak_mb", "cost_per_request_usd",
        "input_tokens", "output_tokens", "total_tokens",
    ]
    existing = [col for col in numeric if col in metrics.columns]
    return metrics.groupby(group_cols, dropna=False)[existing].mean().reset_index()


def estimate_cost(elapsed_s: float, energy_j: Optional[float], hw_name: str) -> float:
    electricity = ((energy_j or 0.0) / 3_600_000.0) * 0.18
    hourly = {"a100_gpu": 2.50, "l40s_gpu": 1.50, "datacenter_gpu": 2.00, "consumer_gpu": 0.50, "cpu": 0.20}.get(hw_name, 0.0)
    return electricity + hourly * (elapsed_s / 3600.0)


def pull_model(model: str, out: Path) -> None:
    log = out / "pull_log.txt"
    started = time.time()
    proc = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
    append_jsonl(log, {"model": model, "returncode": proc.returncode, "seconds": time.time() - started, "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]})
    if proc.returncode != 0:
        raise RuntimeError(f"ollama pull failed for {model}: {proc.stderr[-500:]}")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
