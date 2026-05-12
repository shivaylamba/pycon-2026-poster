from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class WorkloadItem:
    workload: str
    item_id: str
    prompt: str
    max_tokens: int
    metadata: Dict[str, Any]


def generation_workloads() -> List[WorkloadItem]:
    return code_generation_items() + summarization_items() + chat_items() + rag_rank_items()


def code_generation_items() -> List[WorkloadItem]:
    tasks = [
        {
            "id": "code_fib",
            "signature": "def fib(n: int) -> int:",
            "tests": "assert fib(0) == 0\nassert fib(1) == 1\nassert fib(7) == 13\nassert fib(10) == 55",
            "desc": "Return the nth Fibonacci number using zero-based indexing.",
        },
        {
            "id": "code_dedupe",
            "signature": "def dedupe_keep_order(items: list[int]) -> list[int]:",
            "tests": "assert dedupe_keep_order([1,2,1,3,2]) == [1,2,3]\nassert dedupe_keep_order([]) == []\nassert dedupe_keep_order([5,5,5]) == [5]",
            "desc": "Remove duplicates while preserving the first occurrence order.",
        },
        {
            "id": "code_slug",
            "signature": "def slugify(text: str) -> str:",
            "tests": "assert slugify('Hello, PyCon US!') == 'hello-pycon-us'\nassert slugify('  GPU   energy ') == 'gpu-energy'\nassert slugify('A+B=C') == 'a-b-c'",
            "desc": "Lowercase text, replace non-alphanumeric runs with one hyphen, and trim hyphens.",
        },
    ]
    items = []
    for task in tasks:
        prompt = (
            "Write only valid Python code. Do not include Markdown.\n"
            f"Task: {task['desc']}\n"
            f"Complete this function:\n{task['signature']}\n"
        )
        items.append(WorkloadItem("code_generation", task["id"], prompt, 180, task))
    return items


def summarization_items() -> List[WorkloadItem]:
    docs = [
        {
            "id": "sum_energy",
            "keywords": ["tokens", "joules", "latency", "batching"],
            "text": (
                "A Python team is moving from hosted LLM calls to local inference. "
                "They care about latency, joules per request, cost per thousand tokens, "
                "and whether batching can improve GPU utilization. The deployment target "
                "is a mix of laptops, RTX workstations, and A100 servers."
            ),
        },
        {
            "id": "sum_quant",
            "keywords": ["quantization", "VRAM", "accuracy", "throughput"],
            "text": (
                "Quantization compresses model weights so that larger language models fit "
                "inside smaller VRAM budgets. Four-bit models can often preserve enough "
                "accuracy for everyday workloads while improving throughput and reducing "
                "energy per generated token."
            ),
        },
        {
            "id": "sum_ops",
            "keywords": ["observability", "power", "inference", "cost"],
            "text": (
                "Infrastructure teams want observability for LLM inference similar to traces "
                "and dashboards used in distributed systems. They need to see tokenization, "
                "model inference, memory movement, power draw, and final cost in one view."
            ),
        },
    ]
    return [
        WorkloadItem(
            "summarization",
            doc["id"],
            "Summarize the document in exactly three concise bullets. Mention concrete cost or infrastructure factors.\n\n"
            f"Document:\n{doc['text']}",
            150,
            doc,
        )
        for doc in docs
    ]


def chat_items() -> List[WorkloadItem]:
    prompts = [
        {
            "id": "chat_choose_model",
            "terms": ["latency", "cost", "quality"],
            "prompt": "A developer asks whether to deploy a 7B or 14B local LLM. Give a practical 4 sentence answer mentioning latency, cost, and quality.",
        },
        {
            "id": "chat_gpu_cpu",
            "terms": ["batch", "gpu", "cpu"],
            "prompt": "Explain when CPU inference is acceptable and when GPU inference becomes economically better. Use plain language for Python developers.",
        },
        {
            "id": "chat_tokens_joule",
            "terms": ["tokens", "joule", "efficiency"],
            "prompt": "Define tokens per joule and explain why it is useful for choosing LLM infrastructure.",
        },
    ]
    return [WorkloadItem("chat_completion", item["id"], item["prompt"], 140, item) for item in prompts]


def rag_rank_items() -> List[WorkloadItem]:
    corpus = [
        ("doc_latency", "Latency waterfalls split LLM request time into network, tokenization, inference, and post-processing."),
        ("doc_quant", "Quantized models reduce VRAM and memory bandwidth pressure, often improving tokens per joule."),
        ("doc_gpu", "GPU batching increases throughput when multiple prompts can share parallel compute resources."),
        ("doc_cost", "Infrastructure cost depends on hardware rental, electricity, request latency, and utilization."),
    ]
    queries = [
        ("rag_quant", "Which document explains why q4 models can use less memory?", "doc_quant"),
        ("rag_latency", "Which document describes the parts of a latency waterfall?", "doc_latency"),
        ("rag_gpu", "Which document talks about batching and GPU throughput?", "doc_gpu"),
    ]
    items: List[WorkloadItem] = []
    corpus_text = "\n".join(f"{doc_id}: {text}" for doc_id, text in corpus)
    for item_id, query, expected in queries:
        prompt = (
            "Choose the best document id for the query. Return only the document id.\n\n"
            f"Documents:\n{corpus_text}\n\nQuery: {query}\nAnswer:"
        )
        items.append(WorkloadItem("semantic_search", item_id, prompt, 24, {"expected_doc_id": expected, "corpus": corpus, "query": query}))
    return items


def embedding_items() -> List[WorkloadItem]:
    corpus = [
        ("doc_latency", "Latency waterfalls split LLM request time into network, tokenization, inference, and post-processing."),
        ("doc_quant", "Quantized models reduce VRAM and memory bandwidth pressure, often improving tokens per joule."),
        ("doc_gpu", "GPU batching increases throughput when multiple prompts can share parallel compute resources."),
        ("doc_cost", "Infrastructure cost depends on hardware rental, electricity, request latency, and utilization."),
    ]
    queries = [
        ("embed_quant", "memory savings from four bit quantization", "doc_quant"),
        ("embed_latency", "break down request latency into stages", "doc_latency"),
        ("embed_gpu", "concurrency and batching on GPUs", "doc_gpu"),
    ]
    return [
        WorkloadItem("embedding_search", item_id, query, 0, {"expected_doc_id": expected, "corpus": corpus})
        for item_id, query, expected in queries
    ]


def evaluate_generation(item: WorkloadItem, response: str) -> Dict[str, Any]:
    if item.workload == "code_generation":
        return evaluate_code(item, response)
    if item.workload == "summarization":
        keywords = [k.lower() for k in item.metadata["keywords"]]
        text = response.lower()
        hits = sum(1 for keyword in keywords if keyword in text)
        return {"score": hits / max(1, len(keywords)), "is_correct": hits >= math.ceil(len(keywords) / 2), "matched_keywords": hits}
    if item.workload == "chat_completion":
        terms = [k.lower() for k in item.metadata["terms"]]
        text = response.lower()
        hits = sum(1 for term in terms if term in text)
        return {"score": hits / max(1, len(terms)), "is_correct": hits == len(terms), "matched_keywords": hits}
    if item.workload == "semantic_search":
        expected = item.metadata["expected_doc_id"]
        found = re.search(r"doc_[a-z]+", response.lower())
        predicted = found.group(0) if found else response.strip().split()[0].strip(".,:;`")[:32].lower()
        return {"score": float(predicted == expected), "is_correct": predicted == expected, "predicted_doc_id": predicted}
    return {"score": 0.0, "is_correct": False}


def evaluate_code(item: WorkloadItem, response: str) -> Dict[str, Any]:
    code = extract_code(response)
    tests = item.metadata["tests"]
    program = f"{code}\n\n{tests}\n"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(["python", str(path)], capture_output=True, text=True, timeout=4)
        except subprocess.TimeoutExpired:
            return {"score": 0.0, "is_correct": False, "error": "timeout"}
    return {"score": float(proc.returncode == 0), "is_correct": proc.returncode == 0, "error": (proc.stderr or proc.stdout)[-300:]}


def extract_code(text: str) -> str:
    block = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if block:
        text = block.group(1)
    lines = text.strip().splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    va = np.array(list(a), dtype=np.float32)
    vb = np.array(list(b), dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def evaluate_embedding_search(item: WorkloadItem, query_vec: List[float], doc_vectors: Dict[str, List[float]]) -> Dict[str, Any]:
    scores = [(doc_id, cosine(query_vec, vec)) for doc_id, vec in doc_vectors.items()]
    scores.sort(key=lambda row: row[1], reverse=True)
    predicted = scores[0][0] if scores else ""
    expected = item.metadata["expected_doc_id"]
    return {
        "score": float(predicted == expected),
        "is_correct": predicted == expected,
        "predicted_doc_id": predicted,
        "top_similarity": scores[0][1] if scores else 0.0,
    }
