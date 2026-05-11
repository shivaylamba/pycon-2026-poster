from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


def load_workload_items(name: str, data_dir: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    if name == "chat":
        items = _load_jsonl(data_dir / "chat_prompts.jsonl")
        return _limit([_chat_item(item) for item in items], limit)
    if name == "summarization":
        items = []
        for path in sorted((data_dir / "documents").glob("*.md")):
            text = path.read_text(encoding="utf-8")
            items.append(
                {
                    "id": path.stem,
                    "prompt": (
                        "Summarize this document for a Python developer. "
                        "Return 4 bullets: purpose, cost drivers, energy drivers, action item.\n\n"
                        f"{text}"
                    ),
                    "system": "You write crisp technical summaries.",
                    "expected": "",
                    "metadata": {"document_path": str(path)},
                }
            )
        return _limit(items, limit)
    if name == "classification":
        rows = _load_csv(data_dir / "classification_examples.csv")
        labels = ["bug", "how-to", "concept", "performance", "security"]
        items = []
        for row in rows:
            items.append(
                {
                    "id": row["id"],
                    "prompt": (
                        "Choose one label for the user message.\n"
                        f"Labels: {', '.join(labels)}\n"
                        f"Message: {row['text']}\n"
                        "Return only the label."
                    ),
                    "system": "You are a careful classifier.",
                    "expected": row["label"],
                    "metadata": {"labels": labels},
                }
            )
        return _limit(items, limit)
    raise ValueError(f"Unknown generation workload: {name}")


def load_embedding_texts(data_dir: Path, limit: int | None = None) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for row in _load_jsonl(data_dir / "rag_corpus.jsonl"):
        items.append({"id": row["id"], "text": row["text"]})
    for path in sorted((data_dir / "documents").glob("*.md")):
        items.append({"id": f"doc:{path.stem}", "text": path.read_text(encoding="utf-8")})
    return _limit(items, limit)


def load_rag_corpus(data_dir: Path, limit: int | None = None) -> List[Dict[str, str]]:
    return _limit(_load_jsonl(data_dir / "rag_corpus.jsonl"), limit)


def load_rag_queries(data_dir: Path, limit: int | None = None) -> List[Dict[str, str]]:
    return _limit(_load_jsonl(data_dir / "rag_queries.jsonl"), limit)


def postprocess_generation(workload: str, text: str, item: Dict[str, Any]) -> Dict[str, Any]:
    started_label = ""
    is_correct = None
    if workload == "classification":
        labels = item.get("metadata", {}).get("labels", [])
        normalized = text.strip().lower().split()[0].strip(".,:;`'\"") if text.strip() else ""
        started_label = normalized
        is_correct = normalized == str(item.get("expected", "")).lower()
        if normalized not in labels:
            is_correct = False
    return {"parsed_label": started_label, "is_correct": is_correct}


def rank_by_cosine(query_vector: Sequence[float], doc_vectors: Sequence[Sequence[float]], top_k: int) -> List[int]:
    query = np.asarray(query_vector, dtype=np.float32)
    docs = np.asarray(doc_vectors, dtype=np.float32)
    query_norm = np.linalg.norm(query) or 1.0
    doc_norms = np.linalg.norm(docs, axis=1)
    doc_norms[doc_norms == 0] = 1.0
    scores = docs @ query / (doc_norms * query_norm)
    return list(np.argsort(-scores)[:top_k])


def _chat_item(item: Dict[str, Any]) -> Dict[str, Any]:
    messages = item.get("messages", [])
    transcript = "\n".join(f"{msg['role'].title()}: {msg['content']}" for msg in messages)
    return {
        "id": item["id"],
        "prompt": f"Continue the conversation as the assistant.\n\n{transcript}\nAssistant:",
        "system": "You are a practical Python mentor.",
        "expected": "",
        "metadata": {},
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _limit(items: Iterable[Any], limit: int | None) -> List[Any]:
    values = list(items)
    if limit is None:
        return values
    return values[:limit]

