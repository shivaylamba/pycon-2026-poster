from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text)


def deep_merge(*items: Dict[str, Any] | None) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in items:
        if not item:
            continue
        for key, value in item.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
    return result


def configured_names(items: Iterable[Dict[str, Any]]) -> list[str]:
    return [str(item["name"]) for item in items if item.get("enabled", True)]

