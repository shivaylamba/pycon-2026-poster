from __future__ import annotations

from typing import Any, Dict, Optional


def cost_columns(
    *,
    input_tokens: int,
    output_tokens: int,
    total_energy_j: Optional[float],
    pricing: Dict[str, Any] | None,
    global_costs: Dict[str, Any] | None,
    is_embedding: bool = False,
) -> Dict[str, float]:
    pricing = pricing or {}
    global_costs = global_costs or {}
    input_per_1m = float(pricing.get("input_usd_per_1m_tokens", 0.0) or 0.0)
    output_per_1m = float(pricing.get("output_usd_per_1m_tokens", 0.0) or 0.0)
    embedding_per_1m = float(pricing.get("embedding_usd_per_1m_tokens", input_per_1m) or 0.0)
    if is_embedding:
        api_cost = (input_tokens / 1_000_000.0) * embedding_per_1m
    else:
        api_cost = (input_tokens / 1_000_000.0) * input_per_1m
        api_cost += (output_tokens / 1_000_000.0) * output_per_1m

    electricity = float(global_costs.get("electricity_usd_per_kwh", 0.0) or 0.0)
    energy_cost = 0.0
    if total_energy_j is not None:
        energy_cost = (total_energy_j / 3_600_000.0) * electricity

    hardware_cost = float(global_costs.get("hardware_usd_per_request", 0.0) or 0.0)
    total_cost = api_cost + energy_cost + hardware_cost
    total_tokens = max(1, input_tokens + output_tokens)
    return {
        "api_cost_usd": api_cost,
        "energy_cost_usd": energy_cost,
        "hardware_cost_usd": hardware_cost,
        "total_cost_usd": total_cost,
        "cost_per_1k_tokens_usd": total_cost / total_tokens * 1000.0,
    }

