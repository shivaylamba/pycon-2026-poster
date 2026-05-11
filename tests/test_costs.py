from bench.costs import cost_columns


def test_cost_columns_combines_tokens_and_energy() -> None:
    row = cost_columns(
        input_tokens=1000,
        output_tokens=500,
        total_energy_j=3600,
        pricing={"input_usd_per_1m_tokens": 1.0, "output_usd_per_1m_tokens": 2.0},
        global_costs={"electricity_usd_per_kwh": 0.10},
    )
    assert round(row["api_cost_usd"], 6) == 0.002
    assert round(row["energy_cost_usd"], 6) == 0.0001
    assert row["total_cost_usd"] > row["api_cost_usd"]

