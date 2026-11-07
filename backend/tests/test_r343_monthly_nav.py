from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "data/strategies/ai/.iterations/ai_hs300_multi_factor_rank/evidence/run_r343_gold_trend_proxy.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("r343_monthly_nav", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("round_trip_cost", [0.0, 0.01])
def test_monthly_sleeves_preserve_round_trip_nav(round_trip_cost: float) -> None:
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    curves = tuple(
        pl.DataFrame({"date": days, "value": values})
        for values in ([1.0, 2.0, 1.0], [1.0, 0.5, 1.0], [1.0, 2.0, 1.0])
    )

    curve, weights = _load_module()._combine_three(curves, round_trip_cost, 0.10)

    assert len(weights) == 1
    # All three sleeves return to their starting prices without an intervening
    # monthly rebalance, so neither profit nor terminal turnover can arise.
    assert curve["value"].to_list() == pytest.approx([1.0, 1.325, 1.0])


def test_month_boundary_charges_drift_turnover_once() -> None:
    days = [date(2024, 1, 30), date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 2)]
    curves = tuple(
        pl.DataFrame({"date": days, "value": values})
        for values in ([1.0, 2.0, 2.0, 2.0], [1.0] * 4, [1.0] * 4)
    )

    curve, weights = _load_module()._combine_three(curves, 0.01, 0.10)

    # Initial 45/45/10 holdings become 90/45/10 cents. Rebalancing their
    # $1.45 total sells $0.2475 of A, costing $0.002475 at 1% round-trip cost.
    assert curve["value"].to_list() == pytest.approx([1.0, 1.447525, 1.447525, 1.447525])
    assert [row["date"] for row in weights] == [days[0], days[2]]
