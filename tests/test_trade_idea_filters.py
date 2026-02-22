import unittest
from pathlib import Path

import pandas as pd

from equipicker_filters import (
    accel_down_weak,
    accel_up_weak,
    extreme_accel_down,
    extreme_accel_up,
)


def _extreme_up_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": "PASS",
        "market_cap": 2_000_000_000,
        "general_technical_score": 90.0,
        "fundamental_total_score": 80.0,
        "rs_daily": 12.0,
        "rs_sma20": 10.0,
        "rs_monthly": 0.2,
        "obvm_daily": 120.0,
        "obvm_sma20": 100.0,
        "obvm_monthly": 1.0,
        "obvm_weekly": 0.5,
        "rsi_weekly": 65.0,
        "rsi_daily": 75.0,
        "eod_price_used": 106.0,
        "sma_daily_20": 100.0,
        "sma_daily_50": 95.0,
        "sma_daily_200": 90.0,
    }
    row.update(overrides)
    return row


def _weak_up_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": "PASS",
        "market_cap": 800_000_000,
        "general_technical_score": 82.0,
        "fundamental_total_score": 70.0,
        "rs_daily": 9.0,
        "rs_sma20": 10.0,
        "rs_monthly": -0.5,
        "obvm_daily": 120.0,
        "obvm_sma20": 100.0,
        "obvm_monthly": 0.4,
        "obvm_weekly": -0.3,
        "rsi_weekly": 64.0,
        "rsi_daily": 68.0,
        "eod_price_used": 101.0,
        "sma_daily_50": 95.0,
        "sma_daily_200": 90.0,
    }
    row.update(overrides)
    return row


class TradeIdeaFilterTests(unittest.TestCase):
    def test_extreme_accel_up_filters_and_market_cap(self) -> None:
        df = pd.DataFrame(
            [
                _extreme_up_row(ticker="PASS"),
                _extreme_up_row(ticker="BAD_PERF", rs_monthly=0.0),
                _extreme_up_row(ticker="BAD_VOLUME", obvm_weekly=0.0),
                _extreme_up_row(ticker="BAD_MOMENTUM", rsi_daily=69.0),
                _extreme_up_row(ticker="BAD_INTERMEDIATE", eod_price_used=99.0),
                _extreme_up_row(ticker="BAD_LONG", sma_daily_50=89.0),
            ]
        )

        out = extreme_accel_up(df, Path("."), save_output=False)

        self.assertEqual(out["ticker"].tolist(), ["PASS"])
        self.assertTrue(out.iloc[0]["market_cap"].endswith("B"))

    def test_accel_up_weak_or_volume_rule(self) -> None:
        df = pd.DataFrame(
            [
                _weak_up_row(ticker="PASS_BY_WEEKLY", obvm_weekly=-0.2, obvm_daily=120.0, obvm_sma20=100.0),
                _weak_up_row(ticker="PASS_BY_DAILY", obvm_weekly=0.2, obvm_daily=90.0, obvm_sma20=100.0),
                _weak_up_row(ticker="FAIL_OR", obvm_weekly=0.2, obvm_daily=120.0, obvm_sma20=100.0),
            ]
        )

        out = accel_up_weak(df, Path("."), save_output=False)

        self.assertEqual(out["ticker"].tolist(), ["PASS_BY_WEEKLY", "PASS_BY_DAILY"])
        self.assertTrue(all(value.endswith("M") for value in out["market_cap"]))

    def test_extreme_accel_down_mirror_rules(self) -> None:
        pass_row = _extreme_up_row(
            ticker="PASS_DOWN",
            rs_daily=8.0,
            rs_sma20=10.0,
            rs_monthly=-0.2,
            obvm_daily=90.0,
            obvm_sma20=100.0,
            obvm_monthly=-1.0,
            obvm_weekly=-0.5,
            rsi_weekly=35.0,
            rsi_daily=25.0,
            eod_price_used=94.0,
            sma_daily_20=100.0,
            sma_daily_50=105.0,
            sma_daily_200=110.0,
        )
        df = pd.DataFrame(
            [
                pass_row,
                {**pass_row, "ticker": "BAD_PERF", "rs_monthly": 0.0},
                {**pass_row, "ticker": "BAD_VOLUME", "obvm_daily": 101.0},
                {**pass_row, "ticker": "BAD_MOMENTUM", "rsi_daily": 31.0},
                {**pass_row, "ticker": "BAD_INTERMEDIATE", "eod_price_used": 101.0},
                {**pass_row, "ticker": "BAD_LONG", "sma_daily_50": 112.5},
            ]
        )

        out = extreme_accel_down(df, Path("."), save_output=False)

        self.assertEqual(out["ticker"].tolist(), ["PASS_DOWN"])

    def test_accel_down_weak_mirror_and_rs_monthly(self) -> None:
        pass_row = _weak_up_row(
            ticker="PASS_DOWN_WEEKLY",
            rs_daily=12.0,
            rs_sma20=10.0,
            rs_monthly=0.5,
            obvm_daily=80.0,
            obvm_sma20=100.0,
            obvm_monthly=-0.4,
            obvm_weekly=0.2,
            rsi_weekly=35.0,
            rsi_daily=32.0,
            eod_price_used=89.0,
            sma_daily_50=95.0,
            sma_daily_200=100.0,
        )
        df = pd.DataFrame(
            [
                pass_row,
                {**pass_row, "ticker": "PASS_DOWN_DAILY", "obvm_weekly": -0.2, "obvm_daily": 120.0},
                {**pass_row, "ticker": "FAIL_RS_MONTHLY", "rs_monthly": 1.1},
                {**pass_row, "ticker": "FAIL_OR", "obvm_weekly": -0.2, "obvm_daily": 80.0},
            ]
        )

        out = accel_down_weak(df, Path("."), save_output=False)

        self.assertEqual(out["ticker"].tolist(), ["PASS_DOWN_DAILY", "PASS_DOWN_WEEKLY"])

    def test_sort_fallback_without_score_columns(self) -> None:
        up_df = pd.DataFrame(
            [
                _extreme_up_row(ticker="B", fundamental_total_score=50.0),
                _extreme_up_row(ticker="A", fundamental_total_score=90.0),
                _extreme_up_row(ticker="C", fundamental_total_score=90.0),
            ]
        ).drop(columns=["general_technical_score"])

        up_out = extreme_accel_up(up_df, Path("."), save_output=False)
        self.assertEqual(up_out["ticker"].tolist(), ["A", "C", "B"])

        down_df = pd.DataFrame(
            [
                _weak_up_row(
                    ticker="C",
                    rs_daily=12.0,
                    rs_sma20=10.0,
                    rs_monthly=0.5,
                    obvm_daily=80.0,
                    obvm_sma20=100.0,
                    obvm_monthly=-0.4,
                    obvm_weekly=0.2,
                    rsi_weekly=35.0,
                    rsi_daily=32.0,
                    eod_price_used=89.0,
                    sma_daily_50=95.0,
                    sma_daily_200=100.0,
                ),
                _weak_up_row(
                    ticker="A",
                    rs_daily=12.0,
                    rs_sma20=10.0,
                    rs_monthly=0.5,
                    obvm_daily=80.0,
                    obvm_sma20=100.0,
                    obvm_monthly=-0.4,
                    obvm_weekly=0.2,
                    rsi_weekly=35.0,
                    rsi_daily=32.0,
                    eod_price_used=89.0,
                    sma_daily_50=95.0,
                    sma_daily_200=100.0,
                ),
            ]
        ).drop(columns=["general_technical_score", "fundamental_total_score"])

        down_out = accel_down_weak(down_df, Path("."), save_output=False)
        self.assertEqual(down_out["ticker"].tolist(), ["A", "C"])


if __name__ == "__main__":
    unittest.main()
