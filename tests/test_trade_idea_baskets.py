import unittest

import pandas as pd

from equipilot_app import (
    TRADE_IDEA_EXTENSION_ATR_CAP,
    TRADE_IDEA_MIN_ADV_USD,
    TRADE_IDEA_SCORE_COLUMNS,
    _compute_trade_idea_setup_scores,
    _filter_trade_idea_basket,
)


def _acceleration_row(**overrides: object) -> dict[str, object]:
    """A row that clears every Full Acceleration gate; override to break one gate."""
    row: dict[str, object] = {
        "ticker": "PASS",
        "market_cap_bucket": "Large",
        "fundamental_total_score": 60.0,
        "fundamental_risk": 60.0,
        "fundamental_quality": 60.0,
        "fundamental_momentum": 60.0,
        "stock_rsi_regime_score": 80.0,
        "stock_rsi_regime_20d_vs_50d_flag": "positive",
        "rsi_weekly": 60.0,
        "rsi_daily": 75.0,
        "rs_monthly": 0.5,
        "rs_daily": 12.0,
        "rs_sma20": 10.0,
        "obvm_monthly": 1.0,
        "obvm_weekly": 0.5,
        "obvm_daily": 120.0,
        "obvm_sma20": 100.0,
        "eod_price_used": 105.0,
        "sma_daily_20": 100.0,
        "sma_daily_50": 95.0,
        "sma_daily_200": 90.0,
        "rsi_divergence_daily_flag": "none",
        "rsi_divergence_weekly_flag": "none",
        "atr_vs_ma20": 2.0,
        "adv_usd_20": 50_000_000.0,
    }
    row.update(overrides)
    return row


class TradeIdeaLiquidityGateTests(unittest.TestCase):
    def test_liquidity_floor_excludes_illiquid_names(self) -> None:
        df = pd.DataFrame(
            [
                _acceleration_row(ticker="LIQUID", adv_usd_20=50_000_000.0),
                _acceleration_row(ticker="THIN", adv_usd_20=TRADE_IDEA_MIN_ADV_USD - 1.0),
            ]
        )
        out = _filter_trade_idea_basket(df, "acceleration")
        self.assertEqual(out["ticker"].tolist(), ["LIQUID"])

    def test_liquidity_gate_is_noop_when_adv_absent(self) -> None:
        # A cache predating the volume column yields no ADV at all -> gate must not fire.
        df = pd.DataFrame(
            [
                _acceleration_row(ticker="A"),
                _acceleration_row(ticker="B"),
            ]
        ).drop(columns=["adv_usd_20"])
        out = _filter_trade_idea_basket(df, "acceleration")
        self.assertEqual(sorted(out["ticker"].tolist()), ["A", "B"])


class TradeIdeaAccelerationGateTests(unittest.TestCase):
    def test_extension_cap_excludes_stretched_names(self) -> None:
        df = pd.DataFrame(
            [
                _acceleration_row(ticker="CONTROLLED", atr_vs_ma20=2.0),
                _acceleration_row(
                    ticker="STRETCHED", atr_vs_ma20=TRADE_IDEA_EXTENSION_ATR_CAP + 1.0
                ),
            ]
        )
        out = _filter_trade_idea_basket(df, "acceleration")
        self.assertEqual(out["ticker"].tolist(), ["CONTROLLED"])

    def test_missing_extension_is_allowed_through(self) -> None:
        df = pd.DataFrame([_acceleration_row(ticker="NOEXT")]).drop(columns=["atr_vs_ma20"])
        out = _filter_trade_idea_basket(df, "acceleration")
        self.assertEqual(out["ticker"].tolist(), ["NOEXT"])

    def test_requires_positive_monthly_relative_strength(self) -> None:
        df = pd.DataFrame(
            [
                _acceleration_row(ticker="POS", rs_monthly=0.5),
                _acceleration_row(ticker="FLAT", rs_monthly=-0.05),
            ]
        )
        out = _filter_trade_idea_basket(df, "acceleration")
        self.assertEqual(out["ticker"].tolist(), ["POS"])

    def test_price_must_reclaim_ma20(self) -> None:
        df = pd.DataFrame(
            [
                _acceleration_row(ticker="ABOVE", eod_price_used=105.0, sma_daily_20=100.0),
                _acceleration_row(ticker="BELOW", eod_price_used=98.0, sma_daily_20=100.0),
            ]
        )
        out = _filter_trade_idea_basket(df, "acceleration")
        self.assertEqual(out["ticker"].tolist(), ["ABOVE"])


class TradeIdeaSetupScoreTests(unittest.TestCase):
    def test_scores_are_attached_and_bounded(self) -> None:
        df = pd.DataFrame(
            [
                _acceleration_row(ticker="A"),
                _acceleration_row(ticker="B", rs_daily=11.0, obvm_daily=110.0),
            ]
        )
        out = _filter_trade_idea_basket(df, "acceleration")
        for column in TRADE_IDEA_SCORE_COLUMNS:
            self.assertIn(column, out.columns)
        scores = pd.to_numeric(out["trade_idea_setup_score"], errors="coerce")
        self.assertTrue((scores >= 0.0).all() and (scores <= 100.0).all())

    def test_stronger_setup_scores_higher(self) -> None:
        strong = _acceleration_row(
            ticker="STRONG",
            rs_daily=25.0,
            rs_sma20=10.0,
            rs_monthly=3.0,
            obvm_daily=250.0,
            obvm_sma20=100.0,
            obvm_weekly=3.0,
            obvm_monthly=3.0,
            stock_rsi_regime_score=98.0,
            rsi_daily=82.0,
            atr_vs_ma20=1.0,
        )
        weak = _acceleration_row(
            ticker="WEAK",
            rs_daily=11.0,
            rs_sma20=10.5,
            rs_monthly=0.05,
            obvm_daily=105.0,
            obvm_sma20=100.0,
            obvm_weekly=0.05,
            obvm_monthly=0.05,
            stock_rsi_regime_score=72.0,
            rsi_daily=71.0,
            atr_vs_ma20=3.8,
        )
        out = _filter_trade_idea_basket(pd.DataFrame([weak, strong]), "acceleration")
        ranked = out.set_index("ticker")["trade_idea_setup_score"]
        self.assertGreater(ranked.loc["STRONG"], ranked.loc["WEAK"])

    def test_empty_universe_returns_score_schema(self) -> None:
        scores = _compute_trade_idea_setup_scores(pd.DataFrame(), "acceleration")
        self.assertEqual(list(scores.columns), list(TRADE_IDEA_SCORE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
