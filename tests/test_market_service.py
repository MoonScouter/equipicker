import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from market_service import (
    build_market_signature,
    compute_and_save_market_bundle,
    compute_market_bundle,
    compute_stock_rsi_regime,
    get_default_market_anchors,
    load_market_regime_config,
    load_sector_families,
    market_cache_status,
    resolve_anchor_on_or_before,
)


def _report_df(current_prices: dict[str, float], technical_scores: dict[str, float]) -> pd.DataFrame:
    rows = [
        {
            "ticker": "NVDA.US",
            "sector": "Technology",
            "industry": "Semiconductors",
            "general_technical_score": technical_scores["NVDA.US"],
            "fundamental_total_score": 82.0,
            "fundamental_quality": 40.0,
            "fundamental_risk": 35.0,
            "fundamental_value": 45.0,
            "fundamental_growth": 92.0,
            "fundamental_momentum": 88.0,
            "relative_performance": 86.0,
            "relative_volume": 80.0,
            "eod_price_used": current_prices["NVDA.US"],
            "ic_eod_price_used": current_prices["NVDA.US"],
            "1m_close": current_prices["NVDA.US"] - 10.0,
            "market_cap": 1500.0,
            "rs_monthly": 1.0,
            "obvm_monthly": 1.0,
        },
        {
            "ticker": "TSLA.US",
            "sector": "Consumer Cyclical",
            "industry": "Auto Manufacturers",
            "general_technical_score": technical_scores["TSLA.US"],
            "fundamental_total_score": 70.0,
            "fundamental_quality": 30.0,
            "fundamental_risk": 30.0,
            "fundamental_value": 35.0,
            "fundamental_growth": 80.0,
            "fundamental_momentum": 70.0,
            "relative_performance": 72.0,
            "relative_volume": 68.0,
            "eod_price_used": current_prices["TSLA.US"],
            "ic_eod_price_used": current_prices["TSLA.US"],
            "1m_close": current_prices["TSLA.US"] - 8.0,
            "market_cap": 900.0,
            "rs_monthly": 1.0,
            "obvm_monthly": 1.0,
        },
        {
            "ticker": "JNJ.US",
            "sector": "Healthcare",
            "industry": "Drug Manufacturers",
            "general_technical_score": technical_scores["JNJ.US"],
            "fundamental_total_score": 78.0,
            "fundamental_quality": 80.0,
            "fundamental_risk": 72.0,
            "fundamental_value": 70.0,
            "fundamental_growth": 45.0,
            "fundamental_momentum": 54.0,
            "relative_performance": 45.0,
            "relative_volume": 42.0,
            "eod_price_used": current_prices["JNJ.US"],
            "ic_eod_price_used": current_prices["JNJ.US"],
            "1m_close": current_prices["JNJ.US"] + 2.0,
            "market_cap": 1100.0,
            "rs_monthly": -1.0,
            "obvm_monthly": -1.0,
        },
        {
            "ticker": "PG.US",
            "sector": "Consumer Defensive",
            "industry": "Household Products",
            "general_technical_score": technical_scores["PG.US"],
            "fundamental_total_score": 76.0,
            "fundamental_quality": 82.0,
            "fundamental_risk": 74.0,
            "fundamental_value": 68.0,
            "fundamental_growth": 40.0,
            "fundamental_momentum": 50.0,
            "relative_performance": 48.0,
            "relative_volume": 40.0,
            "eod_price_used": current_prices["PG.US"],
            "ic_eod_price_used": current_prices["PG.US"],
            "1m_close": current_prices["PG.US"] + 1.0,
            "market_cap": 950.0,
            "rs_monthly": -1.0,
            "obvm_monthly": -1.0,
        },
    ]
    return pd.DataFrame(rows)


def _build_price_history(eval_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_rows: list[dict[str, object]] = []
    weekly_rows: list[dict[str, object]] = []
    bullish_daily = [58.0 + idx for idx in range(20)]
    bearish_daily = [42.0 - idx for idx in range(20)]
    bullish_weekly = [45.0, 55.0, 65.0, 72.0]
    bearish_weekly = [55.0, 45.0, 35.0, 25.0]
    for ticker in ("NVDA.US", "TSLA.US"):
        for idx, rsi_value in enumerate(bullish_daily):
            daily_rows.append({"ticker": ticker, "date": eval_date - timedelta(days=19 - idx), "rsi_14": rsi_value})
        for idx, rsi_value in enumerate(bullish_weekly):
            weekly_rows.append({"ticker": ticker, "date": eval_date - timedelta(days=21 - (idx * 7)), "rsi_14": rsi_value})
    for ticker in ("JNJ.US", "PG.US"):
        for idx, rsi_value in enumerate(bearish_daily):
            daily_rows.append({"ticker": ticker, "date": eval_date - timedelta(days=19 - idx), "rsi_14": rsi_value})
        for idx, rsi_value in enumerate(bearish_weekly):
            weekly_rows.append({"ticker": ticker, "date": eval_date - timedelta(days=21 - (idx * 7)), "rsi_14": rsi_value})
    return pd.DataFrame(daily_rows), pd.DataFrame(weekly_rows)


class MarketServiceTests(unittest.TestCase):
    def test_default_market_anchors_use_latest_date_and_prior_offsets(self) -> None:
        config = load_market_regime_config()
        available_dates = [
            date(2026, 3, 1),
            date(2026, 3, 8),
            date(2026, 3, 15),
            date(2026, 3, 22),
        ]

        anchors = get_default_market_anchors(available_dates, config)

        self.assertEqual(anchors["evaluation_date"], date(2026, 3, 22))
        self.assertEqual(anchors["rsi_start_date"], date(2025, 12, 22))
        self.assertEqual(anchors["month_anchor_date"], date(2026, 3, 1))
        self.assertEqual(anchors["week_anchor_date"], date(2026, 3, 15))
        self.assertEqual(resolve_anchor_on_or_before(available_dates, date(2026, 3, 10)), date(2026, 3, 8))

    def test_compute_stock_rsi_regime_marks_sparse_weekly_history_unavailable(self) -> None:
        config = load_market_regime_config()
        evaluation_df = pd.DataFrame(
            [{"ticker": "AAPL.US", "sector": "Technology", "industry": "Software"}]
        )
        daily_prices_df = pd.DataFrame(
            [{"ticker": "AAPL.US", "date": "2026-03-01", "rsi_14": 60.0}]
        )
        weekly_prices_df = pd.DataFrame(
            [
                {"ticker": "AAPL.US", "date": "2026-02-14", "rsi_14": 50.0},
                {"ticker": "AAPL.US", "date": "2026-02-21", "rsi_14": 55.0},
                {"ticker": "AAPL.US", "date": "2026-02-28", "rsi_14": 60.0},
            ]
        )

        result = compute_stock_rsi_regime(
            evaluation_df,
            daily_prices_df,
            weekly_prices_df,
            date(2026, 3, 6),
            date(2025, 12, 6),
            config,
        )

        self.assertTrue(pd.isna(result.iloc[0]["stock_rsi_regime_score"]))
        self.assertIn("at least 4 weekly RSI observations", str(result.iloc[0]["missing_data_reason"]))

    def test_compute_market_bundle_uses_family_average_for_market_sector_rotation(self) -> None:
        config = load_market_regime_config()
        sector_families = load_sector_families()
        evaluation_date = date(2026, 3, 22)
        daily_prices_df, weekly_prices_df = _build_price_history(evaluation_date)
        evaluation_df = _report_df(
            {"NVDA.US": 110.0, "TSLA.US": 108.0, "JNJ.US": 95.0, "PG.US": 96.0},
            {"NVDA.US": 88.0, "TSLA.US": 78.0, "JNJ.US": 52.0, "PG.US": 50.0},
        )
        month_df = _report_df(
            {"NVDA.US": 100.0, "TSLA.US": 100.0, "JNJ.US": 97.0, "PG.US": 98.0},
            {"NVDA.US": 70.0, "TSLA.US": 65.0, "JNJ.US": 55.0, "PG.US": 54.0},
        )
        week_df = _report_df(
            {"NVDA.US": 104.0, "TSLA.US": 103.0, "JNJ.US": 96.0, "PG.US": 97.0},
            {"NVDA.US": 80.0, "TSLA.US": 72.0, "JNJ.US": 53.0, "PG.US": 52.0},
        )

        bundle = compute_market_bundle(
            evaluation_df=evaluation_df,
            month_df=month_df,
            week_df=week_df,
            evaluation_date=evaluation_date,
            rsi_start_date=evaluation_date - timedelta(days=90),
            month_anchor_date=evaluation_date - timedelta(days=30),
            week_anchor_date=evaluation_date - timedelta(days=7),
            evaluation_source_path=Path("report_select_eval.xlsx"),
            month_source_path=Path("report_select_month.xlsx"),
            week_source_path=Path("report_select_week.xlsx"),
            config=config,
            sector_families=sector_families,
            daily_prices_df=daily_prices_df,
            weekly_prices_df=weekly_prices_df,
        )

        family_scores = pd.DataFrame(bundle["market_snapshot_payload"]["family_scores"])
        market_rotation = bundle["market_snapshot_payload"]["market_summary"]["market_sector_rotation_score"]
        self.assertAlmostEqual(market_rotation, family_scores["sector_rotation_score"].mean(), places=6)
        self.assertIn("leading_family_classifier", bundle["market_snapshot_payload"]["market_summary"])

    def test_compute_and_save_market_bundle_reuses_existing_cache(self) -> None:
        config = load_market_regime_config()
        sector_families = load_sector_families()
        evaluation_date = date(2026, 3, 22)
        daily_prices_df, weekly_prices_df = _build_price_history(evaluation_date)
        base_eval_df = _report_df(
            {"NVDA.US": 110.0, "TSLA.US": 108.0, "JNJ.US": 95.0, "PG.US": 96.0},
            {"NVDA.US": 88.0, "TSLA.US": 78.0, "JNJ.US": 52.0, "PG.US": 50.0},
        )
        modified_eval_df = _report_df(
            {"NVDA.US": 210.0, "TSLA.US": 208.0, "JNJ.US": 65.0, "PG.US": 66.0},
            {"NVDA.US": 98.0, "TSLA.US": 88.0, "JNJ.US": 42.0, "PG.US": 40.0},
        )
        month_df = _report_df(
            {"NVDA.US": 100.0, "TSLA.US": 100.0, "JNJ.US": 97.0, "PG.US": 98.0},
            {"NVDA.US": 70.0, "TSLA.US": 65.0, "JNJ.US": 55.0, "PG.US": 54.0},
        )
        week_df = _report_df(
            {"NVDA.US": 104.0, "TSLA.US": 103.0, "JNJ.US": 96.0, "PG.US": 97.0},
            {"NVDA.US": 80.0, "TSLA.US": 72.0, "JNJ.US": 53.0, "PG.US": 52.0},
        )

        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            first = compute_and_save_market_bundle(
                evaluation_df=base_eval_df,
                month_df=month_df,
                week_df=week_df,
                evaluation_date=evaluation_date,
                rsi_start_date=evaluation_date - timedelta(days=90),
                month_anchor_date=evaluation_date - timedelta(days=30),
                week_anchor_date=evaluation_date - timedelta(days=7),
                evaluation_source_path=Path("eval.xlsx"),
                month_source_path=Path("month.xlsx"),
                week_source_path=Path("week.xlsx"),
                config=config,
                sector_families=sector_families,
                daily_prices_df=daily_prices_df,
                weekly_prices_df=weekly_prices_df,
                cache_dir=cache_dir,
            )
            second = compute_and_save_market_bundle(
                evaluation_df=modified_eval_df,
                month_df=month_df,
                week_df=week_df,
                evaluation_date=evaluation_date,
                rsi_start_date=evaluation_date - timedelta(days=90),
                month_anchor_date=evaluation_date - timedelta(days=30),
                week_anchor_date=evaluation_date - timedelta(days=7),
                evaluation_source_path=Path("eval.xlsx"),
                month_source_path=Path("month.xlsx"),
                week_source_path=Path("week.xlsx"),
                config=config,
                sector_families=sector_families,
                daily_prices_df=daily_prices_df,
                weekly_prices_df=weekly_prices_df,
                cache_dir=cache_dir,
            )

            status = market_cache_status(first["signature"], cache_dir)

        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertTrue(status["ready"])
        self.assertEqual(
            first["market_snapshot_payload"]["market_summary"]["market_regime_score"],
            second["market_snapshot_payload"]["market_summary"]["market_regime_score"],
        )

    def test_compute_market_bundle_uses_persistence_from_prior_snapshots(self) -> None:
        config = load_market_regime_config()
        sector_families = load_sector_families()
        base_eval_df = _report_df(
            {"NVDA.US": 110.0, "TSLA.US": 108.0, "JNJ.US": 95.0, "PG.US": 96.0},
            {"NVDA.US": 88.0, "TSLA.US": 78.0, "JNJ.US": 52.0, "PG.US": 50.0},
        )
        month_df = _report_df(
            {"NVDA.US": 100.0, "TSLA.US": 100.0, "JNJ.US": 97.0, "PG.US": 98.0},
            {"NVDA.US": 70.0, "TSLA.US": 65.0, "JNJ.US": 55.0, "PG.US": 54.0},
        )
        week_df = _report_df(
            {"NVDA.US": 104.0, "TSLA.US": 103.0, "JNJ.US": 96.0, "PG.US": 97.0},
            {"NVDA.US": 80.0, "TSLA.US": 72.0, "JNJ.US": 53.0, "PG.US": 52.0},
        )

        with TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir)
            for evaluation_date in (date(2026, 3, 8), date(2026, 3, 15), date(2026, 3, 22)):
                daily_prices_df, weekly_prices_df = _build_price_history(evaluation_date)
                bundle = compute_and_save_market_bundle(
                    evaluation_df=base_eval_df,
                    month_df=month_df,
                    week_df=week_df,
                    evaluation_date=evaluation_date,
                    rsi_start_date=evaluation_date - timedelta(days=90),
                    month_anchor_date=evaluation_date - timedelta(days=30),
                    week_anchor_date=evaluation_date - timedelta(days=7),
                    evaluation_source_path=Path("eval.xlsx"),
                    month_source_path=Path("month.xlsx"),
                    week_source_path=Path("week.xlsx"),
                    config=config,
                    sector_families=sector_families,
                    daily_prices_df=daily_prices_df,
                    weekly_prices_df=weekly_prices_df,
                    cache_dir=cache_dir,
                    force_recompute=True,
                )

        component_scores = bundle["market_snapshot_payload"]["component_scores"]
        self.assertIsNotNone(component_scores["persistence_score"])
        self.assertGreater(component_scores["persistence_score"], 0)
        self.assertTrue(component_scores["persistence_values_used"])

    def test_signature_builder_includes_all_anchor_dates(self) -> None:
        signature = build_market_signature(
            date(2026, 3, 22),
            date(2025, 12, 22),
            date(2026, 2, 20),
            date(2026, 3, 13),
        )

        self.assertEqual(signature, "eval_2026-03-22__rsi_2025-12-22__m1_2026-02-20__w1_2026-03-13")


if __name__ == "__main__":
    unittest.main()
