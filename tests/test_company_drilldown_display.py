import json
import unittest

import pandas as pd
from datetime import date
from pathlib import Path
from unittest.mock import patch

from equipilot_app import (
    TREND_FILTER_LABELS,
    TREND_SYMBOL_DOWN,
    TREND_SYMBOL_UP,
    _annotate_company_score_trends,
    _annotate_company_technical_trend,
    _build_all_companies_filter_state,
    _build_company_filter_state,
    _build_all_thematics_company_universe,
    _build_thematics_basket_metrics,
    _build_thematics_lens_frame,
    _build_thematics_basket_table_frame,
    _build_thematics_company_universe,
    _build_company_drilldown_styler,
    _default_sector_regime_fit_range_for_company_scope,
    _enrich_company_universe_with_market_regime,
    _enrich_company_universe_with_rsi_divergence,
    _filter_company_grid_by_ticker_list,
    _filter_thematics_basket_table_for_view,
    _filter_by_optional_label_value,
    _filter_by_optional_numeric_range,
    _format_divergence_flag,
    _latest_divergence_flags_for_frequency,
    _normalize_grid_visible_columns,
    _company_grid_height,
    _compute_company_return_metrics,
    _load_market_regime_company_metrics_for_date,
    _normalize_thematics_selected_basket,
    _prepare_company_drilldown_universe,
    _use_fast_company_grid_render,
    build_thematics_catalog,
    apply_trend_symbols_to_table,
    build_company_drilldown_context,
    format_thematics_company_display,
    format_company_drilldown_display,
)


class CompanyDrilldownDisplayTests(unittest.TestCase):
    def test_prepare_company_universe_uses_company_and_fallbacks_to_ticker(self) -> None:
        report_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "company": "Alpha Inc",
                    "sector": "Technology",
                    "industry": "Software",
                    "market_cap": 12_000_000_000,
                    "fundamental_total_score": 84.0,
                    "fundamental_momentum": 77.0,
                    "general_technical_score": 88.0,
                    "rs_daily": 12.0,
                    "rs_sma20": 10.0,
                    "obvm_daily": 105.0,
                    "obvm_sma20": 100.0,
                    "rs_monthly": 1.2,
                    "obvm_monthly": -0.4,
                },
                {
                    "ticker": "BBB.US",
                    "company": " ",
                    "sector": "Technology",
                    "industry": "Software",
                    "market_cap": 2_000_000_000,
                    "fundamental_total_score": 70.0,
                    "fundamental_momentum": 55.0,
                    "general_technical_score": 60.0,
                    "rs_daily": 8.0,
                    "rs_sma20": 10.0,
                    "obvm_daily": 95.0,
                    "obvm_sma20": 100.0,
                    "rs_monthly": -0.1,
                    "obvm_monthly": 0.3,
                },
            ]
        )

        prepared, error = _prepare_company_drilldown_universe(report_df)

        self.assertIsNone(error)
        assert prepared is not None
        self.assertEqual(prepared["company"].tolist(), ["Alpha Inc", "BBB.US"])
        self.assertEqual(prepared["fundamental_momentum"].tolist(), [77.0, 55.0])
        self.assertEqual(prepared["rel_strength"].tolist(), ["Positive", "Negative"])
        self.assertEqual(prepared["rel_volume"].tolist(), ["Negative", "Positive"])
        self.assertEqual(prepared["short_term_flow"].tolist(), ["positive", "negative"])
        self.assertEqual(prepared["ai_revenue_exposure"].tolist(), ["none", "none"])
        self.assertEqual(prepared["ai_disruption_risk"].tolist(), ["none", "none"])
        self.assertTrue(prepared["stock_rsi_regime_score"].isna().all())
        self.assertTrue(prepared["sector_regime_fit_score"].isna().all())

    def test_prepare_company_universe_backfills_missing_company_column(self) -> None:
        report_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "sector": "Technology",
                    "industry": "Software",
                    "market_cap": 12_000_000_000,
                    "fundamental_total_score": 84.0,
                    "general_technical_score": 88.0,
                }
            ]
        )

        prepared, error = _prepare_company_drilldown_universe(report_df)

        self.assertIsNone(error)
        assert prepared is not None
        self.assertIn("company", prepared.columns)
        self.assertEqual(prepared.iloc[0]["company"], "AAA.US")
        self.assertIn("fundamental_momentum", prepared.columns)
        self.assertTrue(pd.isna(prepared.iloc[0]["fundamental_momentum"]))
        self.assertIn("stock_rsi_regime_score", prepared.columns)
        self.assertIn("sector_regime_fit_score", prepared.columns)
        self.assertEqual(prepared.iloc[0]["rel_strength"], "N/A")
        self.assertEqual(prepared.iloc[0]["rel_volume"], "N/A")
        self.assertTrue(pd.isna(prepared.iloc[0]["short_term_flow"]))

    def test_prepare_company_universe_classifies_neutral_and_missing_short_term_flow(self) -> None:
        report_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "company": "Alpha",
                    "sector": "Technology",
                    "industry": "Software",
                    "market_cap": 1_000_000_000,
                    "fundamental_total_score": 80.0,
                    "general_technical_score": 75.0,
                    "rs_daily": 12.0,
                    "rs_sma20": 10.0,
                    "obvm_daily": 80.0,
                    "obvm_sma20": 100.0,
                },
                {
                    "ticker": "BBB.US",
                    "company": "Beta",
                    "sector": "Technology",
                    "industry": "Software",
                    "market_cap": 1_000_000_000,
                    "fundamental_total_score": 70.0,
                    "general_technical_score": 65.0,
                    "rs_daily": 12.0,
                    "rs_sma20": None,
                    "obvm_daily": 120.0,
                    "obvm_sma20": 100.0,
                },
            ]
        )

        prepared, error = _prepare_company_drilldown_universe(report_df)

        self.assertIsNone(error)
        assert prepared is not None
        by_ticker = prepared.set_index("ticker")
        self.assertEqual(by_ticker.loc["AAA.US", "short_term_flow"], "neutral")
        self.assertTrue(pd.isna(by_ticker.loc["BBB.US", "short_term_flow"]))

    def test_enrich_company_universe_merges_regime_scores_from_market_cache(self) -> None:
        company_df = pd.DataFrame(
            [
                {"ticker": "AAA.US", "company": "Alpha"},
                {"ticker": "BBB.US", "company": "Beta"},
            ]
        )
        setup_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "stock_rsi_regime_score": 81.5,
                    "sector_regime_fit_score": 67.25,
                }
            ]
        )

        _load_market_regime_company_metrics_for_date.clear()
        with patch("equipilot_app.market_cache_status", return_value={"ready": True}), patch(
            "equipilot_app.load_market_bundle",
            return_value={"setup_readiness_df": setup_df},
        ):
            enriched, warning = _enrich_company_universe_with_market_regime(
                company_df,
                date(2026, 3, 13),
            )

        self.assertIsNone(warning)
        by_ticker = enriched.set_index("ticker")
        self.assertAlmostEqual(float(by_ticker.loc["AAA.US", "stock_rsi_regime_score"]), 81.5)
        self.assertAlmostEqual(float(by_ticker.loc["AAA.US", "sector_regime_fit_score"]), 67.25)
        self.assertTrue(pd.isna(by_ticker.loc["BBB.US", "stock_rsi_regime_score"]))
        self.assertTrue(pd.isna(by_ticker.loc["BBB.US", "sector_regime_fit_score"]))

    def test_enrich_company_universe_warns_and_leaves_regime_scores_empty_when_cache_missing(self) -> None:
        company_df = pd.DataFrame([{"ticker": "AAA.US", "company": "Alpha"}])

        _load_market_regime_company_metrics_for_date.clear()
        with patch("equipilot_app.market_cache_status", return_value={"ready": False}):
            enriched, warning = _enrich_company_universe_with_market_regime(
                company_df,
                date(2026, 3, 20),
            )

        assert warning is not None
        self.assertIn("2026-03-20", warning)
        self.assertTrue(enriched["stock_rsi_regime_score"].isna().all())
        self.assertTrue(enriched["sector_regime_fit_score"].isna().all())

    def test_market_regime_company_metrics_refresh_when_cache_signature_changes(self) -> None:
        evaluation_date = date(2026, 3, 23)
        setup_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "stock_rsi_regime_score": 91.5,
                    "sector_regime_fit_score": 64.0,
                }
            ]
        )

        _load_market_regime_company_metrics_for_date.clear()
        with patch(
            "equipilot_app._market_regime_company_metrics_cache_signature",
            side_effect=["sig-missing", "sig-ready"],
        ), patch(
            "equipilot_app.market_cache_status",
            side_effect=[{"ready": False}, {"ready": True}],
        ), patch(
            "equipilot_app.load_market_bundle",
            return_value={"setup_readiness_df": setup_df},
        ) as load_bundle:
            first_df, first_warning = _load_market_regime_company_metrics_for_date(evaluation_date)
            second_df, second_warning = _load_market_regime_company_metrics_for_date(evaluation_date)

        self.assertTrue(first_df.empty)
        assert first_warning is not None
        self.assertIn("2026-03-23", first_warning)
        self.assertIsNone(second_warning)
        self.assertEqual(load_bundle.call_count, 1)
        self.assertEqual(second_df["ticker"].tolist(), ["AAA.US"])
        self.assertAlmostEqual(float(second_df.iloc[0]["stock_rsi_regime_score"]), 91.5)
        self.assertAlmostEqual(float(second_df.iloc[0]["sector_regime_fit_score"]), 64.0)

    def test_market_regime_company_metrics_clear_discards_stale_cached_missing_result(self) -> None:
        evaluation_date = date(2026, 3, 23)
        setup_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "stock_rsi_regime_score": 88.0,
                    "sector_regime_fit_score": 61.0,
                }
            ]
        )

        _load_market_regime_company_metrics_for_date.clear()
        with patch(
            "equipilot_app._market_regime_company_metrics_cache_signature",
            return_value="stable-sig",
        ), patch("equipilot_app.market_cache_status", return_value={"ready": False}):
            missing_df, missing_warning = _load_market_regime_company_metrics_for_date(evaluation_date)

        self.assertTrue(missing_df.empty)
        assert missing_warning is not None

        _load_market_regime_company_metrics_for_date.clear()
        with patch(
            "equipilot_app._market_regime_company_metrics_cache_signature",
            return_value="stable-sig",
        ), patch(
            "equipilot_app.market_cache_status",
            return_value={"ready": True},
        ), patch(
            "equipilot_app.load_market_bundle",
            return_value={"setup_readiness_df": setup_df},
        ):
            refreshed_df, refreshed_warning = _load_market_regime_company_metrics_for_date(evaluation_date)

        self.assertIsNone(refreshed_warning)
        self.assertEqual(refreshed_df["ticker"].tolist(), ["AAA.US"])
        self.assertAlmostEqual(float(refreshed_df.iloc[0]["stock_rsi_regime_score"]), 88.0)
        self.assertAlmostEqual(float(refreshed_df.iloc[0]["sector_regime_fit_score"]), 61.0)

    def test_optional_numeric_range_filter_relaxes_when_column_has_no_values(self) -> None:
        df = pd.DataFrame(
            [
                {"ticker": "AAA.US", "stock_rsi_regime_score": float("nan")},
                {"ticker": "BBB.US", "stock_rsi_regime_score": float("nan")},
            ]
        )

        filtered, applied = _filter_by_optional_numeric_range(
            df,
            column="stock_rsi_regime_score",
            range_value=(70.0, 100.0),
        )

        self.assertFalse(applied)
        self.assertEqual(filtered["ticker"].tolist(), ["AAA.US", "BBB.US"])

    def test_technical_display_includes_company_column_and_preserves_sort(self) -> None:
        company_df = pd.DataFrame(
            [
                {
                    "ticker": "LOW.US",
                    "company": "Low Tech",
                    "sector": "Technology",
                    "industry": "Software",
                    "market_cap": 1_000_000_000,
                    "fundamental_total_score": 95.0,
                    "fundamental_momentum": 60.0,
                    "general_technical_score": 80.0,
                    "stock_rsi_regime_score": 69.0,
                    "sector_regime_fit_score": 58.0,
                },
                {
                    "ticker": "HIGH.US",
                    "company": "High Tech",
                    "sector": "Technology",
                    "industry": "Hardware",
                    "market_cap": 1_000_000_000,
                    "fundamental_total_score": 70.0,
                    "fundamental_momentum": 72.0,
                    "general_technical_score": 90.0,
                    "stock_rsi_regime_score": 82.0,
                    "sector_regime_fit_score": 74.0,
                },
            ]
        )

        rendered = format_company_drilldown_display(company_df, sort_by="technical")

        self.assertEqual(
            rendered.columns.tolist(),
            [
                "Thematic",
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Market Cap",
                "Beta",
                "TS",
                "RSI Regime",
                "Sector Regime Fit",
                "Short Term Flow",
                "RSI Divergence (D)",
                "RSI Divergence (W)",
                "FS",
                "Mom. FS",
                "Growth FS",
                "Value FS",
                "Quality FS",
                "Risk FS",
                "Rel Strength",
                "Rel Volume",
                "AI Revenue Exposure",
                "AI Disruption Risk",
            ],
        )
        self.assertEqual(rendered.iloc[0]["Thematic"], "Unassigned")
        self.assertEqual(rendered.iloc[0]["Ticker"], "HIGH.US")
        self.assertEqual(rendered.iloc[0]["Company"], "High Tech")
        self.assertEqual(rendered.iloc[0]["Beta"], "N/A")
        self.assertEqual(rendered.iloc[0]["FS"], "70.0")
        self.assertEqual(rendered.iloc[0]["Mom. FS"], "72.0")
        self.assertEqual(rendered.iloc[0]["Growth FS"], "N/A")
        self.assertEqual(rendered.iloc[0]["Value FS"], "N/A")
        self.assertEqual(rendered.iloc[0]["Quality FS"], "N/A")
        self.assertEqual(rendered.iloc[0]["Risk FS"], "N/A")
        self.assertEqual(rendered.iloc[0]["TS"], "90.0")
        self.assertEqual(rendered.iloc[0]["RSI Regime"], "82.0")
        self.assertEqual(rendered.iloc[0]["Sector Regime Fit"], "74.0")
        self.assertEqual(rendered.iloc[0]["Short Term Flow"], "N/A")
        self.assertEqual(rendered.iloc[0]["RSI Divergence (D)"], "N/A")
        self.assertEqual(rendered.iloc[0]["RSI Divergence (W)"], "N/A")

    def test_fundamental_display_now_includes_company_and_momentum(self) -> None:
        company_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "company": "Alpha Inc",
                    "sector": "Technology",
                    "industry": "Software",
                    "market_cap": 1_000_000_000,
                    "fundamental_total_score": 92.0,
                    "fundamental_momentum": 85.0,
                    "general_technical_score": 70.0,
                    "stock_rsi_regime_score": 76.0,
                    "sector_regime_fit_score": 68.0,
                }
            ]
        )

        rendered = format_company_drilldown_display(company_df, sort_by="fundamental")

        self.assertEqual(
            rendered.columns.tolist(),
            [
                "Thematic",
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Market Cap",
                "Beta",
                "TS",
                "RSI Regime",
                "Sector Regime Fit",
                "Short Term Flow",
                "RSI Divergence (D)",
                "RSI Divergence (W)",
                "FS",
                "Mom. FS",
                "Growth FS",
                "Value FS",
                "Quality FS",
                "Risk FS",
                "Rel Strength",
                "Rel Volume",
                "AI Revenue Exposure",
                "AI Disruption Risk",
            ],
        )
        self.assertEqual(rendered.iloc[0]["Thematic"], "Unassigned")
        self.assertEqual(rendered.iloc[0]["Company"], "Alpha Inc")
        self.assertEqual(rendered.iloc[0]["FS"], "92.0")
        self.assertEqual(rendered.iloc[0]["Mom. FS"], "85.0")
        self.assertEqual(rendered.iloc[0]["TS"], "70.0")
        self.assertEqual(rendered.iloc[0]["RSI Regime"], "76.0")
        self.assertEqual(rendered.iloc[0]["Sector Regime Fit"], "68.0")
        self.assertEqual(rendered.iloc[0]["Short Term Flow"], "N/A")
        self.assertEqual(rendered.iloc[0]["RSI Divergence (D)"], "N/A")
        self.assertEqual(rendered.iloc[0]["RSI Divergence (W)"], "N/A")

    def test_fundamental_and_technical_display_include_ai_and_signal_fields(self) -> None:
        company_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "company": "Alpha Inc",
                    "sector": "Technology",
                    "industry": "Software",
                    "market_cap": 1_500_000_000,
                    "fundamental_total_score": 88.0,
                    "fundamental_momentum": 77.0,
                    "general_technical_score": 91.0,
                    "stock_rsi_regime_score": 84.0,
                    "sector_regime_fit_score": 72.0,
                    "short_term_flow": "neutral",
                    "rsi_divergence_daily_flag": "positive",
                    "rsi_divergence_weekly_flag": "negative",
                    "rel_strength": "Positive",
                    "rel_volume": "Negative",
                    "ai_revenue_exposure": "direct",
                    "ai_disruption_risk": "low",
                }
            ]
        )

        fundamental_rendered = format_company_drilldown_display(company_df, sort_by="fundamental")
        technical_rendered = format_company_drilldown_display(company_df, sort_by="technical")

        expected_columns = [
            "Thematic",
            "Ticker",
            "Company",
            "Sector",
            "Industry",
            "Market Cap",
            "Beta",
            "TS",
            "RSI Regime",
            "Sector Regime Fit",
            "Short Term Flow",
            "RSI Divergence (D)",
            "RSI Divergence (W)",
            "FS",
            "Mom. FS",
            "Growth FS",
            "Value FS",
            "Quality FS",
            "Risk FS",
            "Rel Strength",
            "Rel Volume",
            "AI Revenue Exposure",
            "AI Disruption Risk",
        ]
        self.assertEqual(fundamental_rendered.columns.tolist(), expected_columns)
        self.assertEqual(technical_rendered.columns.tolist(), expected_columns)
        self.assertEqual(fundamental_rendered.iloc[0]["Thematic"], "Unassigned")
        self.assertEqual(fundamental_rendered.iloc[0]["RSI Regime"], "84.0")
        self.assertEqual(fundamental_rendered.iloc[0]["Sector Regime Fit"], "72.0")
        self.assertEqual(fundamental_rendered.iloc[0]["Short Term Flow"], "Neutral")
        self.assertEqual(fundamental_rendered.iloc[0]["RSI Divergence (D)"], "Positive")
        self.assertEqual(fundamental_rendered.iloc[0]["RSI Divergence (W)"], "Negative")
        self.assertEqual(fundamental_rendered.iloc[0]["Rel Strength"], "Positive")
        self.assertEqual(fundamental_rendered.iloc[0]["AI Revenue Exposure"], "direct")
        self.assertEqual(technical_rendered.iloc[0]["AI Disruption Risk"], "low")

    def test_company_drilldown_styler_applies_score_and_signal_colors(self) -> None:
        display_df = pd.DataFrame(
            [
                {
                    "Thematic": "AI Infra",
                    "Ticker": "AAA.US",
                    "Company": "Alpha Inc",
                    "Sector": "Technology",
                    "Industry": "Software",
                    "Market Cap": "1.50B",
                    "Beta": "1.1",
                    "TS": "91.0",
                    "RSI Regime": "84.0",
                    "Sector Regime Fit": "72.0",
                    "Short Term Flow": "Neutral",
                    "RSI Divergence (D)": "Positive",
                    "RSI Divergence (W)": "None",
                    "FS": "88.0",
                    "Mom. FS": "77.0",
                    "Growth FS": "69.0",
                    "Value FS": "61.0",
                    "Quality FS": "73.0",
                    "Risk FS": "65.0",
                    "Rel Strength": "Positive",
                    "Rel Volume": "Negative",
                    "AI Revenue Exposure": "indirect",
                    "AI Disruption Risk": "medium",
                }
            ]
        )

        html = _build_company_drilldown_styler(display_df).to_html()

        self.assertIn("#15803D", html)
        self.assertIn("#B42318", html)
        self.assertIn("#B45309", html)

    def test_company_drilldown_styler_colors_confirmed_divergence_labels(self) -> None:
        display_df = pd.DataFrame(
            [
                {
                    "Thematic": "AI Infra",
                    "Ticker": "AAA.US",
                    "Company": "Alpha Inc",
                    "Sector": "Technology",
                    "Industry": "Software",
                    "Market Cap": "1.50B",
                    "Beta": "1.1",
                    "TS": "70.0",
                    "RSI Regime": "50.0",
                    "Sector Regime Fit": "50.0",
                    "Short Term Flow": "N/A",
                    "RSI Divergence (D)": "Positive - Confirmed",
                    "RSI Divergence (W)": "Negative - Confirmed",
                    "FS": "70.0",
                    "Mom. FS": "70.0",
                    "Growth FS": "70.0",
                    "Value FS": "70.0",
                    "Quality FS": "70.0",
                    "Risk FS": "70.0",
                    "Rel Strength": "N/A",
                    "Rel Volume": "N/A",
                    "AI Revenue Exposure": "none",
                    "AI Disruption Risk": "none",
                }
            ]
        )

        html = _build_company_drilldown_styler(display_df).to_html()

        self.assertIn("Positive - Confirmed", html)
        self.assertIn("Negative - Confirmed", html)
        self.assertIn("#5FA777", html)
        self.assertIn("#D97B7B", html)

    def test_enrich_company_universe_merges_daily_and_weekly_divergence_flags(self) -> None:
        company_df = pd.DataFrame(
            [
                {"ticker": "AAA.US", "company": "Alpha"},
                {"ticker": "BBB.US", "company": "Beta"},
            ]
        )
        divergence_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "rsi_divergence_daily_flag": "positive",
                    "rsi_divergence_weekly_flag": "negative",
                }
            ]
        )

        with patch(
            "equipilot_app._load_company_divergence_metrics_for_date",
            return_value=divergence_df,
        ):
            enriched = _enrich_company_universe_with_rsi_divergence(
                company_df,
                date(2026, 3, 13),
            )

        by_ticker = enriched.set_index("ticker")
        self.assertEqual(by_ticker.loc["AAA.US", "rsi_divergence_daily_flag"], "positive")
        self.assertEqual(by_ticker.loc["AAA.US", "rsi_divergence_weekly_flag"], "negative")
        self.assertTrue(pd.isna(by_ticker.loc["BBB.US", "rsi_divergence_daily_flag"]))
        self.assertTrue(pd.isna(by_ticker.loc["BBB.US", "rsi_divergence_weekly_flag"]))

    def test_latest_divergence_flags_use_latest_cached_confirmation_state(self) -> None:
        cache_df = pd.DataFrame(
            [
                {"ticker": "AAA.US", "date": "2026-01-05", "rsi_divergence_flag": "negative", "rsi_divergence_confirmed": True},
                {"ticker": "BBB.US", "date": "2026-01-05", "rsi_divergence_flag": "positive", "rsi_divergence_confirmed": True},
                {"ticker": "CCC.US", "date": "2026-01-05", "rsi_divergence_flag": "positive", "rsi_divergence_confirmed": False},
                {"ticker": "DDD.US", "date": "2026-01-05", "rsi_divergence_flag": "none", "rsi_divergence_confirmed": False},
                {"ticker": "EEE.US", "date": "2026-01-05", "rsi_divergence_flag": "positive", "rsi_divergence_confirmed": pd.NA},
            ]
        )

        with patch(
            "equipilot_app.list_prices_cache_paths",
            return_value=[Path("prices_daily_2026.jsonl")],
        ), patch("equipilot_app.load_prices_cache", return_value=cache_df):
            result = _latest_divergence_flags_for_frequency(
                "daily",
                date(2026, 1, 10),
            )

        by_ticker = result.set_index("ticker")
        self.assertEqual(by_ticker.loc["AAA.US", "rsi_divergence_flag"], "negative-confirmed")
        self.assertEqual(by_ticker.loc["BBB.US", "rsi_divergence_flag"], "positive-confirmed")
        self.assertEqual(by_ticker.loc["CCC.US", "rsi_divergence_flag"], "positive")
        self.assertEqual(by_ticker.loc["DDD.US", "rsi_divergence_flag"], "none")
        self.assertEqual(by_ticker.loc["EEE.US", "rsi_divergence_flag"], "positive")

    def test_latest_divergence_flags_handle_older_cache_without_confirmed_column(self) -> None:
        cache_df = pd.DataFrame(
            [
                {"ticker": "AAA.US", "date": "2026-01-05", "rsi_divergence_flag": "positive"},
                {"ticker": "BBB.US", "date": "2026-01-05", "rsi_divergence_flag": "negative"},
                {"ticker": "CCC.US", "date": "2026-01-05", "rsi_divergence_flag": "none"},
            ]
        )

        with patch(
            "equipilot_app.list_prices_cache_paths",
            return_value=[Path("prices_daily_2026.jsonl")],
        ), patch("equipilot_app.load_prices_cache", return_value=cache_df):
            result = _latest_divergence_flags_for_frequency(
                "daily",
                date(2026, 1, 10),
            )

        by_ticker = result.set_index("ticker")
        self.assertEqual(by_ticker.loc["AAA.US", "rsi_divergence_flag"], "positive")
        self.assertEqual(by_ticker.loc["BBB.US", "rsi_divergence_flag"], "negative")
        self.assertEqual(by_ticker.loc["CCC.US", "rsi_divergence_flag"], "none")

    def test_optional_label_filter_handles_positive_negative_neutral_and_none(self) -> None:
        df = pd.DataFrame(
            [
                {"ticker": "AAA.US", "rsi_divergence_daily_flag": "positive"},
                {"ticker": "BBB.US", "rsi_divergence_daily_flag": "negative"},
                {"ticker": "CCC.US", "rsi_divergence_daily_flag": "none"},
                {"ticker": "DDD.US", "rsi_divergence_daily_flag": pd.NA},
                {"ticker": "EEE.US", "rsi_divergence_daily_flag": "positive-confirmed"},
                {"ticker": "FFF.US", "rsi_divergence_daily_flag": "negative-confirmed"},
            ]
        )
        short_term_df = pd.DataFrame(
            [
                {"ticker": "AAA.US", "short_term_flow": "positive"},
                {"ticker": "BBB.US", "short_term_flow": "negative"},
                {"ticker": "CCC.US", "short_term_flow": "neutral"},
                {"ticker": "DDD.US", "short_term_flow": pd.NA},
            ]
        )

        positive_filtered, positive_applied = _filter_by_optional_label_value(
            df,
            column="rsi_divergence_daily_flag",
            selected_value="Positive",
            label_map={"Positive": "positive", "Negative": "negative", "None": "none"},
        )
        none_filtered, none_applied = _filter_by_optional_label_value(
            df,
            column="rsi_divergence_daily_flag",
            selected_value="None",
            label_map={"Positive": "positive", "Negative": "negative", "None": "none"},
        )
        positive_confirmed_filtered, positive_confirmed_applied = _filter_by_optional_label_value(
            df,
            column="rsi_divergence_daily_flag",
            selected_value="Positive - Confirmed",
            label_map={
                "Positive": "positive",
                "Positive - Confirmed": "positive-confirmed",
                "Negative": "negative",
                "Negative - Confirmed": "negative-confirmed",
                "None": "none",
            },
        )
        neutral_filtered, neutral_applied = _filter_by_optional_label_value(
            short_term_df,
            column="short_term_flow",
            selected_value="Neutral",
            label_map={"Positive": "positive", "Negative": "negative", "Neutral": "neutral"},
        )

        self.assertTrue(positive_applied)
        self.assertEqual(positive_filtered["ticker"].tolist(), ["AAA.US"])
        self.assertTrue(none_applied)
        self.assertEqual(none_filtered["ticker"].tolist(), ["CCC.US"])
        self.assertTrue(positive_confirmed_applied)
        self.assertEqual(positive_confirmed_filtered["ticker"].tolist(), ["EEE.US"])
        self.assertTrue(neutral_applied)
        self.assertEqual(neutral_filtered["ticker"].tolist(), ["CCC.US"])

    def test_format_divergence_flag_supports_confirmed_labels(self) -> None:
        self.assertEqual(_format_divergence_flag("positive-confirmed"), "Positive - Confirmed")
        self.assertEqual(_format_divergence_flag("negative-confirmed"), "Negative - Confirmed")
        self.assertEqual(_format_divergence_flag("positive"), "Positive")
        self.assertEqual(_format_divergence_flag("negative"), "Negative")
        self.assertEqual(_format_divergence_flag("none"), "None")

    def test_annotate_company_technical_trend_builds_symbol_and_direction(self) -> None:
        company_df = pd.DataFrame(
            [
                {"ticker": "UP.US", "general_technical_score": 80.0},
                {"ticker": "DOWN.US", "general_technical_score": 70.0},
                {"ticker": "FLAT.US", "general_technical_score": 65.0},
                {"ticker": "MISS.US", "general_technical_score": 90.0},
            ]
        )
        previous_df = pd.DataFrame(
            [
                {"ticker": "UP.US", "general_technical_score": 70.0},
                {"ticker": "DOWN.US", "general_technical_score": 80.0},
                {"ticker": "FLAT.US", "general_technical_score": 63.0},
            ]
        )

        annotated = _annotate_company_technical_trend(company_df, previous_df, threshold=5.0)

        by_ticker = annotated.set_index("ticker")
        self.assertEqual(by_ticker.loc["UP.US", "technical_trend_symbol"], TREND_SYMBOL_UP)
        self.assertEqual(by_ticker.loc["UP.US", "technical_trend_direction"], "up")
        self.assertEqual(by_ticker.loc["DOWN.US", "technical_trend_symbol"], TREND_SYMBOL_DOWN)
        self.assertEqual(by_ticker.loc["DOWN.US", "technical_trend_direction"], "down")
        self.assertEqual(by_ticker.loc["FLAT.US", "technical_trend_symbol"], "")
        self.assertEqual(by_ticker.loc["FLAT.US", "technical_trend_direction"], "flat")
        self.assertEqual(by_ticker.loc["MISS.US", "technical_trend_direction"], "none")

    def test_technical_display_appends_trend_symbol_to_technical_score(self) -> None:
        company_df = pd.DataFrame(
            [
                {
                    "ticker": "HIGH.US",
                    "company": "High Tech",
                    "sector": "Technology",
                    "industry": "Hardware",
                    "market_cap": 1_000_000_000,
                    "fundamental_total_score": 70.0,
                    "fundamental_momentum": 72.0,
                    "general_technical_score": 90.0,
                    "technical_trend_symbol": TREND_SYMBOL_UP,
                },
                {
                    "ticker": "LOW.US",
                    "company": "Low Tech",
                    "sector": "Technology",
                    "industry": "Software",
                    "market_cap": 1_000_000_000,
                    "fundamental_total_score": 95.0,
                    "fundamental_momentum": 60.0,
                    "general_technical_score": 80.0,
                    "technical_trend_symbol": "",
                },
            ]
        )

        rendered = format_company_drilldown_display(company_df, sort_by="technical")

        self.assertEqual(rendered.iloc[0]["Ticker"], "HIGH.US")
        self.assertEqual(str(rendered.iloc[0]["TS"]).strip(), f"90.0 {TREND_SYMBOL_UP}")
        html = _build_company_drilldown_styler(rendered).to_html()
        self.assertIn(f"90.0 {TREND_SYMBOL_UP}", html)

    def test_thematics_company_display_appends_score_trend_symbols(self) -> None:
        company_df = pd.DataFrame(
            [
                {
                    "thematic": "AI Infra",
                    "ticker": "AAA.US",
                    "company": "Alpha Inc",
                    "sector": "Technology",
                    "industry": "Hardware",
                    "market_cap": 1_000_000_000,
                    "beta": 1.2,
                    "1w_perf": 2.0,
                    "1m_perf": -1.0,
                    "3m_perf": 5.0,
                    "ytd_perf": 10.0,
                    "general_technical_score": 81.0,
                    "stock_rsi_regime_score": 74.0,
                    "sector_regime_fit_score": 66.0,
                    "short_term_flow": "positive",
                    "fundamental_total_score": 76.0,
                    "fundamental_momentum": 68.0,
                    "fundamental_growth": 71.0,
                    "fundamental_value": 62.0,
                    "fundamental_quality": 74.0,
                    "fundamental_risk": 69.0,
                    "technical_trend_symbol": TREND_SYMBOL_UP,
                    "fundamental_trend_symbol": TREND_SYMBOL_DOWN,
                    "fundamental_momentum_trend_symbol": "",
                    "fundamental_growth_trend_symbol": TREND_SYMBOL_UP,
                    "fundamental_value_trend_symbol": TREND_SYMBOL_DOWN,
                    "fundamental_quality_trend_symbol": "",
                    "fundamental_risk_trend_symbol": TREND_SYMBOL_UP,
                    "rel_strength": "Positive",
                    "rel_volume": "Positive",
                    "ai_revenue_exposure": "direct",
                    "ai_disruption_risk": "low",
                }
            ]
        )

        rendered = format_thematics_company_display(company_df)

        self.assertEqual(
            rendered.columns.tolist(),
            [
                "Thematic",
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Market Cap",
                "Beta",
                "1W",
                "1M",
                "3M",
                "YTD",
                "TS",
                "RSI Regime",
                "Sector Regime Fit",
                "Short Term Flow",
                "RSI Divergence (D)",
                "RSI Divergence (W)",
                "FS",
                "Mom. FS",
                "Growth FS",
                "Value FS",
                "Quality FS",
                "Risk FS",
                "Rel Strength",
                "Rel Volume",
                "AI Revenue Exposure",
                "AI Disruption Risk",
            ],
        )
        self.assertEqual(str(rendered.iloc[0]["TS"]).strip(), f"81.0 {TREND_SYMBOL_UP}")
        self.assertAlmostEqual(float(rendered.iloc[0]["RSI Regime"]), 74.0)
        self.assertAlmostEqual(float(rendered.iloc[0]["Sector Regime Fit"]), 66.0)
        self.assertEqual(rendered.iloc[0]["Short Term Flow"], "Positive")
        self.assertEqual(rendered.iloc[0]["RSI Divergence (D)"], "N/A")
        self.assertEqual(rendered.iloc[0]["RSI Divergence (W)"], "N/A")
        self.assertEqual(str(rendered.iloc[0]["FS"]).strip(), f"76.0 {TREND_SYMBOL_DOWN}")
        self.assertEqual(str(rendered.iloc[0]["Mom. FS"]).strip(), "68.0")
        self.assertEqual(str(rendered.iloc[0]["Growth FS"]).strip(), f"71.0 {TREND_SYMBOL_UP}")
        self.assertEqual(str(rendered.iloc[0]["Value FS"]).strip(), f"62.0 {TREND_SYMBOL_DOWN}")
        self.assertEqual(str(rendered.iloc[0]["Quality FS"]).strip(), "74.0")
        self.assertEqual(str(rendered.iloc[0]["Risk FS"]).strip(), f"69.0 {TREND_SYMBOL_UP}")

    def test_thematics_basket_table_frame_includes_fundamental_pillar_scores(self) -> None:
        catalog = {
            "items": {
                "AI": {
                    "parent": "",
                    "is_parent": True,
                    "is_ai_super_parent": True,
                    "is_ai_group_child": False,
                }
            },
            "roots": ["AI"],
        }
        basket_metrics_df = pd.DataFrame(
            [
                {
                    "name": "AI",
                    "beta": 1.1,
                    "1w_perf": 2.0,
                    "1m_perf": 3.0,
                    "3m_perf": 4.0,
                    "ytd_perf": 5.0,
                    "rel_strength_breadth": 60.0,
                    "rel_volume_breadth": 55.0,
                    "technical_scoring": 81.0,
                    "fundamental_scoring": 76.0,
                    "fundamental_momentum_scoring": 68.0,
                    "fundamental_growth_scoring": 71.0,
                    "fundamental_value_scoring": 62.0,
                    "fundamental_quality_scoring": 74.0,
                    "fundamental_risk_scoring": 69.0,
                    "technical_trend_symbol": TREND_SYMBOL_UP,
                    "fundamental_trend_symbol": TREND_SYMBOL_DOWN,
                    "fundamental_momentum_trend_symbol": "",
                    "fundamental_growth_trend_symbol": TREND_SYMBOL_UP,
                    "fundamental_value_trend_symbol": TREND_SYMBOL_DOWN,
                    "fundamental_quality_trend_symbol": "",
                    "fundamental_risk_trend_symbol": TREND_SYMBOL_UP,
                }
            ]
        )

        display_df, meta_df = _build_thematics_basket_table_frame(basket_metrics_df, catalog)

        self.assertEqual(
            display_df.columns.tolist(),
            [
                "Name",
                "Beta",
                "1W",
                "1M",
                "3M",
                "YTD",
                "RS %",
                "Vol %",
                "TS",
                "FS",
                "Mom. FS",
                "Growth FS",
                "Value FS",
                "Quality FS",
                "Risk FS",
            ],
        )
        self.assertEqual(str(display_df.iloc[0]["TS"]).strip(), f"81.0 {TREND_SYMBOL_UP}")
        self.assertEqual(str(display_df.iloc[0]["FS"]).strip(), f"76.0 {TREND_SYMBOL_DOWN}")
        self.assertEqual(str(display_df.iloc[0]["Mom. FS"]).strip(), "68.0")
        self.assertEqual(str(display_df.iloc[0]["Growth FS"]).strip(), f"71.0 {TREND_SYMBOL_UP}")
        self.assertEqual(str(display_df.iloc[0]["Value FS"]).strip(), f"62.0 {TREND_SYMBOL_DOWN}")
        self.assertEqual(str(display_df.iloc[0]["Quality FS"]).strip(), "74.0")
        self.assertEqual(str(display_df.iloc[0]["Risk FS"]).strip(), f"69.0 {TREND_SYMBOL_UP}")
        self.assertEqual(meta_df.iloc[0]["basket_name"], "AI")

    def test_annotate_company_score_trends_adds_pillar_trend_symbols(self) -> None:
        company_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "general_technical_score": 81.0,
                    "fundamental_total_score": 76.0,
                    "fundamental_momentum": 68.0,
                    "fundamental_growth": 71.0,
                    "fundamental_value": 62.0,
                    "fundamental_quality": 74.0,
                    "fundamental_risk": 69.0,
                }
            ]
        )
        previous_report_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "general_technical_score": 72.0,
                    "fundamental_total_score": 83.0,
                    "fundamental_momentum": 68.0,
                    "fundamental_growth": 63.0,
                    "fundamental_value": 69.0,
                    "fundamental_quality": 72.0,
                    "fundamental_risk": 61.0,
                }
            ]
        )

        annotated = _annotate_company_score_trends(company_df, previous_report_df, threshold=5.0)

        self.assertEqual(annotated.iloc[0]["fundamental_trend_symbol"], TREND_SYMBOL_DOWN)
        self.assertEqual(annotated.iloc[0]["fundamental_momentum_trend_symbol"], "")
        self.assertEqual(annotated.iloc[0]["fundamental_growth_trend_symbol"], TREND_SYMBOL_UP)
        self.assertEqual(annotated.iloc[0]["fundamental_value_trend_symbol"], TREND_SYMBOL_DOWN)
        self.assertEqual(annotated.iloc[0]["fundamental_quality_trend_symbol"], "")
        self.assertEqual(annotated.iloc[0]["fundamental_risk_trend_symbol"], TREND_SYMBOL_UP)

    def test_normalize_grid_visible_columns_falls_back_to_defaults_for_invalid_saved_layouts(self) -> None:
        available_columns = ["Ticker", "Company", "FS", "TS"]

        self.assertEqual(
            _normalize_grid_visible_columns([], available_columns),
            available_columns,
        )
        self.assertEqual(
            _normalize_grid_visible_columns(["Unknown"], available_columns),
            available_columns,
        )
        self.assertEqual(
            _normalize_grid_visible_columns(["TS", "Ticker", "Missing"], available_columns),
            ["TS", "Ticker"],
        )

    def test_normalize_grid_visible_columns_keeps_locked_columns_and_order(self) -> None:
        available_columns = ["Name", "TS", "FS", "Mom. FS"]

        self.assertEqual(
            _normalize_grid_visible_columns(
                ["FS", "TS"],
                available_columns,
                locked_columns=["Name"],
            ),
            ["Name", "FS", "TS"],
        )
        self.assertEqual(
            _normalize_grid_visible_columns(
                [],
                available_columns,
                locked_columns=["Name"],
            ),
            available_columns,
        )


    def test_apply_trend_symbols_to_table_formats_threshold_crossings_and_missing_previous(self) -> None:
        current_table = pd.DataFrame(
            [
                {"Sector": "Up", "Total": 20.0},
                {"Sector": "Down", "Total": 10.0},
                {"Sector": "Flat", "Total": 15.0},
                {"Sector": "Missing", "Total": 18.0},
            ]
        )
        previous_table = pd.DataFrame(
            [
                {"Sector": "Up", "Total": 10.0},
                {"Sector": "Down", "Total": 20.0},
                {"Sector": "Flat", "Total": 12.0},
            ]
        )

        rendered = apply_trend_symbols_to_table(current_table, previous_table, ["Total"], threshold=5.0)

        by_sector = rendered.set_index("Sector")
        self.assertEqual(by_sector.loc["Up", "Total"], f"20.0 {TREND_SYMBOL_UP}")
        self.assertEqual(by_sector.loc["Down", "Total"], f"10.0 {TREND_SYMBOL_DOWN}")
        self.assertEqual(by_sector.loc["Flat", "Total"], "15.0")
        self.assertEqual(by_sector.loc["Missing", "Total"], "18.0")

    def test_build_all_companies_filter_state_removes_scope_restrictions(self) -> None:
        state = _build_all_companies_filter_state()

        self.assertEqual(state["thematic"], [])
        self.assertEqual(state["sector"], [])
        self.assertEqual(state["industry"], [])
        self.assertEqual(state["cap"], [])
        self.assertEqual(state["fund_range"], (0.0, 100.0))
        self.assertEqual(state["tech_range"], (0.0, 100.0))
        self.assertEqual(state["rsi_regime_range"], (0.0, 100.0))
        self.assertEqual(state["sector_regime_fit_range"], (0.0, 100.0))
        self.assertEqual(state["fund_momentum_range"], (0.0, 100.0))
        self.assertEqual(state["tech_trend_dir"], "All")
        self.assertEqual(state["daily_rsi_divergence"], "All")
        self.assertEqual(state["weekly_rsi_divergence"], "All")
        self.assertEqual(state["short_term_flow"], "All")
        self.assertEqual(state["rel_strength"], "All")
        self.assertEqual(state["rel_volume"], "All")
        self.assertEqual(state["ai_revenue_exposure"], "All")
        self.assertEqual(state["ai_disruption_risk"], "All")
        self.assertEqual(state["beta_range"], (0.0, 5.0))
        self.assertEqual(state["ticker"], "")

    def test_build_company_filter_state_preserves_default_reset_values(self) -> None:
        state = _build_company_filter_state(
            caps=["Large", "Mega"],
            fund_range=(50.0, 100.0),
            tech_range=(60.0, 100.0),
            rsi_regime_range=(70.0, 100.0),
            sector_regime_fit_range=(60.0, 100.0),
            fund_momentum_range=(60.0, 100.0),
            tech_trend_dir=TREND_FILTER_LABELS["down"],
        )

        self.assertEqual(state["cap"], ["Large", "Mega"])
        self.assertEqual(state["fund_range"], (50.0, 100.0))
        self.assertEqual(state["tech_range"], (60.0, 100.0))
        self.assertEqual(state["rsi_regime_range"], (70.0, 100.0))
        self.assertEqual(state["sector_regime_fit_range"], (60.0, 100.0))
        self.assertEqual(state["fund_momentum_range"], (60.0, 100.0))
        self.assertEqual(state["tech_trend_dir"], TREND_FILTER_LABELS["down"])

    def test_filter_company_grid_by_ticker_list_matches_exact_normalized_tickers(self) -> None:
        company_df = pd.DataFrame(
            [
                {"ticker": "MSFT.US", "company": "Microsoft"},
                {"ticker": "AMN.US", "company": "AMN"},
                {"ticker": "GOOG.US", "company": "Google"},
                {"ticker": "MSFTW.US", "company": "Microsoft Warrant"},
            ]
        )

        filtered = _filter_company_grid_by_ticker_list(company_df, "msft, AMN\ngoog")

        self.assertEqual(filtered["ticker"].tolist(), ["MSFT.US", "AMN.US", "GOOG.US"])

    def test_filter_company_grid_by_ticker_list_does_not_use_partial_matches(self) -> None:
        company_df = pd.DataFrame(
            [
                {"ticker": "MSFT.US"},
                {"ticker": "MSFTW.US"},
            ]
        )

        filtered = _filter_company_grid_by_ticker_list(company_df, "MSF")

        self.assertTrue(filtered.empty)

    def test_company_grid_height_caps_large_result_sets(self) -> None:
        height = _company_grid_height(1000, row_height=34, min_height=220)

        self.assertLess(height, 1000 * 34)
        self.assertGreaterEqual(height, 220)

    def test_use_fast_company_grid_render_turns_on_for_large_results(self) -> None:
        self.assertFalse(_use_fast_company_grid_render(50))
        self.assertTrue(_use_fast_company_grid_render(900))

    def test_company_scope_sector_regime_fit_defaults_match_show_all_and_selected_modes(self) -> None:
        self.assertEqual(
            _default_sector_regime_fit_range_for_company_scope("show_all"),
            (60.0, 100.0),
        )
        self.assertEqual(
            _default_sector_regime_fit_range_for_company_scope("selected"),
            (0.0, 100.0),
        )
        self.assertEqual(
            _default_sector_regime_fit_range_for_company_scope("thematic"),
            (0.0, 100.0),
        )

    def test_build_company_drilldown_context_all_mode_returns_full_universe_without_default_scope_filters(self) -> None:
        report_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "company": "Alpha",
                    "sector": "Technology",
                    "industry": "Software",
                    "market_cap": 10_000_000_000,
                    "fundamental_total_score": 80.0,
                    "general_technical_score": 82.0,
                },
                {
                    "ticker": "BBB.US",
                    "company": "Beta",
                    "sector": "Industrials",
                    "industry": "Machinery",
                    "market_cap": 8_000_000_000,
                    "fundamental_total_score": 75.0,
                    "general_technical_score": 79.0,
                },
            ]
        )
        empty_regime_df = pd.DataFrame(columns=["ticker", "stock_rsi_regime_score", "sector_regime_fit_score"])

        _load_market_regime_company_metrics_for_date.clear()
        with patch(
            "equipilot_app._load_market_regime_company_metrics_for_date",
            return_value=(empty_regime_df, None),
        ):
            title, company_universe, default_sectors, default_industries, error, warning = build_company_drilldown_context(
                report_df,
                evaluation_date=date(2026, 3, 13),
                selected_sector="All sectors",
                selected_mode="all",
                selected_key="__all__",
            )

        self.assertEqual(title, "All companies")
        self.assertIsNone(error)
        self.assertIsNone(warning)
        assert company_universe is not None
        self.assertEqual(sorted(company_universe["ticker"].tolist()), ["AAA.US", "BBB.US"])
        self.assertEqual(default_sectors, [])
        self.assertEqual(default_industries, [])

    def test_compute_company_return_metrics_uses_exact_anchor_and_prior_targets(self) -> None:
        price_lookup = {
            "AAA.US": {
                "dates": [date(2026, 1, 2), date(2026, 1, 30), date(2026, 3, 13)],
                "closes": [100.0, 110.0, 121.0],
            }
        }

        metrics_df, anchor_missing = _compute_company_return_metrics(
            ["AAA.US"],
            price_lookup,
            date(2026, 3, 13),
        )

        self.assertFalse(anchor_missing)
        row = metrics_df.iloc[0]
        self.assertAlmostEqual(float(row["1m_perf"]), 10.0)
        self.assertAlmostEqual(float(row["ytd_perf"]), 21.0)

    def test_build_thematics_company_universe_adds_child_memberships_for_parent_scope(self) -> None:
        catalog = {
            "items": {
                "Parent Basket": {
                    "name": "Parent Basket",
                    "is_parent": True,
                    "children": ["Child A", "Child B"],
                    "tickers": ["AAA.US", "BBB.US"],
                },
                "Child A": {"name": "Child A", "tickers": ["AAA.US"]},
                "Child B": {"name": "Child B", "tickers": ["BBB.US"]},
            }
        }
        report_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "company": "Alpha",
                    "sector": "Tech",
                    "industry": "Software",
                    "market_cap": 10_000_000_000,
                    "beta": 1.2,
                    "fundamental_total_score": 80.0,
                    "fundamental_growth": 72.0,
                    "fundamental_value": 61.0,
                    "fundamental_quality": 74.0,
                    "fundamental_risk": 68.0,
                    "general_technical_score": 82.0,
                    "fundamental_momentum": 70.0,
                    "rs_monthly": 0.5,
                    "obvm_monthly": 0.4,
                },
                {
                    "ticker": "BBB.US",
                    "company": "Beta",
                    "sector": "Tech",
                    "industry": "Hardware",
                    "market_cap": 8_000_000_000,
                    "beta": 0.9,
                    "fundamental_total_score": 75.0,
                    "fundamental_growth": 64.0,
                    "fundamental_value": 59.0,
                    "fundamental_quality": 71.0,
                    "fundamental_risk": 63.0,
                    "general_technical_score": 79.0,
                    "fundamental_momentum": 68.0,
                    "rs_monthly": -0.2,
                    "obvm_monthly": 0.3,
                },
            ]
        )
        price_lookup = {
            "AAA.US": {
                "dates": [date(2026, 3, 6), date(2026, 3, 13)],
                "closes": [100.0, 110.0],
            },
            "BBB.US": {
                "dates": [date(2026, 3, 6), date(2026, 3, 13)],
                "closes": [200.0, 220.0],
            },
        }

        company_universe, anchor_missing = _build_thematics_company_universe(
            "Parent Basket",
            catalog,
            report_df,
            price_lookup,
            date(2026, 3, 13),
        )

        self.assertFalse(anchor_missing)
        thematic_by_ticker = company_universe.set_index("ticker")["thematic"].to_dict()
        self.assertEqual(thematic_by_ticker["AAA.US"], "Child A")
        self.assertEqual(thematic_by_ticker["BBB.US"], "Child B")
        by_ticker = company_universe.set_index("ticker")
        self.assertAlmostEqual(float(by_ticker.loc["AAA.US", "fundamental_growth"]), 72.0)
        self.assertAlmostEqual(float(by_ticker.loc["AAA.US", "fundamental_value"]), 61.0)
        self.assertAlmostEqual(float(by_ticker.loc["AAA.US", "fundamental_quality"]), 74.0)
        self.assertAlmostEqual(float(by_ticker.loc["AAA.US", "fundamental_risk"]), 68.0)

    def test_build_thematics_company_universe_normalizes_raw_tickers_for_price_metrics(self) -> None:
        catalog = {
            "items": {
                "Storage": {
                    "name": "Storage",
                    "is_parent": False,
                    "children": [],
                    "tickers": ["STX"],
                }
            }
        }
        report_df = pd.DataFrame(
            [
                {
                    "ticker": "STX.US",
                    "company": "Seagate Technology PLC",
                    "sector": "Technology",
                    "industry": "Computer Hardware",
                    "market_cap": 85_000_000_000,
                    "beta": 1.6,
                    "fundamental_total_score": 64.3,
                    "fundamental_growth": 55.0,
                    "fundamental_value": 44.0,
                    "fundamental_quality": 67.0,
                    "fundamental_risk": 49.0,
                    "general_technical_score": 70.5,
                    "fundamental_momentum": 100.0,
                    "rs_monthly": 0.4,
                    "obvm_monthly": 0.3,
                }
            ]
        )
        price_lookup = {
            "STX.US": {
                "dates": [date(2026, 2, 11), date(2026, 3, 1), date(2026, 3, 31)],
                "closes": [300.0, 350.0, 385.0],
            }
        }

        company_universe, anchor_missing = _build_thematics_company_universe(
            "Storage",
            catalog,
            report_df,
            price_lookup,
            date(2026, 3, 31),
        )

        self.assertFalse(anchor_missing)
        row = company_universe.iloc[0]
        self.assertEqual(row["ticker"], "STX.US")
        self.assertAlmostEqual(float(row["1w_perf"]), 10.0)
        self.assertAlmostEqual(float(row["1m_perf"]), 10.0)
        self.assertAlmostEqual(float(row["anchor_close"]), 385.0)

    def test_build_all_thematics_company_universe_deduplicates_tickers_and_preserves_memberships(self) -> None:
        catalog = {
            "items": {
                "AI": {
                    "name": "AI",
                    "is_ai_super_parent": True,
                    "tickers": ["AAA.US", "BBB.US"],
                },
                "Basket A": {
                    "name": "Basket A",
                    "tickers": ["AAA.US", "BBB.US"],
                },
                "Basket B": {
                    "name": "Basket B",
                    "tickers": ["AAA.US"],
                },
            }
        }
        report_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "company": "Alpha",
                    "sector": "Tech",
                    "industry": "Software",
                    "market_cap": 10_000_000_000,
                    "beta": 1.2,
                    "fundamental_total_score": 80.0,
                    "general_technical_score": 82.0,
                    "fundamental_momentum": 70.0,
                    "rs_monthly": 0.5,
                    "obvm_monthly": 0.4,
                },
                {
                    "ticker": "BBB.US",
                    "company": "Beta",
                    "sector": "Tech",
                    "industry": "Hardware",
                    "market_cap": 8_000_000_000,
                    "beta": 0.9,
                    "fundamental_total_score": 75.0,
                    "general_technical_score": 79.0,
                    "fundamental_momentum": 68.0,
                    "rs_monthly": -0.2,
                    "obvm_monthly": 0.3,
                },
            ]
        )
        price_lookup = {
            "AAA.US": {
                "dates": [date(2026, 3, 6), date(2026, 3, 13)],
                "closes": [100.0, 110.0],
            },
            "BBB.US": {
                "dates": [date(2026, 3, 6), date(2026, 3, 13)],
                "closes": [200.0, 220.0],
            },
        }

        company_universe, anchor_missing = _build_all_thematics_company_universe(
            catalog,
            report_df,
            price_lookup,
            date(2026, 3, 13),
        )

        self.assertFalse(anchor_missing)
        self.assertEqual(sorted(company_universe["ticker"].tolist()), ["AAA.US", "BBB.US"])
        memberships_by_ticker = company_universe.set_index("ticker")["thematic_memberships"].to_dict()
        self.assertEqual(memberships_by_ticker["AAA.US"], ["Basket A", "Basket B"])
        self.assertEqual(memberships_by_ticker["BBB.US"], ["Basket A"])

    def test_build_thematics_basket_metrics_computes_breadth_and_average_scores(self) -> None:
        catalog = {
            "items": {
                "Basket A": {
                    "name": "Basket A",
                    "description": "",
                    "article_narrative": "",
                    "tier": 1,
                    "tier_label": "AI",
                    "value_chain_layer": 1,
                    "parent": "",
                    "is_parent": False,
                    "children": [],
                    "tickers": ["AAA.US", "BBB.US"],
                }
            }
        }
        report_df = pd.DataFrame(
            [
                {
                    "ticker": "AAA.US",
                    "beta": 1.0,
                    "fundamental_total_score": 80.0,
                    "general_technical_score": 90.0,
                    "fundamental_momentum": 70.0,
                    "rs_monthly": 0.2,
                    "obvm_monthly": -0.1,
                },
                {
                    "ticker": "BBB.US",
                    "beta": 1.4,
                    "fundamental_total_score": 60.0,
                    "general_technical_score": 70.0,
                    "fundamental_momentum": 50.0,
                    "rs_monthly": -0.4,
                    "obvm_monthly": 0.5,
                },
            ]
        )
        price_lookup = {
            "AAA.US": {
                "dates": [date(2026, 3, 6), date(2026, 3, 13)],
                "closes": [100.0, 110.0],
            },
            "BBB.US": {
                "dates": [date(2026, 3, 6), date(2026, 3, 13)],
                "closes": [100.0, 120.0],
            },
        }

        metrics_df, anchor_missing = _build_thematics_basket_metrics(
            catalog,
            report_df,
            None,
            price_lookup,
            date(2026, 3, 13),
        )

        self.assertFalse(anchor_missing)
        row = metrics_df.iloc[0]
        self.assertAlmostEqual(float(row["beta"]), 1.2)
        self.assertAlmostEqual(float(row["technical_scoring"]), 80.0)
        self.assertAlmostEqual(float(row["fundamental_scoring"]), 70.0)
        self.assertAlmostEqual(float(row["rel_strength_breadth"]), 50.0)
        self.assertAlmostEqual(float(row["rel_volume_breadth"]), 50.0)

    def test_thematics_energy_config_matches_final_taxonomy(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "thematics.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        thematics = payload["thematics"]

        self.assertNotIn("Oil & Gas E&P", thematics)
        self.assertNotIn("Oilfield Services", thematics)
        self.assertEqual(list(thematics).count("Energy Sector"), 1)
        self.assertEqual(
            thematics["Energy Sector"]["sub_baskets"],
            [
                "Energy: Oil & Gas Integrated",
                "Energy: Oil & Gas E&P",
                "Energy: Oil & Gas Midstream",
                "Energy: Oil & Gas Refining & Marketing",
                "Energy: Oilfield Services & Drilling",
                "Energy: Uranium",
            ],
        )

        child_union = []
        for child_name in thematics["Energy Sector"]["sub_baskets"]:
            child_union.extend(thematics[child_name]["tickers"])
        self.assertEqual(len(thematics["Energy Sector"]["tickers"]), len(set(thematics["Energy Sector"]["tickers"])))
        self.assertEqual(set(thematics["Energy Sector"]["tickers"]), set(child_union))

        self.assertTrue(thematics["Energy: Oil & Gas Midstream"]["is_parent"])
        self.assertEqual(
            thematics["Energy: Oil & Gas Midstream"]["sub_baskets"],
            ["Energy Midstream & LNG", "Energy Midstream: Core Pipelines & Cash Flows"],
        )
        self.assertEqual(thematics["Energy Midstream & LNG"]["parent"], "Energy: Oil & Gas Midstream")
        self.assertEqual(
            thematics["Energy Midstream: Core Pipelines & Cash Flows"]["parent"],
            "Energy: Oil & Gas Midstream",
        )
        self.assertEqual(thematics["Energy Midstream & LNG"]["tickers"], ["LNG", "NEXT", "KMI", "TRGP", "ET"])
        self.assertEqual(
            thematics["Energy Midstream: Core Pipelines & Cash Flows"]["tickers"],
            ["WMB", "EPD", "ENB", "OKE", "MPLX", "VNOM"],
        )
        midstream_union = []
        for child_name in thematics["Energy: Oil & Gas Midstream"]["sub_baskets"]:
            midstream_union.extend(thematics[child_name]["tickers"])
        self.assertEqual(set(thematics["Energy: Oil & Gas Midstream"]["tickers"]), set(midstream_union))
        self.assertEqual(
            thematics["Energy: Oilfield Services & Drilling"]["tickers"],
            ["SLB", "HAL", "BKR", "NOV", "FTI", "LBRT", "PTEN", "NBR"],
        )
        self.assertEqual(thematics["Energy Security & Geopolitics"]["tickers"], ["LNG", "NEXT", "CCJ", "LEU", "TPL"])
        self.assertEqual(
            thematics["Nuclear Renaissance"],
            {
                "description": "Nuclear generation, fuel-cycle exposure, and SMR-linked beneficiaries driven by clean baseload demand.",
                "tier": 2,
                "tickers": ["CEG", "VST", "TLN", "CCJ", "LEU", "OKLO", "SMR", "BWXT"],
            },
        )
        self.assertIn("Energy Transition", thematics)

    def test_thematics_ai_view_modes_filter_expected_rows(self) -> None:
        catalog = {
            "items": {
                "AI": {
                    "name": "AI",
                    "parent": "",
                    "is_parent": True,
                    "children": ["AI Infrastructure", "AI Silicon", "AI Cloud Platforms"],
                    "tickers": [],
                    "is_ai_super_parent": True,
                    "is_ai_group_child": False,
                },
                "AI Infrastructure": {
                    "name": "AI Infrastructure",
                    "parent": "AI",
                    "is_parent": False,
                    "children": ["AI Infra: Power Generation"],
                    "tickers": [],
                    "is_ai_super_parent": False,
                    "is_ai_group_child": True,
                },
                "AI Infra: Power Generation": {
                    "name": "AI Infra: Power Generation",
                    "parent": "AI Infrastructure",
                    "is_parent": False,
                    "children": [],
                    "tickers": [],
                    "is_ai_super_parent": False,
                    "is_ai_group_child": False,
                },
                "AI Silicon": {
                    "name": "AI Silicon",
                    "parent": "AI",
                    "is_parent": False,
                    "children": [],
                    "tickers": [],
                    "is_ai_super_parent": False,
                    "is_ai_group_child": True,
                },
                "AI Cloud Platforms": {
                    "name": "AI Cloud Platforms",
                    "parent": "AI",
                    "is_parent": True,
                    "children": ["AI Cloud Platforms: Core"],
                    "tickers": [],
                    "is_ai_super_parent": False,
                    "is_ai_group_child": True,
                },
                "AI Cloud Platforms: Core": {
                    "name": "AI Cloud Platforms: Core",
                    "parent": "AI Cloud Platforms",
                    "is_parent": False,
                    "children": [],
                    "tickers": [],
                    "is_ai_super_parent": False,
                    "is_ai_group_child": False,
                },
                "Utilities Basket": {
                    "name": "Utilities Basket",
                    "parent": "",
                    "is_parent": False,
                    "children": [],
                    "tickers": [],
                    "is_ai_super_parent": False,
                    "is_ai_group_child": False,
                },
            },
            "roots": ["AI", "Utilities Basket"],
        }
        basket_metrics_df = pd.DataFrame(
            [
                {"name": "AI", "beta": 1.0},
                {"name": "AI Infrastructure", "beta": 1.0},
                {"name": "AI Infra: Power Generation", "beta": 1.0},
                {"name": "AI Silicon", "beta": 1.0},
                {"name": "AI Cloud Platforms", "beta": 1.0},
                {"name": "AI Cloud Platforms: Core", "beta": 1.0},
                {"name": "Utilities Basket", "beta": 1.0},
            ]
        )

        display_df, meta_df = _build_thematics_basket_table_frame(basket_metrics_df, catalog)

        all_names = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "all")[0]["Name"].tolist()
        ai_vs_rest = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "ai_vs_rest")[0]["Name"].tolist()
        ai_layers_vs_rest = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "ai_layers_vs_rest")[0]["Name"].tolist()
        ai_sub_layers_vs_rest = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "ai_sub_layers_vs_rest")[0]["Name"].tolist()
        ai_layers = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "ai_layers")[0]["Name"].tolist()
        ai_sub_layers = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "ai_sub_layers")[0]["Name"].tolist()

        self.assertEqual(
            all_names,
            [
                "AI",
                "AI Infrastructure",
                "AI Infra: Power Generation",
                "AI Silicon",
                "AI Cloud Platforms",
                "AI Cloud Platforms: Core",
                "Utilities Basket",
            ],
        )
        self.assertEqual(ai_vs_rest, ["AI", "Utilities Basket"])
        self.assertEqual(ai_layers_vs_rest, ["AI Infrastructure", "AI Silicon", "AI Cloud Platforms", "Utilities Basket"])
        self.assertEqual(ai_sub_layers_vs_rest, ["AI Infra: Power Generation", "AI Silicon", "AI Cloud Platforms: Core", "Utilities Basket"])
        self.assertEqual(ai_layers, ["AI Infrastructure", "AI Silicon", "AI Cloud Platforms"])
        self.assertEqual(ai_sub_layers, ["AI Infra: Power Generation", "AI Cloud Platforms: Core"])

    def test_thematics_energy_view_modes_filter_expected_rows(self) -> None:
        items = {}

        def add_item(
            name: str,
            *,
            parent: str = "",
            children: list[str] | None = None,
            family: str = "",
            role: str = "standalone",
            is_parent: bool = False,
            is_ai_super_parent: bool = False,
            is_ai_group_child: bool = False,
        ) -> None:
            items[name] = {
                "name": name,
                "parent": parent,
                "is_parent": is_parent,
                "children": list(children or []),
                "tickers": [],
                "is_ai_super_parent": is_ai_super_parent,
                "is_ai_group_child": is_ai_group_child,
                "family": family,
                "hierarchy_role": role,
                "is_family_parent": role == "family_parent",
                "is_family_layer": role == "layer",
                "is_family_sub_layer": role == "sub_layer",
            }

        add_item(
            "AI",
            children=["AI Infrastructure", "AI Silicon", "AI Cloud Platforms"],
            family="AI",
            role="family_parent",
            is_parent=True,
            is_ai_super_parent=True,
        )
        add_item(
            "AI Infrastructure",
            parent="AI",
            children=["AI Infra: Power Generation"],
            family="AI",
            role="layer",
            is_ai_group_child=True,
        )
        add_item("AI Infra: Power Generation", parent="AI Infrastructure", family="AI", role="sub_layer")
        add_item("AI Silicon", parent="AI", family="AI", role="layer", is_ai_group_child=True)
        add_item(
            "AI Cloud Platforms",
            parent="AI",
            children=["AI Cloud Platforms: Core"],
            family="AI",
            role="layer",
            is_parent=True,
            is_ai_group_child=True,
        )
        add_item("AI Cloud Platforms: Core", parent="AI Cloud Platforms", family="AI", role="sub_layer")
        energy_layers = [
            "Energy: Oil & Gas Integrated",
            "Energy: Oil & Gas E&P",
            "Energy: Oil & Gas Midstream",
            "Energy: Oil & Gas Refining & Marketing",
            "Energy: Oilfield Services & Drilling",
            "Energy: Uranium",
        ]
        add_item("Energy Sector", children=energy_layers, family="Energy", role="family_parent", is_parent=True)
        for layer_name in energy_layers:
            add_item(
                layer_name,
                parent="Energy Sector",
                children=[
                    "Energy Midstream & LNG",
                    "Energy Midstream: Core Pipelines & Cash Flows",
                ] if layer_name == "Energy: Oil & Gas Midstream" else [],
                family="Energy",
                role="layer",
                is_parent=layer_name == "Energy: Oil & Gas Midstream",
            )
        add_item("Energy Midstream & LNG", parent="Energy: Oil & Gas Midstream", family="Energy", role="sub_layer")
        add_item(
            "Energy Midstream: Core Pipelines & Cash Flows",
            parent="Energy: Oil & Gas Midstream",
            family="Energy",
            role="sub_layer",
        )
        add_item("High-Yield Energy Income", family="Energy", role="overlay")
        add_item("Energy Security & Geopolitics", family="Energy", role="overlay")
        add_item("Nuclear Renaissance", family="Energy", role="related")
        add_item("Energy Transition")
        add_item("Utilities Basket")

        catalog = {
            "items": items,
            "roots": [
                "AI",
                "Energy Sector",
                "High-Yield Energy Income",
                "Energy Security & Geopolitics",
                "Nuclear Renaissance",
                "Energy Transition",
                "Utilities Basket",
            ],
        }
        basket_metrics_df = pd.DataFrame([{"name": name, "beta": 1.0} for name in items])
        display_df, meta_df = _build_thematics_basket_table_frame(basket_metrics_df, catalog)

        family_roots = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "family_roots")[0]["Name"].tolist()
        layers = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "layers")[0]["Name"].tolist()
        sub_layers = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "sub_layers")[0]["Name"].tolist()
        energy_all = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "energy_all")[0]["Name"].tolist()
        energy_layer_rows = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "energy_layers")[0]["Name"].tolist()
        energy_sub_layer_rows = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, "energy_sub_layers")[0]["Name"].tolist()

        self.assertEqual(family_roots, ["AI", "Energy Sector", "Energy Transition", "Utilities Basket"])
        self.assertEqual(
            layers,
            [
                "AI Infrastructure",
                "AI Silicon",
                "AI Cloud Platforms",
                "Energy: Oil & Gas Integrated",
                "Energy: Oil & Gas E&P",
                "Energy: Oil & Gas Midstream",
                "Energy: Oil & Gas Refining & Marketing",
                "Energy: Oilfield Services & Drilling",
                "Energy: Uranium",
                "High-Yield Energy Income",
                "Energy Security & Geopolitics",
                "Nuclear Renaissance",
                "Energy Transition",
                "Utilities Basket",
            ],
        )
        self.assertEqual(
            sub_layers,
            [
                "AI Infra: Power Generation",
                "AI Silicon",
                "AI Cloud Platforms: Core",
                "Energy: Oil & Gas Integrated",
                "Energy: Oil & Gas E&P",
                "Energy Midstream & LNG",
                "Energy Midstream: Core Pipelines & Cash Flows",
                "Energy: Oil & Gas Refining & Marketing",
                "Energy: Oilfield Services & Drilling",
                "Energy: Uranium",
                "High-Yield Energy Income",
                "Energy Security & Geopolitics",
                "Nuclear Renaissance",
                "Energy Transition",
                "Utilities Basket",
            ],
        )
        self.assertEqual(
            energy_all,
            [
                "Energy Sector",
                "Energy: Oil & Gas Integrated",
                "Energy: Oil & Gas E&P",
                "Energy: Oil & Gas Midstream",
                "Energy Midstream & LNG",
                "Energy Midstream: Core Pipelines & Cash Flows",
                "Energy: Oil & Gas Refining & Marketing",
                "Energy: Oilfield Services & Drilling",
                "Energy: Uranium",
                "High-Yield Energy Income",
                "Energy Security & Geopolitics",
                "Nuclear Renaissance",
            ],
        )
        self.assertNotIn("Energy Transition", energy_all)
        self.assertEqual(energy_layer_rows, energy_layers)
        self.assertEqual(
            energy_sub_layer_rows,
            [
                "Energy: Oil & Gas Integrated",
                "Energy: Oil & Gas E&P",
                "Energy Midstream & LNG",
                "Energy Midstream: Core Pipelines & Cash Flows",
                "Energy: Oil & Gas Refining & Marketing",
                "Energy: Oilfield Services & Drilling",
                "Energy: Uranium",
            ],
        )
        self.assertNotIn("Energy: Oil & Gas Midstream", energy_sub_layer_rows)

    def test_thematics_catalog_marks_energy_hierarchy_for_table_and_lens(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "thematics.json"
        signature = f"test-{config_path.stat().st_mtime_ns}"
        catalog = build_thematics_catalog(str(config_path), signature)
        items = catalog["items"]
        self.assertEqual(items["Energy Sector"]["family"], "Energy")
        self.assertEqual(items["Energy Sector"]["hierarchy_role"], "family_parent")
        self.assertEqual(items["Energy: Oil & Gas Midstream"]["hierarchy_role"], "layer")
        self.assertEqual(items["Energy Midstream & LNG"]["hierarchy_role"], "sub_layer")
        self.assertEqual(items["Energy Midstream: Core Pipelines & Cash Flows"]["hierarchy_role"], "sub_layer")
        self.assertEqual(items["High-Yield Energy Income"]["hierarchy_role"], "overlay")
        self.assertEqual(items["Energy Security & Geopolitics"]["hierarchy_role"], "overlay")
        self.assertEqual(items["Nuclear Renaissance"]["hierarchy_role"], "related")
        self.assertEqual(items["Energy Transition"]["hierarchy_role"], "standalone")

        basket_metrics_df = pd.DataFrame([{"name": name, "beta": 1.0} for name in items])
        _, table_meta = _build_thematics_basket_table_frame(basket_metrics_df, catalog)
        _, lens_meta = _build_thematics_lens_frame(catalog)
        table_roles = table_meta.set_index("basket_name")["hierarchy_role"].to_dict()
        lens_roles = lens_meta.set_index("basket_name")["hierarchy_role"].to_dict()
        self.assertEqual(table_roles["Energy Sector"], "family_parent")
        self.assertEqual(lens_roles["Energy Sector"], "family_parent")
        self.assertEqual(table_roles["Energy Midstream & LNG"], "sub_layer")
        self.assertEqual(lens_roles["Energy Midstream & LNG"], "sub_layer")
        self.assertEqual(table_roles["Energy Midstream: Core Pipelines & Cash Flows"], "sub_layer")
        self.assertEqual(lens_roles["Energy Midstream: Core Pipelines & Cash Flows"], "sub_layer")

    def test_thematics_selected_basket_is_cleared_when_hidden(self) -> None:
        self.assertEqual(
            _normalize_thematics_selected_basket("AI Infrastructure", {"AI", "Utilities Basket"}),
            None,
        )
        self.assertEqual(
            _normalize_thematics_selected_basket("AI Infrastructure", {"AI Infrastructure", "Utilities Basket"}),
            "AI Infrastructure",
        )
if __name__ == "__main__":
    unittest.main()
