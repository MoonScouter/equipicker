import unittest

import pandas as pd
from datetime import date
from unittest.mock import patch

from equipilot_app import (
    TREND_FILTER_LABELS,
    TREND_SYMBOL_DOWN,
    TREND_SYMBOL_UP,
    _annotate_company_technical_trend,
    _build_all_thematics_company_universe,
    _build_thematics_basket_metrics,
    _build_thematics_basket_table_frame,
    _build_thematics_company_universe,
    _build_company_drilldown_styler,
    _default_sector_regime_fit_range_for_company_scope,
    _enrich_company_universe_with_market_regime,
    _filter_thematics_basket_table_for_view,
    _filter_by_optional_numeric_range,
    _company_filter_presets,
    _compute_company_return_metrics,
    _load_market_regime_company_metrics_for_date,
    _normalize_thematics_selected_basket,
    _prepare_company_drilldown_universe,
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
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Market Cap",
                "Fundamental Score",
                "Fundamental Momentum",
                "Technical Score",
                "RSI Regime Score",
                "Sector Regime Fit",
                "Rel Strength",
                "Rel Volume",
                "AI Revenue Exposure",
                "AI Disruption Risk",
            ],
        )
        self.assertEqual(rendered.iloc[0]["Ticker"], "HIGH.US")
        self.assertEqual(rendered.iloc[0]["Company"], "High Tech")
        self.assertTrue(pd.api.types.is_numeric_dtype(rendered["Fundamental Score"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(rendered["Fundamental Momentum"]))
        self.assertAlmostEqual(float(rendered.iloc[0]["Fundamental Momentum"]), 72.0)
        self.assertEqual(rendered.iloc[0]["RSI Regime Score"], "82.0")
        self.assertEqual(rendered.iloc[0]["Sector Regime Fit"], "74.0")

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
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Market Cap",
                "Fundamental Score",
                "Fundamental Momentum",
                "Technical Score",
                "RSI Regime Score",
                "Sector Regime Fit",
                "Rel Strength",
                "Rel Volume",
                "AI Revenue Exposure",
                "AI Disruption Risk",
            ],
        )
        self.assertEqual(rendered.iloc[0]["Company"], "Alpha Inc")
        self.assertTrue(pd.api.types.is_numeric_dtype(rendered["Fundamental Score"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(rendered["Fundamental Momentum"]))
        self.assertAlmostEqual(float(rendered.iloc[0]["Fundamental Momentum"]), 85.0)
        self.assertEqual(rendered.iloc[0]["RSI Regime Score"], "76.0")
        self.assertEqual(rendered.iloc[0]["Sector Regime Fit"], "68.0")

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
            "Ticker",
            "Company",
            "Sector",
            "Industry",
            "Market Cap",
            "Fundamental Score",
            "Fundamental Momentum",
            "Technical Score",
            "RSI Regime Score",
            "Sector Regime Fit",
            "Rel Strength",
            "Rel Volume",
            "AI Revenue Exposure",
            "AI Disruption Risk",
        ]
        self.assertEqual(fundamental_rendered.columns.tolist(), expected_columns)
        self.assertEqual(technical_rendered.columns.tolist(), expected_columns)
        self.assertEqual(fundamental_rendered.iloc[0]["RSI Regime Score"], "84.0")
        self.assertEqual(fundamental_rendered.iloc[0]["Sector Regime Fit"], "72.0")
        self.assertEqual(fundamental_rendered.iloc[0]["Rel Strength"], "Positive")
        self.assertEqual(fundamental_rendered.iloc[0]["AI Revenue Exposure"], "direct")
        self.assertEqual(technical_rendered.iloc[0]["AI Disruption Risk"], "low")

    def test_company_drilldown_styler_applies_score_and_signal_colors(self) -> None:
        display_df = pd.DataFrame(
            [
                {
                    "Ticker": "AAA.US",
                    "Company": "Alpha Inc",
                    "Sector": "Technology",
                    "Industry": "Software",
                    "Market Cap": "1.50B",
                    "Fundamental Score": "88.0",
                    "Fundamental Momentum": "77.0",
                    "Technical Score": "91.0",
                    "RSI Regime Score": "84.0",
                    "Sector Regime Fit": "72.0",
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
        self.assertTrue(pd.api.types.is_numeric_dtype(rendered["Technical Score"]))
        self.assertAlmostEqual(float(rendered.iloc[0]["Technical Score"]), 90.0)
        self.assertEqual(rendered.attrs.get("technical_trend_symbols", [""])[0], TREND_SYMBOL_UP)
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
                    "fundamental_total_score": 76.0,
                    "fundamental_momentum": 68.0,
                    "technical_trend_symbol": TREND_SYMBOL_UP,
                    "fundamental_trend_symbol": TREND_SYMBOL_DOWN,
                    "fundamental_momentum_trend_symbol": "",
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
                "FS",
                "Mom. FS",
                "Rel Strength",
                "Rel Volume",
                "AI Revenue Exposure",
                "AI Disruption Risk",
            ],
        )
        self.assertEqual(str(rendered.iloc[0]["TS"]).strip(), f"81.0 {TREND_SYMBOL_UP}")
        self.assertAlmostEqual(float(rendered.iloc[0]["RSI Regime"]), 74.0)
        self.assertAlmostEqual(float(rendered.iloc[0]["Sector Regime Fit"]), 66.0)
        self.assertEqual(str(rendered.iloc[0]["FS"]).strip(), f"76.0 {TREND_SYMBOL_DOWN}")
        self.assertEqual(str(rendered.iloc[0]["Mom. FS"]).strip(), "68.0")


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

    def test_company_filter_presets_use_shared_trend_labels(self) -> None:
        presets = _company_filter_presets()
        trend_labels = {str(preset["trend_dir"]) for preset in presets.values()}

        self.assertEqual(
            trend_labels,
            {
                TREND_FILTER_LABELS["up"],
                TREND_FILTER_LABELS["flat"],
                TREND_FILTER_LABELS["down"],
            },
        )
        self.assertTrue(all(tuple(preset["rsi_regime_range"]) == (70.0, 100.0) for preset in presets.values()))
        self.assertTrue(
            all(tuple(preset["sector_regime_fit_range"]) == (60.0, 100.0) for preset in presets.values())
        )

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
        self.assertEqual(ai_sub_layers_vs_rest, ["AI Infra: Power Generation", "AI Cloud Platforms: Core", "Utilities Basket"])
        self.assertEqual(ai_layers, ["AI Infrastructure", "AI Silicon", "AI Cloud Platforms"])
        self.assertEqual(ai_sub_layers, ["AI Infra: Power Generation", "AI Cloud Platforms: Core"])

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
