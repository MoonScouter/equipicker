import unittest

import pandas as pd
from datetime import date

from equipilot_app import (
    TREND_FILTER_LABELS,
    TREND_SYMBOL_DOWN,
    TREND_SYMBOL_UP,
    _annotate_company_technical_trend,
    _build_thematics_basket_metrics,
    _build_thematics_company_universe,
    _company_filter_presets,
    _compute_company_return_metrics,
    _prepare_company_drilldown_universe,
    apply_trend_symbols_to_table,
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
        self.assertEqual(prepared.iloc[0]["rel_strength"], "N/A")
        self.assertEqual(prepared.iloc[0]["rel_volume"], "N/A")

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
            ],
        )
        self.assertEqual(rendered.iloc[0]["Ticker"], "HIGH.US")
        self.assertEqual(rendered.iloc[0]["Company"], "High Tech")
        self.assertEqual(rendered.iloc[0]["Fundamental Momentum"], "72.0")

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
            ],
        )
        self.assertEqual(rendered.iloc[0]["Company"], "Alpha Inc")
        self.assertEqual(rendered.iloc[0]["Fundamental Momentum"], "85.0")

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
        self.assertEqual(rendered.iloc[0]["Technical Score"], f"90.0 {TREND_SYMBOL_UP}")

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

        self.assertEqual(str(rendered.iloc[0]["TS"]).strip(), f"81.0 {TREND_SYMBOL_UP}")
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
if __name__ == "__main__":
    unittest.main()
