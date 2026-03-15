import unittest

import pandas as pd

from equipilot_app import (
    TREND_FILTER_LABELS,
    TREND_SYMBOL_DOWN,
    TREND_SYMBOL_UP,
    _annotate_company_technical_trend,
    _company_filter_presets,
    _prepare_company_drilldown_universe,
    apply_trend_symbols_to_table,
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
                },
            ]
        )

        prepared, error = _prepare_company_drilldown_universe(report_df)

        self.assertIsNone(error)
        assert prepared is not None
        self.assertEqual(prepared["company"].tolist(), ["Alpha Inc", "BBB.US"])
        self.assertEqual(prepared["fundamental_momentum"].tolist(), [77.0, 55.0])

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
if __name__ == "__main__":
    unittest.main()
