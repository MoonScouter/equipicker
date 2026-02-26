import unittest

import pandas as pd

from equipilot_app import (
    _prepare_company_drilldown_universe,
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


if __name__ == "__main__":
    unittest.main()
