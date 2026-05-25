from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from equipilot_app import _portfolio_segment, _portfolio_segment_overrides, build_portfolio_open_positions


class PortfolioTabTests(unittest.TestCase):
    def test_build_open_positions_treats_realized_return_as_sell(self) -> None:
        journal_df = pd.DataFrame(
            [
                {
                    "Data": "2025-10-31",
                    "Ticker": "NEE",
                    "Decizie": "BUY",
                    "Text": "NextEra Energy intră în portofoliu la 81,40 USD.",
                    "Randament": pd.NA,
                },
                {
                    "Data": "2026-04-11",
                    "Ticker": "NEE",
                    "Decizie": "BUY",
                    "Text": "NEE - SELL - Facem exit la prețul de 94,08 USD.",
                    "Randament": 0.156,
                },
                {
                    "Data": "2026-04-11",
                    "Ticker": "AAPL",
                    "Decizie": "BUY",
                    "Text": "AAPL - BUY - Am cumpărat Apple la prețul de 260,48 USD.",
                    "Randament": pd.NA,
                },
            ]
        )

        positions = build_portfolio_open_positions(journal_df, reference_date=date(2026, 5, 1))

        self.assertEqual(positions["ticker"].tolist(), ["AAPL.US"])
        self.assertAlmostEqual(float(positions.iloc[0]["transaction_price"]), 260.48)

    def test_build_open_positions_applies_snapshot_exclusions(self) -> None:
        journal_df = pd.DataFrame(
            [
                {
                    "Data": "2026-02-08",
                    "Ticker": "AA",
                    "Decizie": "BUY",
                    "Text": "Am cumpărat Alcoa la 59,16 USD.",
                    "Randament": pd.NA,
                },
                {
                    "Data": "2026-02-08",
                    "Ticker": "BMY",
                    "Decizie": "BUY",
                    "Text": "Am cumpărat Bristol Myers Squibb la 61,99 USD.",
                    "Randament": pd.NA,
                },
            ]
        )

        positions = build_portfolio_open_positions(
            journal_df,
            reference_date=date(2026, 5, 1),
            excluded_tickers={"AA"},
        )

        self.assertEqual(positions["ticker"].tolist(), ["BMY.US"])

    def test_portfolio_segment_uses_config_override(self) -> None:
        config = {
            "segment_overrides": {
                "dinamic": {
                    "ASTS": "Aggressive",
                    "MSFT": "Growth",
                }
            }
        }

        overrides = _portfolio_segment_overrides(config, "dinamic")

        self.assertEqual(
            _portfolio_segment(
                pd.Series({"ticker": "ASTS.US", "fundamental_growth": 20, "fundamental_value": 30}),
                overrides,
            ),
            "Aggressive",
        )
        self.assertEqual(
            _portfolio_segment(
                pd.Series({"ticker": "MSFT.US", "fundamental_growth": 20, "fundamental_value": 30}),
                overrides,
            ),
            "Growth",
        )


if __name__ == "__main__":
    unittest.main()
