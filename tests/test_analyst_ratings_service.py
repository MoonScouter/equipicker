import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from analyst_ratings_service import (
    ANALYST_RATINGS_COLUMNS,
    analyst_rating_row,
    analyst_ratings_path,
    enrich_analyst_ratings_with_latest_prices,
    import_analyst_ratings,
    latest_analyst_ratings_file,
    load_analyst_ratings_file,
    save_analyst_ratings_workbook,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.requested_tickers: list[str] = []
        self.closed = False

    def get(self, url, params, timeout):
        del params, timeout
        ticker = url.rsplit("/", 1)[-1]
        self.requested_tickers.append(ticker)
        return _FakeResponse(
            {
                "AnalystRatings": {
                    "StrongBuy": 2,
                    "Buy": 1,
                    "Hold": 1,
                    "Sell": 0,
                    "StrongSell": 0,
                    "Rating": 4.2,
                    "TargetPrice": 123.45,
                }
            }
        )

    def close(self) -> None:
        self.closed = True


class AnalystRatingsServiceTests(unittest.TestCase):
    def test_analyst_rating_row_matches_legacy_columns_and_percentages(self) -> None:
        row = analyst_rating_row(
            "AAPL.US",
            {
                "StrongBuy": 2,
                "Buy": 1,
                "Hold": 1,
                "Sell": 0,
                "StrongSell": 0,
                "Rating": 4.2,
                "TargetPrice": 123.45,
            },
            metadata={"Name": "Apple", "Sector": "Technology", "Industry": "Consumer Electronics"},
        )

        self.assertEqual(list(row.keys()), list(ANALYST_RATINGS_COLUMNS))
        self.assertEqual(row["Ticker"], "AAPL.US")
        self.assertEqual(row["TargetPrice"], 123.45)
        self.assertEqual(row["TotalRatings"], 4)
        self.assertEqual(row["%StrongBuy"], 50)
        self.assertEqual(row["%Buy"], 25)

    def test_save_and_load_workbook_uses_dated_analyst_ratings_filename(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            with patch("analyst_ratings_service.ANALYST_RATINGS_DIR", Path(tmp_dir)):
                output_path = analyst_ratings_path(date(2026, 6, 20))
                df = pd.DataFrame(
                    [
                        analyst_rating_row(
                            "MSFT.US",
                            {
                                "StrongBuy": 1,
                                "Buy": 2,
                                "Hold": 0,
                                "Sell": 0,
                                "StrongSell": 0,
                                "Rating": 4.0,
                                "TargetPrice": 500,
                            },
                            metadata={"Name": "Microsoft", "Sector": "Technology", "Industry": "Software"},
                        )
                    ]
                )

                save_analyst_ratings_workbook(df, output_path)
                loaded = load_analyst_ratings_file(output_path)

                self.assertEqual(output_path.name, "analyst_ratings_2026_06_20.xlsx")
                self.assertEqual(latest_analyst_ratings_file(), output_path)
                self.assertEqual(list(loaded.columns), list(ANALYST_RATINGS_COLUMNS))
                self.assertEqual(loaded.loc[0, "Ticker"], "MSFT.US")

    def test_enrich_analyst_ratings_with_latest_prices_filters_requested_tickers(self) -> None:
        ratings_df = pd.DataFrame(
            [
                analyst_rating_row("AAPL.US", {"TargetPrice": 210}),
                analyst_rating_row("MSFT.US", {"TargetPrice": 500}),
            ]
        )
        prices_df = pd.DataFrame(
            [
                {"ticker": "AAPL.US", "date": "2026-06-18", "adjusted_close": 198.0},
                {"ticker": "AAPL.US", "date": "2026-06-19", "adjusted_close": 200.0},
                {"ticker": "MSFT.US", "date": "2026-06-19", "adjusted_close": 480.0},
            ]
        )

        enriched = enrich_analyst_ratings_with_latest_prices(ratings_df, prices_df, tickers=["aapl"])

        self.assertEqual(enriched["Ticker"].tolist(), ["AAPL.US"])
        self.assertEqual(enriched.loc[0, "LastClose"], 200.0)
        self.assertEqual(enriched.loc[0, "CloseDate"], "2026-06-19")

    def test_import_analyst_ratings_uses_selected_tickers_and_saves_workbook(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            fake_session = _FakeSession()
            with patch("analyst_ratings_service.ANALYST_RATINGS_DIR", Path(tmp_dir)):
                result = import_analyst_ratings(
                    scope="specific",
                    manual_tickers=["aapl", "MSFT.US"],
                    run_date=date(2026, 6, 20),
                    metadata_df=pd.DataFrame(
                        [
                            {"ticker": "AAPL.US", "company": "Apple", "sector": "Technology", "industry": "Hardware"},
                            {"ticker": "MSFT.US", "company": "Microsoft", "sector": "Technology", "industry": "Software"},
                        ]
                    ),
                    sleep_seconds=0,
                    session=fake_session,
                    eodhd_api_token="test-token",
                )

                saved_path = Path(result["saved_path"])
                loaded = load_analyst_ratings_file(saved_path)

        self.assertEqual(fake_session.requested_tickers, ["AAPL.US", "MSFT.US"])
        self.assertEqual(saved_path.name, "analyst_ratings_2026_06_20.xlsx")
        self.assertEqual(result["saved_rows"], 2)
        self.assertEqual(loaded["Name"].tolist(), ["Apple", "Microsoft"])


if __name__ == "__main__":
    unittest.main()
