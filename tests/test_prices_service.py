import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from prices_service import (
    build_prices_cache_dataframe,
    compute_wilder_rsi,
    enrich_prices_with_rsi,
    fetch_prices_history,
    get_price_history_query,
    intersect_ticker_universe,
    normalize_price_tickers,
    parse_manual_price_tickers,
    save_prices_cache,
    load_prices_cache,
)


class _DummyConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyEngine:
    def connect(self):
        return _DummyConnection()


class PricesServiceTests(unittest.TestCase):
    def test_normalize_price_tickers_adds_suffix_and_deduplicates(self) -> None:
        normalized = normalize_price_tickers(["aapl", "AAPL.US", "brk.b", " msft "])

        self.assertEqual(normalized, ["AAPL.US", "BRK.B.US", "MSFT.US"])

    def test_parse_manual_price_tickers_accepts_mixed_separators(self) -> None:
        parsed = parse_manual_price_tickers("AAPL, msft\nNVDA   brk.b")

        self.assertEqual(parsed, ["AAPL.US", "MSFT.US", "NVDA.US", "BRK.B.US"])

    def test_intersect_ticker_universe_uses_normalized_overlap(self) -> None:
        resolved = intersect_ticker_universe(["aapl", "msft", "nvda"], ["AAPL.US", "NVDA.US", "AMD.US"])

        self.assertEqual(resolved, ["AAPL.US", "NVDA.US"])

    def test_get_price_history_query_chooses_expected_source_table(self) -> None:
        self.assertIn("FROM eod_data", get_price_history_query("daily"))
        self.assertIn("FROM eod_weekly", get_price_history_query("weekly"))

    def test_fetch_prices_history_chunks_tickers_and_concatenates_results(self) -> None:
        calls: list[list[str]] = []

        def fake_read_sql(query, conn, params):
            tickers = list(params["tickers"])
            calls.append(tickers)
            return pd.DataFrame(
                [
                    {
                        "ticker": ticker,
                        "date": "2026-01-02",
                        "adjusted_close": float(index + 1),
                        "adjusted_high": float(index + 2),
                        "adjusted_low": float(index),
                        "rs": 1.0,
                        "obvm": 2.0,
                    }
                    for index, ticker in enumerate(tickers)
                ]
            )

        with patch("prices_service.make_engine", return_value=_DummyEngine()), patch(
            "prices_service.pd.read_sql", side_effect=fake_read_sql
        ):
            df = fetch_prices_history(
                "daily",
                ["aapl", "MSFT.US", "nvda", "amd", "tsla"],
                date(2026, 1, 1),
                chunk_size=2,
            )

        self.assertEqual(calls, [["AAPL.US", "MSFT.US"], ["NVDA.US", "AMD.US"], ["TSLA.US"]])
        self.assertEqual(df["ticker"].tolist(), ["AAPL.US", "AMD.US", "MSFT.US", "NVDA.US", "TSLA.US"])
        self.assertEqual(df["date"].tolist(), ["2026-01-02"] * 5)

    def test_build_prices_cache_dataframe_all_scope_overwrites_existing_rows(self) -> None:
        existing_df = pd.DataFrame(
            [{
                "ticker": "AAPL.US",
                "date": "2026-01-05",
                "adjusted_close": 100.0,
                "adjusted_high": 101.0,
                "adjusted_low": 99.0,
                "rs": 1.0,
                "obvm": 2.0,
            }]
        )
        fetched_df = pd.DataFrame(
            [{
                "ticker": "MSFT.US",
                "date": "2026-02-05",
                "adjusted_close": 200.0,
                "adjusted_high": 201.0,
                "adjusted_low": 198.0,
                "rs": 3.0,
                "obvm": 4.0,
            }]
        )

        result = build_prices_cache_dataframe(
            existing_df,
            fetched_df,
            scope="all",
            selected_tickers=[],
            cutoff_date=date(2026, 1, 1),
        )

        self.assertEqual(result["ticker"].tolist(), ["MSFT.US"])
        self.assertEqual(result["date"].tolist(), ["2026-02-05"])

    def test_build_prices_cache_dataframe_specific_scope_replaces_only_selected_rows_from_start_date(self) -> None:
        existing_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL.US",
                    "date": "2026-01-10",
                    "adjusted_close": 95.0,
                    "adjusted_high": 96.0,
                    "adjusted_low": 94.0,
                    "rs": 0.8,
                    "obvm": 1.8,
                },
                {
                    "ticker": "AAPL.US",
                    "date": "2026-02-10",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                },
                {
                    "ticker": "MSFT.US",
                    "date": "2026-02-10",
                    "adjusted_close": 200.0,
                    "adjusted_high": 201.0,
                    "adjusted_low": 198.0,
                    "rs": 3.0,
                    "obvm": 4.0,
                },
            ]
        )
        fetched_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "date": "2026-02-10",
                    "adjusted_close": 111.0,
                    "adjusted_high": 112.0,
                    "adjusted_low": 109.0,
                    "rs": 1.5,
                    "obvm": 2.5,
                },
                {
                    "ticker": "AAPL.US",
                    "date": "2026-03-10",
                    "adjusted_close": 120.0,
                    "adjusted_high": 121.0,
                    "adjusted_low": 118.0,
                    "rs": 1.8,
                    "obvm": 2.8,
                },
            ]
        )

        result = build_prices_cache_dataframe(
            existing_df,
            fetched_df,
            scope="specific",
            selected_tickers=["AAPL"],
            cutoff_date=date(2026, 2, 10),
        )

        self.assertEqual(result["ticker"].tolist(), ["AAPL.US", "AAPL.US", "AAPL.US", "MSFT.US"])
        self.assertEqual(result["date"].tolist(), ["2026-01-10", "2026-02-10", "2026-03-10", "2026-02-10"])
        self.assertEqual(result.loc[result["date"] == "2026-02-10", "adjusted_close"].tolist(), [111.0, 200.0])

    def test_compute_wilder_rsi_returns_expected_values_for_monotonic_gain_sequence(self) -> None:
        closes = pd.Series([float(value) for value in range(1, 18)])

        result = compute_wilder_rsi(closes)

        self.assertTrue(result.iloc[:14].isna().all())
        self.assertEqual(result.iloc[14], 100.0)
        self.assertEqual(result.iloc[15], 100.0)
        self.assertEqual(result.iloc[16], 100.0)

    def test_enrich_prices_with_rsi_updates_only_selected_tickers(self) -> None:
        target_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL.US",
                    "date": f"2026-01-{day:02d}",
                    "adjusted_close": float(day),
                    "adjusted_high": float(day) + 1.0,
                    "adjusted_low": float(day) - 1.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }
                for day in range(1, 17)
            ]
            + [
                {
                    "ticker": "MSFT.US",
                    "date": f"2026-01-{day:02d}",
                    "adjusted_close": float(200 - day),
                    "adjusted_high": float(201 - day),
                    "adjusted_low": float(199 - day),
                    "rs": 3.0,
                    "obvm": 4.0,
                    "rsi_14": 42.0,
                }
                for day in range(1, 17)
            ]
        )

        result = enrich_prices_with_rsi(target_df, selected_tickers=["AAPL"])

        aapl_rsi = result.loc[result["ticker"] == "AAPL.US", "rsi_14"]
        msft_rsi = result.loc[result["ticker"] == "MSFT.US", "rsi_14"]
        self.assertTrue(aapl_rsi.iloc[:14].isna().all())
        self.assertEqual(aapl_rsi.iloc[14], 100.0)
        self.assertEqual(aapl_rsi.iloc[15], 100.0)
        self.assertTrue((msft_rsi == 42.0).all())

    def test_enrich_prices_with_rsi_can_use_seed_history_for_early_rows(self) -> None:
        seed_history_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL.US",
                    "date": f"2025-12-{day:02d}",
                    "adjusted_close": float(day),
                    "adjusted_high": float(day) + 1.0,
                    "adjusted_low": float(day) - 1.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }
                for day in range(17, 31)
            ]
        )
        target_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL.US",
                    "date": "2026-01-04",
                    "adjusted_close": 31.0,
                    "adjusted_high": 32.0,
                    "adjusted_low": 30.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                },
                {
                    "ticker": "AAPL.US",
                    "date": "2026-01-05",
                    "adjusted_close": 32.0,
                    "adjusted_high": 33.0,
                    "adjusted_low": 31.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                },
            ]
        )

        result = enrich_prices_with_rsi(
            target_df,
            selected_tickers=["AAPL"],
            seed_history_df=seed_history_df,
        )

        self.assertEqual(result["rsi_14"].tolist(), [100.0, 100.0])

    def test_save_and_load_prices_cache_round_trip_jsonl(self) -> None:
        df = pd.DataFrame(
            [{
                "ticker": "AAPL.US",
                "date": "2026-02-10",
                "adjusted_close": 111.0,
                "adjusted_high": 112.0,
                "adjusted_low": 109.0,
                "rs": 1.5,
                "obvm": 2.5,
                "rsi_14": 75.0,
            }]
        )
        with TemporaryDirectory() as tmp_dir:
            with patch("prices_service.CACHE_DIR", Path(tmp_dir)):
                saved_path = save_prices_cache(df, "daily", 2026)
                loaded = load_prices_cache(saved_path)

        self.assertEqual(saved_path.name, "prices_daily_2026.jsonl")
        self.assertEqual(loaded["ticker"].tolist(), ["AAPL.US"])
        self.assertEqual(loaded["date"].tolist(), ["2026-02-10"])
        self.assertEqual(loaded["rsi_14"].tolist(), [75.0])


if __name__ == "__main__":
    unittest.main()

