import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from prices_service import (
    ActiveDivergence,
    build_prices_cache_dataframe,
    compute_rsi_divergence_state,
    compute_rsi_divergence_flags,
    compute_wilder_rsi,
    divergence_seed_history_rows,
    enrich_prices_with_rsi,
    fetch_prices_history,
    get_price_history_query,
    import_prices_cache,
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
    @staticmethod
    def _active_divergence(
        *,
        anchor_index: int,
        pivot2_index: int,
        confirmation_index: int,
        pivot2_price: float,
        pivot2_rsi: float,
    ) -> ActiveDivergence:
        return ActiveDivergence(
            anchor_index=anchor_index,
            pivot2_index=pivot2_index,
            confirmation_index=confirmation_index,
            pivot2_price=pivot2_price,
            pivot2_rsi=pivot2_rsi,
        )

    @staticmethod
    def _build_divergence_frame(
        highs: list[float],
        lows: list[float],
        rsi_values: list[float],
        *,
        ticker: str = "AAA.US",
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for idx, (high_value, low_value, rsi_value) in enumerate(zip(highs, lows, rsi_values), start=1):
            close_value = (high_value + low_value) / 2.0
            rows.append(
                {
                    "ticker": ticker,
                    "date": f"2026-01-{idx:02d}",
                    "adjusted_close": close_value,
                    "adjusted_high": high_value,
                    "adjusted_low": low_value,
                    "rs": 1.0,
                    "obvm": 2.0,
                    "rsi_14": rsi_value,
                }
            )
        return pd.DataFrame(rows)

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
                    "rsi_divergence_flag": "negative",
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
        self.assertTrue(
            (
                result.loc[result["ticker"] == "MSFT.US", "rsi_divergence_flag"]
                == "negative"
            ).all()
        )
        self.assertTrue(
            result.loc[result["ticker"] == "MSFT.US", "rsi_divergence_confirmed"].isna().all()
        )

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
        self.assertEqual(result["rsi_divergence_flag"].tolist(), ["none", "none"])

    def test_compute_rsi_divergence_flags_detects_bearish_daily_divergence(self) -> None:
        highs = [8, 9, 10, 11, 10, 9, 14, 9, 8, 9, 10, 11, 12, 13, 17, 12, 10, 9, 8, 8]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 58, 56, 54, 74, 55, 54, 53, 54, 56, 58, 60, 66, 58, 55, 52, 50, 49]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertTrue((flags.iloc[:14] == "none").all())
        self.assertEqual(flags.iloc[14], "potential-negative")
        self.assertEqual(flags.iloc[15], "potential-negative")
        self.assertEqual(flags.iloc[16], "negative")
        self.assertEqual(flags.iloc[17], "negative")
        self.assertEqual(flags.iloc[18], "negative")
        self.assertEqual(flags.iloc[19], "negative")

    def test_compute_rsi_divergence_state_exposes_active_anchor_and_pivot_metadata(self) -> None:
        highs = [8, 9, 10, 11, 10, 9, 14, 9, 8, 9, 10, 11, 12, 13, 17, 12, 10, 9, 8, 8]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 58, 56, 54, 74, 55, 54, 53, 54, 56, 58, 60, 66, 58, 55, 52, 50, 49]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_anchor_date"].iloc[16], "2026-01-07")
        self.assertEqual(state["rsi_divergence_anchor_price"].iloc[16], 14.0)
        self.assertEqual(state["rsi_divergence_pivot_date"].iloc[16], "2026-01-15")
        self.assertEqual(state["rsi_divergence_pivot_price"].iloc[16], 17.0)

    def test_compute_rsi_divergence_state_tracks_last_pivot_reference_and_distance_coeff(self) -> None:
        highs = [8, 9, 10, 11, 10, 9, 14, 9, 8, 9, 10, 11, 12, 13, 17, 12, 10, 9, 8, 8]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 58, 56, 54, 74, 55, 54, 53, 54, 56, 58, 60, 66, 58, 55, 52, 50, 49]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_last_pivot_type"].iloc[8], "bear")
        self.assertEqual(state["rsi_divergence_last_pivot_price"].iloc[8], 14.0)
        self.assertAlmostEqual(state["rsi_divergence_pivot_distance_coeff"].iloc[8], 6.0 / 14.0)
        self.assertEqual(state["rsi_divergence_last_pivot_type"].iloc[16], "bear")
        self.assertEqual(state["rsi_divergence_last_pivot_price"].iloc[16], 17.0)
        self.assertAlmostEqual(state["rsi_divergence_pivot_distance_coeff"].iloc[16], 8.0 / 17.0)
        self.assertEqual(state["rsi_divergence_last_pivot_type"].iloc[19], "bear")
        self.assertEqual(state["rsi_divergence_last_pivot_price"].iloc[19], 17.0)
        self.assertAlmostEqual(state["rsi_divergence_pivot_distance_coeff"].iloc[19], 6.0 / 17.0)

    def test_compute_rsi_divergence_state_tracks_bullish_last_pivot_reference_coeff(self) -> None:
        lows = [12, 11, 10, 9, 10, 11, 6, 11, 12, 11, 10, 9, 8, 7, 5, 8, 10, 11, 12, 12]
        highs = [value + 4 for value in lows]
        rsi_values = [50, 48, 45, 42, 44, 46, 26, 44, 45, 46, 44, 42, 40, 38, 34, 40, 44, 46, 48, 49]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_last_pivot_type"].iloc[8], "bull")
        self.assertEqual(state["rsi_divergence_last_pivot_price"].iloc[8], 6.0)
        self.assertAlmostEqual(state["rsi_divergence_pivot_distance_coeff"].iloc[8], 6.0 / 14.0)
        self.assertEqual(state["rsi_divergence_last_pivot_type"].iloc[16], "bull")
        self.assertEqual(state["rsi_divergence_last_pivot_price"].iloc[16], 5.0)
        self.assertAlmostEqual(state["rsi_divergence_pivot_distance_coeff"].iloc[16], 5.0 / 12.0)
        self.assertEqual(state["rsi_divergence_last_pivot_type"].iloc[19], "bull")
        self.assertEqual(state["rsi_divergence_last_pivot_price"].iloc[19], 5.0)
        self.assertAlmostEqual(state["rsi_divergence_pivot_distance_coeff"].iloc[19], 5.0 / 14.0)

    def test_compute_rsi_divergence_flags_detects_bullish_daily_divergence(self) -> None:
        lows = [12, 11, 10, 9, 10, 11, 6, 11, 12, 11, 10, 9, 8, 7, 5, 8, 10, 11, 12, 12]
        highs = [value + 4 for value in lows]
        rsi_values = [50, 48, 45, 42, 44, 46, 26, 44, 45, 46, 44, 42, 40, 38, 34, 40, 44, 46, 48, 49]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertTrue((flags.iloc[:14] == "none").all())
        self.assertEqual(flags.iloc[14], "potential-positive")
        self.assertEqual(flags.iloc[15], "potential-positive")
        self.assertEqual(flags.iloc[16], "positive")
        self.assertEqual(flags.iloc[17], "positive")
        self.assertEqual(flags.iloc[18], "positive")
        self.assertEqual(flags.iloc[19], "positive")

    def test_compute_rsi_divergence_flags_uses_weekly_defaults(self) -> None:
        highs = [8, 9, 10, 12, 10, 9, 11, 13, 18, 12, 10, 9]
        lows = [value - 4 for value in highs]
        rsi_values = [48, 50, 52, 74, 58, 56, 57, 60, 66, 58, 55, 52]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="weekly")

        self.assertTrue((flags.iloc[:7] == "none").all())
        self.assertEqual(flags.iloc[7], "potential-negative")
        self.assertEqual(flags.iloc[8], "potential-negative")
        self.assertEqual(flags.iloc[9], "potential-negative")
        self.assertEqual(flags.iloc[10], "negative")
        self.assertEqual(flags.iloc[11], "negative")

    def test_compute_rsi_divergence_flags_detects_unconfirmed_bearish_potential(self) -> None:
        highs = [8, 9, 10, 11, 10, 9, 14, 9, 8, 9, 10, 11, 12, 13, 17, 18]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 58, 56, 54, 74, 55, 54, 53, 54, 56, 58, 60, 66, 65]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_flag"].iloc[-2], "potential-negative")
        self.assertEqual(state["rsi_divergence_flag"].iloc[-1], "potential-negative")
        self.assertFalse(bool(state["rsi_divergence_confirmed"].iloc[-1]))

    def test_compute_rsi_divergence_flags_rejects_stale_bearish_potential_anchor(self) -> None:
        highs = [8, 9, 10, 14, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 17, 16, 15]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 74, 55, 54, 54, 54, 54, 54, 54, 54, 54, 54, 66, 60, 58]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="weekly")

        self.assertEqual(flags.iloc[14], "none")

    def test_compute_rsi_divergence_flags_detects_unconfirmed_bullish_potential(self) -> None:
        lows = [12, 11, 10, 9, 10, 11, 6, 11, 12, 11, 10, 9, 8, 7, 5, 4]
        highs = [value + 4 for value in lows]
        rsi_values = [50, 48, 45, 42, 44, 46, 26, 44, 45, 46, 44, 42, 40, 38, 34, 35]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_flag"].iloc[-2], "potential-positive")
        self.assertEqual(state["rsi_divergence_flag"].iloc[-1], "potential-positive")
        self.assertFalse(bool(state["rsi_divergence_confirmed"].iloc[-1]))

    def test_compute_rsi_divergence_flags_rejects_stale_bullish_potential_anchor(self) -> None:
        lows = [12, 11, 10, 6, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 4, 5, 6]
        highs = [value + 4 for value in lows]
        rsi_values = [50, 48, 45, 26, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 34, 40, 42]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="weekly")

        self.assertEqual(flags.iloc[14], "none")

    def test_compute_rsi_divergence_flags_clears_bearish_potential_when_rsi_reclaims_anchor(self) -> None:
        highs = [8, 9, 10, 11, 10, 9, 14, 9, 8, 9, 10, 11, 12, 13, 17, 18]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 58, 56, 54, 74, 55, 54, 53, 54, 56, 58, 60, 66, 75]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertEqual(flags.iloc[-2], "potential-negative")
        self.assertEqual(flags.iloc[-1], "none")

    def test_compute_rsi_divergence_flags_invalidates_bearish_anchor_after_live_rsi_reclaim(self) -> None:
        highs = [8, 9, 10, 11, 10, 9, 14, 9, 8, 9, 10, 11, 12, 13, 17, 18, 19]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 58, 56, 54, 74, 55, 54, 53, 54, 56, 58, 60, 66, 75, 65]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertEqual(flags.iloc[-3], "potential-negative")
        self.assertEqual(flags.iloc[-2], "none")
        self.assertEqual(flags.iloc[-1], "none")

    def test_compute_rsi_divergence_flags_invalidates_bearish_anchor_when_only_rsi_reclaims(self) -> None:
        highs = [8, 9, 10, 11, 10, 9, 14, 9, 8, 9, 10, 11, 12, 13, 17, 12, 10, 9, 13, 18]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 58, 56, 54, 74, 55, 54, 53, 54, 56, 58, 60, 66, 58, 55, 52, 75, 65]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertEqual(flags.iloc[17], "negative")
        self.assertEqual(flags.iloc[18], "none")
        self.assertEqual(flags.iloc[19], "none")

    def test_compute_rsi_divergence_flags_clears_bullish_potential_when_rsi_breaks_anchor(self) -> None:
        lows = [12, 11, 10, 9, 10, 11, 6, 11, 12, 11, 10, 9, 8, 7, 5, 4]
        highs = [value + 4 for value in lows]
        rsi_values = [50, 48, 45, 42, 44, 46, 26, 44, 45, 46, 44, 42, 40, 38, 34, 25]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertEqual(flags.iloc[-2], "potential-positive")
        self.assertEqual(flags.iloc[-1], "none")

    def test_compute_rsi_divergence_flags_invalidates_bullish_anchor_after_live_rsi_break(self) -> None:
        lows = [12, 11, 10, 9, 10, 11, 6, 11, 12, 11, 10, 9, 8, 7, 5, 4, 3]
        highs = [value + 4 for value in lows]
        rsi_values = [50, 48, 45, 42, 44, 46, 26, 44, 45, 46, 44, 42, 40, 38, 34, 25, 35]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertEqual(flags.iloc[-3], "potential-positive")
        self.assertEqual(flags.iloc[-2], "none")
        self.assertEqual(flags.iloc[-1], "none")

    def test_compute_rsi_divergence_flags_invalidates_bullish_anchor_when_only_rsi_breaks(self) -> None:
        lows = [12, 11, 10, 9, 10, 11, 6, 11, 12, 11, 10, 9, 8, 7, 5, 8, 10, 11, 7, 4]
        highs = [value + 4 for value in lows]
        rsi_values = [50, 48, 45, 42, 44, 46, 26, 44, 45, 46, 44, 42, 40, 38, 34, 40, 44, 46, 25, 35]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertEqual(flags.iloc[17], "positive")
        self.assertEqual(flags.iloc[18], "none")
        self.assertEqual(flags.iloc[19], "none")

    def test_compute_rsi_divergence_flags_keeps_same_anchor_only_when_new_high_is_more_extreme(self) -> None:
        highs = [
            8, 9, 10, 11, 10, 9, 14, 9, 8, 9,
            10, 11, 12, 13, 17, 12, 10, 9, 8, 9,
            10, 11, 12, 13, 16, 12, 10, 9, 8, 8,
        ] + [8] * 50
        lows = [value - 4 for value in highs]
        rsi_values = [
            50, 52, 55, 58, 56, 54, 74, 55, 54, 53,
            54, 56, 58, 60, 66, 58, 55, 52, 50, 51,
            52, 54, 56, 58, 68, 56, 54, 52, 50, 49,
        ] + [48] * 50
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertEqual(flags.iloc[19], "negative")
        self.assertEqual(flags.iloc[20], "negative")

    def test_compute_rsi_divergence_flags_resets_when_rsi_breaks_anchor(self) -> None:
        highs = [8, 9, 10, 11, 10, 9, 14, 9, 8, 9, 10, 11, 12, 13, 17, 12, 10, 9, 10, 11, 12, 13, 18, 12, 10, 9]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 58, 56, 54, 74, 55, 54, 53, 54, 56, 58, 60, 66, 58, 55, 52, 54, 56, 58, 60, 76, 60, 56, 52]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertEqual(flags.iloc[17], "negative")
        self.assertEqual(flags.iloc[22], "none")
        self.assertEqual(flags.iloc[24], "none")
        self.assertEqual(flags.iloc[25], "none")

    def test_compute_rsi_divergence_flags_expires_after_max_active_age(self) -> None:
        highs = [8, 9, 10, 11, 10, 9, 14, 9, 8, 9, 10, 11, 12, 13, 17, 12, 10, 9, 8, 8] + [8] * 58
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 58, 56, 54, 74, 55, 54, 53, 54, 56, 58, 60, 66, 58, 55, 52, 50, 49] + [48] * 58
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertEqual(flags.iloc[19], "negative")
        self.assertEqual(flags.iloc[29], "negative")
        self.assertEqual(flags.iloc[30], "none")

    def test_compute_rsi_divergence_flags_expires_weekly_after_6_bars(self) -> None:
        highs = [8, 9, 10, 14, 10, 9, 11, 13, 17, 12, 10, 9] + [9] * 10
        lows = [value - 4 for value in highs]
        rsi_values = [48, 50, 52, 74, 58, 56, 57, 60, 66, 58, 55, 52] + [50] * 10
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="weekly")

        self.assertEqual(flags.iloc[14], "negative")
        self.assertEqual(flags.iloc[15], "none")

    def test_compute_rsi_divergence_flags_rejects_candidates_beyond_pair_span(self) -> None:
        highs = [8, 9, 10, 11, 10, 9, 14] + [9] * 60 + [10, 11, 12, 13, 17, 12, 10, 9]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 58, 56, 54, 74] + [52] * 60 + [54, 56, 58, 60, 66, 58, 55, 52]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="daily")

        self.assertTrue((flags == "none").all())

    def test_compute_rsi_divergence_flags_rejects_weekly_candidates_beyond_10_bar_pair_span(self) -> None:
        highs = [8, 9, 10, 14] + [9] * 17 + [10, 11, 12, 17, 12, 10, 9]
        lows = [value - 4 for value in highs]
        rsi_values = [50, 52, 55, 74] + [52] * 17 + [54, 56, 58, 66, 58, 55, 52]
        df = self._build_divergence_frame(highs, lows, rsi_values)

        flags = compute_rsi_divergence_flags(df, frequency="weekly")

        self.assertTrue((flags == "none").all())

    def test_compute_rsi_divergence_state_confirms_bullish_instance_after_post_activation_cross(self) -> None:
        df = self._build_divergence_frame(
            [10, 10, 10, 10, 10],
            [6, 6, 6, 6, 6],
            [45, 47, 49, 51, 52],
        )
        bullish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=6.0,
            pivot2_rsi=49.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None, None, None, None, None],
                [None, bullish_state, bullish_state, bullish_state, bullish_state],
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_flag"].tolist(), ["none", "positive", "positive", "positive", "positive"])
        self.assertEqual(state["rsi_divergence_confirmed"].tolist(), [False, False, False, True, True])

    def test_compute_rsi_divergence_state_confirms_bearish_instance_after_post_activation_cross(self) -> None:
        df = self._build_divergence_frame(
            [10, 10, 10, 10, 10],
            [6, 6, 6, 6, 6],
            [55, 53, 51, 49, 48],
        )
        bearish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=10.0,
            pivot2_rsi=51.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None, bearish_state, bearish_state, bearish_state, bearish_state],
                [None, None, None, None, None],
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_flag"].tolist(), ["none", "negative", "negative", "negative", "negative"])
        self.assertEqual(state["rsi_divergence_confirmed"].tolist(), [False, False, False, True, True])

    def test_compute_rsi_divergence_state_persists_confirmed_metadata_until_expiry(self) -> None:
        row_count = 15
        df = self._build_divergence_frame(
            [10] * row_count,
            [6] * row_count,
            [55, 53, 51, 49] + [48] * (row_count - 4),
        )
        bearish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=10.0,
            pivot2_rsi=51.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None] + [bearish_state] * (row_count - 1),
                [None] * row_count,
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_flag"].iloc[13], "negative")
        self.assertTrue(bool(state["rsi_divergence_confirmed"].iloc[13]))
        self.assertEqual(state["rsi_divergence_anchor_date"].iloc[13], "2026-01-02")
        self.assertEqual(state["rsi_divergence_anchor_price"].iloc[13], 10.0)
        self.assertEqual(state["rsi_divergence_pivot_date"].iloc[13], "2026-01-03")
        self.assertEqual(state["rsi_divergence_pivot_price"].iloc[13], 10.0)
        self.assertEqual(state["rsi_divergence_flag"].iloc[14], "none")
        self.assertFalse(bool(state["rsi_divergence_confirmed"].iloc[14]))
        self.assertTrue(pd.isna(state["rsi_divergence_anchor_date"].iloc[14]))
        self.assertTrue(pd.isna(state["rsi_divergence_anchor_price"].iloc[14]))
        self.assertTrue(pd.isna(state["rsi_divergence_pivot_date"].iloc[14]))
        self.assertTrue(pd.isna(state["rsi_divergence_pivot_price"].iloc[14]))

    def test_compute_rsi_divergence_state_marks_bearish_extension_after_confirmation(self) -> None:
        df = self._build_divergence_frame(
            [10, 10, 10, 10, 12],
            [6, 6, 6, 6, 6],
            [55, 53, 51, 49, 48],
        )
        bearish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=10.0,
            pivot2_rsi=51.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None, bearish_state, bearish_state, bearish_state, bearish_state],
                [None, None, None, None, None],
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(
            state["rsi_divergence_flag"].tolist(),
            ["none", "negative", "negative", "negative", "extension-negative"],
        )
        self.assertEqual(state["rsi_divergence_confirmed"].tolist(), [False, False, False, True, False])

    def test_compute_rsi_divergence_state_marks_bullish_extension_after_confirmation(self) -> None:
        df = self._build_divergence_frame(
            [10, 10, 10, 10, 10],
            [6, 6, 6, 6, 5],
            [45, 47, 49, 51, 52],
        )
        bullish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=6.0,
            pivot2_rsi=49.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None, None, None, None, None],
                [None, bullish_state, bullish_state, bullish_state, bullish_state],
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(
            state["rsi_divergence_flag"].tolist(),
            ["none", "positive", "positive", "positive", "extension-positive"],
        )
        self.assertEqual(state["rsi_divergence_confirmed"].tolist(), [False, False, False, True, False])

    def test_compute_rsi_divergence_state_does_not_mark_stale_bearish_extension(self) -> None:
        highs = [10] * 31
        highs[-1] = 12
        lows = [6] * 31
        rsi_values = [55, 53, 51, 49] + [48] * 27
        df = self._build_divergence_frame(highs, lows, rsi_values)
        bearish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=10.0,
            pivot2_rsi=51.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None] + [bearish_state] * 30,
                [None] * 31,
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_flag"].iloc[-1], "none")
        self.assertFalse(bool(state["rsi_divergence_confirmed"].iloc[-1]))

    def test_compute_rsi_divergence_state_marks_same_anchor_bearish_replacement_as_extension(self) -> None:
        df = self._build_divergence_frame(
            [10, 10, 10, 10, 11, 12, 12],
            [6, 6, 6, 6, 6, 6, 6],
            [55, 53, 51, 49, 55, 55, 54],
        )
        first_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=10.0,
            pivot2_rsi=51.0,
        )
        refreshed_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=4,
            confirmation_index=5,
            pivot2_price=12.0,
            pivot2_rsi=50.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None, first_state, first_state, first_state, first_state, refreshed_state, refreshed_state],
                [None, None, None, None, None, None, None],
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(
            state["rsi_divergence_flag"].tolist(),
            ["none", "negative", "negative", "negative", "negative", "extension-negative", "extension-negative"],
        )
        self.assertEqual(state["rsi_divergence_confirmed"].tolist(), [False, False, False, True, True, False, False])

    def test_compute_rsi_divergence_state_does_not_refresh_lifecycle_after_weekly_divergence_expires(self) -> None:
        highs = [10] * 31
        lows = [6] * 31
        rsi_values = [55] * 31
        rsi_values[9] = 51
        rsi_values[10] = 49
        df = self._build_divergence_frame(highs, lows, rsi_values)
        first_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=10.0,
            pivot2_rsi=51.0,
        )
        refreshed_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=17,
            confirmation_index=17,
            pivot2_price=12.0,
            pivot2_rsi=50.0,
        )

        bearish_states = [None] + [first_state] * 14 + [None, None] + [refreshed_state] * 13 + [None]
        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                bearish_states,
                [None] * 31,
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="weekly")

        self.assertEqual(state["rsi_divergence_flag"].iloc[14], "none")
        self.assertFalse(bool(state["rsi_divergence_confirmed"].iloc[14]))
        self.assertEqual(state["rsi_divergence_flag"].iloc[15], "none")
        self.assertEqual(state["rsi_divergence_flag"].iloc[17], "negative")
        self.assertEqual(state["rsi_divergence_flag"].iloc[23], "negative")
        self.assertEqual(state["rsi_divergence_flag"].iloc[24], "negative")
        self.assertFalse(bool(state["rsi_divergence_confirmed"].iloc[17]))

    def test_compute_rsi_divergence_state_refreshes_lifecycle_for_active_same_anchor_weekly_extension(self) -> None:
        highs = [10] * 29
        lows = [6] * 29
        rsi_values = [55] * 29
        rsi_values[9] = 51
        rsi_values[10] = 49
        df = self._build_divergence_frame(highs, lows, rsi_values)
        first_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=10.0,
            pivot2_rsi=51.0,
        )
        refreshed_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=14,
            confirmation_index=14,
            pivot2_price=12.0,
            pivot2_rsi=50.0,
        )

        bearish_states = [None] + [first_state] * 13 + [refreshed_state] * 14 + [None]
        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                bearish_states,
                [None] * 29,
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="weekly")

        self.assertEqual(state["rsi_divergence_flag"].iloc[13], "negative")
        self.assertTrue(bool(state["rsi_divergence_confirmed"].iloc[13]))
        self.assertEqual(state["rsi_divergence_flag"].iloc[14], "negative")
        self.assertEqual(state["rsi_divergence_flag"].iloc[20], "negative")
        self.assertEqual(state["rsi_divergence_flag"].iloc[28], "none")
        self.assertFalse(bool(state["rsi_divergence_confirmed"].iloc[14]))

    def test_compute_rsi_divergence_state_resets_confirmation_when_newer_same_sign_instance_replaces_old(self) -> None:
        df = self._build_divergence_frame(
            [10, 10, 10, 10, 10, 10, 10],
            [6, 6, 6, 6, 6, 6, 6],
            [44, 48, 51, 52, 48, 49, 51],
        )
        first_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=6.0,
            pivot2_rsi=49.0,
        )
        refreshed_state = self._active_divergence(
            anchor_index=3,
            pivot2_index=4,
            confirmation_index=5,
            pivot2_price=5.5,
            pivot2_rsi=48.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None, None, None, None, None, None, None],
                [None, first_state, first_state, first_state, refreshed_state, refreshed_state, refreshed_state],
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_flag"].tolist(), ["none", "positive", "positive", "positive", "positive", "positive", "positive"])
        self.assertEqual(state["rsi_divergence_confirmed"].tolist(), [False, False, True, True, False, False, True])

    def test_compute_rsi_divergence_state_resets_confirmation_when_divergence_changes_sign(self) -> None:
        df = self._build_divergence_frame(
            [10, 10, 10, 10, 10],
            [6, 6, 6, 6, 6],
            [45, 48, 51, 55, 49],
        )
        bullish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=6.0,
            pivot2_rsi=49.0,
        )
        bearish_state = self._active_divergence(
            anchor_index=3,
            pivot2_index=4,
            confirmation_index=4,
            pivot2_price=10.0,
            pivot2_rsi=52.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None, None, None, bearish_state, bearish_state],
                [None, bullish_state, bullish_state, None, None],
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_flag"].tolist(), ["none", "positive", "positive", "negative", "negative"])
        self.assertEqual(state["rsi_divergence_confirmed"].tolist(), [False, False, True, False, True])

    def test_compute_rsi_divergence_state_resets_confirmation_when_divergence_clears(self) -> None:
        df = self._build_divergence_frame(
            [10, 10, 10, 10, 10],
            [6, 6, 6, 6, 6],
            [44, 48, 51, 49, 52],
        )
        bullish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=6.0,
            pivot2_rsi=49.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None, None, None, None, None],
                [None, bullish_state, bullish_state, None, None],
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_flag"].tolist(), ["none", "positive", "positive", "none", "none"])
        self.assertEqual(state["rsi_divergence_confirmed"].tolist(), [False, False, True, False, False])

    def test_compute_rsi_divergence_state_does_not_confirm_without_post_activation_cross(self) -> None:
        df = self._build_divergence_frame(
            [10, 10, 10, 10],
            [6, 6, 6, 6],
            [49, 51, 53, 54],
        )
        bullish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=6.0,
            pivot2_rsi=49.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None, None, None, None],
                [None, bullish_state, bullish_state, bullish_state],
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(state["rsi_divergence_flag"].tolist(), ["none", "positive", "positive", "positive"])
        self.assertEqual(state["rsi_divergence_confirmed"].tolist(), [False, False, False, False])

    def test_compute_rsi_divergence_state_marks_bullish_extension_on_post_confirmation_refresh(self) -> None:
        df = self._build_divergence_frame(
            [10, 10, 10, 10, 10, 10, 10],
            [6.4, 6.2, 6.0, 6.3, 5.8, 6.0, 6.2],
            [44, 48, 49, 51, 50, 47, 51],
        )
        bullish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=6.0,
            pivot2_rsi=49.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None, None, None, None, None, None, None],
                [None, bullish_state, bullish_state, bullish_state, bullish_state, bullish_state, bullish_state],
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(
            state["rsi_divergence_flag"].tolist(),
            [
                "none",
                "positive",
                "positive",
                "positive",
                "extension-positive",
                "extension-positive",
                "extension-positive",
            ],
        )
        self.assertEqual(state["rsi_divergence_confirmed"].tolist(), [False, False, False, True, False, False, True])

    def test_compute_rsi_divergence_state_allows_confirmed_extension_to_extend_again(self) -> None:
        df = self._build_divergence_frame(
            [10] * 9,
            [6.4, 6.2, 6.0, 6.3, 5.8, 6.0, 6.2, 5.6, 6.0],
            [44, 48, 49, 51, 50, 47, 51, 50, 51],
        )
        bullish_state = self._active_divergence(
            anchor_index=1,
            pivot2_index=2,
            confirmation_index=3,
            pivot2_price=6.0,
            pivot2_rsi=49.0,
        )

        with patch(
            "prices_service._build_active_divergence_series",
            side_effect=[
                [None] * 9,
                [None] + [bullish_state] * 8,
            ],
        ):
            state = compute_rsi_divergence_state(df, frequency="daily")

        self.assertEqual(
            state["rsi_divergence_flag"].tolist(),
            [
                "none",
                "positive",
                "positive",
                "positive",
                "extension-positive",
                "extension-positive",
                "extension-positive",
                "extension-positive",
                "extension-positive",
            ],
        )
        self.assertEqual(
            state["rsi_divergence_confirmed"].tolist(),
            [False, False, False, True, False, False, True, False, True],
        )

    def test_import_prices_cache_uses_frequency_specific_divergence_seed_history_lengths(self) -> None:
        build_seed_calls: list[int] = []

        def fake_build_seed_history(existing_df, target_df, selected_tickers, *, lookback_rows):
            del existing_df, target_df, selected_tickers
            build_seed_calls.append(int(lookback_rows))
            return pd.DataFrame()

        fetched_df = pd.DataFrame(
            [
                {
                    "ticker": "AAPL.US",
                    "date": "2026-01-01",
                    "adjusted_close": 10.0,
                    "adjusted_high": 11.0,
                    "adjusted_low": 9.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }
            ]
        )

        with patch("prices_service.resolve_all_price_tickers", return_value=["AAPL.US"]), patch(
            "prices_service.fetch_prices_history",
            return_value=fetched_df,
        ), patch(
            "prices_service.save_prices_cache",
            side_effect=lambda df, frequency, cache_year: Path(f"{frequency}_{cache_year}.jsonl"),
        ), patch(
            "prices_service._build_rsi_seed_history",
            side_effect=fake_build_seed_history,
        ), patch(
            "prices_service.prices_cache_path",
            side_effect=lambda frequency, cache_year: Path(f"{frequency}_{cache_year}.jsonl"),
        ), patch("pathlib.Path.exists", return_value=False):
            import_prices_cache("daily", date(2026, 1, 1), scope="all", cache_year=2026)
            import_prices_cache("weekly", date(2026, 1, 1), scope="all", cache_year=2026)

        self.assertEqual(
            build_seed_calls,
            [
                divergence_seed_history_rows("daily"),
                divergence_seed_history_rows("weekly"),
            ],
        )

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
                "rsi_divergence_flag": "negative",
                "rsi_divergence_confirmed": True,
                "rsi_divergence_anchor_date": "2026-01-20",
                "rsi_divergence_anchor_price": 100.0,
                "rsi_divergence_pivot_date": "2026-02-03",
                "rsi_divergence_pivot_price": 112.0,
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
        self.assertEqual(loaded["rsi_divergence_flag"].tolist(), ["negative"])
        self.assertEqual(loaded["rsi_divergence_confirmed"].tolist(), [True])
        self.assertEqual(loaded["rsi_divergence_anchor_date"].tolist(), ["2026-01-20"])
        self.assertEqual(loaded["rsi_divergence_anchor_price"].tolist(), [100.0])
        self.assertEqual(loaded["rsi_divergence_pivot_date"].tolist(), ["2026-02-03"])
        self.assertEqual(loaded["rsi_divergence_pivot_price"].tolist(), [112.0])


if __name__ == "__main__":
    unittest.main()

