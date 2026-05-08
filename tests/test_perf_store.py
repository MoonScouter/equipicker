import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import perf_store
from perf_store import (
    is_cache_fresh,
    load_prices_cached,
    load_report_select_cached,
    save_company_universe_cached,
    source_signature,
)


class PerfStoreTests(unittest.TestCase):
    def test_report_select_parquet_cache_reuses_fresh_source_and_rebuilds_when_stale(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "report_select_2026-05-07.xlsx"
            pd.DataFrame([{"ticker": "AAA", "sector": "Tech"}]).to_excel(source_path, index=False)
            cache_root = tmp_path / "perf_cache" / "report_select"
            calls = 0

            def loader(path: Path) -> pd.DataFrame:
                nonlocal calls
                calls += 1
                return pd.read_excel(path)

            with patch.object(perf_store, "REPORT_SELECT_CACHE_DIR", cache_root):
                first = load_report_select_cached(source_path, loader)
                second = load_report_select_cached(source_path, loader)

                self.assertEqual(calls, 1)
                pd.testing.assert_frame_equal(first, second)

                pd.DataFrame([{"ticker": "BBB", "sector": "Health"}]).to_excel(source_path, index=False)
                rebuilt = load_report_select_cached(source_path, loader)

            self.assertEqual(calls, 2)
            self.assertEqual(rebuilt["ticker"].tolist(), ["BBB"])

    def test_price_cache_parquet_output_matches_loader_output(self) -> None:
        rows = [
            {"ticker": "AAA.US", "date": "2026-01-02", "adjusted_close": 10.0},
            {"ticker": "BBB.US", "date": "2026-01-03", "adjusted_close": 20.0},
        ]
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "prices_daily_2026.jsonl"
            pd.DataFrame(rows).to_json(source_path, orient="records", lines=True)
            cache_root = tmp_path / "perf_cache" / "prices"

            with patch.object(perf_store, "PRICES_CACHE_DIR", cache_root):
                loaded = load_prices_cached(
                    source_path,
                    frequency="daily",
                    cache_year=2026,
                    loader=lambda path: pd.read_json(path, orient="records", lines=True),
                )
                loaded_again = load_prices_cached(
                    source_path,
                    frequency="daily",
                    cache_year=2026,
                    loader=lambda path: pd.DataFrame([{"ticker": "SHOULD_NOT_LOAD"}]),
                )

        expected = pd.DataFrame(rows)
        loaded_compare = loaded.copy()
        loaded_again_compare = loaded_again.copy()
        loaded_compare["date"] = pd.to_datetime(loaded_compare["date"]).dt.strftime("%Y-%m-%d")
        loaded_again_compare["date"] = pd.to_datetime(loaded_again_compare["date"]).dt.strftime("%Y-%m-%d")
        pd.testing.assert_frame_equal(loaded_compare.reset_index(drop=True), expected, check_dtype=False)
        pd.testing.assert_frame_equal(loaded_again_compare.reset_index(drop=True), expected, check_dtype=False)

    def test_cache_freshness_tracks_schema_source_and_extra_metadata(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "source.csv"
            source_path.write_text("a\n1\n", encoding="utf-8")
            parquet_path = tmp_path / "cache.parquet"
            signature = source_signature(source_path)

            perf_store._write_parquet_cache(  # type: ignore[attr-defined]
                pd.DataFrame([{"a": 1}]),
                parquet_path,
                source=signature,
                extra={"kind": "unit"},
            )

            self.assertTrue(is_cache_fresh(parquet_path, source=signature, extra={"kind": "unit"}))
            self.assertFalse(is_cache_fresh(parquet_path, source=signature, extra={"kind": "other"}))

            source_path.write_text("a\n2\n", encoding="utf-8")
            self.assertFalse(is_cache_fresh(parquet_path, source=source_signature(source_path), extra={"kind": "unit"}))

    def test_company_universe_cache_round_trips_warning_and_list_columns(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            cache_root = Path(tmp_dir) / "perf_cache" / "company_universe"
            signatures = {"report": "r1", "prices": "p1"}
            source_df = pd.DataFrame(
                [
                    {
                        "ticker": "AAA.US",
                        "sector": "Tech",
                        "thematic_memberships": ["AI", "Cloud"],
                    }
                ]
            )
            with patch.object(perf_store, "COMPANY_UNIVERSE_CACHE_DIR", cache_root):
                save_company_universe_cached(
                    source_df,
                    eod_date=date(2026, 5, 7),
                    signatures=signatures,
                    warning_message="fallback used",
                )
                cached = perf_store.load_company_universe_cached(date(2026, 5, 7), signatures)

            self.assertIsNotNone(cached)
            cached_df, warning_message = cached
            self.assertEqual(warning_message, "fallback used")
            self.assertEqual(cached_df["ticker"].tolist(), ["AAA.US"])
            self.assertEqual(cached_df["thematic_memberships"].iloc[0].tolist(), ["AI", "Cloud"])


if __name__ == "__main__":
    unittest.main()
