import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from equipilot_app import (
    evaluate_home_import_checks,
    get_latest_indices_cache_date,
    get_latest_prices_cache_date,
    get_report_select_import_state,
    invalidate_prices_cache_views,
)


class HomeImportCheckTests(unittest.TestCase):
    def _write_indices_cache(self, path: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_excel(path, index=False)

    def _write_prices_cache(self, path: Path, rows: list[dict[str, object]]) -> None:
        normalized_rows: list[dict[str, object]] = []
        for row in rows:
            updated = dict(row)
            has_legacy_price_schema = all(
                key in updated
                for key in ["adjusted_high", "adjusted_low", "rs", "obvm"]
            )
            if has_legacy_price_schema:
                updated.setdefault("rsi_14", 50.0)
                updated.setdefault("rsi_divergence_flag", "none")
            normalized_rows.append(updated)
        pd.DataFrame(normalized_rows).to_json(path, orient="records", lines=True)

    def test_report_select_state_detects_existing_repo_snapshot(self) -> None:
        state = get_report_select_import_state(date(2026, 3, 6))

        self.assertTrue(state["report_select_exists"])
        self.assertEqual(Path(state["report_select_path"]).name, "report_select_2026-03-06.xlsx")

    def test_report_select_state_detects_missing_repo_snapshot(self) -> None:
        state = get_report_select_import_state(date(2026, 3, 5))

        self.assertFalse(state["report_select_exists"])
        candidates = state["report_select_candidates"]
        self.assertEqual(candidates[0].name, "report_select_2026-03-05.xlsx")
        self.assertEqual(candidates[1].name, "report_select_2026-03-05.csv")

    def test_latest_indices_cache_date_aggregates_across_yearly_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            first_path = Path(tmp_dir) / "indices-prices-2025.xlsx"
            second_path = Path(tmp_dir) / "indices-prices-2026.xlsx"
            self._write_indices_cache(
                first_path,
                [{"ticker": "GSPC.INDX", "date": pd.Timestamp("2025-12-31"), "adjusted_close": 100.0}],
            )
            self._write_indices_cache(
                second_path,
                [{"ticker": "IXIC.INDX", "date": pd.Timestamp("2026-01-03"), "adjusted_close": 200.0}],
            )

            latest_date, errors = get_latest_indices_cache_date([first_path, second_path])

        self.assertEqual(latest_date, date(2026, 1, 3))
        self.assertEqual(errors, [])

    def test_latest_indices_cache_date_collects_invalid_file_errors(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            valid_path = Path(tmp_dir) / "indices-prices-2026.xlsx"
            invalid_path = Path(tmp_dir) / "indices-prices-2027.xlsx"
            self._write_indices_cache(
                valid_path,
                [{"ticker": "GSPC.INDX", "date": pd.Timestamp("2026-03-07"), "adjusted_close": 100.0}],
            )
            pd.DataFrame([{"ticker": "GSPC.INDX", "date": pd.Timestamp("2026-03-08")}]).to_excel(
                invalid_path,
                index=False,
            )

            latest_date, errors = get_latest_indices_cache_date([valid_path, invalid_path])

        self.assertEqual(latest_date, date(2026, 3, 7))
        self.assertEqual(len(errors), 1)
        self.assertIn("missing required columns", errors[0].lower())

    def test_latest_prices_cache_date_aggregates_daily_jsonl_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            first_path = Path(tmp_dir) / "prices_daily_2025.jsonl"
            second_path = Path(tmp_dir) / "prices_daily_2026.jsonl"
            self._write_prices_cache(
                first_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2025-12-31",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )
            self._write_prices_cache(
                second_path,
                [{
                    "ticker": "MSFT.US",
                    "date": "2026-01-05",
                    "adjusted_close": 200.0,
                    "adjusted_high": 201.0,
                    "adjusted_low": 198.0,
                    "rs": 3.0,
                    "obvm": 4.0,
                }],
            )

            latest_date, errors = get_latest_prices_cache_date("daily", [first_path, second_path])

        self.assertEqual(latest_date, date(2026, 1, 5))
        self.assertEqual(errors, [])

    def test_latest_prices_cache_date_uses_latest_weekly_date_without_filter(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            weekly_path = Path(tmp_dir) / "prices_weekly_2026.jsonl"
            self._write_prices_cache(
                weekly_path,
                [
                    {
                        "ticker": "AAPL.US",
                        "date": "2026-03-06",
                        "adjusted_close": 100.0,
                        "adjusted_high": 101.0,
                        "adjusted_low": 99.0,
                        "rs": 1.0,
                        "obvm": 2.0,
                    },
                    {
                        "ticker": "AAPL.US",
                        "date": "2026-03-13",
                        "adjusted_close": 102.0,
                        "adjusted_high": 103.0,
                        "adjusted_low": 100.0,
                        "rs": 1.5,
                        "obvm": 2.5,
                    },
                ],
            )

            latest_date, errors = get_latest_prices_cache_date(
                "weekly",
                [weekly_path],
            )

        self.assertEqual(latest_date, date(2026, 3, 13))
        self.assertEqual(errors, [])

    def test_latest_prices_cache_date_collects_invalid_file_errors(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            valid_path = Path(tmp_dir) / "prices_daily_2026.jsonl"
            invalid_path = Path(tmp_dir) / "prices_daily_2027.jsonl"
            self._write_prices_cache(
                valid_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-07",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )
            self._write_prices_cache(
                invalid_path,
                [{"ticker": "AAPL.US", "date": "2026-03-08", "adjusted_close": 100.0}],
            )

            latest_date, errors = get_latest_prices_cache_date("daily", [valid_path, invalid_path])

        self.assertEqual(latest_date, date(2026, 3, 7))
        self.assertEqual(len(errors), 1)
        self.assertIn("missing required columns", errors[0].lower())

    def test_evaluate_home_import_checks_accepts_equal_or_newer_dates_for_indices_and_prices(self) -> None:
        selected_date = date(2026, 3, 6)
        with TemporaryDirectory() as tmp_dir:
            indices_path = Path(tmp_dir) / "indices-prices-2026.xlsx"
            daily_path = Path(tmp_dir) / "prices_daily_2026.jsonl"
            weekly_path = Path(tmp_dir) / "prices_weekly_2026.jsonl"
            self._write_indices_cache(
                indices_path,
                [{"ticker": "GSPC.INDX", "date": pd.Timestamp("2026-03-07"), "adjusted_close": 100.0}],
            )
            self._write_prices_cache(
                daily_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-07",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )
            self._write_prices_cache(
                weekly_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-06",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )

            state = evaluate_home_import_checks(
                selected_date,
                cache_paths=[indices_path],
                daily_price_cache_paths=[daily_path],
                weekly_price_cache_paths=[weekly_path],
            )

        self.assertTrue(state["indices_check_passed"])
        self.assertTrue(state["daily_prices_check_passed"])
        self.assertTrue(state["weekly_prices_check_passed"])
        self.assertTrue(state["overall_ready"])

    def test_evaluate_home_import_checks_passes_when_weekly_cache_is_after_selected_date(self) -> None:
        selected_date = date(2026, 3, 6)
        with TemporaryDirectory() as tmp_dir:
            indices_path = Path(tmp_dir) / "indices-prices-2026.xlsx"
            daily_path = Path(tmp_dir) / "prices_daily_2026.jsonl"
            weekly_path = Path(tmp_dir) / "prices_weekly_2026.jsonl"
            self._write_indices_cache(
                indices_path,
                [{"ticker": "GSPC.INDX", "date": pd.Timestamp("2026-03-07"), "adjusted_close": 100.0}],
            )
            self._write_prices_cache(
                daily_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-07",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )
            self._write_prices_cache(
                weekly_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-13",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )

            state = evaluate_home_import_checks(
                selected_date,
                cache_paths=[indices_path],
                daily_price_cache_paths=[daily_path],
                weekly_price_cache_paths=[weekly_path],
            )

        self.assertTrue(state["weekly_prices_check_passed"])
        self.assertTrue(state["overall_ready"])

    def test_invalidate_prices_cache_views_clears_raw_and_lookup_caches(self) -> None:
        with patch("equipilot_app.load_prices_cache_file.clear") as clear_prices_cache_file:
            with patch("equipilot_app.build_price_history_lookup.clear") as clear_price_history_lookup:
                with patch("equipilot_app._load_company_divergence_metrics_for_date.clear") as clear_divergence_metrics:
                    invalidate_prices_cache_views()

        clear_prices_cache_file.assert_called_once_with()
        clear_price_history_lookup.assert_called_once_with()
        clear_divergence_metrics.assert_called_once_with()

    def test_evaluate_home_import_checks_passes_when_weekly_date_plus_four_days_covers_selected_date(self) -> None:
        selected_date = date(2026, 3, 10)
        with TemporaryDirectory() as tmp_dir:
            indices_path = Path(tmp_dir) / "indices-prices-2026.xlsx"
            daily_path = Path(tmp_dir) / "prices_daily_2026.jsonl"
            weekly_path = Path(tmp_dir) / "prices_weekly_2026.jsonl"
            self._write_indices_cache(
                indices_path,
                [{"ticker": "GSPC.INDX", "date": pd.Timestamp("2026-03-10"), "adjusted_close": 100.0}],
            )
            self._write_prices_cache(
                daily_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-10",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )
            self._write_prices_cache(
                weekly_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-06",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )

            with patch(
                "equipilot_app.get_report_select_import_state",
                return_value={
                    "selected_date": selected_date,
                    "report_select_exists": True,
                    "report_select_path": Path("report_select_2026-03-10.xlsx"),
                    "report_select_candidates": (
                        Path("report_select_2026-03-10.xlsx"),
                        Path("report_select_2026-03-10.csv"),
                    ),
                },
            ):
                state = evaluate_home_import_checks(
                    selected_date,
                    cache_paths=[indices_path],
                    daily_price_cache_paths=[daily_path],
                    weekly_price_cache_paths=[weekly_path],
                )

        self.assertTrue(state["weekly_prices_check_passed"])
        self.assertTrue(state["overall_ready"])

    def test_evaluate_home_import_checks_fails_when_weekly_date_plus_four_days_is_still_before_selected_date(self) -> None:
        selected_date = date(2026, 3, 11)
        with TemporaryDirectory() as tmp_dir:
            indices_path = Path(tmp_dir) / "indices-prices-2026.xlsx"
            daily_path = Path(tmp_dir) / "prices_daily_2026.jsonl"
            weekly_path = Path(tmp_dir) / "prices_weekly_2026.jsonl"
            self._write_indices_cache(
                indices_path,
                [{"ticker": "GSPC.INDX", "date": pd.Timestamp("2026-03-11"), "adjusted_close": 100.0}],
            )
            self._write_prices_cache(
                daily_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-11",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )
            self._write_prices_cache(
                weekly_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-06",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )

            state = evaluate_home_import_checks(
                selected_date,
                cache_paths=[indices_path],
                daily_price_cache_paths=[daily_path],
                weekly_price_cache_paths=[weekly_path],
            )

        self.assertFalse(state["weekly_prices_check_passed"])
        self.assertFalse(state["overall_ready"])

    def test_evaluate_home_import_checks_passes_for_equal_daily_indices_and_weekly_dates(self) -> None:
        selected_date = date(2026, 3, 6)
        with TemporaryDirectory() as tmp_dir:
            indices_path = Path(tmp_dir) / "indices-prices-2026.xlsx"
            daily_path = Path(tmp_dir) / "prices_daily_2026.jsonl"
            weekly_path = Path(tmp_dir) / "prices_weekly_2026.jsonl"
            self._write_indices_cache(
                indices_path,
                [{"ticker": "GSPC.INDX", "date": pd.Timestamp("2026-03-06"), "adjusted_close": 100.0}],
            )
            self._write_prices_cache(
                daily_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-06",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )
            self._write_prices_cache(
                weekly_path,
                [{
                    "ticker": "AAPL.US",
                    "date": "2026-03-06",
                    "adjusted_close": 100.0,
                    "adjusted_high": 101.0,
                    "adjusted_low": 99.0,
                    "rs": 1.0,
                    "obvm": 2.0,
                }],
            )

            state = evaluate_home_import_checks(
                selected_date,
                cache_paths=[indices_path],
                daily_price_cache_paths=[daily_path],
                weekly_price_cache_paths=[weekly_path],
            )

        self.assertTrue(state["indices_check_passed"])
        self.assertTrue(state["daily_prices_check_passed"])
        self.assertTrue(state["weekly_prices_check_passed"])
        self.assertTrue(state["overall_ready"])


if __name__ == "__main__":
    unittest.main()



