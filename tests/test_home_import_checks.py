import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from equipilot_app import (
    evaluate_home_import_checks,
    get_latest_indices_cache_date,
    get_latest_prices_cache_date,
    get_report_select_import_state,
)


class HomeImportCheckTests(unittest.TestCase):
    def _write_indices_cache(self, path: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_excel(path, index=False)

    def _write_prices_cache(self, path: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_json(path, orient="records", lines=True)

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

    def test_latest_prices_cache_date_applies_weekly_on_or_before_filter(self) -> None:
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
                on_or_before=date(2026, 3, 10),
            )

        self.assertEqual(latest_date, date(2026, 3, 6))
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

    def test_evaluate_home_import_checks_requires_fresh_daily_and_indices_plus_weekly_on_or_before(self) -> None:
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

    def test_evaluate_home_import_checks_fails_when_weekly_cache_is_only_after_selected_date(self) -> None:
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

        self.assertFalse(state["weekly_prices_check_passed"])
        self.assertFalse(state["overall_ready"])

    def test_evaluate_home_import_checks_fails_for_equal_daily_or_indices_dates(self) -> None:
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

        self.assertFalse(state["indices_check_passed"])
        self.assertFalse(state["daily_prices_check_passed"])
        self.assertTrue(state["weekly_prices_check_passed"])
        self.assertFalse(state["overall_ready"])


if __name__ == "__main__":
    unittest.main()
