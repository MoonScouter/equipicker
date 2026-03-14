import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from equipilot_app import (
    evaluate_home_import_checks,
    get_latest_indices_cache_date,
    get_report_select_import_state,
)


class HomeImportCheckTests(unittest.TestCase):
    def _write_indices_cache(self, path: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_excel(path, index=False)

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

    def test_evaluate_home_import_checks_accepts_equal_or_newer_indices_date(self) -> None:
        selected_date = date(2026, 3, 6)
        with TemporaryDirectory() as tmp_dir:
            equal_path = Path(tmp_dir) / "indices-prices-2026.xlsx"
            newer_path = Path(tmp_dir) / "indices-prices-2027.xlsx"
            self._write_indices_cache(
                equal_path,
                [{"ticker": "GSPC.INDX", "date": pd.Timestamp("2026-03-06"), "adjusted_close": 100.0}],
            )
            self._write_indices_cache(
                newer_path,
                [{"ticker": "GSPC.INDX", "date": pd.Timestamp("2026-03-07"), "adjusted_close": 101.0}],
            )

            equal_state = evaluate_home_import_checks(selected_date, cache_paths=[equal_path])
            newer_state = evaluate_home_import_checks(selected_date, cache_paths=[equal_path, newer_path])

        self.assertTrue(equal_state["indices_check_passed"])
        self.assertTrue(newer_state["indices_check_passed"])


if __name__ == "__main__":
    unittest.main()
