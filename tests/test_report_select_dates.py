import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import equipilot_app
from equipilot_app import (
    _report_select_calendar_label_fragments,
    get_available_report_select_dates,
    list_report_select_dates,
)


class ReportSelectDateTests(unittest.TestCase):
    def _write_report_select(self, path: Path) -> None:
        path.write_text("placeholder", encoding="utf-8")

    def test_list_report_select_dates_dedupes_and_sorts_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            self._write_report_select(data_dir / "report_select_2026-03-10.xlsx")
            self._write_report_select(data_dir / "report_select_2026-03-08.xlsx")
            self._write_report_select(data_dir / "report_select_2026-03-10.csv")
            (data_dir / "report_select_not-a-date.xlsx").write_text("ignored", encoding="utf-8")

            dates = list_report_select_dates(data_dir)

        self.assertEqual(dates, [date(2026, 3, 8), date(2026, 3, 10)])

    def test_cached_available_report_select_dates_reads_from_configured_data_dir(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            self._write_report_select(data_dir / "report_select_2026-03-07.xlsx")
            self._write_report_select(data_dir / "report_select_2026-03-09.xlsx")
            self._write_report_select(data_dir / "report_select_2026-03-07.csv")

            with patch.object(equipilot_app, "DATA_DIR", data_dir):
                equipilot_app._cached_available_report_select_dates.clear()
                dates = get_available_report_select_dates()

        self.assertEqual(dates, (date(2026, 3, 7), date(2026, 3, 9)))

    def test_calendar_label_fragments_cover_streamlit_adapter_variants(self) -> None:
        fragments = _report_select_calendar_label_fragments(date(2026, 3, 13))

        self.assertEqual(
            fragments,
            (
                "Friday, March 13th 2026",
                "Friday, March 13th, 2026",
                "Friday, March 13 2026",
                "Friday, March 13, 2026",
            ),
        )
