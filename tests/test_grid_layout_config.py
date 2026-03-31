import tempfile
import unittest
from pathlib import Path

from grid_layout_config import clear_grid_layout, load_grid_layout, load_grid_layouts, save_grid_layout


class GridLayoutConfigTests(unittest.TestCase):
    def test_save_and_load_grid_layout_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "grid_layouts.json"

            save_grid_layout("thematics_company_grid", ["Ticker", "FS", "TS"], config_path)

            self.assertEqual(
                load_grid_layout("thematics_company_grid", config_path),
                ["Ticker", "FS", "TS"],
            )

    def test_grid_layouts_are_isolated_per_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "grid_layouts.json"

            save_grid_layout("technical_company_grid", ["Ticker", "TS"], config_path)
            save_grid_layout("fundamental_company_grid", ["Ticker", "FS"], config_path)

            layouts = load_grid_layouts(config_path)
            self.assertEqual(layouts["technical_company_grid"]["visible_columns"], ["Ticker", "TS"])
            self.assertEqual(layouts["fundamental_company_grid"]["visible_columns"], ["Ticker", "FS"])

    def test_clear_grid_layout_removes_only_target_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "grid_layouts.json"

            save_grid_layout("technical_company_grid", ["Ticker", "TS"], config_path)
            save_grid_layout("fundamental_company_grid", ["Ticker", "FS"], config_path)

            clear_grid_layout("technical_company_grid", config_path)

            self.assertIsNone(load_grid_layout("technical_company_grid", config_path))
            self.assertEqual(
                load_grid_layout("fundamental_company_grid", config_path),
                ["Ticker", "FS"],
            )

    def test_load_grid_layouts_ignores_invalid_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "grid_layouts.json"
            config_path.write_text(
                """
                {
                  "valid_surface": {"visible_columns": ["Ticker", "Ticker", "TS", ""]},
                  "invalid_surface": {"visible_columns": "Ticker"},
                  "also_invalid": [],
                  "": {"visible_columns": ["FS"]}
                }
                """,
                encoding="utf-8",
            )

            layouts = load_grid_layouts(config_path)

            self.assertEqual(layouts, {"valid_surface": {"visible_columns": ["Ticker", "TS"]}})


if __name__ == "__main__":
    unittest.main()
