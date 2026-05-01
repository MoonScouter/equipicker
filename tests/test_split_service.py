import json
import os
import shutil
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import split_service as service


class SplitServiceTests(unittest.TestCase):
    def test_normalize_split_symbol_removes_us_suffix_and_share_class_dot(self) -> None:
        self.assertEqual(service.normalize_split_symbol("brk.b.us"), "BRK-B")
        self.assertEqual(service.normalize_split_symbol(" AAPL.US "), "AAPL")
        self.assertEqual(service.normalize_split_symbol("GOOG"), "GOOG")

    def test_run_split_check_matches_bulk_records_against_equipicker_universe(self) -> None:
        universe = {
            "AAPL": {"name": "Apple", "sector": "Technology", "industry": "Consumer Electronics"},
            "BRK-B": {"name": "Berkshire", "sector": "Financial Services", "industry": "Insurance"},
        }

        def fake_fetch_records(session, eodhd_api_token, exchange, split_date):
            del session, eodhd_api_token, exchange
            if split_date == date(2026, 4, 1):
                return [
                    {"code": "AAPL.US", "date": "2026-04-01", "split": "2/1", "exchange": "US"},
                    {"code": "MSFT.US", "date": "2026-04-01", "split": "3/1", "exchange": "US"},
                ]
            return [{"code": "BRK.B.US", "date": "2026-04-02", "split": "50/1", "exchange": "US"}]

        with patch.dict(
            os.environ,
            {
                service.EODHD_API_TOKEN_ENV: "eod-test",
                service.EQUIPICKER_API_KEY_ENV: "equipicker-test",
            },
            clear=True,
        ), patch("split_service.fetch_equipicker_universe", return_value=universe), patch(
            "split_service.fetch_bulk_splits_for_date", side_effect=fake_fetch_records
        ):
            result = service.run_split_check(date(2026, 4, 1), date(2026, 4, 2))

        self.assertEqual(result["universe_size"], 2)
        self.assertEqual(result["total_split_records_seen"], 3)
        self.assertEqual([match["code"] for match in result["matches"]], ["AAPL", "BRK-B"])
        self.assertEqual(
            result["page_summary_text"],
            "For the time between 2026-04-01 and 2026-04-02 inclusive, we had splits on 2 Equipicker universe stocks.",
        )

    def test_run_split_check_stops_before_next_eodhd_day_call(self) -> None:
        stop_state = {"stop": False}
        called_dates = []

        def fake_fetch_records(session, eodhd_api_token, exchange, split_date):
            del session, eodhd_api_token, exchange
            called_dates.append(split_date)
            return [{"code": "AAPL.US", "date": split_date.isoformat(), "split": "2/1", "exchange": "US"}]

        def progress_callback(_current_date, _records_seen, _total_seen):
            stop_state["stop"] = True

        with patch.dict(
            os.environ,
            {
                service.EODHD_API_TOKEN_ENV: "eod-test",
                service.EQUIPICKER_API_KEY_ENV: "equipicker-test",
            },
            clear=True,
        ), patch(
            "split_service.fetch_equipicker_universe",
            return_value={"AAPL": {"name": "Apple", "sector": "Technology", "industry": "Consumer Electronics"}},
        ), patch("split_service.fetch_bulk_splits_for_date", side_effect=fake_fetch_records):
            result = service.run_split_check(
                date(2026, 4, 1),
                date(2026, 4, 3),
                stop_requested=lambda: stop_state["stop"],
                progress_callback=progress_callback,
            )

        self.assertEqual(called_dates, [date(2026, 4, 1)])
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["checked_days"], 1)
        self.assertEqual(result["total_split_records_seen"], 1)
        self.assertIn("stopped early", result["page_summary_text"])

    def test_run_split_check_accepts_existing_lowercase_eodhd_env_alias(self) -> None:
        seen_tokens = []

        def fake_fetch_records(session, eodhd_api_token, exchange, split_date):
            del session, exchange, split_date
            seen_tokens.append(eodhd_api_token)
            return []

        with patch.dict(os.environ, {"eodhd": "legacy-token"}, clear=True), patch(
            "split_service.fetch_equipicker_universe",
            return_value={"AAPL": {"name": "Apple"}},
        ), patch("split_service.fetch_bulk_splits_for_date", side_effect=fake_fetch_records):
            result = service.run_split_check(date(2026, 4, 1), date(2026, 4, 1))

        self.assertEqual(seen_tokens, ["legacy-token"])
        self.assertEqual(result["universe_size"], 1)

    def test_run_split_check_emits_progress_log_messages(self) -> None:
        log_messages = []

        def fake_fetch_records(session, eodhd_api_token, exchange, split_date):
            del session, eodhd_api_token, exchange, split_date
            return [{"code": "AAPL.US", "date": "2026-04-01", "split": "2/1", "exchange": "US"}]

        with patch.dict(
            os.environ,
            {
                service.EODHD_API_TOKEN_ENV: "eod-test",
                service.EQUIPICKER_API_KEY_ENV: "equipicker-test",
            },
            clear=True,
        ), patch(
            "split_service.fetch_equipicker_universe",
            return_value={"AAPL": {"name": "Apple", "sector": "Technology", "industry": "Consumer Electronics"}},
        ), patch("split_service.fetch_bulk_splits_for_date", side_effect=fake_fetch_records):
            service.run_split_check(
                date(2026, 4, 1),
                date(2026, 4, 1),
                log_callback=log_messages.append,
            )

        joined_logs = "\n".join(log_messages)
        self.assertIn("Starting split check", joined_logs)
        self.assertIn("Loading Equipicker universe", joined_logs)
        self.assertIn("Loaded Equipicker universe: 1 tickers.", joined_logs)
        self.assertIn("Checking split records for 2026-04-01", joined_logs)
        self.assertIn("Covered ticker AAPL", joined_logs)
        self.assertIn("Matched Equipicker ticker AAPL", joined_logs)
        self.assertIn("Split check completed", joined_logs)

    def test_save_and_load_split_outputs_round_trip_json_csv_and_html(self) -> None:
        result = {
            "start_date": "2026-04-01",
            "end_date": "2026-04-02",
            "exchange": "US",
            "generated_at": "2026-04-02T10:00:00",
            "universe_size": 1,
            "total_split_records_seen": 1,
            "matches": [
                {
                    "code": "AAPL",
                    "name": "Apple",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "exchange": "US",
                    "date": "2026-04-01",
                    "split": "2/1",
                    "raw_record": json.dumps({"code": "AAPL.US"}),
                }
            ],
            "page_summary_text": "For the time between 2026-04-01 and 2026-04-02 inclusive, split found.",
        }

        output_dir = Path("data") / "split_checks_test_runtime"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        try:
            output_paths = service.save_split_outputs(result, output_dir=output_dir)
            names = service.list_saved_split_outputs(output_dir=output_dir)
            loaded = service.load_split_output(names[0], output_dir=output_dir)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

        self.assertTrue(output_paths["json"].name.endswith(".json"))
        self.assertTrue(output_paths["csv"].name.endswith(".csv"))
        self.assertTrue(output_paths["html"].name.endswith(".html"))
        self.assertEqual(names, ["equipicker_splits_2026-04-01_to_2026-04-02"])
        self.assertEqual(loaded["matches"][0]["code"], "AAPL")
        self.assertEqual(
            loaded["page_summary_text"],
            "For the time between 2026-04-01 and 2026-04-02 inclusive, we had splits on 1 Equipicker universe stock.",
        )


if __name__ == "__main__":
    unittest.main()
