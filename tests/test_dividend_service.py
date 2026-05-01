import json
import os
import shutil
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import dividend_service as service


class DividendServiceTests(unittest.TestCase):
    def test_run_dividend_check_matches_bulk_records_against_equipicker_universe(self) -> None:
        universe = {
            "AAPL": {"name": "Apple", "sector": "Technology", "industry": "Consumer Electronics"},
            "MSFT": {"name": "Microsoft", "sector": "Technology", "industry": "Software"},
        }

        def fake_fetch_records(session, eodhd_api_token, exchange, dividend_date):
            del session, eodhd_api_token, exchange
            if dividend_date == date(2026, 4, 1):
                return [
                    {"code": "AAPL.US", "date": "2026-04-01", "value": "0.25", "exchange": "US"},
                    {"code": "NVDA.US", "date": "2026-04-01", "value": "0.01", "exchange": "US"},
                ]
            return [
                {"code": "MSFT.US", "date": "2026-04-02", "value": "0.83", "exchange": "US"}
            ]

        with patch.dict(
            os.environ,
            {
                service.EODHD_API_TOKEN_ENV_ALIASES[0]: "eod-test",
                service.EQUIPICKER_API_KEY_ENV_ALIASES[0]: "equipicker-test",
            },
            clear=True,
        ), patch("dividend_service.fetch_equipicker_universe", return_value=universe), patch(
            "dividend_service.fetch_bulk_dividends_for_date", side_effect=fake_fetch_records
        ):
            result = service.run_dividend_check(date(2026, 4, 1), date(2026, 4, 2))

        self.assertEqual(result["universe_size"], 2)
        self.assertEqual(result["total_dividend_records_seen"], 3)
        self.assertEqual([match["code"] for match in result["matches"]], ["AAPL", "MSFT"])
        self.assertEqual(
            result["page_summary_text"],
            "For the time between 2026-04-01 and 2026-04-02 inclusive, we had dividends on 2 Equipicker universe stocks.",
        )

    def test_run_dividend_check_stops_before_next_eodhd_day_call(self) -> None:
        stop_state = {"stop": False}
        called_dates = []

        def fake_fetch_records(session, eodhd_api_token, exchange, dividend_date):
            del session, eodhd_api_token, exchange
            called_dates.append(dividend_date)
            return [{"code": "AAPL.US", "date": dividend_date.isoformat(), "value": "0.25", "exchange": "US"}]

        def progress_callback(_current_date, _records_seen, _total_seen):
            stop_state["stop"] = True

        with patch.dict(
            os.environ,
            {
                service.EODHD_API_TOKEN_ENV_ALIASES[0]: "eod-test",
                service.EQUIPICKER_API_KEY_ENV_ALIASES[0]: "equipicker-test",
            },
            clear=True,
        ), patch(
            "dividend_service.fetch_equipicker_universe",
            return_value={"AAPL": {"name": "Apple", "sector": "Technology", "industry": "Consumer Electronics"}},
        ), patch("dividend_service.fetch_bulk_dividends_for_date", side_effect=fake_fetch_records):
            result = service.run_dividend_check(
                date(2026, 4, 1),
                date(2026, 4, 3),
                stop_requested=lambda: stop_state["stop"],
                progress_callback=progress_callback,
            )

        self.assertEqual(called_dates, [date(2026, 4, 1)])
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["checked_days"], 1)
        self.assertEqual(result["total_dividend_records_seen"], 1)
        self.assertIn("stopped early", result["page_summary_text"])

    def test_run_dividend_check_accepts_existing_lowercase_eodhd_env_alias(self) -> None:
        seen_tokens = []

        def fake_fetch_records(session, eodhd_api_token, exchange, dividend_date):
            del session, exchange, dividend_date
            seen_tokens.append(eodhd_api_token)
            return []

        with patch.dict(os.environ, {"eodhd": "legacy-token"}, clear=True), patch(
            "dividend_service.fetch_equipicker_universe",
            return_value={"AAPL": {"name": "Apple"}},
        ), patch("dividend_service.fetch_bulk_dividends_for_date", side_effect=fake_fetch_records):
            result = service.run_dividend_check(date(2026, 4, 1), date(2026, 4, 1))

        self.assertEqual(seen_tokens, ["legacy-token"])
        self.assertEqual(result["universe_size"], 1)

    def test_run_dividend_check_emits_progress_log_messages(self) -> None:
        log_messages = []

        def fake_fetch_records(session, eodhd_api_token, exchange, dividend_date):
            del session, eodhd_api_token, exchange, dividend_date
            return [{"code": "AAPL.US", "date": "2026-04-01", "value": "0.25", "exchange": "US"}]

        with patch.dict(
            os.environ,
            {
                service.EODHD_API_TOKEN_ENV_ALIASES[0]: "eod-test",
                service.EQUIPICKER_API_KEY_ENV_ALIASES[0]: "equipicker-test",
            },
            clear=True,
        ), patch(
            "dividend_service.fetch_equipicker_universe",
            return_value={"AAPL": {"name": "Apple", "sector": "Technology", "industry": "Consumer Electronics"}},
        ), patch("dividend_service.fetch_bulk_dividends_for_date", side_effect=fake_fetch_records):
            service.run_dividend_check(
                date(2026, 4, 1),
                date(2026, 4, 1),
                log_callback=log_messages.append,
            )

        joined_logs = "\n".join(log_messages)
        self.assertIn("Starting dividend check", joined_logs)
        self.assertIn("Loading Equipicker universe", joined_logs)
        self.assertIn("Loaded Equipicker universe: 1 tickers.", joined_logs)
        self.assertIn("Checking dividend records for 2026-04-01", joined_logs)
        self.assertIn("Covered ticker AAPL", joined_logs)
        self.assertIn("Matched Equipicker ticker AAPL", joined_logs)
        self.assertIn("Dividend check completed", joined_logs)

    def test_save_and_load_dividend_outputs_round_trip_json_csv_and_html(self) -> None:
        result = {
            "start_date": "2026-04-01",
            "end_date": "2026-04-02",
            "exchange": "US",
            "generated_at": "2026-04-02T10:00:00",
            "universe_size": 1,
            "total_dividend_records_seen": 1,
            "matches": [
                {
                    "code": "AAPL",
                    "name": "Apple",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "exchange": "US",
                    "date": "2026-04-01",
                    "period": "Quarterly",
                    "value": "0.25",
                    "raw_record": json.dumps({"code": "AAPL.US"}),
                }
            ],
            "page_summary_text": "placeholder",
        }

        output_dir = Path("data") / "dividend_checks_test_runtime"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        try:
            output_paths = service.save_dividend_outputs(result, output_dir=output_dir)
            names = service.list_saved_dividend_outputs(output_dir=output_dir)
            loaded = service.load_dividend_output(names[0], output_dir=output_dir)
        finally:
            shutil.rmtree(output_dir, ignore_errors=True)

        self.assertTrue(output_paths["json"].name.endswith(".json"))
        self.assertTrue(output_paths["csv"].name.endswith(".csv"))
        self.assertTrue(output_paths["html"].name.endswith(".html"))
        self.assertEqual(names, ["equipicker_dividends_2026-04-01_to_2026-04-02"])
        self.assertEqual(loaded["matches"][0]["code"], "AAPL")
        self.assertEqual(
            loaded["page_summary_text"],
            "For the time between 2026-04-01 and 2026-04-02 inclusive, we had dividends on 1 Equipicker universe stock.",
        )


if __name__ == "__main__":
    unittest.main()
