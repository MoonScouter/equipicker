"""CLI entrypoint for scheduled report_select cache generation."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from report_select_service import generate_report_select_cache, resolve_anchor_date

LOGGER = logging.getLogger("report_select_scheduler")


def configure_logging(log_file: Path | None) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(stream_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOGGER.addHandler(file_handler)

    # Propagate to root so lower-level loggers (SQL/cache) are also captured.
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().handlers.clear()
    for handler in LOGGER.handlers:
        logging.getLogger().addHandler(handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scheduled report_select cache generator")
    parser.add_argument(
        "--mode",
        default="previous-us-trading-day",
        choices=["previous-us-trading-day", "today-bucharest", "explicit-date"],
        help="Anchor date resolution mode.",
    )
    parser.add_argument(
        "--run-sql",
        action="store_true",
        help="Force SQL execution even when cache file exists.",
    )
    parser.add_argument(
        "--time-zone",
        default="Europe/Bucharest",
        help="IANA timezone for date resolution.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Explicit anchor date in YYYY-MM-DD (required if --mode explicit-date).",
    )
    parser.add_argument(
        "--log-file",
        default="logs/report_select_scheduler.log",
        help="Optional log file path. Pass empty string to disable file logs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_file = Path(args.log_file) if args.log_file else None
    configure_logging(log_file)

    start = time.perf_counter()
    LOGGER.info("Scheduler run started")
    LOGGER.info("Args: mode=%s run_sql=%s time_zone=%s", args.mode, args.run_sql, args.time_zone)

    try:
        anchor_date = resolve_anchor_date(
            mode=args.mode,
            time_zone=args.time_zone,
            explicit_date=args.date,
        )
        LOGGER.info("Resolved anchor date: %s", anchor_date.isoformat())
        output_path = generate_report_select_cache(anchor_date=anchor_date, run_sql=args.run_sql)
        if not output_path.exists():
            raise FileNotFoundError(f"Output file missing after run: {output_path}")
    except Exception:
        LOGGER.exception("Scheduler run failed")
        return 1

    duration = time.perf_counter() - start
    LOGGER.info("Scheduler run finished successfully in %.2fs", duration)
    LOGGER.info("Output path: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
