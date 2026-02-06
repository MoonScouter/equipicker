"""Shared helpers for report_select cache generation and scheduling."""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from equipicker_connect import bucharest_today_str, get_report_dataframe, report_cache_path

logger = logging.getLogger(__name__)


def generate_report_select_cache(anchor_date: date, run_sql: bool) -> Path:
    """Generate (or load) report_select cache for an anchor date and return the output path."""
    cache_file = report_cache_path(cache_date=anchor_date)
    logger.info(
        "Starting report_select generation for %s (run_sql=%s) -> %s",
        anchor_date.isoformat(),
        run_sql,
        cache_file,
    )
    df = get_report_dataframe(
        run_sql=run_sql,
        use_cache=True,
        cache_date=anchor_date,
        eod_as_of_date=anchor_date,
    )
    logger.info("Rows loaded: %s", len(df))
    if not cache_file.exists():
        raise FileNotFoundError(f"report_select output not found after generation: {cache_file}")
    logger.info("report_select generation completed: %s", cache_file)
    return cache_file


def resolve_anchor_date(mode: str, time_zone: str, explicit_date: str | None = None) -> date:
    """Resolve anchor date for scheduled runs."""
    mode_normalized = mode.strip().lower()
    if mode_normalized == "previous-us-trading-day":
        return previous_us_trading_day(time_zone)
    if mode_normalized == "today-bucharest":
        if time_zone != "Europe/Bucharest":
            now = datetime.now(ZoneInfo(time_zone))
            return now.date()
        return date.fromisoformat(bucharest_today_str())
    if mode_normalized == "explicit-date":
        if not explicit_date:
            raise ValueError("--date is required when --mode explicit-date")
        return date.fromisoformat(explicit_date)
    raise ValueError(f"Unsupported mode: {mode}")


def previous_us_trading_day(time_zone: str = "Europe/Bucharest") -> date:
    """Return previous NYSE trading day relative to the local date in given timezone."""
    try:
        import pandas_market_calendars as mcal
    except ImportError as exc:
        raise RuntimeError(
            "Missing optional dependency 'pandas_market_calendars'. "
            "Install requirements and rerun scheduler."
        ) from exc

    local_today = datetime.now(ZoneInfo(time_zone)).date()
    # 30-day lookback safely covers weekends + long holiday periods.
    start_date = local_today - timedelta(days=30)
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date.isoformat(), end_date=local_today.isoformat())
    if schedule.empty:
        raise RuntimeError("NYSE schedule is empty for lookback window; cannot resolve previous trading day.")

    sessions = [ts.date() for ts in schedule.index]
    prior_sessions = [session for session in sessions if session < local_today]
    if not prior_sessions:
        raise RuntimeError(f"No previous NYSE trading day found before {local_today.isoformat()}.")
    return prior_sessions[-1]
