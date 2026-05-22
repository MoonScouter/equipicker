#!/usr/bin/env python3
"""Fetch latest ROIC.ai earnings-call transcripts into local text files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from calendar import month_abbr
from pathlib import Path
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import parse, request


ROIC_API_KEY_ENV = "ROIC_API_KEY"
EQUIPICKER_API_KEY_ENV = "EQUIPICKER_API_KEY"
ROIC_BASE_URL = "https://api.roic.ai/v2/company/earnings-calls/latest"
EQUIPICKER_BASE_URL = "https://ci.equipicker.com/api"
DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "transcripts"

MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


class TranscriptFetchError(RuntimeError):
    """Raised when transcript retrieval or normalization fails."""


def normalize_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if ticker.endswith(".US"):
        ticker = ticker[:-3]
    ticker = re.sub(r"[^A-Z0-9.\-]", "", ticker)
    if not ticker:
        raise TranscriptFetchError("Ticker cannot be empty.")
    return ticker


def parse_tickers(values: list[str], comma_values: str | None) -> list[str]:
    raw_values: list[str] = []
    raw_values.extend(values)
    if comma_values:
        raw_values.extend(comma_values.split(","))
    tickers = [normalize_ticker(value) for value in raw_values if value.strip()]
    unique_tickers: list[str] = []
    seen = set()
    for ticker in tickers:
        if ticker not in seen:
            unique_tickers.append(ticker)
            seen.add(ticker)
    if not unique_tickers:
        raise TranscriptFetchError("Provide at least one ticker, e.g. RKLB or --tickers RKLB,AAPL.")
    return unique_tickers


def resolve_api_key(cli_value: str | None, env_name: str) -> str:
    value = (cli_value or os.environ.get(env_name) or read_windows_env_var(env_name) or "").strip()
    if not value:
        raise TranscriptFetchError(f"Missing API key. Set {env_name} or pass the CLI override.")
    return value


def read_windows_env_var(name: str) -> str:
    if os.name != "nt":
        return ""
    try:
        import winreg
    except ImportError:
        return ""

    locations = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    )
    for root, subkey in locations:
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _ = winreg.QueryValueEx(key, name)
                return str(value)
        except OSError:
            continue
    return ""


def get_json(url: str, timeout: int) -> Any:
    req = request.Request(url, headers={"Accept": "application/json", "User-Agent": "equipicker-transcript-fetcher/1.0"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise TranscriptFetchError(f"HTTP {exc.code}: {detail[:500]}") from exc
    except urllib_error.URLError as exc:
        raise TranscriptFetchError(f"Request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise TranscriptFetchError(f"Request timed out after {timeout}s.") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise TranscriptFetchError(f"Invalid JSON response: {body[:500]}") from exc


def fetch_company_overview(ticker: str, api_key: str, timeout: int) -> Mapping[str, Any]:
    params = parse.urlencode({"api_key": api_key, "ticker": f"{ticker}.US"})
    url = f"{EQUIPICKER_BASE_URL}/company_overview?{params}"
    data = get_json(url, timeout)
    if not isinstance(data, Mapping):
        raise TranscriptFetchError(f"Unexpected company_overview payload for {ticker}.")
    return data


def fetch_latest_transcript(ticker: str, api_key: str, timeout: int) -> Mapping[str, Any]:
    params = parse.urlencode({"apikey": api_key})
    url = f"{ROIC_BASE_URL}/{parse.quote(ticker)}?{params}"
    data = get_json(url, timeout)
    if not isinstance(data, Mapping):
        raise TranscriptFetchError(f"Unexpected transcript payload for {ticker}.")
    return data


def month_number(value: Any) -> int:
    text = str(value or "").strip().lower()
    if text.isdigit():
        month = int(text)
        if 1 <= month <= 12:
            return month
    if text in MONTHS:
        return MONTHS[text]
    raise TranscriptFetchError(f"Could not parse FiscalYearEnd month: {value!r}.")


def fiscal_period_label(year: Any, quarter: Any, fiscal_year_end: Any) -> str:
    fiscal_year = int(year)
    fiscal_quarter = int(quarter)
    if fiscal_quarter not in {1, 2, 3, 4}:
        raise TranscriptFetchError(f"Quarter must be 1-4, received {quarter!r}.")

    fye_month = month_number(fiscal_year_end)
    quarter_end_month = ((fye_month - 1 - (4 - fiscal_quarter) * 3) % 12) + 1
    calendar_year = fiscal_year - 1 if quarter_end_month > fye_month else fiscal_year
    return f"{month_abbr[quarter_end_month].upper()}{str(calendar_year)[-2:]}"


def transcript_filename(ticker: str, period: str) -> str:
    return f"{ticker}_{period}_transcript.txt"


def save_transcript(output_dir: Path, ticker: str, period: str, content: str) -> Path:
    if not content.strip():
        raise TranscriptFetchError(f"Transcript content for {ticker} is empty.")
    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    path = ticker_dir / transcript_filename(ticker, period)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def fetch_and_save_one(args: argparse.Namespace, ticker: str, roic_key: str, equipicker_key: str) -> dict[str, Any]:
    overview = fetch_company_overview(ticker, equipicker_key, args.timeout)
    transcript = fetch_latest_transcript(ticker, roic_key, args.timeout)
    fiscal_year_end = overview.get("FiscalYearEnd")
    period = fiscal_period_label(transcript.get("year"), transcript.get("quarter"), fiscal_year_end)
    content = str(transcript.get("content") or "")
    path = save_transcript(args.output_dir, ticker, period, content)
    return {
        "ticker": ticker,
        "period": period,
        "roic_year": transcript.get("year"),
        "roic_quarter": transcript.get("quarter"),
        "call_date": transcript.get("date"),
        "fiscal_year_end": fiscal_year_end,
        "path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch latest ROIC.ai earnings-call transcripts and save local .txt files."
    )
    parser.add_argument("ticker", nargs="*", help="Ticker(s), without .US preferred. .US suffix is stripped if present.")
    parser.add_argument("--tickers", help="Comma-separated ticker list, e.g. RKLB,AAPL,MSFT.")
    parser.add_argument("--roic-api-key", help=f"ROIC API key. Defaults to {ROIC_API_KEY_ENV}.")
    parser.add_argument("--equipicker-api-key", help=f"Equipicker API key. Defaults to {EQUIPICKER_API_KEY_ENV}.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help=f"Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Request timeout in seconds.")
    args = parser.parse_args()

    try:
        tickers = parse_tickers(args.ticker, args.tickers)
        roic_key = resolve_api_key(args.roic_api_key, ROIC_API_KEY_ENV)
        equipicker_key = resolve_api_key(args.equipicker_api_key, EQUIPICKER_API_KEY_ENV)
        results = [fetch_and_save_one(args, ticker, roic_key, equipicker_key) for ticker in tickers]
        print(json.dumps({"saved": results}, ensure_ascii=False, indent=2))
        return 0
    except (TranscriptFetchError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
