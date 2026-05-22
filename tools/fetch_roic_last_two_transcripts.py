#!/usr/bin/env python3
"""Fetch the last two ROIC.ai earnings-call transcripts into local text files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping
from urllib import parse

try:
    from .fetch_roic_latest_transcripts import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_TIMEOUT_SECONDS,
        EQUIPICKER_API_KEY_ENV,
        ROIC_API_KEY_ENV,
        TranscriptFetchError,
        fetch_company_overview,
        fiscal_period_label,
        get_json,
        parse_tickers,
        resolve_api_key,
        save_transcript,
    )
except ImportError:
    from fetch_roic_latest_transcripts import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_TIMEOUT_SECONDS,
        EQUIPICKER_API_KEY_ENV,
        ROIC_API_KEY_ENV,
        TranscriptFetchError,
        fetch_company_overview,
        fiscal_period_label,
        get_json,
        parse_tickers,
        resolve_api_key,
        save_transcript,
    )


ROIC_LIST_URL = "https://api.roic.ai/v2/company/earnings-calls/list"
ROIC_TRANSCRIPT_URL = "https://api.roic.ai/v2/company/earnings-calls/transcript"
DEFAULT_TRANSCRIPT_COUNT = 2


def fetch_transcript_list(ticker: str, api_key: str, timeout: int) -> list[Mapping[str, Any]]:
    params = parse.urlencode({"apikey": api_key, "limit": DEFAULT_TRANSCRIPT_COUNT})
    url = f"{ROIC_LIST_URL}/{parse.quote(ticker)}?{params}"
    data = get_json(url, timeout)
    if not isinstance(data, list):
        raise TranscriptFetchError(f"Unexpected transcript list payload for {ticker}.")
    calls: list[Mapping[str, Any]] = []
    for item in data[:DEFAULT_TRANSCRIPT_COUNT]:
        if not isinstance(item, Mapping):
            raise TranscriptFetchError(f"Unexpected transcript list item for {ticker}: {item!r}")
        calls.append(item)
    if not calls:
        raise TranscriptFetchError(f"No transcripts found for {ticker}.")
    return calls


def fetch_transcript_by_period(ticker: str, api_key: str, year: Any, quarter: Any, timeout: int) -> Mapping[str, Any]:
    params = parse.urlencode({"apikey": api_key, "year": int(year), "quarter": int(quarter)})
    url = f"{ROIC_TRANSCRIPT_URL}/{parse.quote(ticker)}?{params}"
    data = get_json(url, timeout)
    if not isinstance(data, Mapping):
        raise TranscriptFetchError(f"Unexpected transcript payload for {ticker} {year} Q{quarter}.")
    return data


def fetch_and_save_last_two(args: argparse.Namespace, ticker: str, roic_key: str, equipicker_key: str) -> list[dict[str, Any]]:
    overview = fetch_company_overview(ticker, equipicker_key, args.timeout)
    fiscal_year_end = overview.get("FiscalYearEnd")
    calls = fetch_transcript_list(ticker, roic_key, args.timeout)
    results = []

    for call in calls:
        year = call.get("year")
        quarter = call.get("quarter")
        transcript = fetch_transcript_by_period(ticker, roic_key, year, quarter, args.timeout)
        period = fiscal_period_label(transcript.get("year", year), transcript.get("quarter", quarter), fiscal_year_end)
        content = str(transcript.get("content") or "")
        path = save_transcript(args.output_dir, ticker, period, content)
        results.append(
            {
                "ticker": ticker,
                "period": period,
                "roic_year": transcript.get("year", year),
                "roic_quarter": transcript.get("quarter", quarter),
                "call_date": transcript.get("date", call.get("date")),
                "fiscal_year_end": fiscal_year_end,
                "path": str(path),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the last two ROIC.ai earnings-call transcripts and save local .txt files."
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
        results = []
        for ticker in tickers:
            results.extend(fetch_and_save_last_two(args, ticker, roic_key, equipicker_key))
        print(json.dumps({"saved": results}, ensure_ascii=False, indent=2))
        return 0
    except (TranscriptFetchError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
