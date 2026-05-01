from __future__ import annotations

import csv
import html
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import requests

from split_service import (
    DATA_DIR,
    DEFAULT_END_DATE,
    DEFAULT_EQUIPICKER_API_KEY,
    DEFAULT_EXCHANGE,
    DEFAULT_START_DATE,
    EODHD_API_TOKEN_ENV_ALIASES,
    EQUIPICKER_API_KEY_ENV_ALIASES,
    fetch_equipicker_universe,
    iter_dates,
    normalize_split_symbol,
    parse_split_date,
    require_config_value,
)

DIVIDEND_OUTPUT_DIR = DATA_DIR / "dividend_checks"


def ensure_dividend_output_dir(output_dir: Path | None = None) -> Path:
    directory = output_dir or DIVIDEND_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_dividend_output_prefix(start_date: date, end_date: date) -> str:
    return f"equipicker_dividends_{start_date.isoformat()}_to_{end_date.isoformat()}"


def _coerce_dividend_value(record: dict[str, Any]) -> Any:
    for field_name in ("value", "adjusted_value", "unadjusted_value", "dividend", "amount"):
        value = record.get(field_name)
        if value not in (None, ""):
            return value
    return ""


def fetch_bulk_dividends_for_date(
    session: requests.Session,
    eodhd_api_token: str,
    exchange: str,
    dividend_date: date,
) -> list[dict[str, Any]]:
    response = session.get(
        f"https://eodhd.com/api/eod-bulk-last-day/{exchange}",
        params={
            "api_token": eodhd_api_token,
            "type": "dividends",
            "date": dividend_date.isoformat(),
            "fmt": "json",
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"EODHD API error on {dividend_date}: {data}")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected EODHD response on {dividend_date}: {data}")

    return data


def build_page_summary_text(
    start_date: date,
    end_date: date,
    matches: list[dict[str, Any]],
) -> str:
    period = f"{start_date.isoformat()} and {end_date.isoformat()} inclusive"
    if not matches:
        return f"For the time between {period}, we had no dividends on Equipicker universe stocks."

    match_count = len(matches)
    noun = "stock" if match_count == 1 else "stocks"
    return f"For the time between {period}, we had dividends on {match_count} Equipicker universe {noun}."


def refresh_dividend_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    start_date = parse_split_date(str(result.get("start_date", "")))
    end_date = parse_split_date(str(result.get("end_date", "")))
    matches = result.get("matches", [])
    if not isinstance(matches, list):
        matches = []

    result["page_summary_text"] = build_page_summary_text(start_date, end_date, matches)
    if result.get("cancelled"):
        checked_days = int(result.get("checked_days", 0) or 0)
        result["page_summary_text"] = (
            f"Dividend check stopped early after {checked_days} checked day(s). "
            + result["page_summary_text"]
        )
    return result


def build_html_report(result: dict[str, Any]) -> str:
    matches = result["matches"]
    rows = ""
    for match in matches:
        rows += f"""
        <tr>
            <td>{html.escape(str(match.get("code", "")))}</td>
            <td>{html.escape(str(match.get("name", "")))}</td>
            <td>{html.escape(str(match.get("sector", "")))}</td>
            <td>{html.escape(str(match.get("industry", "")))}</td>
            <td>{html.escape(str(match.get("date", "")))}</td>
            <td>{html.escape(str(match.get("period", "")))}</td>
            <td>{html.escape(str(match.get("value", "")))}</td>
            <td>{html.escape(str(match.get("exchange", "")))}</td>
        </tr>
        """

    if not rows:
        rows = """
        <tr>
            <td colspan="8">No Equipicker universe dividend matches found.</td>
        </tr>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Equipicker Dividend Check</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 32px;
            color: #111827;
        }}
        .summary {{
            padding: 16px;
            background: #f3f4f6;
            border-radius: 8px;
            margin-bottom: 24px;
            font-size: 16px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
        }}
        th, td {{
            border: 1px solid #d1d5db;
            padding: 8px;
            text-align: left;
            font-size: 14px;
        }}
        th {{
            background: #e5e7eb;
        }}
        .meta {{
            margin-top: 24px;
            color: #4b5563;
            font-size: 13px;
        }}
    </style>
</head>
<body>
    <h1>Equipicker Dividend Check</h1>
    <div class="summary">{html.escape(result["page_summary_text"])}</div>
    <table>
        <thead>
            <tr>
                <th>Code</th>
                <th>Name</th>
                <th>Sector</th>
                <th>Industry</th>
                <th>Ex Date</th>
                <th>Period</th>
                <th>Value</th>
                <th>Exchange</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    <div class="meta">
        Period: {html.escape(result["start_date"])} to {html.escape(result["end_date"])} inclusive<br>
        Exchange: {html.escape(result["exchange"])}<br>
        Universe size: {result["universe_size"]}<br>
        Total dividend records seen: {result["total_dividend_records_seen"]}<br>
        Universe dividend matches: {len(matches)}<br>
        Generated at: {html.escape(result["generated_at"])}
    </div>
</body>
</html>
"""


def save_dividend_outputs(
    result: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    directory = ensure_dividend_output_dir(output_dir)
    result = refresh_dividend_result_summary(dict(result))
    prefix = build_dividend_output_prefix(parse_split_date(result["start_date"]), parse_split_date(result["end_date"]))

    json_path = directory / f"{prefix}.json"
    csv_path = directory / f"{prefix}.csv"
    html_path = directory / f"{prefix}.html"
    latest_json_path = directory / "equipicker_dividends_latest.json"
    latest_html_path = directory / "equipicker_dividends_latest.html"

    json_text = json.dumps(result, indent=2, ensure_ascii=True)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json_path.write_text(json_text, encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "code",
            "name",
            "sector",
            "industry",
            "exchange",
            "date",
            "period",
            "value",
            "raw_record",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["matches"])

    html_report = build_html_report(result)
    html_path.write_text(html_report, encoding="utf-8")
    latest_html_path.write_text(html_report, encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "html": html_path,
        "latest_json": latest_json_path,
        "latest_html": latest_html_path,
    }


def list_saved_dividend_outputs(output_dir: Path | None = None) -> list[str]:
    directory = ensure_dividend_output_dir(output_dir)
    names = [
        path.stem
        for path in directory.glob("equipicker_dividends_*.json")
        if path.is_file() and path.stem != "equipicker_dividends_latest"
    ]
    return sorted(names, key=str.lower, reverse=True)


def load_dividend_output(name: str, output_dir: Path | None = None) -> dict[str, Any]:
    directory = ensure_dividend_output_dir(output_dir)
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Previous output file not found: {path}")
    return refresh_dividend_result_summary(json.loads(path.read_text(encoding="utf-8")))


def run_dividend_check(
    start_date: date,
    end_date: date,
    exchange: str = DEFAULT_EXCHANGE,
    stop_requested: Callable[[], bool] | None = None,
    progress_callback: Callable[[date, int, int], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if end_date < start_date:
        raise RuntimeError("End date cannot be earlier than start date.")

    if log_callback:
        log_callback(
            f"Starting dividend check for {start_date.isoformat()} to {end_date.isoformat()} on exchange {exchange}."
        )

    eodhd_api_token = require_config_value(EODHD_API_TOKEN_ENV_ALIASES, label="EODHD API token")
    equipicker_api_key = require_config_value(
        EQUIPICKER_API_KEY_ENV_ALIASES,
        label="Equipicker API key",
        default=DEFAULT_EQUIPICKER_API_KEY,
    )
    matches: list[dict[str, Any]] = []
    total_dividend_records_seen = 0
    checked_days = 0
    cancelled = False

    with requests.Session() as session:
        if log_callback:
            log_callback("Loading Equipicker universe...")
        universe = fetch_equipicker_universe(session, equipicker_api_key)
        if log_callback:
            log_callback(f"Loaded Equipicker universe: {len(universe)} tickers.")
        for current_date in iter_dates(start_date, end_date):
            if stop_requested and stop_requested():
                cancelled = True
                if log_callback:
                    log_callback("Stop requested before the next EODHD daily call.")
                break

            if log_callback:
                log_callback(f"Checking dividend records for {current_date.isoformat()}...")
            records = fetch_bulk_dividends_for_date(
                session=session,
                eodhd_api_token=eodhd_api_token,
                exchange=exchange,
                dividend_date=current_date,
            )
            if log_callback:
                log_callback(
                    f"Fetched {len(records)} dividend record(s) from EODHD for {current_date.isoformat()}."
                )
            total_dividend_records_seen += len(records)
            for record in records:
                code = normalize_split_symbol(str(record.get("code", "")))
                if log_callback and code:
                    log_callback(f"Covered ticker {code} for {current_date.isoformat()}.")
                if code not in universe:
                    continue

                company = universe[code]
                value = _coerce_dividend_value(record)
                if log_callback:
                    log_callback(
                        f"Matched Equipicker ticker {code} on {record.get('date')} with dividend value {value}."
                    )
                matches.append(
                    {
                        "code": code,
                        "name": company.get("name") or record.get("name"),
                        "sector": company.get("sector"),
                        "industry": company.get("industry"),
                        "exchange": record.get("exchange"),
                        "date": record.get("date"),
                        "period": record.get("period"),
                        "value": value,
                        "raw_record": json.dumps(record, ensure_ascii=True),
                    }
                )

            checked_days += 1
            if progress_callback:
                progress_callback(current_date, len(records), total_dividend_records_seen)

    matches = sorted(matches, key=lambda item: (str(item.get("date") or ""), str(item.get("code") or "")))
    result = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "exchange": exchange,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe_size": len(universe),
        "checked_days": checked_days,
        "cancelled": cancelled,
        "total_dividend_records_seen": total_dividend_records_seen,
        "matches": matches,
    }
    if log_callback:
        if cancelled:
            log_callback(
                f"Dividend check stopped after {checked_days} checked day(s), {total_dividend_records_seen} dividend record(s), "
                f"and {len(matches)} match(es)."
            )
        else:
            log_callback(
                f"Dividend check completed with {checked_days} checked day(s), {total_dividend_records_seen} dividend record(s), "
                f"and {len(matches)} match(es)."
            )
    return refresh_dividend_result_summary(result)
