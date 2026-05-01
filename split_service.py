from __future__ import annotations

import csv
import html
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SPLIT_OUTPUT_DIR = DATA_DIR / "split_checks"

DEFAULT_START_DATE = date(2026, 3, 31)
DEFAULT_END_DATE = date(2026, 4, 30)
DEFAULT_EXCHANGE = "US"

EODHD_API_TOKEN_ENV = "EODHD_API_TOKEN"
EQUIPICKER_API_KEY_ENV = "EQUIPICKER_API_KEY"
EODHD_API_TOKEN_ENV_ALIASES = (EODHD_API_TOKEN_ENV, "eodhd", "EODHD", "EODHD_API_KEY")
EQUIPICKER_API_KEY_ENV_ALIASES = (
    EQUIPICKER_API_KEY_ENV,
    "equipicker_api_key",
    "EQUIPICKER",
    "EQUIPICKER_API_TOKEN",
)
DEFAULT_EQUIPICKER_API_KEY = "3b3703b83"
DOTENV_CANDIDATES = (BASE_DIR / ".env", BASE_DIR.parent / ".env")


def parse_split_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def normalize_split_symbol(symbol: str) -> str:
    if not symbol:
        return ""

    normalized = symbol.strip().upper()
    parts = normalized.split(".")
    if len(parts) > 1 and parts[-1] in {"US", "NYSE", "NASDAQ", "AMEX", "BATS"}:
        normalized = ".".join(parts[:-1])

    return normalized.replace(".", "-")


def iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def _read_dotenv_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in DOTENV_CANDIDATES:
        if not path.exists() or not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            key = key.strip()
            value = raw_value.strip().strip("\"'")
            if key and value:
                values.setdefault(key, value)
    return values


def get_config_value(names: Iterable[str], *, default: str = "") -> str:
    dotenv_values = _read_dotenv_values()
    for name in names:
        value = os.getenv(name) or dotenv_values.get(name)
        if value:
            return value.strip()
    return default.strip()


def require_config_value(names: Iterable[str], *, label: str, default: str = "") -> str:
    names_tuple = tuple(names)
    value = get_config_value(names_tuple, default=default)
    if not value:
        accepted_names = ", ".join(names_tuple)
        raise RuntimeError(
            f"Missing required {label}. Set one of: {accepted_names}. "
            "If the Streamlit server was already running when you set it, restart Streamlit."
        )
    return value


def ensure_split_output_dir(output_dir: Path | None = None) -> Path:
    directory = output_dir or SPLIT_OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_split_output_prefix(start_date: date, end_date: date) -> str:
    return f"equipicker_splits_{start_date.isoformat()}_to_{end_date.isoformat()}"


def fetch_equipicker_universe(
    session: requests.Session,
    equipicker_api_key: str,
) -> dict[str, dict[str, Any]]:
    response = session.get(
        "https://ci.equipicker.com/api/company_overview_list",
        params={
            "show_all": 1,
            "api_key": equipicker_api_key,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected Equipicker universe response: {data}")

    universe: dict[str, dict[str, Any]] = {}
    for symbol, payload in data.items():
        if str(symbol).lower() == "count":
            continue

        normalized_symbol = normalize_split_symbol(str(symbol))
        if not normalized_symbol:
            continue

        if isinstance(payload, dict):
            universe[normalized_symbol] = payload
        else:
            universe[normalized_symbol] = {
                "name": None,
                "sector": None,
                "industry": None,
            }

    if not universe:
        raise RuntimeError("Equipicker universe is empty after parsing API response.")

    return universe


def fetch_bulk_splits_for_date(
    session: requests.Session,
    eodhd_api_token: str,
    exchange: str,
    split_date: date,
) -> list[dict[str, Any]]:
    response = session.get(
        f"https://eodhd.com/api/eod-bulk-last-day/{exchange}",
        params={
            "api_token": eodhd_api_token,
            "type": "splits",
            "date": split_date.isoformat(),
            "fmt": "json",
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(f"EODHD API error on {split_date}: {data}")
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected EODHD response on {split_date}: {data}")

    return data


def build_page_summary_text(
    start_date: date,
    end_date: date,
    matches: list[dict[str, Any]],
) -> str:
    period = f"{start_date.isoformat()} and {end_date.isoformat()} inclusive"
    if not matches:
        return f"For the time between {period}, we had no splits on Equipicker universe stocks."

    match_count = len(matches)
    noun = "stock" if match_count == 1 else "stocks"
    return f"For the time between {period}, we had splits on {match_count} Equipicker universe {noun}."


def refresh_split_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    start_date = parse_split_date(str(result.get("start_date", "")))
    end_date = parse_split_date(str(result.get("end_date", "")))
    matches = result.get("matches", [])
    if not isinstance(matches, list):
        matches = []

    result["page_summary_text"] = build_page_summary_text(start_date, end_date, matches)
    if result.get("cancelled"):
        checked_days = int(result.get("checked_days", 0) or 0)
        result["page_summary_text"] = (
            f"Split check stopped early after {checked_days} checked day(s). "
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
            <td>{html.escape(str(match.get("split", "")))}</td>
            <td>{html.escape(str(match.get("exchange", "")))}</td>
        </tr>
        """

    if not rows:
        rows = """
        <tr>
            <td colspan="7">No Equipicker universe split matches found.</td>
        </tr>
        """

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Equipicker Split Check</title>
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
    <h1>Equipicker Split Check</h1>
    <div class="summary">{html.escape(result["page_summary_text"])}</div>
    <table>
        <thead>
            <tr>
                <th>Code</th>
                <th>Name</th>
                <th>Sector</th>
                <th>Industry</th>
                <th>Date</th>
                <th>Split</th>
                <th>Exchange</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    <div class="meta">
        Period: {html.escape(result["start_date"])} to {html.escape(result["end_date"])} inclusive<br>
        Exchange: {html.escape(result["exchange"])}<br>
        Universe size: {result["universe_size"]}<br>
        Total split records seen: {result["total_split_records_seen"]}<br>
        Universe split matches: {len(matches)}<br>
        Generated at: {html.escape(result["generated_at"])}
    </div>
</body>
</html>
"""


def save_split_outputs(
    result: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Path]:
    directory = ensure_split_output_dir(output_dir)
    result = refresh_split_result_summary(dict(result))
    prefix = build_split_output_prefix(parse_split_date(result["start_date"]), parse_split_date(result["end_date"]))

    json_path = directory / f"{prefix}.json"
    csv_path = directory / f"{prefix}.csv"
    html_path = directory / f"{prefix}.html"
    latest_json_path = directory / "equipicker_splits_latest.json"
    latest_html_path = directory / "equipicker_splits_latest.html"

    json_text = json.dumps(result, indent=2, ensure_ascii=True)
    json_path.write_text(json_text, encoding="utf-8")
    latest_json_path.write_text(json_text, encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["code", "name", "sector", "industry", "exchange", "date", "split", "raw_record"]
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


def list_saved_split_outputs(output_dir: Path | None = None) -> list[str]:
    directory = ensure_split_output_dir(output_dir)
    names = [
        path.stem
        for path in directory.glob("equipicker_splits_*.json")
        if path.is_file() and path.stem != "equipicker_splits_latest"
    ]
    return sorted(names, key=str.lower, reverse=True)


def load_split_output(name: str, output_dir: Path | None = None) -> dict[str, Any]:
    directory = ensure_split_output_dir(output_dir)
    path = directory / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Previous output file not found: {path}")
    return refresh_split_result_summary(json.loads(path.read_text(encoding="utf-8")))


def run_split_check(
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
            f"Starting split check for {start_date.isoformat()} to {end_date.isoformat()} on exchange {exchange}."
        )

    eodhd_api_token = require_config_value(EODHD_API_TOKEN_ENV_ALIASES, label="EODHD API token")
    equipicker_api_key = require_config_value(
        EQUIPICKER_API_KEY_ENV_ALIASES,
        label="Equipicker API key",
        default=DEFAULT_EQUIPICKER_API_KEY,
    )
    matches: list[dict[str, Any]] = []
    total_split_records_seen = 0
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
                log_callback(f"Checking split records for {current_date.isoformat()}...")
            records = fetch_bulk_splits_for_date(
                session=session,
                eodhd_api_token=eodhd_api_token,
                exchange=exchange,
                split_date=current_date,
            )
            if log_callback:
                log_callback(
                    f"Fetched {len(records)} split record(s) from EODHD for {current_date.isoformat()}."
                )
            total_split_records_seen += len(records)
            for record in records:
                code = normalize_split_symbol(str(record.get("code", "")))
                if log_callback and code:
                    log_callback(f"Covered ticker {code} for {current_date.isoformat()}.")
                if code not in universe:
                    continue

                company = universe[code]
                if log_callback:
                    log_callback(
                        f"Matched Equipicker ticker {code} on {record.get('date')} with split {record.get('split')}."
                    )
                matches.append(
                    {
                        "code": code,
                        "name": company.get("name") or record.get("name"),
                        "sector": company.get("sector"),
                        "industry": company.get("industry"),
                        "exchange": record.get("exchange"),
                        "date": record.get("date"),
                        "split": record.get("split"),
                        "raw_record": json.dumps(record, ensure_ascii=True),
                    }
                )

            checked_days += 1
            if progress_callback:
                progress_callback(current_date, len(records), total_split_records_seen)

    matches = sorted(matches, key=lambda item: (str(item.get("date") or ""), str(item.get("code") or "")))
    result = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "exchange": exchange,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe_size": len(universe),
        "checked_days": checked_days,
        "cancelled": cancelled,
        "total_split_records_seen": total_split_records_seen,
        "matches": matches,
    }
    if log_callback:
        if cancelled:
            log_callback(
                f"Split check stopped after {checked_days} checked day(s), {total_split_records_seen} split record(s), "
                f"and {len(matches)} match(es)."
            )
        else:
            log_callback(
                f"Split check completed with {checked_days} checked day(s), {total_split_records_seen} split record(s), "
                f"and {len(matches)} match(es)."
            )
    return refresh_split_result_summary(result)
