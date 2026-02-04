from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "report_config.json"


@dataclass(frozen=True)
class ReportConfig:
    report_date: date
    eod_as_of_date: Optional[date] = None
    cache_date: Optional[date] = None


def _parse_iso_date(value: Any, field_name: str) -> Optional[date]:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be YYYY-MM-DD") from exc
    raise ValueError(f"{field_name} must be YYYY-MM-DD or null")


def load_report_config(
    path: Path | str | None = None,
    *,
    default_report_date: Optional[date] = None,
) -> ReportConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    else:
        raw = {}

    report_date = _parse_iso_date(raw.get("report_date"), "report_date")
    report_date = report_date or default_report_date or date.today()
    eod_as_of_date = _parse_iso_date(raw.get("eod_as_of_date"), "eod_as_of_date")
    cache_date = _parse_iso_date(raw.get("cache_date"), "cache_date")
    return ReportConfig(
        report_date=report_date,
        eod_as_of_date=eod_as_of_date,
        cache_date=cache_date,
    )


def save_report_config(config: ReportConfig, path: Path | str | None = None) -> Path:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report_date": config.report_date.isoformat(),
        "eod_as_of_date": config.eod_as_of_date.isoformat() if config.eod_as_of_date else None,
        "cache_date": config.cache_date.isoformat() if config.cache_date else None,
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return config_path
