"""Local DuckDB/Parquet performance cache helpers for Equipilot."""
from __future__ import annotations

from datetime import date
from functools import lru_cache
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import pandas as pd

from equipicker_connect import CACHE_DIR


PERF_CACHE_SCHEMA_VERSION = "2026-05-08.1"
PERF_CACHE_DIR = CACHE_DIR / "perf_cache"
REPORT_SELECT_CACHE_DIR = PERF_CACHE_DIR / "report_select"
PRICES_CACHE_DIR = PERF_CACHE_DIR / "prices"
COMPANY_UNIVERSE_CACHE_DIR = PERF_CACHE_DIR / "company_universe"
MARKET_BUNDLE_CACHE_DIR = PERF_CACHE_DIR / "market_bundle"


class PerfCacheUnavailable(RuntimeError):
    """Raised when the optional local performance cache backend cannot be used."""


def source_signature(path: str | Path | None) -> dict[str, object]:
    if path is None:
        return {"path": "", "exists": False, "size": 0, "mtime_ns": 0}
    resolved = Path(path)
    try:
        stats = resolved.stat()
    except OSError:
        return {"path": str(resolved), "exists": False, "size": 0, "mtime_ns": 0}
    return {
        "path": str(resolved.resolve()),
        "exists": True,
        "size": int(stats.st_size),
        "mtime_ns": int(stats.st_mtime_ns),
    }


def _metadata_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(f"{parquet_path.suffix}.meta.json")


def _stable_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def cache_key_for_payload(payload: Mapping[str, object], *, length: int = 16) -> str:
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()[:length]


def _cache_metadata(source: Mapping[str, object], extra: Mapping[str, object] | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "schema_version": PERF_CACHE_SCHEMA_VERSION,
        "source": dict(source),
    }
    if extra:
        metadata["extra"] = dict(extra)
    return metadata


def is_cache_fresh(
    parquet_path: Path,
    *,
    source: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> bool:
    if not parquet_path.exists():
        return False
    metadata_path = _metadata_path(parquet_path)
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return metadata == _cache_metadata(source, extra)


def _write_parquet_cache(
    df: pd.DataFrame,
    parquet_path: Path,
    *,
    source: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> Path:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    _metadata_path(parquet_path).write_text(
        json.dumps(_cache_metadata(source, extra), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return parquet_path


def report_select_parquet_path(eod_date: date) -> Path:
    return REPORT_SELECT_CACHE_DIR / f"date={eod_date.isoformat()}" / "report.parquet"


def _parse_report_select_date(source_path: Path) -> Optional[date]:
    stem = source_path.stem
    if not stem.startswith("report_select_"):
        return None
    try:
        return date.fromisoformat(stem.replace("report_select_", "", 1))
    except ValueError:
        return None


def load_report_select_cached(
    source_path: str | Path,
    loader: Callable[[Path], pd.DataFrame],
    *,
    normalizer: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    source = Path(source_path)
    eod_date = _parse_report_select_date(source)
    if eod_date is None:
        return normalizer(loader(source)) if normalizer else loader(source)

    parquet_path = report_select_parquet_path(eod_date)
    signature = source_signature(source)
    extra = {"kind": "report_select"}
    try:
        if not is_cache_fresh(parquet_path, source=signature, extra=extra):
            loaded = loader(source)
            if normalizer:
                loaded = normalizer(loaded)
            _write_parquet_cache(loaded, parquet_path, source=signature, extra=extra)
        return pd.read_parquet(parquet_path)
    except Exception:
        loaded = loader(source)
        return normalizer(loaded) if normalizer else loaded


def prices_parquet_path(frequency: str, cache_year: int) -> Path:
    return PRICES_CACHE_DIR / f"frequency={frequency}" / f"year={cache_year}" / "prices.parquet"


def load_prices_cached(
    source_path: str | Path,
    *,
    frequency: str,
    cache_year: int,
    loader: Callable[[Path], pd.DataFrame],
) -> pd.DataFrame:
    source = Path(source_path)
    parquet_path = prices_parquet_path(frequency, cache_year)
    signature = source_signature(source)
    extra = {"kind": "prices", "frequency": frequency, "cache_year": int(cache_year)}
    try:
        if not is_cache_fresh(parquet_path, source=signature, extra=extra):
            _write_parquet_cache(loader(source), parquet_path, source=signature, extra=extra)
        return pd.read_parquet(parquet_path)
    except Exception:
        return loader(source)


def _company_universe_cache_key(
    *,
    eod_date: date,
    signatures: Mapping[str, object],
) -> str:
    return cache_key_for_payload(
        {
            "kind": "company_universe",
            "schema_version": PERF_CACHE_SCHEMA_VERSION,
            "eod_date": eod_date.isoformat(),
            "signatures": dict(signatures),
        }
    )


def company_universe_parquet_path(eod_date: date, signatures: Mapping[str, object]) -> Path:
    cache_key = _company_universe_cache_key(eod_date=eod_date, signatures=signatures)
    return COMPANY_UNIVERSE_CACHE_DIR / f"eod={eod_date.isoformat()}" / f"company_universe_{cache_key}.parquet"


def load_company_universe_cached(
    eod_date: date,
    signatures: Mapping[str, object],
) -> Optional[tuple[pd.DataFrame, Optional[str]]]:
    parquet_path = company_universe_parquet_path(eod_date, signatures)
    source = {"eod_date": eod_date.isoformat(), "signatures": dict(signatures)}
    extra = {"kind": "company_universe"}
    if not is_cache_fresh(parquet_path, source=source, extra=extra):
        return None
    try:
        df = pd.read_parquet(parquet_path)
        warning_path = parquet_path.with_suffix(f"{parquet_path.suffix}.warning.json")
        warning_message = None
        if warning_path.exists():
            warning_payload = json.loads(warning_path.read_text(encoding="utf-8"))
            warning_message = warning_payload.get("warning_message")
        return df, str(warning_message) if warning_message else None
    except Exception:
        return None


def save_company_universe_cached(
    df: pd.DataFrame,
    *,
    eod_date: date,
    signatures: Mapping[str, object],
    warning_message: Optional[str] = None,
) -> Path:
    parquet_path = company_universe_parquet_path(eod_date, signatures)
    source = {"eod_date": eod_date.isoformat(), "signatures": dict(signatures)}
    extra = {"kind": "company_universe"}
    saved_path = _write_parquet_cache(df, parquet_path, source=source, extra=extra)
    saved_path.with_suffix(f"{saved_path.suffix}.warning.json").write_text(
        json.dumps({"warning_message": warning_message}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return saved_path


def market_bundle_parquet_path(signature: str, name: str) -> Path:
    return MARKET_BUNDLE_CACHE_DIR / signature / f"{name}.parquet"


def save_market_bundle_frame(signature: str, name: str, df: pd.DataFrame) -> Path:
    return _write_parquet_cache(
        df,
        market_bundle_parquet_path(signature, name),
        source={"signature": signature, "name": name},
        extra={"kind": "market_bundle"},
    )


def load_market_bundle_frame(signature: str, name: str) -> Optional[pd.DataFrame]:
    parquet_path = market_bundle_parquet_path(signature, name)
    source = {"signature": signature, "name": name}
    extra = {"kind": "market_bundle"}
    if not is_cache_fresh(parquet_path, source=source, extra=extra):
        return None
    try:
        return pd.read_parquet(parquet_path)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_duckdb_connection():
    try:
        import duckdb
    except Exception as exc:  # pragma: no cover - optional dependency feedback
        raise PerfCacheUnavailable("DuckDB is not installed.") from exc
    PERF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(PERF_CACHE_DIR / "equipilot.duckdb"))


def query_parquet(parquet_path: str | Path, sql: str, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    conn = get_duckdb_connection()
    conn.execute("CREATE OR REPLACE TEMP VIEW parquet_source AS SELECT * FROM read_parquet(?)", [str(parquet_path)])
    return conn.execute(sql, params or {}).df()


def clear_performance_cache() -> None:
    get_duckdb_connection.cache_clear()
    if PERF_CACHE_DIR.exists():
        shutil.rmtree(PERF_CACHE_DIR)
