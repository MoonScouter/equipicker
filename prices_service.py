"""Helpers for importing and storing yearly daily/weekly prices caches."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Literal, Sequence
import re

import numpy as np
import pandas as pd
import requests
from sqlalchemy import bindparam, text

from equipicker_connect import CACHE_DIR, make_engine, run_query_to_df

PricesFrequency = Literal["daily", "weekly"]
RSI_PERIOD = 14

PRICE_CACHE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "adjusted_close",
    "adjusted_high",
    "adjusted_low",
    "rs",
    "obvm",
    "rsi_14",
)
PRICE_CACHE_REQUIRED_COLUMNS: set[str] = set(PRICE_CACHE_COLUMNS)
PRICE_TICKER_ENDPOINT = (
    "https://ci.equipicker.com/api/company_overview_list"
    "?show_all=1&api_key=3b3703b83"
)
DB_ELIGIBLE_TICKERS_SQL = """
SELECT t.ticker
FROM tickers t
WHERE t.exclude_from_screener = 0
ORDER BY t.ticker ASC
"""
SQL_QUERY_PRICES_DAILY = """
SELECT
    ed.ticker,
    ed.date,
    ed.adjusted_close,
    ed.adjusted_high,
    ed.adjusted_low,
    ed.rs,
    ed.obvm
FROM eod_data ed
WHERE ed.date > :cutoff_datetime
  AND ed.ticker IN :tickers
ORDER BY ed.ticker ASC, ed.date ASC
"""
SQL_QUERY_PRICES_WEEKLY = """
SELECT
    ew.ticker,
    ew.date,
    ew.adjusted_close,
    ew.adjusted_high,
    ew.adjusted_low,
    ew.rs,
    ew.obvm
FROM eod_weekly ew
WHERE ew.date > :cutoff_datetime
  AND ew.ticker IN :tickers
ORDER BY ew.ticker ASC, ew.date ASC
"""


def _validate_frequency(frequency: PricesFrequency) -> PricesFrequency:
    if frequency not in {"daily", "weekly"}:
        raise ValueError(f"Unsupported prices frequency: {frequency}")
    return frequency


def prices_cache_path(frequency: PricesFrequency, cache_year: int) -> Path:
    validated_frequency = _validate_frequency(frequency)
    return CACHE_DIR / f"prices_{validated_frequency}_{cache_year}.jsonl"


def list_prices_cache_paths(frequency: PricesFrequency) -> list[Path]:
    validated_frequency = _validate_frequency(frequency)
    return sorted(CACHE_DIR.glob(f"prices_{validated_frequency}_*.jsonl"))


def empty_prices_cache_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(PRICE_CACHE_COLUMNS))


def normalize_price_ticker(ticker: object) -> str:
    value = str(ticker or "").strip().upper()
    if not value:
        return ""
    if value.endswith(".US"):
        return value
    return f"{value}.US"


def normalize_price_tickers(tickers: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        normalized_ticker = normalize_price_ticker(ticker)
        if not normalized_ticker or normalized_ticker in seen:
            continue
        normalized.append(normalized_ticker)
        seen.add(normalized_ticker)
    return normalized


def parse_manual_price_tickers(raw_text: str) -> list[str]:
    return normalize_price_tickers(re.split(r"[\s,;]+", raw_text or ""))


def fetch_company_overview_tickers() -> list[str]:
    response = requests.get(PRICE_TICKER_ENDPOINT, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected ticker endpoint payload; expected JSON object keyed by ticker.")
    return normalize_price_tickers(payload.keys())


def fetch_db_eligible_tickers() -> list[str]:
    db_df = run_query_to_df(DB_ELIGIBLE_TICKERS_SQL)
    return normalize_price_tickers(db_df.get("ticker", pd.Series(dtype=str)).tolist())


def intersect_ticker_universe(
    endpoint_tickers: Sequence[object],
    db_tickers: Sequence[object],
) -> list[str]:
    endpoint_set = set(normalize_price_tickers(endpoint_tickers))
    db_set = set(normalize_price_tickers(db_tickers))
    return sorted(endpoint_set.intersection(db_set))


def resolve_all_price_tickers() -> list[str]:
    return intersect_ticker_universe(fetch_company_overview_tickers(), fetch_db_eligible_tickers())


def get_price_history_query(frequency: PricesFrequency) -> str:
    validated_frequency = _validate_frequency(frequency)
    if validated_frequency == "daily":
        return SQL_QUERY_PRICES_DAILY
    return SQL_QUERY_PRICES_WEEKLY


def _chunked(values: Sequence[str], chunk_size: int) -> list[list[str]]:
    return [list(values[index:index + chunk_size]) for index in range(0, len(values), chunk_size)]


def _canonicalize_prices_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return empty_prices_cache_df()

    working_df = df.copy()
    for column in PRICE_CACHE_COLUMNS:
        if column not in working_df.columns:
            working_df[column] = pd.NA
    working_df = working_df.loc[:, list(PRICE_CACHE_COLUMNS)]

    working_df["ticker"] = working_df["ticker"].map(normalize_price_ticker)
    date_series = pd.to_datetime(working_df["date"], errors="coerce").dt.date
    working_df["date"] = date_series.map(lambda value: value.isoformat() if pd.notna(value) else None)

    numeric_columns = ["adjusted_close", "adjusted_high", "adjusted_low", "rs", "obvm", "rsi_14"]
    for column in numeric_columns:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")

    working_df = working_df[
        working_df["ticker"].astype(str).str.len() > 0
    ]
    working_df = working_df.dropna(subset=["date"])
    working_df = working_df.drop_duplicates(subset=["ticker", "date"], keep="last")
    working_df = working_df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    return working_df


def fetch_prices_history(
    frequency: PricesFrequency,
    tickers: Sequence[object],
    cutoff_date: date,
    *,
    chunk_size: int = 250,
) -> pd.DataFrame:
    normalized_tickers = normalize_price_tickers(tickers)
    if not normalized_tickers:
        return empty_prices_cache_df()

    query = text(get_price_history_query(frequency)).bindparams(bindparam("tickers", expanding=True))
    cutoff_datetime = f"{cutoff_date.isoformat()} 00:00:00"
    parts: list[pd.DataFrame] = []
    with make_engine().connect() as conn:
        for ticker_chunk in _chunked(normalized_tickers, chunk_size):
            part_df = pd.read_sql(
                query,
                conn,
                params={
                    "cutoff_datetime": cutoff_datetime,
                    "tickers": ticker_chunk,
                },
            )
            if not part_df.empty:
                parts.append(part_df)

    if not parts:
        return empty_prices_cache_df()
    return _canonicalize_prices_df(pd.concat(parts, ignore_index=True))


def load_prices_cache(cache_file: Path) -> pd.DataFrame:
    if not cache_file.exists() or cache_file.stat().st_size == 0:
        return empty_prices_cache_df()
    loaded_df = pd.read_json(cache_file, orient="records", lines=True)
    return _canonicalize_prices_df(loaded_df)


def save_prices_cache(df: pd.DataFrame, frequency: PricesFrequency, cache_year: int) -> Path:
    cache_file = prices_cache_path(frequency, cache_year)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    prepared_df = _canonicalize_prices_df(df)
    if prepared_df.empty:
        cache_file.write_text("", encoding="utf-8")
        return cache_file
    prepared_df.to_json(
        cache_file,
        orient="records",
        lines=True,
        force_ascii=False,
    )
    return cache_file


def build_prices_cache_dataframe(
    existing_df: pd.DataFrame,
    fetched_df: pd.DataFrame,
    *,
    scope: str,
    selected_tickers: Sequence[object],
    cutoff_date: date,
) -> pd.DataFrame:
    normalized_fetched_df = _canonicalize_prices_df(fetched_df)
    if scope == "all":
        return normalized_fetched_df

    normalized_existing_df = _canonicalize_prices_df(existing_df)
    normalized_selected_tickers = set(normalize_price_tickers(selected_tickers))
    if normalized_existing_df.empty:
        return normalized_fetched_df

    existing_dates = pd.to_datetime(normalized_existing_df["date"], errors="coerce").dt.date
    replace_mask = normalized_existing_df["ticker"].isin(normalized_selected_tickers) & (
        existing_dates >= cutoff_date
    )
    preserved_df = normalized_existing_df.loc[~replace_mask].copy()
    merged_df = pd.concat([preserved_df, normalized_fetched_df], ignore_index=True)
    return _canonicalize_prices_df(merged_df)


def compute_wilder_rsi(close_series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    close = pd.to_numeric(close_series, errors="coerce")
    rsi_values = np.full(len(close), np.nan, dtype=float)
    if len(close) <= period:
        return pd.Series(rsi_values, index=close_series.index)

    close_values = close.to_numpy(dtype=float)
    if np.isnan(close_values).any():
        return pd.Series(rsi_values, index=close_series.index)

    deltas = np.diff(close_values)
    gains = np.maximum(deltas, 0.0)
    losses = np.maximum(-deltas, 0.0)

    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    if avg_loss == 0.0:
        rsi_values[period] = 100.0 if avg_gain > 0.0 else 50.0
    else:
        rs_value = avg_gain / avg_loss
        rsi_values[period] = 100.0 - (100.0 / (1.0 + rs_value))

    for idx in range(period + 1, len(close_values)):
        gain = gains[idx - 1]
        loss = losses[idx - 1]
        avg_gain = ((period - 1) * avg_gain + gain) / period
        avg_loss = ((period - 1) * avg_loss + loss) / period
        if avg_loss == 0.0:
            rsi_values[idx] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            rs_value = avg_gain / avg_loss
            rsi_values[idx] = 100.0 - (100.0 / (1.0 + rs_value))

    return pd.Series(rsi_values, index=close_series.index)


def _build_rsi_seed_history(
    existing_df: pd.DataFrame,
    target_df: pd.DataFrame,
    selected_tickers: Sequence[object],
    *,
    period: int = RSI_PERIOD,
) -> pd.DataFrame:
    normalized_existing_df = _canonicalize_prices_df(existing_df)
    normalized_target_df = _canonicalize_prices_df(target_df)
    normalized_selected_tickers = normalize_price_tickers(selected_tickers)
    if (
        normalized_existing_df.empty
        or normalized_target_df.empty
        or not normalized_selected_tickers
    ):
        return empty_prices_cache_df()

    selected_set = set(normalized_selected_tickers)
    target_dates = pd.to_datetime(normalized_target_df["date"], errors="coerce")
    min_target_dates = (
        normalized_target_df.loc[normalized_target_df["ticker"].isin(selected_set)]
        .assign(_target_date=target_dates)
        .dropna(subset=["_target_date"])
        .groupby("ticker")["_target_date"]
        .min()
    )
    if min_target_dates.empty:
        return empty_prices_cache_df()

    existing_subset = normalized_existing_df.loc[
        normalized_existing_df["ticker"].isin(selected_set)
    ].copy()
    if existing_subset.empty:
        return empty_prices_cache_df()

    existing_subset["_existing_date"] = pd.to_datetime(existing_subset["date"], errors="coerce")
    seed_parts: list[pd.DataFrame] = []
    for ticker, min_target_date in min_target_dates.items():
        ticker_history = existing_subset.loc[
            (existing_subset["ticker"] == ticker)
            & existing_subset["_existing_date"].notna()
            & (existing_subset["_existing_date"] < min_target_date)
        ].copy()
        if ticker_history.empty:
            continue
        seed_parts.append(ticker_history.sort_values("_existing_date").tail(period))

    if not seed_parts:
        return empty_prices_cache_df()

    combined_seed = pd.concat(seed_parts, ignore_index=True).drop(columns="_existing_date", errors="ignore")
    return _canonicalize_prices_df(combined_seed)


def enrich_prices_with_rsi(
    target_df: pd.DataFrame,
    *,
    selected_tickers: Sequence[object] | None = None,
    seed_history_df: pd.DataFrame | None = None,
    period: int = RSI_PERIOD,
) -> pd.DataFrame:
    normalized_target_df = _canonicalize_prices_df(target_df)
    if normalized_target_df.empty:
        return normalized_target_df

    if selected_tickers is None:
        impacted_tickers = normalized_target_df["ticker"].dropna().astype(str).unique().tolist()
    else:
        impacted_tickers = normalize_price_tickers(selected_tickers)
    if not impacted_tickers:
        return normalized_target_df

    impacted_set = set(impacted_tickers)
    impacted_target_df = normalized_target_df.loc[normalized_target_df["ticker"].isin(impacted_set)].copy()
    if impacted_target_df.empty:
        return normalized_target_df

    working_frames = []
    normalized_seed_history = _canonicalize_prices_df(seed_history_df) if seed_history_df is not None else empty_prices_cache_df()
    if not normalized_seed_history.empty:
        working_frames.append(normalized_seed_history.loc[normalized_seed_history["ticker"].isin(impacted_set)])
    working_frames.append(impacted_target_df)
    working_df = _canonicalize_prices_df(pd.concat(working_frames, ignore_index=True))
    if working_df.empty:
        return normalized_target_df

    working_df["rsi_14"] = np.nan
    for ticker, group in working_df.groupby("ticker", sort=False):
        if ticker not in impacted_set:
            continue
        rsi_series = compute_wilder_rsi(group["adjusted_close"], period=period)
        working_df.loc[group.index, "rsi_14"] = rsi_series.to_numpy()

    impacted_rsi_df = working_df.loc[
        working_df["ticker"].isin(impacted_set),
        ["ticker", "date", "rsi_14"],
    ].copy()
    impacted_rsi_df = impacted_rsi_df.drop_duplicates(subset=["ticker", "date"], keep="last")

    preserved_df = normalized_target_df.loc[~normalized_target_df["ticker"].isin(impacted_set)].copy()
    updated_impacted_df = impacted_target_df.drop(columns="rsi_14").merge(
        impacted_rsi_df,
        on=["ticker", "date"],
        how="left",
    )
    final_df = pd.concat([preserved_df, updated_impacted_df], ignore_index=True)
    return _canonicalize_prices_df(final_df)


def import_prices_cache(
    frequency: PricesFrequency,
    cutoff_date: date,
    *,
    scope: str,
    manual_tickers: Sequence[object] | None = None,
    cache_year: int | None = None,
    chunk_size: int = 250,
) -> dict[str, object]:
    resolved_year = cache_year or date.today().year
    normalized_scope = scope.strip().lower()
    if normalized_scope == "all":
        requested_tickers = resolve_all_price_tickers()
    elif normalized_scope == "specific":
        requested_tickers = normalize_price_tickers(manual_tickers or [])
    else:
        raise ValueError(f"Unsupported prices import scope: {scope}")

    if not requested_tickers:
        raise ValueError("No tickers available for import.")

    cache_file = prices_cache_path(frequency, resolved_year)
    existing_df = empty_prices_cache_df()
    if cache_file.exists():
        existing_df = load_prices_cache(cache_file)

    fetched_df = fetch_prices_history(
        frequency,
        requested_tickers,
        cutoff_date,
        chunk_size=chunk_size,
    )

    final_df = build_prices_cache_dataframe(
        existing_df,
        fetched_df,
        scope=normalized_scope,
        selected_tickers=requested_tickers,
        cutoff_date=cutoff_date,
    )
    previous_year_seed_df = empty_prices_cache_df()
    previous_year_cache_file = prices_cache_path(frequency, resolved_year - 1)
    if previous_year_cache_file.exists():
        previous_year_seed_df = _build_rsi_seed_history(
            load_prices_cache(previous_year_cache_file),
            final_df,
            requested_tickers,
        )
    current_year_seed_df = _build_rsi_seed_history(existing_df, final_df, requested_tickers)
    seed_history_df = _canonicalize_prices_df(
        pd.concat([previous_year_seed_df, current_year_seed_df], ignore_index=True)
    )
    final_df = enrich_prices_with_rsi(
        final_df,
        selected_tickers=requested_tickers,
        seed_history_df=seed_history_df,
    )
    saved_path = save_prices_cache(final_df, frequency, resolved_year)
    latest_date = None
    if not final_df.empty:
        latest_date = max(pd.to_datetime(final_df["date"], errors="coerce").dt.date.dropna().tolist())
    return {
        "saved_path": saved_path,
        "frequency": frequency,
        "cache_year": resolved_year,
        "scope": normalized_scope,
        "requested_tickers": requested_tickers,
        "requested_tickers_count": len(requested_tickers),
        "fetched_rows": int(len(fetched_df)),
        "saved_rows": int(len(final_df)),
        "latest_date": latest_date,
    }
