"""Helpers for importing and storing yearly daily/weekly prices caches."""
from __future__ import annotations

from dataclasses import dataclass
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
DIVERGENCE_OB_LEVEL = 70.0
DIVERGENCE_OS_LEVEL = 30.0
DIVERGENCE_SAME_SWING_TOLERANCE_PCT = 0.001


@dataclass(frozen=True)
class DivergenceSettings:
    pivot_window_l: int
    pivot_window_r: int
    min_bars_between_same_type_pivots: int
    min_pivot_move_pct: float
    max_pivot_pair_span: int
    max_age_pivot2_from_last: int


@dataclass(frozen=True)
class PivotPoint:
    index: int
    price: float
    rsi: float
    confirmation_index: int


@dataclass(frozen=True)
class ActiveDivergence:
    anchor_index: int
    pivot2_index: int
    confirmation_index: int
    pivot2_price: float
    pivot2_rsi: float

    @property
    def recency_key(self) -> tuple[int, int]:
        return (self.pivot2_index, self.confirmation_index)

    @property
    def instance_key(self) -> tuple[int, int, int]:
        return (self.anchor_index, self.pivot2_index, self.confirmation_index)


DIVERGENCE_SETTINGS_BY_FREQUENCY: dict[PricesFrequency, DivergenceSettings] = {
    "daily": DivergenceSettings(
        pivot_window_l=3,
        pivot_window_r=3,
        min_bars_between_same_type_pivots=4,
        min_pivot_move_pct=0.01,
        max_pivot_pair_span=60,
        max_age_pivot2_from_last=60,
    ),
    "weekly": DivergenceSettings(
        pivot_window_l=2,
        pivot_window_r=2,
        min_bars_between_same_type_pivots=3,
        min_pivot_move_pct=0.01,
        max_pivot_pair_span=30,
        max_age_pivot2_from_last=30,
    ),
}

DIVERGENCE_SEED_HISTORY_ROWS: dict[PricesFrequency, int] = {
    "daily": 144,
    "weekly": 81,
}

PRICE_CACHE_COLUMNS: tuple[str, ...] = (
    "ticker",
    "date",
    "adjusted_close",
    "adjusted_high",
    "adjusted_low",
    "rs",
    "obvm",
    "rsi_14",
    "rsi_divergence_flag",
    "rsi_divergence_confirmed",
)
PRICE_CACHE_REQUIRED_COLUMNS: set[str] = set(PRICE_CACHE_COLUMNS).difference(
    {"rsi_divergence_confirmed"}
)
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


def divergence_settings_for_frequency(frequency: PricesFrequency) -> DivergenceSettings:
    validated_frequency = _validate_frequency(frequency)
    return DIVERGENCE_SETTINGS_BY_FREQUENCY[validated_frequency]


def divergence_seed_history_rows(frequency: PricesFrequency) -> int:
    validated_frequency = _validate_frequency(frequency)
    return DIVERGENCE_SEED_HISTORY_ROWS[validated_frequency]


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
    divergence_series = working_df["rsi_divergence_flag"].where(
        working_df["rsi_divergence_flag"].notna(),
        pd.NA,
    )
    working_df["rsi_divergence_flag"] = divergence_series.astype("string").str.strip().str.lower()
    working_df.loc[
        ~working_df["rsi_divergence_flag"].isin({"positive", "negative", "none"}),
        "rsi_divergence_flag",
    ] = pd.NA
    confirmed_series = working_df["rsi_divergence_confirmed"].map(_normalize_optional_boolean_value)
    working_df["rsi_divergence_confirmed"] = pd.array(confirmed_series, dtype="boolean")

    working_df = working_df[
        working_df["ticker"].astype(str).str.len() > 0
    ]
    working_df = working_df.dropna(subset=["date"])
    working_df = working_df.drop_duplicates(subset=["ticker", "date"], keep="last")
    working_df = working_df.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    return working_df


def _normalize_optional_boolean_value(value: object) -> object:
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return pd.NA


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


def _is_price_pivot_candidate(price_series: pd.Series, index: int, *, left_window: int, right_window: int, kind: str) -> bool:
    window = pd.to_numeric(
        price_series.iloc[index - left_window:index + right_window + 1],
        errors="coerce",
    )
    if window.isna().any():
        return False
    value = float(price_series.iloc[index])
    if kind == "high":
        return value >= float(window.max())
    return value <= float(window.min())


def _relative_same_type_move(candidate_price: float, previous_price: float, *, kind: str) -> float:
    if previous_price == 0:
        return float("inf")
    if kind == "high":
        return (candidate_price / previous_price) - 1.0
    return (previous_price / candidate_price) - 1.0


def _find_price_pivots(
    price_series: pd.Series,
    rsi_series: pd.Series,
    *,
    kind: Literal["high", "low"],
    settings: DivergenceSettings,
) -> list[PivotPoint]:
    pivots: list[PivotPoint] = []
    last_pivot_index: int | None = None
    last_pivot_price: float | None = None
    series_length = len(price_series)
    if series_length <= settings.pivot_window_l + settings.pivot_window_r:
        return pivots

    for index in range(settings.pivot_window_l, series_length - settings.pivot_window_r):
        rsi_value = pd.to_numeric(rsi_series.iloc[index], errors="coerce")
        price_value = pd.to_numeric(price_series.iloc[index], errors="coerce")
        if pd.isna(rsi_value) or pd.isna(price_value):
            continue
        if not _is_price_pivot_candidate(
            price_series,
            index,
            left_window=settings.pivot_window_l,
            right_window=settings.pivot_window_r,
            kind=kind,
        ):
            continue

        candidate_price = float(price_value)
        candidate_rsi = float(rsi_value)
        candidate = PivotPoint(
            index=index,
            price=candidate_price,
            rsi=candidate_rsi,
            confirmation_index=index + settings.pivot_window_r,
        )
        if last_pivot_index is None or last_pivot_price is None:
            pivots.append(candidate)
            last_pivot_index = index
            last_pivot_price = candidate_price
            continue

        relative_move = _relative_same_type_move(candidate_price, last_pivot_price, kind=kind)
        is_same_swing_update = 0.0 < relative_move < DIVERGENCE_SAME_SWING_TOLERANCE_PCT
        if (index - last_pivot_index) < settings.min_bars_between_same_type_pivots:
            if is_same_swing_update:
                pivots[-1] = candidate
                last_pivot_index = index
                last_pivot_price = candidate_price
            continue

        if relative_move >= settings.min_pivot_move_pct:
            pivots.append(candidate)
            last_pivot_index = index
            last_pivot_price = candidate_price
            continue

        if is_same_swing_update:
            pivots[-1] = candidate
            last_pivot_index = index
            last_pivot_price = candidate_price

    return pivots


def _build_active_divergence_series(
    *,
    pivots: list[PivotPoint],
    settings: DivergenceSettings,
    row_count: int,
    anchor_threshold: float,
    anchor_cmp,
    anchor_break_cmp,
    price_divergence_cmp,
    rsi_divergence_cmp,
    prefer_more_extreme_cmp,
) -> list[ActiveDivergence | None]:
    events: dict[int, list[PivotPoint]] = {}
    for pivot in pivots:
        if pivot.confirmation_index >= row_count:
            continue
        events.setdefault(pivot.confirmation_index, []).append(pivot)

    active_by_row: list[ActiveDivergence | None] = [None] * row_count
    anchor: PivotPoint | None = None
    best_divergence: tuple[PivotPoint, PivotPoint] | None = None
    for row_index in range(row_count):
        for pivot in events.get(row_index, []):
            if anchor is None:
                if anchor_cmp(pivot.rsi, anchor_threshold):
                    anchor = pivot
                best_divergence = None
                continue

            if (pivot.index - anchor.index) > settings.max_pivot_pair_span:
                anchor = pivot if anchor_cmp(pivot.rsi, anchor_threshold) else None
                best_divergence = None
                continue

            if anchor_break_cmp(pivot.rsi, anchor.rsi):
                anchor = pivot if anchor_cmp(pivot.rsi, anchor_threshold) else None
                best_divergence = None
                continue

            if price_divergence_cmp(pivot.price, anchor.price) and rsi_divergence_cmp(pivot.rsi, anchor.rsi):
                if best_divergence is None or prefer_more_extreme_cmp(pivot.price, best_divergence[1].price):
                    best_divergence = (anchor, pivot)

        if best_divergence is None:
            active_by_row[row_index] = None
            continue

        pivot2 = best_divergence[1]
        if (row_index - pivot2.index) > settings.max_age_pivot2_from_last:
            active_by_row[row_index] = None
            continue

        active_by_row[row_index] = ActiveDivergence(
            anchor_index=best_divergence[0].index,
            pivot2_index=pivot2.index,
            confirmation_index=pivot2.confirmation_index,
            pivot2_price=pivot2.price,
            pivot2_rsi=pivot2.rsi,
        )

    return active_by_row


def _rsi_crosses_confirmation_threshold(
    previous_rsi: object,
    current_rsi: object,
    *,
    divergence_flag: str,
) -> bool:
    previous_numeric = pd.to_numeric(previous_rsi, errors="coerce")
    current_numeric = pd.to_numeric(current_rsi, errors="coerce")
    if pd.isna(previous_numeric) or pd.isna(current_numeric):
        return False
    if divergence_flag == "positive":
        return float(previous_numeric) <= 50.0 and float(current_numeric) > 50.0
    if divergence_flag == "negative":
        return float(previous_numeric) >= 50.0 and float(current_numeric) < 50.0
    return False


def _is_developing_refresh_triggered(
    *,
    divergence_flag: str,
    selected_state: ActiveDivergence,
    row_index: int,
    current_low: object,
    current_high: object,
    current_rsi: object,
) -> tuple[bool, float | None]:
    if row_index <= selected_state.pivot2_index:
        return False, None

    current_rsi_numeric = pd.to_numeric(current_rsi, errors="coerce")
    if pd.isna(current_rsi_numeric):
        return False, None

    if divergence_flag == "positive":
        current_low_numeric = pd.to_numeric(current_low, errors="coerce")
        if pd.isna(current_low_numeric):
            return False, None
        if (
            float(current_low_numeric) < selected_state.pivot2_price
            and float(current_rsi_numeric) > selected_state.pivot2_rsi
        ):
            return True, float(current_low_numeric)
        return False, None

    if divergence_flag == "negative":
        current_high_numeric = pd.to_numeric(current_high, errors="coerce")
        if pd.isna(current_high_numeric):
            return False, None
        if (
            float(current_high_numeric) > selected_state.pivot2_price
            and float(current_rsi_numeric) < selected_state.pivot2_rsi
        ):
            return True, float(current_high_numeric)
        return False, None

    return False, None


def compute_rsi_divergence_state(
    price_df: pd.DataFrame,
    *,
    frequency: PricesFrequency,
    rsi_column: str = "rsi_14",
) -> pd.DataFrame:
    settings = divergence_settings_for_frequency(frequency)
    working_df = price_df.reset_index(drop=True).copy()
    row_count = len(working_df)
    if row_count == 0:
        return pd.DataFrame(
            {
                "rsi_divergence_flag": pd.Series(dtype=object),
                "rsi_divergence_confirmed": pd.Series(dtype="boolean"),
            }
        )

    highs = pd.to_numeric(working_df.get("adjusted_high"), errors="coerce")
    lows = pd.to_numeric(working_df.get("adjusted_low"), errors="coerce")
    rsi_values = pd.to_numeric(working_df.get(rsi_column), errors="coerce")

    pivot_highs = _find_price_pivots(highs, rsi_values, kind="high", settings=settings)
    pivot_lows = _find_price_pivots(lows, rsi_values, kind="low", settings=settings)

    bearish_active = _build_active_divergence_series(
        pivots=pivot_highs,
        settings=settings,
        row_count=row_count,
        anchor_threshold=DIVERGENCE_OB_LEVEL,
        anchor_cmp=lambda current, threshold: current > threshold,
        anchor_break_cmp=lambda current, anchor: current > anchor,
        price_divergence_cmp=lambda current, anchor: current > anchor,
        rsi_divergence_cmp=lambda current, anchor: current < anchor,
        prefer_more_extreme_cmp=lambda current, previous: current > previous,
    )
    bullish_active = _build_active_divergence_series(
        pivots=pivot_lows,
        settings=settings,
        row_count=row_count,
        anchor_threshold=DIVERGENCE_OS_LEVEL,
        anchor_cmp=lambda current, threshold: current < threshold,
        anchor_break_cmp=lambda current, anchor: current < anchor,
        price_divergence_cmp=lambda current, anchor: current < anchor,
        rsi_divergence_cmp=lambda current, anchor: current > anchor,
        prefer_more_extreme_cmp=lambda current, previous: current < previous,
    )

    divergence_flags = np.full(row_count, "none", dtype=object)
    divergence_confirmed = pd.array([False] * row_count, dtype="boolean")
    active_instance_key: tuple[str, tuple[int, int, int]] | None = None
    active_instance_start: int | None = None
    confirmation_latched = False
    refresh_extreme_price: float | None = None

    for row_index in range(row_count):
        bullish_state = bullish_active[row_index]
        bearish_state = bearish_active[row_index]
        selected_flag = "none"
        selected_state: ActiveDivergence | None = None
        if bullish_state is not None and bearish_state is None:
            selected_flag = "positive"
            selected_state = bullish_state
        elif bearish_state is not None and bullish_state is None:
            selected_flag = "negative"
            selected_state = bearish_state
        elif bullish_state is not None and bearish_state is not None:
            if bullish_state.recency_key > bearish_state.recency_key:
                selected_flag = "positive"
                selected_state = bullish_state
            else:
                selected_flag = "negative"
                selected_state = bearish_state

        divergence_flags[row_index] = selected_flag
        if selected_state is None:
            active_instance_key = None
            active_instance_start = None
            confirmation_latched = False
            refresh_extreme_price = None
            divergence_confirmed[row_index] = False
            continue

        current_instance_key = (selected_flag, selected_state.instance_key)
        if current_instance_key != active_instance_key:
            active_instance_key = current_instance_key
            active_instance_start = row_index
            confirmation_latched = False
            refresh_extreme_price = None

        refresh_triggered, refresh_price = _is_developing_refresh_triggered(
            divergence_flag=selected_flag,
            selected_state=selected_state,
            row_index=row_index,
            current_low=lows.iloc[row_index],
            current_high=highs.iloc[row_index],
            current_rsi=rsi_values.iloc[row_index],
        )
        if refresh_triggered and refresh_price is not None:
            should_refresh = False
            if refresh_extreme_price is None:
                should_refresh = True
            elif selected_flag == "positive" and refresh_price < refresh_extreme_price:
                should_refresh = True
            elif selected_flag == "negative" and refresh_price > refresh_extreme_price:
                should_refresh = True
            if should_refresh:
                active_instance_start = row_index
                confirmation_latched = False
                refresh_extreme_price = refresh_price

        if (
            not confirmation_latched
            and active_instance_start is not None
            and row_index > active_instance_start
            and _rsi_crosses_confirmation_threshold(
                rsi_values.iloc[row_index - 1],
                rsi_values.iloc[row_index],
                divergence_flag=selected_flag,
            )
        ):
            confirmation_latched = True

        divergence_confirmed[row_index] = confirmation_latched

    return pd.DataFrame(
        {
            "rsi_divergence_flag": pd.Series(divergence_flags, index=price_df.index, dtype=object),
            "rsi_divergence_confirmed": pd.Series(divergence_confirmed, index=price_df.index, dtype="boolean"),
        }
    )


def compute_rsi_divergence_flags(
    price_df: pd.DataFrame,
    *,
    frequency: PricesFrequency,
    rsi_column: str = "rsi_14",
) -> pd.Series:
    return compute_rsi_divergence_state(
        price_df,
        frequency=frequency,
        rsi_column=rsi_column,
    )["rsi_divergence_flag"]


def _build_rsi_seed_history(
    existing_df: pd.DataFrame,
    target_df: pd.DataFrame,
    selected_tickers: Sequence[object],
    *,
    lookback_rows: int = RSI_PERIOD,
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
        seed_parts.append(ticker_history.sort_values("_existing_date").tail(lookback_rows))

    if not seed_parts:
        return empty_prices_cache_df()

    combined_seed = pd.concat(seed_parts, ignore_index=True).drop(columns="_existing_date", errors="ignore")
    return _canonicalize_prices_df(combined_seed)


def enrich_prices_with_rsi(
    target_df: pd.DataFrame,
    *,
    frequency: PricesFrequency = "daily",
    selected_tickers: Sequence[object] | None = None,
    seed_history_df: pd.DataFrame | None = None,
    period: int = RSI_PERIOD,
) -> pd.DataFrame:
    validated_frequency = _validate_frequency(frequency)
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
    working_df["rsi_divergence_flag"] = "none"
    working_df["rsi_divergence_confirmed"] = False
    for ticker, group in working_df.groupby("ticker", sort=False):
        if ticker not in impacted_set:
            continue
        rsi_series = compute_wilder_rsi(group["adjusted_close"], period=period)
        working_df.loc[group.index, "rsi_14"] = rsi_series.to_numpy()
        divergence_state_df = compute_rsi_divergence_state(
            group.assign(rsi_14=rsi_series),
            frequency=validated_frequency,
            rsi_column="rsi_14",
        )
        working_df.loc[group.index, "rsi_divergence_flag"] = (
            divergence_state_df["rsi_divergence_flag"].to_numpy()
        )
        working_df.loc[group.index, "rsi_divergence_confirmed"] = (
            divergence_state_df["rsi_divergence_confirmed"].to_numpy()
        )

    impacted_rsi_df = working_df.loc[
        working_df["ticker"].isin(impacted_set),
        ["ticker", "date", "rsi_14", "rsi_divergence_flag", "rsi_divergence_confirmed"],
    ].copy()
    impacted_rsi_df = impacted_rsi_df.drop_duplicates(subset=["ticker", "date"], keep="last")

    preserved_df = normalized_target_df.loc[~normalized_target_df["ticker"].isin(impacted_set)].copy()
    updated_impacted_df = impacted_target_df.drop(
        columns=["rsi_14", "rsi_divergence_flag", "rsi_divergence_confirmed"],
        errors="ignore",
    ).merge(
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
    seed_lookback_rows = divergence_seed_history_rows(frequency)
    if previous_year_cache_file.exists():
        previous_year_seed_df = _build_rsi_seed_history(
            load_prices_cache(previous_year_cache_file),
            final_df,
            requested_tickers,
            lookback_rows=seed_lookback_rows,
        )
    current_year_seed_df = _build_rsi_seed_history(
        existing_df,
        final_df,
        requested_tickers,
        lookback_rows=seed_lookback_rows,
    )
    seed_history_df = _canonicalize_prices_df(
        pd.concat([previous_year_seed_df, current_year_seed_df], ignore_index=True)
    )
    final_df = enrich_prices_with_rsi(
        final_df,
        frequency=frequency,
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
