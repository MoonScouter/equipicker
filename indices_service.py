"""Helpers for loading/storing yearly index OHLC cache used by Equipilot."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from equipicker_connect import run_query_to_df

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

INDEX_TICKER_TO_NAME: dict[str, str] = {
    "GSPC.INDX": "SP500",
    "DJI.INDX": "Dow Jones",
    "IXIC.INDX": "Nasdaq Composite",
    "GDAXI.INDX": "DAX40",
    "000001.SHG": "Shanghai Exchange Index",
    "N225.INDX": "Nikkei 225",
    "NSEI.INDX": "Nifty 50",
}
INDEX_TICKERS: tuple[str, ...] = tuple(INDEX_TICKER_TO_NAME.keys())

SQL_QUERY_INDICES_OHLC = """
SELECT
    ticker,
    date,
    adjusted_open,
    adjusted_high,
    adjusted_low,
    adjusted_close,
    volume
FROM eod_data
WHERE ticker IN (
    'GSPC.INDX',
    'DJI.INDX',
    'IXIC.INDX',
    '000001.SHG',
    'GDAXI.INDX',
    'N225.INDX',
    'NSEI.INDX'
)
AND date > :cutoff_datetime
ORDER BY date DESC, ticker DESC
"""


def indices_cache_path(cache_year: int) -> Path:
    return DATA_DIR / f"indices-prices-{cache_year}.xlsx"


def fetch_indices_ohlc_since(cutoff_date: date) -> pd.DataFrame:
    cutoff_datetime = f"{cutoff_date.isoformat()} 00:00:00"
    return run_query_to_df(SQL_QUERY_INDICES_OHLC, params={"cutoff_datetime": cutoff_datetime})


def save_indices_cache(df: pd.DataFrame, cache_year: int) -> Path:
    cache_file = indices_cache_path(cache_year)
    df.to_excel(cache_file, index=False)
    return cache_file


def load_indices_cache(cache_year: int) -> pd.DataFrame:
    cache_file = indices_cache_path(cache_year)
    return pd.read_excel(cache_file)
