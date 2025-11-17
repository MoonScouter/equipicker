# pip install sqlalchemy mysql-connector-python pandas openpyxl
import logging
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine

from equipicker_filters import extreme_accel, accel_normal, accel_weak, wake_up
from sql_query import SQL_QUERY  # your big SQL string
from sql_query_scoring import SQL_QUERY_SCORING

#RUN_SQL = False  # False = load cache if exists

logger = logging.getLogger(__name__)

# --- paths anchored to this script ---
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data"
CACHE_DIR.mkdir(exist_ok=True)
def bucharest_today_str(): return datetime.now(ZoneInfo("Europe/Bucharest")).date().isoformat()
def cache_path(ext="xlsx"): return CACHE_DIR / f"select_{bucharest_today_str()}.{ext}"
def scoring_cache_path(ext="xlsx"): return CACHE_DIR / f"scoring_{bucharest_today_str()}.{ext}"

# optional: force cwd to script dir (guards PyCharm mis-config)
os.chdir(BASE_DIR)

# --- DB ---
def make_engine():
    url = f"mysql+mysqlconnector://equipicker_ci:{quote_plus('-$VRdy1~D;Vn')}@equipicker.com:3306/equipicker_ci?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True)

def run_query_to_df(sql: str) -> pd.DataFrame:
    with make_engine().connect() as conn:
        return pd.read_sql(sql, conn)

def save_cache(df: pd.DataFrame, path: Path):
    if path.suffix == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)

def load_cache(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)

def _get_dataframe_with_cache(sql: str, cache_file: Path, run_sql: bool, use_cache: bool = True) -> pd.DataFrame:
    """
    Shared helper that optionally loads/saves from cache around a SQL query execution.
    """
    if use_cache and not run_sql and cache_file and cache_file.exists():
        logger.info("Loading cached data from %s", cache_file)
        return load_cache(cache_file)

    logger.info("Executing SQL query (cache target: %s).", cache_file)
    df = run_query_to_df(sql)
    if cache_file and use_cache:
        save_cache(df, cache_file)
        logger.info("Saved query results to %s", cache_file)
    return df

def get_dataframe(run_sql: bool = False) -> pd.DataFrame:
    """
    Returns the main screener dataframe (existing behavior). Set run_sql=True to bypass cache.
    """
    cp = cache_path("xlsx")
    return _get_dataframe_with_cache(SQL_QUERY, cp, run_sql, use_cache=True)

def get_scoring_dataframe(run_sql: bool = False, use_cache: bool = True) -> pd.DataFrame:
    """
    Returns the scoring dataframe powering the Weekly Scoring Board report.
    """
    cp = scoring_cache_path("xlsx")
    return _get_dataframe_with_cache(SQL_QUERY_SCORING, cp, run_sql, use_cache=use_cache)

if __name__ == "__main__":
    print("script:", BASE_DIR)
    print("cwd:", Path.cwd())
    out = cache_path("xlsx")
    print("will save to:", out)
    df = get_dataframe()
    print(f"rows: {len(df)}  saved: {out}")
    extreme = extreme_accel(df, CACHE_DIR)
    print("Extreme Accel:", len(extreme))
    print(extreme.head())
