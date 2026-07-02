# pip install sqlalchemy mysql-connector-python pandas openpyxl
import logging
import os
from pathlib import Path
from datetime import date, datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text

from equipicker_filters import extreme_accel, accel_normal, accel_weak, wake_up
from sql_query import SQL_QUERY  # your big SQL string
from sql_query_scoring import SQL_QUERY_SCORING
from sql_query_report import SQL_QUERY_REPORT

#RUN_SQL = False  # False = load cache if exists

logger = logging.getLogger(__name__)

# --- paths anchored to this script ---
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data"
CACHE_DIR.mkdir(exist_ok=True)
DOTENV_CANDIDATES = (BASE_DIR / ".env", BASE_DIR.parent / ".env")


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


def _get_config_value(*names: str, default: str = "") -> str:
    dotenv_values = _read_dotenv_values()
    for name in names:
        value = os.getenv(name) or dotenv_values.get(name)
        if value:
            return value.strip()
    return default.strip()


def bucharest_today_str(): return datetime.now(ZoneInfo("Europe/Bucharest")).date().isoformat()

def _date_to_str(value: date | str | None) -> str:
    if value is None:
        return bucharest_today_str()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return value
    raise ValueError("cache_date must be YYYY-MM-DD or date")

def cache_path(ext="xlsx", cache_date: date | str | None = None):
    return CACHE_DIR / f"select_{_date_to_str(cache_date)}.{ext}"

def scoring_cache_path(ext="xlsx", cache_date: date | str | None = None):
    return CACHE_DIR / f"scoring_{_date_to_str(cache_date)}.{ext}"

def report_cache_path(ext="xlsx", cache_date: date | str | None = None):
    return CACHE_DIR / f"report_select_{_date_to_str(cache_date)}.{ext}"

# optional: force cwd to script dir (guards PyCharm mis-config)
os.chdir(BASE_DIR)

# --- DB ---
def build_database_url() -> str:
    explicit_url = _get_config_value("EQUIPICKER_DATABASE_URL", "DATABASE_URL")
    if explicit_url:
        return explicit_url

    host = _get_config_value("EQUIPICKER_DB_HOST", "MYSQL_HOST", default="equipicker.com")
    port = _get_config_value("EQUIPICKER_DB_PORT", "MYSQL_PORT", default="3306")
    database = _get_config_value("EQUIPICKER_DB_NAME", "MYSQL_DATABASE", default="equipicker_ci")
    username = _get_config_value("EQUIPICKER_DB_USER", "MYSQL_USER", default="equipicker_ci")
    password = _get_config_value("EQUIPICKER_DB_PASSWORD", "MYSQL_PASSWORD", default="-$VRdy1~D;Vn")
    charset = _get_config_value("EQUIPICKER_DB_CHARSET", "MYSQL_CHARSET", default="utf8mb4")

    return (
        "mysql+mysqlconnector://"
        f"{quote_plus(username)}:{quote_plus(password)}@{host}:{port}/{database}"
        f"?charset={quote_plus(charset)}"
    )


def make_engine():
    return create_engine(build_database_url(), pool_pre_ping=True)

def run_query_to_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with make_engine().connect() as conn:
        if params:
            return pd.read_sql(text(sql), conn, params=params)
        return pd.read_sql(sql, conn)

def save_cache(df: pd.DataFrame, path: Path):
    if path.suffix == ".xlsx":
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False)

def load_cache(path: Path) -> pd.DataFrame:
    return pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)

def _get_dataframe_with_cache(
    sql: str,
    cache_file: Path,
    run_sql: bool,
    use_cache: bool = True,
    sql_params: dict | None = None,
) -> pd.DataFrame:
    """
    Shared helper that optionally loads/saves from cache around a SQL query execution.
    """
    if use_cache and not run_sql and cache_file and cache_file.exists():
        logger.info("Loading cached data from %s", cache_file)
        return load_cache(cache_file)

    logger.info("Executing SQL query (cache target: %s).", cache_file)
    df = run_query_to_df(sql, params=sql_params)
    if cache_file and use_cache:
        save_cache(df, cache_file)
        logger.info("Saved query results to %s", cache_file)
    return df

def get_dataframe(
    run_sql: bool = False,
    use_cache: bool = True,
    cache_date: date | str | None = None,
) -> pd.DataFrame:
    """
    Returns the main screener dataframe (existing behavior). Set run_sql=True to bypass cache.
    """
    cp = cache_path("xlsx", cache_date=cache_date)
    return _get_dataframe_with_cache(SQL_QUERY, cp, run_sql, use_cache=use_cache)

def get_scoring_dataframe(
    run_sql: bool = False,
    use_cache: bool = True,
    cache_date: date | str | None = None,
) -> pd.DataFrame:
    """
    Returns the scoring dataframe powering the Weekly Scoring Board report.
    """
    cp = scoring_cache_path("xlsx", cache_date=cache_date)
    return _get_dataframe_with_cache(SQL_QUERY_SCORING, cp, run_sql, use_cache=use_cache)

def get_report_dataframe(
    run_sql: bool = False,
    use_cache: bool = True,
    cache_date: date | str | None = None,
    eod_as_of_date: date | str | None = None,
) -> pd.DataFrame:
    """
    Returns the report dataframe, with optional EOD as-of date for 30-day window fields.
    """
    cp = report_cache_path("xlsx", cache_date=cache_date)
    params = {"eod_as_of_date": _date_to_str(eod_as_of_date) if eod_as_of_date else None}
    return _get_dataframe_with_cache(SQL_QUERY_REPORT, cp, run_sql, use_cache=use_cache, sql_params=params)

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
