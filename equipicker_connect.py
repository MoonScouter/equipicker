# pip install sqlalchemy mysql-connector-python pandas openpyxl
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus
from equipicker_filters import extreme_accel, accel_normal, accel_weak, wake_up

import pandas as pd
from sqlalchemy import create_engine
from sql_query import SQL_QUERY  # your big SQL string

#RUN_SQL = False  # False = load cache if exists

# --- paths anchored to this script ---
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data"
CACHE_DIR.mkdir(exist_ok=True)
def bucharest_today_str(): return datetime.now(ZoneInfo("Europe/Bucharest")).date().isoformat()
def cache_path(ext="xlsx"): return CACHE_DIR / f"select_{bucharest_today_str()}.{ext}"

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

def get_dataframe(run_sql: bool) -> pd.DataFrame:
    cp = cache_path("xlsx")
    if not run_sql and cp.exists():
        return load_cache(cp)
    df = run_query_to_df(SQL_QUERY)
    save_cache(df, cp)
    return df

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
