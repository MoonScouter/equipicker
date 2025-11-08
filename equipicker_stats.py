# statistics.py
# pip install openpyxl
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

def _today():
    return datetime.now(ZoneInfo("Europe/Bucharest")).date().isoformat()

def _yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("yes")

def compute_market_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    out["near_1y_high_count"] = int(_yes(df["near_1y_high_5pct"]).sum()) if "near_1y_high_5pct" in df else np.nan
    out["near_1y_low_count"]  = int(_yes(df["near_1y_low_5pct"]).sum())  if "near_1y_low_5pct"  in df else np.nan
    out["avg_rs_coeff"]   = float((df["rs_daily"]/df["rs_sma20"].replace(0,np.nan)).mean())   if {"rs_daily","rs_sma20"}.issubset(df) else np.nan
    out["avg_obvm_coeff"] = float((df["obvm_daily"]/df["obvm_sma20"].replace(0,np.nan)).mean()) if {"obvm_daily","obvm_sma20"}.issubset(df) else np.nan
    rename = {
        "fundamental_total_score":"fundamental_scoring","general_technical_score":"technical_scoring",
        "relative_volume":"relative_volume","relative_performance":"relative_performance","momentum":"momentum",
        "intermediate_trend":"medium_trend","long_term_trend":"long_trend",
        "fundamental_value":"fundamental_value","fundamental_growth":"fundamental_growth",
        "fundamental_risk":"fundamental_risk","fundamental_quality":"fundamental_quality",
        "fundamental_momentum":"fundamental_momentum","rsi_daily":"rsi_daily",
        "rsi_weekly":"rsi_weekly","1w_variation":"w1_variation",
    }
    for col,new in rename.items():
        out[new] = float(pd.to_numeric(df[col], errors="coerce").mean()) if col in df else np.nan
    out["date"] = _today()
    return pd.DataFrame([out])

def _compute_group_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if any(c not in df.columns for c in group_cols): return pd.DataFrame()
    w = df.copy()
    if {"rs_daily","rs_sma20"}.issubset(w):   w["rs_coeff"] = w["rs_daily"]/w["rs_sma20"].replace(0,np.nan)
    if {"obvm_daily","obvm_sma20"}.issubset(w): w["obvm_coeff"] = w["obvm_daily"]/w["obvm_sma20"].replace(0,np.nan)
    if "near_1y_high_5pct" in w: w["near_high_flag"] = _yes(w["near_1y_high_5pct"]).astype(int)
    if "near_1y_low_5pct"  in w: w["near_low_flag"]  = _yes(w["near_1y_low_5pct"]).astype(int)

    agg = {
      "rs_coeff":"mean","obvm_coeff":"mean",
      "fundamental_total_score":"mean","fundamental_value":"mean","fundamental_growth":"mean",
      "fundamental_risk":"mean","fundamental_quality":"mean","fundamental_momentum":"mean",
      "general_technical_score":"mean","relative_volume":"mean","relative_performance":"mean",
      "momentum":"mean","intermediate_trend":"mean","long_term_trend":"mean",
      "rsi_daily":"mean","rsi_weekly":"mean","1w_variation":"mean",
      "near_high_flag":"sum","near_low_flag":"sum"
    }
    agg = {k:v for k,v in agg.items() if k in w.columns}
    g = (w.groupby(group_cols, dropna=False).agg(agg).reset_index()
          .rename(columns={
            "fundamental_total_score":"fundamental_scoring",
            "general_technical_score":"technical_scoring",
            "relative_volume":"relative_volume","relative_performance":"relative_performance",
            "intermediate_trend":"medium_trend","long_term_trend":"long_trend",
            "near_high_flag":"near_1y_high_count","near_low_flag":"near_1y_low_count",
            "1w_variation":"w1_variation",
          }))
    g.insert(0,"date",_today())
    return g

def compute_sector_only_stats(df: pd.DataFrame) -> pd.DataFrame:
    return _compute_group_stats(df, ["gic_sector"])

def compute_sector_industry_stats(df: pd.DataFrame) -> pd.DataFrame:
    return _compute_group_stats(df, ["gic_sector","gic_industry"])

def save_sector_industry_daily(sector_ind_df: pd.DataFrame, out_dir: Path) -> Path|None:
    if sector_ind_df.empty: return None
    path = out_dir/"sector_stats.xlsx"
    if path.exists():
        old = pd.read_excel(path)
        today = pd.to_datetime(_today()).date()
        old["date"] = pd.to_datetime(old["date"]).dt.date
        new = (pd.concat([old, sector_ind_df], ignore_index=True)
                 .drop_duplicates(subset=["date","gic_sector"], keep="last"))
    else:
        new = sector_ind_df
    new.to_excel(path, index=False)
    return path

def compute_and_save_stats(df: pd.DataFrame, out_dir: Path):
    market = compute_market_stats(df)
    sector_only = compute_sector_only_stats(df)
    sector_ind  = compute_sector_industry_stats(df)
    saved = save_sector_industry_daily(sector_only, out_dir)
    return market, sector_only, sector_ind, saved