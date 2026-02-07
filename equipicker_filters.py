import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, datetime
from zoneinfo import ZoneInfo


def _format_market_cap_series(series: pd.Series) -> pd.Series:
    mc = pd.to_numeric(series, errors='coerce')
    return np.where(
        mc >= 1_000_000_000,
        (mc/1_000_000_000).round(2).astype(str) + "B",
        (mc/1_000_000).round(2).astype(str) + "M"
    )


def extreme_accel_up(
    df: pd.DataFrame,
    cache_dir: Path,
    *,
    save_output: bool = True,
    output_date: date | None = None,
) -> pd.DataFrame:
    """Full-blown extreme acceleration filter + save results to Excel."""
    m = pd.Series(True, index=df.index)

    # Scores pinned
    m &= df['general_technical_score'] >= 99.9

    m &= df['relative_performance']    >= 99.9
    if 'rs_monthly' in df.columns:
        m &= df['rs_monthly'] >= 2.0

    m &= df['relative_volume']         >= 99.9
    if {'obvm_weekly','obvm_monthly'}.issubset(df.columns):
        m &= df['obvm_weekly']  > 0
        m &= df['obvm_monthly'] > 0

    m &= df['momentum']                >= 99.9
    if {'rsi_weekly','rsi_daily'}.issubset(df.columns):
        m &= (df['rsi_weekly'] >= 70) & (df['rsi_daily'] >= 80)

    m &= df['intermediate_trend']      >= 99.9

    m &= df['long_term_trend']         >= 99.9

    if {'sma_daily_20','sma_daily_50','sma_daily_200','eod_price_used'}.issubset(df.columns):
        m &= df['sma_daily_20'] >= 1.02 * df['sma_daily_50']
        m &= df['sma_daily_50'] >= 1.02 * df['sma_daily_200']
        m &= df['eod_price_used'] >= 1.03 * df['sma_daily_20']

    # RS/OBVM thrust
    m &= df['rs_daily']   > df['rs_sma20']
    m &= df['obvm_daily'] > 1.5 * df['obvm_sma20']

    out = df.loc[m].copy()

    # Ranking
    if 'fundamental_total_score' in out.columns:
        out = out.sort_values(
            by=['fundamental_total_score', 'general_technical_score'],
            ascending=[False, False]
        )
    else:
        out = out.sort_values(by='general_technical_score', ascending=False)

    # Market cap display: ≥$1B in B, else in M
    if 'market_cap' in out.columns:
        out['market_cap'] = _format_market_cap_series(out['market_cap'])

    if save_output:
        run_day = output_date or datetime.now(ZoneInfo("Europe/Bucharest")).date()
        path = cache_dir / f"extreme_accel_up_{run_day.isoformat()}.xlsx"
        out.to_excel(path, index=False)
        print(f"[extreme_accel_up] saved {len(out)} rows to {path}")
    return out

# placeholders for other variants
def accel_normal(df: pd.DataFrame, out_dir: Path, tighten: bool = True) -> pd.DataFrame:
    """
    Acceleration (normal, no weakening). Optional tighten adds rs_monthly >= 0 and Trend gate.
    Saves accel_normal_{YYYY-MM-DD}.xlsx in out_dir.
    """
    m = pd.Series(True, index=df.index)

    # Core scores
    m &= df['general_technical_score'] >= 82
    m &= df['general_technical_score'] < 100 # Exclude extreme accel stocks

    m &= df['relative_performance']    >= 70 # Same as saying that df['rs_monthly'] >= 0.8
    # Optional tighten
    if tighten and ('rs_monthly' in df.columns):
        m &= df['rs_monthly'] >= 0

    m &= df['relative_volume']         >= 75 # Same as saying that only df['obvm_monthly'] > 0
    # Volume trend
    if {'obvm_weekly','obvm_monthly'}.issubset(df.columns):
        m &= (df['obvm_monthly'] > 0)

    m &= df['momentum']                >= 78 # Same as below filter for RSI
    # RSI confirms
    if {'rsi_weekly','rsi_daily'}.issubset(df.columns):
        m &= (df['rsi_weekly'] >= 60) & (df['rsi_daily'].between(65, 85))


    m &= df['intermediate_trend']      >= 80 # Positive crossover & (price above MA50 OR price in between MA20-MA50, but at least at @60% of the gap)

    m &= df['long_term_trend']         >= 70 # Positive crossover & (price above MA50 OR price in between MA50-MA200, but at least at @40% of the gap)

    # RS / OBVM thrust
    if {'rs_daily','rs_sma20'}.issubset(df.columns):
        m &= df['rs_daily'] > df['rs_sma20']
    if {'obvm_daily','obvm_sma20'}.issubset(df.columns):
        m &= df['obvm_daily'] > 1.10 * df['obvm_sma20']

    out = df.loc[m].copy()

    # Ranking: technical first, then fundamental
    if 'fundamental_total_score' in out.columns:
        out = out.sort_values(
            by=['general_technical_score', 'fundamental_total_score'],
            ascending=[False, False]
        )
    else:
        out = out.sort_values(by='general_technical_score', ascending=False)

    # Format market cap for display
    if 'market_cap' in out.columns:
        mc = pd.to_numeric(out['market_cap'], errors='coerce')
        out['market_cap'] = np.where(
            mc >= 1_000_000_000,
            (mc/1_000_000_000).round(2).astype(str) + "B",
            (mc/1_000_000).round(2).astype(str) + "M"
        )

    # Save
    today = datetime.now(ZoneInfo("Europe/Bucharest")).date().isoformat()
    path = out_dir / f"accel_normal_{today}.xlsx"
    out.to_excel(path, index=False)
    return out

def accel_up_weak(
    df: pd.DataFrame,
    out_dir: Path,
    tighten: bool = True,
    *,
    save_output: bool = True,
    output_date: date | None = None,
) -> pd.DataFrame:
    """
    Acceleration (weak up). Early-to-moderate upside strength, not extreme.
    Saves accel_up_weak_{YYYY-MM-DD}.xlsx in out_dir.
    """
    m = pd.Series(True, index=df.index)

    # Core scores
    m &= df['general_technical_score'] >= 72

    m &= df['relative_volume']         >= 75 # Same as saying that only df['obvm_monthly'] > 0
    # Volume trend
    if {'obvm_weekly','obvm_monthly'}.issubset(df.columns):
        m &= (df['obvm_monthly'] > 0)

    m &= df['relative_performance']    >= 65 # Same as saying that df['rs_monthly'] >= 0.6
    # Optional tighten
    if tighten and ('rs_monthly' in df.columns):
        m &= df['rs_monthly'] >= 0

    m &= df['momentum']                >= 72 #Same as below filter for RSI
    # RSI confirms (cooling but still constructive)
    if {'rsi_weekly','rsi_daily'}.issubset(df.columns):
        m &= df['rsi_weekly'].between(60, 75)
        m &= df['rsi_daily'].between(60, 70)
        m &= df['rsi_daily'] <= df['rsi_weekly']

    m &= df['intermediate_trend']      >= 80 # Positive crossover & (price above MA50 OR price in between MA20-MA50, but at least at @60% of the gap)

    m &= df['long_term_trend']         >= 70 # Positive crossover & (price above MA50 OR price in between MA50-MA200, but at least at @40% of the gap)

    # RS / OBVM thrust
    if {'rs_daily','rs_sma20'}.issubset(df.columns):
        m &= df['rs_daily'] < df['rs_sma20']
    if {'obvm_daily','obvm_sma20'}.issubset(df.columns):
        m &= df['obvm_daily'] < df['obvm_sma20'] # Exclude extreme accel & normal accel stocks

    out = df.loc[m].copy()

    # Ranking: technical then fundamental
    if 'fundamental_total_score' in out.columns:
        out = out.sort_values(['general_technical_score','fundamental_total_score'], ascending=[False, False])
    else:
        out = out.sort_values('general_technical_score', ascending=False)

    # Market cap display
    if 'market_cap' in out.columns:
        out['market_cap'] = _format_market_cap_series(out['market_cap'])

    if save_output:
        run_day = output_date or datetime.now(ZoneInfo("Europe/Bucharest")).date()
        path = out_dir / f"accel_up_weak_{run_day.isoformat()}.xlsx"
        out.to_excel(path, index=False)
    return out


def extreme_accel_down(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    save_output: bool = True,
    output_date: date | None = None,
) -> pd.DataFrame:
    """Full-blown downside acceleration filter + optional save to Excel."""
    m = pd.Series(True, index=df.index)

    m &= df['general_technical_score'] <= 0.1

    m &= df['relative_performance'] <= 0.1
    if 'rs_monthly' in df.columns:
        m &= df['rs_monthly'] <= -2.0

    m &= df['relative_volume'] <= 0.1
    if {'obvm_weekly', 'obvm_monthly'}.issubset(df.columns):
        m &= df['obvm_weekly'] < 0
        m &= df['obvm_monthly'] < 0

    m &= df['momentum'] <= 0.1
    if {'rsi_weekly', 'rsi_daily'}.issubset(df.columns):
        m &= (df['rsi_weekly'] <= 30) & (df['rsi_daily'] <= 20)

    m &= df['intermediate_trend'] <= 0.1
    m &= df['long_term_trend'] <= 0.1

    if {'sma_daily_20', 'sma_daily_50', 'sma_daily_200', 'eod_price_used'}.issubset(df.columns):
        m &= df['sma_daily_20'] <= 0.98 * df['sma_daily_50']
        m &= df['sma_daily_50'] <= 0.98 * df['sma_daily_200']
        m &= df['eod_price_used'] <= 0.97 * df['sma_daily_20']

    m &= df['rs_daily'] < df['rs_sma20']
    m &= df['obvm_daily'] < (df['obvm_sma20'] / 1.5)

    out = df.loc[m].copy()

    if 'fundamental_total_score' in out.columns:
        out = out.sort_values(
            by=['general_technical_score', 'fundamental_total_score'],
            ascending=[True, True],
        )
    else:
        out = out.sort_values(by='general_technical_score', ascending=True)

    if 'market_cap' in out.columns:
        out['market_cap'] = _format_market_cap_series(out['market_cap'])

    if save_output:
        run_day = output_date or datetime.now(ZoneInfo("Europe/Bucharest")).date()
        path = out_dir / f"extreme_accel_down_{run_day.isoformat()}.xlsx"
        out.to_excel(path, index=False)
    return out


def accel_down_weak(
    df: pd.DataFrame,
    out_dir: Path,
    tighten: bool = True,
    *,
    save_output: bool = True,
    output_date: date | None = None,
) -> pd.DataFrame:
    """
    Acceleration (weak down). Early-to-moderate downside strength, not extreme.
    Saves accel_down_weak_{YYYY-MM-DD}.xlsx in out_dir.
    """
    m = pd.Series(True, index=df.index)

    m &= df['general_technical_score'] <= 28

    m &= df['relative_volume'] <= 25
    if {'obvm_weekly', 'obvm_monthly'}.issubset(df.columns):
        m &= df['obvm_monthly'] < 0

    m &= df['relative_performance'] <= 35
    if tighten and ('rs_monthly' in df.columns):
        m &= df['rs_monthly'] <= 0

    m &= df['momentum'] <= 28
    if {'rsi_weekly', 'rsi_daily'}.issubset(df.columns):
        m &= df['rsi_weekly'].between(25, 40)
        m &= df['rsi_daily'].between(30, 40)
        m &= df['rsi_daily'] >= df['rsi_weekly']

    m &= df['intermediate_trend'] <= 20
    m &= df['long_term_trend'] <= 30

    if {'rs_daily', 'rs_sma20'}.issubset(df.columns):
        m &= df['rs_daily'] > df['rs_sma20']
    if {'obvm_daily', 'obvm_sma20'}.issubset(df.columns):
        m &= df['obvm_daily'] > df['obvm_sma20']

    out = df.loc[m].copy()

    if 'fundamental_total_score' in out.columns:
        out = out.sort_values(
            by=['general_technical_score', 'fundamental_total_score'],
            ascending=[True, True],
        )
    else:
        out = out.sort_values(by='general_technical_score', ascending=True)

    if 'market_cap' in out.columns:
        out['market_cap'] = _format_market_cap_series(out['market_cap'])

    if save_output:
        run_day = output_date or datetime.now(ZoneInfo("Europe/Bucharest")).date()
        path = out_dir / f"accel_down_weak_{run_day.isoformat()}.xlsx"
        out.to_excel(path, index=False)
    return out


def extreme_accel(
    df: pd.DataFrame,
    cache_dir: Path,
    *,
    save_output: bool = True,
    output_date: date | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper for extreme_accel_up."""
    return extreme_accel_up(
        df,
        cache_dir,
        save_output=save_output,
        output_date=output_date,
    )


def accel_weak(
    df: pd.DataFrame,
    out_dir: Path,
    tighten: bool = True,
    *,
    save_output: bool = True,
    output_date: date | None = None,
) -> pd.DataFrame:
    """Backward-compatible wrapper for accel_up_weak."""
    return accel_up_weak(
        df,
        out_dir,
        tighten=tighten,
        save_output=save_output,
        output_date=output_date,
    )


def wake_up(df: pd.DataFrame, out_dir: Path, tighten: bool = True) -> pd.DataFrame:
    """
    Wake-up (stalling -> upside). Early turn, not overlapping with accel_weak/normal/extreme.
    Saves wake_up_{YYYY-MM-DD}.xlsx in out_dir.
    """
    m = pd.Series(True, index=df.index)

    # Partition the space below accel_weak
    m &= df['general_technical_score'].between(60, 78, inclusive="left")   # [60,78)

    m &= df['relative_performance'].between(50, 65, inclusive="left")      # [50,65) ; RS between 0 - 0.6

    m &= df['relative_volume'].between(50, 75, inclusive="left")           # [50,75); OBVM monthly is [0,inf) & OBVM weekly can be either pos or neg

    m &= df['momentum'].between(45, 78, inclusive="left")                  # [45,78)
    # RSI confirms: daily leads weekly at low-60s
    if {'rsi_weekly','rsi_daily'}.issubset(df.columns):
        m &= df['rsi_weekly'].between(45, 60, inclusive="both")
        m &= df['rsi_daily'].between(50, 65, inclusive="both")
        m &= df['rsi_daily'] >= df['rsi_weekly']

    m &= df['intermediate_trend'].between(40, 100, inclusive="left")        # [40,80) ; Positive crossover and price is very close to MA50 even if below it

    m &= df['long_term_trend'].between(50, 70, inclusive="left")           # [50,70); Positive crossover and price is above MA200

    # Price vs MAs: 50d close-by or just reclaimed
    if {'eod_price_used','sma_daily_50'}.issubset(df.columns):
        cond_close_50 = df['eod_price_used'] >= df['sma_daily_50']
        if 'near_ma50_5pct' in df.columns:
            cond_close_50 |= df['near_ma50_5pct'].astype(str).str.lower().eq('yes')
        m &= cond_close_50

    # RS / OBVM early thrust
    if {'rs_daily','rs_sma20'}.issubset(df.columns):
        m &= df['rs_daily'] > df['rs_sma20']
    if {'obvm_daily','obvm_sma20'}.issubset(df.columns):
        m &= df['obvm_daily'] > df['obvm_sma20'] # Ensure early wakers are positive on volume

    out = df.loc[m].copy()

    # Ranking: technical first, then fundamental
    if 'fundamental_total_score' in out.columns:
        out = out.sort_values(['general_technical_score','fundamental_total_score'], ascending=[False, False])
    else:
        out = out.sort_values('general_technical_score', ascending=False)

    # Market cap display
    if 'market_cap' in out.columns:
        mc = pd.to_numeric(out['market_cap'], errors='coerce')
        out['market_cap'] = np.where(
            mc >= 1_000_000_000, (mc/1_000_000_000).round(2).astype(str) + "B",
                                  (mc/1_000_000).round(2).astype(str) + "M"
        )

    today = datetime.now(ZoneInfo("Europe/Bucharest")).date().isoformat()
    path = out_dir / f"wake_up_{today}.xlsx"
    out.to_excel(path, index=False)
    return out


# --- PEG value screen ---------------------------------------------------------

_CAP_ORDER = ["Nano","Micro","Small","Mid","Large","Mega","Unknown"]

def _cap_category_from_usd(x: float) -> str:
    """x is market_cap in USD. Bins in USD: 50M, 300M, 2B, 10B, 200B."""
    if pd.isna(x): return "Unknown"
    v = float(x) / 1_000_000_000.0  # billions
    if v < 0.05: return "Nano"                  # < $50m
    if v < 0.3:  return "Micro"                 # $50m–$300m
    if v < 2:    return "Small"                 # $300m–$2b
    if v < 10:   return "Mid"                   # $2b–$10b
    if v < 200:  return "Large"                 # $10b–$200b
    return "Mega"                                # >= $200b

def peg_value(df: pd.DataFrame, out_dir: Path,
              min_fund: float = 60,
              min_peg_score: float = 85,
              peg_abs_lt: float = 1.0
              ) -> pd.DataFrame:
    """
    PEG value screen: absolute PEG < peg_abs_lt, PEG scoring >= min_peg_score,
    Fundamental total >= min_fund.
    Saves peg_value_{YYYY-MM-DD}.xlsx.
    """
    req_cols = {"peg_ratio","peg_ratio_score","fundamental_total_score","gic_sector","market_cap"}
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise KeyError(f"peg_value: missing columns: {missing}")

    w = df.copy()
    # numeric guards
    w["peg_ratio"] = pd.to_numeric(w["peg_ratio"], errors="coerce")
    w["peg_ratio_score"] = pd.to_numeric(w["peg_ratio_score"], errors="coerce")
    w["fundamental_total_score"] = pd.to_numeric(w["fundamental_total_score"], errors="coerce")
    w["market_cap"] = pd.to_numeric(w["market_cap"], errors="coerce")

    m = pd.Series(True, index=w.index)
    m &= w["fundamental_total_score"] >= min_fund
    m &= w["peg_ratio"].notna() & (w["peg_ratio"] > 0) & (w["peg_ratio"] < peg_abs_lt)
    m &= w["peg_ratio_score"] >= min_peg_score

    out = w.loc[m].copy()

    # cap category
    out["cap_category"] = out["market_cap"].apply(_cap_category_from_usd)

    # numeric market cap for sorting
    out["market_cap_n"] = pd.to_numeric(out["market_cap"], errors="coerce")

    # sort: cap bucket, Market Cap desc, Fundamental score desc, PEG asc, Fundamental momentum desc
    sort_cols = ["cap_category", "market_cap_n", "fundamental_total_score", "peg_ratio", "fundamental_momentum"]
    keep = [c for c in sort_cols if c in out.columns]
    out = out.sort_values(by=keep, ascending=[True, False, False, True, False][:len(keep)])

    # display-friendly market cap (after sorting)
    mc = out["market_cap_n"]
    out["market_cap"] = np.where(mc >= 1_000_000_000,
                                 (mc/1_000_000_000).round(2).astype(str) + "B",
                                 (mc/1_000_000).round(2).astype(str) + "M")
    out = out.drop(columns=["market_cap_n"])

    # save
    today = datetime.now(ZoneInfo("Europe/Bucharest")).date().isoformat()
    path = out_dir / f"peg_value_{today}.xlsx"
    out.to_excel(path, index=False)
    return out
