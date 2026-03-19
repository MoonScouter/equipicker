from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from prices_service import list_prices_cache_paths, load_prices_cache, normalize_price_ticker
from weekly_scoring_board import compute_sector_overview_stats

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
DEFAULT_MARKET_CONFIG_PATH = CONFIG_DIR / "market_regime_config.json"
DEFAULT_SECTOR_FAMILIES_PATH = CONFIG_DIR / "sector_families.json"
DEFAULT_MARKET_METHODOLOGY_PATH = CONFIG_DIR / "market_methodology.md"
MARKET_CACHE_DIR = DATA_DIR / "market"


def load_market_regime_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_MARKET_CONFIG_PATH
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_sector_families(path: Path | str | None = None) -> dict[str, list[str]]:
    config_path = Path(path) if path else DEFAULT_SECTOR_FAMILIES_PATH
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return {str(key): [str(value) for value in values] for key, values in payload.items()}


def load_market_methodology_text(path: Path | str | None = None) -> str:
    methodology_path = Path(path) if path else DEFAULT_MARKET_METHODOLOGY_PATH
    return methodology_path.read_text(encoding="utf-8")


def build_market_signature(
    evaluation_date: date,
    rsi_start_date: date,
    month_anchor_date: date,
    week_anchor_date: date,
) -> str:
    return (
        f"eval_{evaluation_date.isoformat()}__"
        f"rsi_{rsi_start_date.isoformat()}__"
        f"m1_{month_anchor_date.isoformat()}__"
        f"w1_{week_anchor_date.isoformat()}"
    )


def market_snapshot_path(signature: str, cache_dir: Path | None = None) -> Path:
    directory = cache_dir or MARKET_CACHE_DIR
    return directory / f"market_snapshot_{signature}.json"


def stock_rsi_regime_path(signature: str, cache_dir: Path | None = None) -> Path:
    directory = cache_dir or MARKET_CACHE_DIR
    return directory / f"stock_rsi_regime_{signature}.jsonl"


def setup_readiness_path(signature: str, cache_dir: Path | None = None) -> Path:
    directory = cache_dir or MARKET_CACHE_DIR
    return directory / f"setup_readiness_{signature}.jsonl"


def resolve_anchor_on_or_before(available_dates: Sequence[date], target_date: date) -> date:
    eligible = sorted(entry for entry in available_dates if entry <= target_date)
    if eligible:
        return eligible[-1]
    if available_dates:
        return sorted(available_dates)[0]
    return target_date


def get_default_market_anchors(
    available_dates: Sequence[date],
    config: Mapping[str, Any],
) -> dict[str, date]:
    sorted_dates = sorted(available_dates)
    evaluation_date = sorted_dates[-1] if sorted_dates else date.today()
    interval_config = config.get("default_intervals_days", {})
    rsi_window_days = int(interval_config.get("rsi_window_days", 90))
    month_offset_days = int(interval_config.get("month_offset_days", 30))
    week_offset_days = int(interval_config.get("week_offset_days", 7))
    return {
        "evaluation_date": evaluation_date,
        "rsi_start_date": evaluation_date - timedelta(days=rsi_window_days),
        "month_anchor_date": resolve_anchor_on_or_before(
            sorted_dates,
            evaluation_date - timedelta(days=month_offset_days),
        ),
        "week_anchor_date": resolve_anchor_on_or_before(
            sorted_dates,
            evaluation_date - timedelta(days=week_offset_days),
        ),
    }


def _ensure_market_cache_dir(cache_dir: Path | None = None) -> Path:
    directory = cache_dir or MARKET_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def market_cache_status(signature: str, cache_dir: Path | None = None) -> dict[str, object]:
    directory = _ensure_market_cache_dir(cache_dir)
    snapshot_file = market_snapshot_path(signature, directory)
    stock_file = stock_rsi_regime_path(signature, directory)
    setup_file = setup_readiness_path(signature, directory)
    return {
        "signature": signature,
        "cache_dir": directory,
        "market_snapshot": snapshot_file,
        "stock_rsi_regime": stock_file,
        "setup_readiness": setup_file,
        "ready": snapshot_file.exists() and stock_file.exists() and setup_file.exists(),
    }


def _to_float(value: object) -> Optional[float]:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _to_iso(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.date().isoformat()


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(np.clip(value, lower, upper))


def _weighted_average_available(weighted_values: Sequence[tuple[Optional[float], float]]) -> Optional[float]:
    total_weight = 0.0
    weighted_sum = 0.0
    for value, weight in weighted_values:
        if value is None:
            continue
        weighted_sum += float(value) * float(weight)
        total_weight += float(weight)
    if total_weight <= 0:
        return None
    return weighted_sum / total_weight


def _weighted_sum_available(weighted_values: Sequence[tuple[Optional[float], float]]) -> Optional[float]:
    weighted_sum = 0.0
    found_value = False
    for value, weight in weighted_values:
        if value is None:
            continue
        found_value = True
        weighted_sum += float(value) * float(weight)
    if not found_value:
        return None
    return weighted_sum


def _score_to_label(score: Optional[float], bands: Sequence[Mapping[str, object]]) -> Optional[str]:
    if score is None:
        return None
    for band in bands:
        minimum = float(band.get("min", 0))
        maximum = float(band.get("max", 100))
        if minimum <= score <= maximum:
            label = str(band.get("label", "")).strip()
            return label or None
    return None


def _safe_mean(values: Sequence[object]) -> Optional[float]:
    numeric = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _longest_consecutive(mask: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _normalize_report_df(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "Sector" in working.columns and "sector" not in working.columns:
        working = working.rename(columns={"Sector": "sector"})
    if "General_Technical_Score" in working.columns and "general_technical_score" not in working.columns:
        working = working.rename(columns={"General_Technical_Score": "general_technical_score"})
    if "General Technical Score" in working.columns and "general_technical_score" not in working.columns:
        working = working.rename(columns={"General Technical Score": "general_technical_score"})
    return working


def _prepare_report_df(df: pd.DataFrame) -> pd.DataFrame:
    working = _normalize_report_df(df)
    if "ticker" in working.columns:
        working["ticker"] = working["ticker"].map(normalize_price_ticker)
    if "sector" in working.columns:
        working["sector"] = working["sector"].fillna("Unspecified")
    if "industry" in working.columns:
        working["industry"] = working["industry"].fillna("Unspecified")
    numeric_columns = [
        "general_technical_score",
        "fundamental_total_score",
        "fundamental_quality",
        "fundamental_risk",
        "fundamental_value",
        "fundamental_growth",
        "fundamental_momentum",
        "relative_performance",
        "relative_volume",
        "eod_price_used",
        "1m_close",
    ]
    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    return working


def _load_price_rsi_history(
    frequency: str,
    start_date: date,
    end_date: date,
    tickers: Sequence[str],
    *,
    price_df_override: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, list[str]]:
    normalized_tickers = {normalize_price_ticker(ticker) for ticker in tickers}
    if price_df_override is not None:
        combined = price_df_override.copy()
        source_names = ["override"]
    else:
        parts: list[pd.DataFrame] = []
        source_names: list[str] = []
        for cache_path in list_prices_cache_paths(frequency):
            loaded = load_prices_cache(cache_path)
            if loaded.empty:
                continue
            parts.append(loaded)
            source_names.append(cache_path.name)
        if not parts:
            return pd.DataFrame(columns=["ticker", "date", "rsi_14"]), source_names
        combined = pd.concat(parts, ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=["ticker", "date", "rsi_14"]), source_names
    working = combined.copy()
    working["ticker"] = working["ticker"].map(normalize_price_ticker)
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.date
    working["rsi_14"] = pd.to_numeric(working.get("rsi_14"), errors="coerce")
    working = working[
        working["ticker"].isin(normalized_tickers)
        & working["date"].notna()
        & (working["date"] >= start_date)
        & (working["date"] <= end_date)
    ]
    working = working.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)
    return working.loc[:, ["ticker", "date", "rsi_14"]], source_names


def _build_family_lookup(sector_families: Mapping[str, Sequence[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for family, sectors in sector_families.items():
        for sector in sectors:
            lookup[str(sector)] = str(family)
    return lookup


def _compute_stock_rsi_row(
    ticker: str,
    sector: str,
    industry: str,
    weekly_values: list[float],
    daily_values: list[float],
    evaluation_date: date,
    rsi_start_date: date,
    config: Mapping[str, Any],
) -> dict[str, object]:
    minimum_data = config.get("minimum_data", {})
    min_weekly_obs = int(minimum_data.get("min_weekly_rsi_observations", 4))
    max_daily_lookback = int(minimum_data.get("max_daily_rsi_lookback", 20))
    weights = config.get("stock_rsi_weights", {})
    bands = config.get("score_bands", {}).get("stock_rsi_regime", [])
    row: dict[str, object] = {
        "ticker": ticker,
        "sector": sector,
        "industry": industry,
        "evaluation_date": evaluation_date.isoformat(),
        "rsi_start_date": rsi_start_date.isoformat(),
        "weekly_observation_count": len(weekly_values),
        "daily_observation_count": len(daily_values),
        "stock_rsi_regime_score": np.nan,
        "stock_rsi_regime_label": None,
        "missing_data_reason": None,
    }
    if len(weekly_values) < min_weekly_obs:
        row["missing_data_reason"] = f"Need at least {min_weekly_obs} weekly RSI observations in the selected interval."
        return row

    weekly_series = pd.Series(weekly_values, dtype=float)
    daily_series = pd.Series(daily_values, dtype=float)
    pct_w_ge_40 = float((weekly_series >= 40).mean() * 100.0)
    pct_w_le_60 = float((weekly_series <= 60).mean() * 100.0)
    cnt_w_ge_60 = int((weekly_series >= 60).sum())
    cnt_w_ge_70 = int((weekly_series >= 70).sum())
    cnt_w_le_40 = int((weekly_series <= 40).sum())
    cnt_w_le_30 = int((weekly_series <= 30).sum())
    cnt_w_lt_40 = int((weekly_series < 40).sum())
    cnt_w_gt_60 = int((weekly_series > 60).sum())
    max_consec_w_lt_40 = _longest_consecutive((weekly_series < 40).tolist())
    max_consec_w_gt_60 = _longest_consecutive((weekly_series > 60).tolist())

    bull_persist = _clamp((pct_w_ge_40 - 50.0) / 40.0) * 100.0
    bear_persist = _clamp((pct_w_le_60 - 50.0) / 40.0) * 100.0
    bull_expand = (
        0.6 * _clamp(cnt_w_ge_60 / 4.0) * 100.0
        + 0.4 * _clamp(cnt_w_ge_70 / 2.0) * 100.0
    )
    bear_expand = (
        0.6 * _clamp(cnt_w_le_40 / 4.0) * 100.0
        + 0.4 * _clamp(cnt_w_le_30 / 2.0) * 100.0
    )
    bull_resilience = 100.0 - (
        0.6 * _clamp(cnt_w_lt_40 / 4.0)
        + 0.4 * _clamp(max_consec_w_lt_40 / 3.0)
    ) * 100.0
    bear_resilience = 100.0 - (
        0.6 * _clamp(cnt_w_gt_60 / 4.0)
        + 0.4 * _clamp(max_consec_w_gt_60 / 3.0)
    ) * 100.0

    bull_daily: Optional[float] = None
    bear_daily: Optional[float] = None
    daily_avg_recent: Optional[float] = None
    daily_last: Optional[float] = None
    if not daily_series.empty:
        recent_daily = daily_series.tail(max_daily_lookback)
        daily_avg_recent = float(recent_daily.mean())
        daily_last = float(recent_daily.iloc[-1])
        bull_daily = (
            0.5 * _clamp((daily_avg_recent - 50.0) / 15.0) * 100.0
            + 0.5 * _clamp((daily_last - 55.0) / 15.0) * 100.0
        )
        bear_daily = (
            0.5 * _clamp((50.0 - daily_avg_recent) / 15.0) * 100.0
            + 0.5 * _clamp((45.0 - daily_last) / 15.0) * 100.0
        )

    bull_evidence = _weighted_average_available(
        [
            (bull_persist, float(weights.get("weekly_range_persistence", 0.45))),
            (bull_expand, float(weights.get("expansion_frequency", 0.25))),
            (bull_resilience, float(weights.get("pullback_resilience", 0.2))),
            (bull_daily, float(weights.get("daily_confirmation", 0.1))),
        ]
    )
    bear_evidence = _weighted_average_available(
        [
            (bear_persist, float(weights.get("weekly_range_persistence", 0.45))),
            (bear_expand, float(weights.get("expansion_frequency", 0.25))),
            (bear_resilience, float(weights.get("pullback_resilience", 0.2))),
            (bear_daily, float(weights.get("daily_confirmation", 0.1))),
        ]
    )
    score = None
    if bull_evidence is not None and bear_evidence is not None:
        score = float(np.clip(50.0 + 0.5 * (bull_evidence - bear_evidence), 0.0, 100.0))

    row.update(
        {
            "pct_w_ge_40": pct_w_ge_40,
            "pct_w_le_60": pct_w_le_60,
            "cnt_w_ge_60": cnt_w_ge_60,
            "cnt_w_ge_70": cnt_w_ge_70,
            "cnt_w_le_40": cnt_w_le_40,
            "cnt_w_le_30": cnt_w_le_30,
            "cnt_w_lt_40": cnt_w_lt_40,
            "cnt_w_gt_60": cnt_w_gt_60,
            "max_consec_w_lt_40": max_consec_w_lt_40,
            "max_consec_w_gt_60": max_consec_w_gt_60,
            "bull_persist": bull_persist,
            "bear_persist": bear_persist,
            "bull_expand": bull_expand,
            "bear_expand": bear_expand,
            "bull_resilience": bull_resilience,
            "bear_resilience": bear_resilience,
            "bull_daily": bull_daily if bull_daily is not None else np.nan,
            "bear_daily": bear_daily if bear_daily is not None else np.nan,
            "bull_evidence": bull_evidence if bull_evidence is not None else np.nan,
            "bear_evidence": bear_evidence if bear_evidence is not None else np.nan,
            "daily_avg_recent": daily_avg_recent if daily_avg_recent is not None else np.nan,
            "daily_last": daily_last if daily_last is not None else np.nan,
            "stock_rsi_regime_score": score if score is not None else np.nan,
            "stock_rsi_regime_label": _score_to_label(score, bands),
        }
    )
    return row


def compute_stock_rsi_regime(
    evaluation_df: pd.DataFrame,
    daily_prices_df: pd.DataFrame,
    weekly_prices_df: pd.DataFrame,
    evaluation_date: date,
    rsi_start_date: date,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    prepared_eval = _prepare_report_df(evaluation_df)
    if prepared_eval.empty:
        return pd.DataFrame()
    daily_lookup = {
        ticker: group.sort_values("date", kind="stable")["rsi_14"].dropna().astype(float).tolist()
        for ticker, group in daily_prices_df.groupby("ticker", sort=False)
    }
    weekly_lookup = {
        ticker: group.sort_values("date", kind="stable")["rsi_14"].dropna().astype(float).tolist()
        for ticker, group in weekly_prices_df.groupby("ticker", sort=False)
    }
    universe = (
        prepared_eval[["ticker", "sector", "industry"]]
        .dropna(subset=["ticker"])
        .drop_duplicates(subset=["ticker"], keep="first")
        .fillna({"sector": "Unspecified", "industry": "Unspecified"})
    )
    rows: list[dict[str, object]] = []
    for item in universe.itertuples(index=False):
        ticker = str(item.ticker)
        rows.append(
            _compute_stock_rsi_row(
                ticker,
                str(item.sector),
                str(item.industry),
                weekly_lookup.get(ticker, []),
                daily_lookup.get(ticker, []),
                evaluation_date,
                rsi_start_date,
                config,
            )
        )
    return pd.DataFrame(rows)


def _sector_participation_from_stock_rsi(stock_rsi_df: pd.DataFrame) -> pd.DataFrame:
    if stock_rsi_df.empty:
        return pd.DataFrame(columns=[
            "sector",
            "sector_rsi_breadth_pct_60",
            "sector_rsi_breadth_pct_75",
            "sector_rsi_breadth_pct_lt40",
            "sector_rsi_participation_composite_score",
        ])
    working = stock_rsi_df.copy()
    working["stock_rsi_regime_score"] = pd.to_numeric(working["stock_rsi_regime_score"], errors="coerce")
    working = working.dropna(subset=["stock_rsi_regime_score"])
    rows: list[dict[str, object]] = []
    for sector, group in working.groupby("sector", dropna=False):
        scores = pd.to_numeric(group["stock_rsi_regime_score"], errors="coerce").dropna()
        if scores.empty:
            continue
        rows.append(
            {
                "sector": sector,
                "sector_rsi_breadth_pct_60": float((scores >= 60).mean() * 100.0),
                "sector_rsi_breadth_pct_75": float((scores >= 75).mean() * 100.0),
                "sector_rsi_breadth_pct_lt40": float((scores < 40).mean() * 100.0),
                "sector_rsi_participation_composite_score": float(scores.mean()),
            }
        )
    return pd.DataFrame(rows)


def _market_participation_from_stock_rsi(stock_rsi_df: pd.DataFrame) -> dict[str, Optional[float]]:
    if stock_rsi_df.empty:
        return {
            "market_rsi_breadth_pct_60": None,
            "market_rsi_breadth_pct_75": None,
            "market_rsi_breadth_pct_lt40": None,
            "market_rsi_participation_composite_score": None,
        }
    scores = pd.to_numeric(stock_rsi_df["stock_rsi_regime_score"], errors="coerce").dropna()
    if scores.empty:
        return {
            "market_rsi_breadth_pct_60": None,
            "market_rsi_breadth_pct_75": None,
            "market_rsi_breadth_pct_lt40": None,
            "market_rsi_participation_composite_score": None,
        }
    return {
        "market_rsi_breadth_pct_60": float((scores >= 60).mean() * 100.0),
        "market_rsi_breadth_pct_75": float((scores >= 75).mean() * 100.0),
        "market_rsi_breadth_pct_lt40": float((scores < 40).mean() * 100.0),
        "market_rsi_participation_composite_score": float(scores.mean()),
    }


def compute_sector_p_from_report(report_df: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    prepared = _prepare_report_df(report_df)
    stats = compute_sector_overview_stats(prepared)
    if stats.empty:
        return pd.DataFrame(columns=[
            "sector",
            "one_month_pct",
            "mk_breadth",
            "rel_perf_breadth",
            "rel_vol_breadth",
            "mc_var_score",
            "mc_var_score_discounted",
            "P",
        ])
    components = stats[[
        "sector",
        "sector_1m_var_pct_num",
        "market_breadth_num",
        "rs_breadth_num",
        "obvm_breadth_num",
    ]].copy()
    components = components.rename(
        columns={
            "sector_1m_var_pct_num": "one_month_pct",
            "market_breadth_num": "mk_breadth",
            "rs_breadth_num": "rel_perf_breadth",
            "obvm_breadth_num": "rel_vol_breadth",
        }
    )
    p_config = config.get("quadrants_participation", {})
    strong_pos = float(p_config.get("strong_positive_threshold", 5.0))
    weak_pos = float(p_config.get("weak_positive_threshold", 0.0))
    weak_neg = float(p_config.get("weak_negative_threshold", -5.0))
    mult_strong_pos = float(p_config.get("multiplier_strong_positive", 1.0))
    mult_weak_pos = float(p_config.get("multiplier_weak_positive", 0.9))
    mult_weak_neg = float(p_config.get("multiplier_weak_negative", 0.8))
    mult_strong_neg = float(p_config.get("multiplier_strong_negative", 0.7))
    components["mc_var_score"] = components["one_month_pct"].rank(pct=True) * 100.0

    def apply_discount(value: object, score: object) -> Optional[float]:
        numeric_value = _to_float(value)
        numeric_score = _to_float(score)
        if numeric_value is None or numeric_score is None:
            return None
        if numeric_value > strong_pos:
            multiplier = mult_strong_pos
        elif numeric_value > weak_pos:
            multiplier = mult_weak_pos
        elif numeric_value > weak_neg:
            multiplier = mult_weak_neg
        else:
            multiplier = mult_strong_neg
        return float(np.clip(numeric_score * multiplier, 0.0, 100.0))

    components["mc_var_score_discounted"] = components.apply(
        lambda row: apply_discount(row["one_month_pct"], row["mc_var_score"]),
        axis=1,
    )
    components["P"] = components[[
        "mc_var_score_discounted",
        "mk_breadth",
        "rel_perf_breadth",
        "rel_vol_breadth",
    ]].mean(axis=1)
    return components


def compute_sector_t_from_report(report_df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_report_df(report_df)
    if prepared.empty or "general_technical_score" not in prepared.columns:
        return pd.DataFrame(columns=["sector", "T"])
    return (
        prepared.groupby("sector", dropna=False)["general_technical_score"]
        .mean()
        .reset_index()
        .rename(columns={"general_technical_score": "T"})
    )


def _compute_sector_state(
    report_df: pd.DataFrame,
    stock_rsi_sector_df: Optional[pd.DataFrame],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    p_df = compute_sector_p_from_report(report_df, config)
    t_df = compute_sector_t_from_report(report_df)
    merged = t_df.merge(p_df, on="sector", how="outer")
    if stock_rsi_sector_df is not None and not stock_rsi_sector_df.empty:
        merged = merged.merge(stock_rsi_sector_df, on="sector", how="left")
    return merged


def _flatten_family_scores(sector_rows: pd.DataFrame, sector_families: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    family_lookup = _build_family_lookup(sector_families)
    working = sector_rows.copy()
    working["family"] = working["sector"].map(lambda sector: family_lookup.get(str(sector), "unassigned"))
    rows: list[dict[str, object]] = []
    for family, group in working.groupby("family", dropna=False):
        rows.append(
            {
                "family": family,
                "sector_rotation_score": _safe_mean(group["sector_rotation_score"].tolist()),
                "sector_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _leading_family_label(family_scores_df: pd.DataFrame, config: Mapping[str, Any]) -> Optional[str]:
    if family_scores_df.empty:
        return None
    working = family_scores_df.copy()
    working["sector_rotation_score"] = pd.to_numeric(working["sector_rotation_score"], errors="coerce")
    working = working.dropna(subset=["sector_rotation_score"]).sort_values("sector_rotation_score", ascending=False)
    if working.empty:
        return None
    tie_band = float(config.get("leader_tie_band", 3.0))
    top_score = float(working.iloc[0]["sector_rotation_score"])
    tied = working[working["sector_rotation_score"] >= top_score - tie_band]
    if len(tied) > 1:
        return "Mixed leadership: " + " / ".join(tied["family"].astype(str).tolist())
    return str(working.iloc[0]["family"])


def _lookup_preference_score(
    regime_label: Optional[str],
    family: str,
    sector: str,
    config: Mapping[str, Any],
) -> Optional[float]:
    if not regime_label:
        return None
    regime_table = config.get("regime_preference_scores", {}).get(regime_label, {})
    sector_overrides = regime_table.get("sector_overrides", {})
    if sector in sector_overrides:
        return _to_float(sector_overrides.get(sector))
    return _to_float(regime_table.get("family_defaults", {}).get(family))


def _lookup_market_alignment_score(
    regime_label: Optional[str],
    family: str,
    sector: str,
    stock_label: Optional[str],
    config: Mapping[str, Any],
) -> Optional[float]:
    if not regime_label or not stock_label:
        return None
    regime_table = config.get("market_alignment_scores", {}).get(regime_label, {})
    sector_override = regime_table.get("sector_overrides", {}).get(sector)
    if isinstance(sector_override, Mapping):
        return _to_float(sector_override.get(stock_label))
    return _to_float(regime_table.get("family_defaults", {}).get(family, {}).get(stock_label))


def _calculate_risk_appetite(
    evaluation_df: pd.DataFrame,
    month_df: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, object]:
    eval_df = _prepare_report_df(evaluation_df)
    month_prepared = _prepare_report_df(month_df)
    needed_eval = {"ticker", "fundamental_quality", "fundamental_risk", "general_technical_score", "eod_price_used"}
    needed_month = {"ticker", "eod_price_used"}
    if not needed_eval.issubset(eval_df.columns) or not needed_month.issubset(month_prepared.columns):
        return {
            "risk_appetite_score": None,
            "quality_defensive_count": 0,
            "speculative_count": 0,
            "ret_spread_1m": None,
            "tech_spread": None,
            "ret_component": None,
            "tech_component": None,
            "warning": "Missing required columns for risk appetite.",
        }
    merged = eval_df[list(needed_eval)].merge(
        month_prepared[["ticker", "eod_price_used"]].rename(columns={"eod_price_used": "month_anchor_close"}),
        on="ticker",
        how="left",
    )
    merged["return_1m"] = np.where(
        merged["month_anchor_close"].replace(0, np.nan).notna(),
        100.0 * (merged["eod_price_used"] / merged["month_anchor_close"] - 1.0),
        np.nan,
    )
    quality_mask = (merged["fundamental_quality"] >= 70) & (merged["fundamental_risk"] >= 60)
    speculative_mask = (
        ((merged["fundamental_quality"] <= 45) | (merged["fundamental_risk"] <= 40))
        & (merged["general_technical_score"] >= 55)
    )
    quality_df = merged.loc[quality_mask].copy()
    speculative_df = merged.loc[speculative_mask].copy()
    quality_return = _safe_mean(quality_df["return_1m"].tolist())
    speculative_return = _safe_mean(speculative_df["return_1m"].tolist())
    quality_tech = _safe_mean(quality_df["general_technical_score"].tolist())
    speculative_tech = _safe_mean(speculative_df["general_technical_score"].tolist())
    ret_spread = None if speculative_return is None or quality_return is None else speculative_return - quality_return
    tech_spread = None if speculative_tech is None or quality_tech is None else speculative_tech - quality_tech
    risk_config = config.get("risk_appetite", {})
    ret_component = None if ret_spread is None else float(
        np.clip(50.0 + 50.0 * (ret_spread / float(risk_config.get("return_spread_divisor", 8.0))), 0.0, 100.0)
    )
    tech_component = None if tech_spread is None else float(
        np.clip(50.0 + 50.0 * (tech_spread / float(risk_config.get("technical_spread_divisor", 15.0))), 0.0, 100.0)
    )
    score = _weighted_average_available(
        [
            (ret_component, float(risk_config.get("return_component_weight", 0.6))),
            (tech_component, float(risk_config.get("technical_component_weight", 0.4))),
        ]
    )
    min_warning = int(config.get("minimum_data", {}).get("risk_appetite_warning_min_cohort_size", 5))
    warning = None
    if len(quality_df) < min_warning or len(speculative_df) < min_warning:
        warning = (
            f"Cohort sizes are small for risk appetite: quality/defensive={len(quality_df)}, "
            f"speculative={len(speculative_df)}."
        )
    return {
        "risk_appetite_score": score,
        "quality_defensive_count": int(len(quality_df)),
        "speculative_count": int(len(speculative_df)),
        "ret_spread_1m": ret_spread,
        "tech_spread": tech_spread,
        "ret_component": ret_component,
        "tech_component": tech_component,
        "warning": warning,
    }


def _compute_persistence_score(
    evaluation_date: date,
    interval_days: Mapping[str, int],
    current_market_regime_score: Optional[float],
    cache_dir: Path,
) -> tuple[Optional[float], list[float]]:
    if current_market_regime_score is None or not cache_dir.exists():
        return None, []
    matching_scores: list[tuple[date, float]] = []
    for snapshot_file in cache_dir.glob("market_snapshot_*.json"):
        try:
            payload = json.loads(snapshot_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        metadata = payload.get("metadata", {})
        try:
            snapshot_eval_date = date.fromisoformat(str(metadata.get("evaluation_date", "")))
        except ValueError:
            continue
        if snapshot_eval_date >= evaluation_date:
            continue
        if int(metadata.get("rsi_window_days", -1)) != int(interval_days.get("rsi_window_days", -1)):
            continue
        if int(metadata.get("month_offset_days", -1)) != int(interval_days.get("month_offset_days", -1)):
            continue
        if int(metadata.get("week_offset_days", -1)) != int(interval_days.get("week_offset_days", -1)):
            continue
        snapshot_score = _to_float(payload.get("market_summary", {}).get("market_regime_score"))
        if snapshot_score is None:
            continue
        matching_scores.append((snapshot_eval_date, snapshot_score))
    if not matching_scores:
        return None, []
    matching_scores.sort(key=lambda item: item[0], reverse=True)
    latest_scores = [score for _, score in matching_scores[:4]]
    same_side_count = sum(
        1
        for score in latest_scores
        if (score >= 50 and current_market_regime_score >= 50) or (score < 50 and current_market_regime_score < 50)
    )
    return 100.0 * same_side_count / len(latest_scores), latest_scores


def compute_market_bundle(
    *,
    evaluation_df: pd.DataFrame,
    month_df: pd.DataFrame,
    week_df: pd.DataFrame,
    evaluation_date: date,
    rsi_start_date: date,
    month_anchor_date: date,
    week_anchor_date: date,
    evaluation_source_path: Path,
    month_source_path: Path,
    week_source_path: Path,
    config: Mapping[str, Any],
    sector_families: Mapping[str, Sequence[str]],
    daily_prices_df: Optional[pd.DataFrame] = None,
    weekly_prices_df: Optional[pd.DataFrame] = None,
    cache_dir: Path | None = None,
) -> dict[str, object]:
    prepared_eval = _prepare_report_df(evaluation_df)
    prepared_month = _prepare_report_df(month_df)
    prepared_week = _prepare_report_df(week_df)
    tickers = prepared_eval.get("ticker", pd.Series(dtype=str)).dropna().astype(str).tolist()

    daily_rsi_df, daily_sources = _load_price_rsi_history(
        "daily",
        rsi_start_date,
        evaluation_date,
        tickers,
        price_df_override=daily_prices_df,
    )
    weekly_rsi_df, weekly_sources = _load_price_rsi_history(
        "weekly",
        rsi_start_date,
        evaluation_date,
        tickers,
        price_df_override=weekly_prices_df,
    )
    stock_rsi_df = compute_stock_rsi_regime(
        prepared_eval,
        daily_rsi_df,
        weekly_rsi_df,
        evaluation_date,
        rsi_start_date,
        config,
    )
    sector_rsi_df = _sector_participation_from_stock_rsi(stock_rsi_df)
    market_participation = _market_participation_from_stock_rsi(stock_rsi_df)

    current_sector_state = _compute_sector_state(prepared_eval, sector_rsi_df, config).rename(
        columns={"P": "P_now", "T": "T_now"}
    )
    month_sector_state = _compute_sector_state(prepared_month, None, config).rename(
        columns={"P": "P_1m_ago", "T": "T_1m_ago"}
    )
    week_sector_state = _compute_sector_state(prepared_week, None, config).rename(
        columns={"P": "P_1w_ago", "T": "T_1w_ago"}
    )
    sector_rows = current_sector_state.merge(
        month_sector_state[["sector", "P_1m_ago", "T_1m_ago"]],
        on="sector",
        how="left",
    ).merge(
        week_sector_state[["sector", "P_1w_ago", "T_1w_ago"]],
        on="sector",
        how="left",
    )
    family_lookup = _build_family_lookup(sector_families)
    sector_rows["family"] = sector_rows["sector"].map(lambda sector: family_lookup.get(str(sector), "unassigned"))

    trend_config = config.get("trend_of_change", {})
    weekly_weight = float(trend_config.get("weekly_weight", 0.6))
    monthly_weight = float(trend_config.get("monthly_weight", 0.4))
    monthly_divisor = float(trend_config.get("monthly_divisor", 4.0))
    score_factor = float(trend_config.get("score_factor", 8.0))
    sector_rows["dT_1w"] = sector_rows["T_now"] - sector_rows["T_1w_ago"]
    sector_rows["dP_1w"] = sector_rows["P_now"] - sector_rows["P_1w_ago"]
    sector_rows["dT_1m"] = sector_rows["T_now"] - sector_rows["T_1m_ago"]
    sector_rows["dP_1m"] = sector_rows["P_now"] - sector_rows["P_1m_ago"]
    sector_rows["dT"] = weekly_weight * sector_rows["dT_1w"] + monthly_weight * (sector_rows["dT_1m"] / monthly_divisor)
    sector_rows["dP"] = weekly_weight * sector_rows["dP_1w"] + monthly_weight * (sector_rows["dP_1m"] / monthly_divisor)
    sector_rows["dT_score"] = (50.0 + score_factor * sector_rows["dT"]).clip(lower=0.0, upper=100.0)
    sector_rows["dP_score"] = (50.0 + score_factor * sector_rows["dP"]).clip(lower=0.0, upper=100.0)
    sector_rows["trend_of_change_score"] = sector_rows[["dT_score", "dP_score"]].mean(axis=1)

    rotation_weights = config.get("sector_rotation_weights", {})
    sector_rows["sector_rotation_score"] = sector_rows.apply(
        lambda row: _weighted_average_available(
            [
                (_to_float(row.get("P_now")), float(rotation_weights.get("participation_score", 0.35))),
                (_to_float(row.get("T_now")), float(rotation_weights.get("technical_score", 0.35))),
                (
                    _to_float(row.get("sector_rsi_participation_composite_score")),
                    float(rotation_weights.get("rsi_participation_score", 0.2)),
                ),
                (_to_float(row.get("trend_of_change_score")), float(rotation_weights.get("trend_of_change_score", 0.1))),
            ]
        ),
        axis=1,
    )
    family_scores_df = _flatten_family_scores(sector_rows, sector_families)
    market_sector_rotation_score = _safe_mean(family_scores_df["sector_rotation_score"].tolist())
    leading_family = _leading_family_label(family_scores_df, config)

    risk_appetite = _calculate_risk_appetite(prepared_eval, prepared_month, config)
    market_weights = config.get("market_regime_weights", {})
    market_regime_score = _weighted_average_available(
        [
            (
                _to_float(market_participation.get("market_rsi_participation_composite_score")),
                float(market_weights.get("market_rsi_participation_composite_score", 0.4)),
            ),
            (_to_float(risk_appetite.get("risk_appetite_score")), float(market_weights.get("risk_appetite_score", 0.3))),
            (_to_float(market_sector_rotation_score), float(market_weights.get("market_sector_rotation_score", 0.3))),
        ]
    )
    market_regime_label = _score_to_label(
        market_regime_score,
        config.get("score_bands", {}).get("market_regime", []),
    )

    component_agreement_score = None
    if market_regime_score is not None:
        component_values = [
            (
                _to_float(market_participation.get("market_rsi_participation_composite_score")),
                float(market_weights.get("market_rsi_participation_composite_score", 0.4)),
            ),
            (_to_float(risk_appetite.get("risk_appetite_score")), float(market_weights.get("risk_appetite_score", 0.3))),
            (_to_float(market_sector_rotation_score), float(market_weights.get("market_sector_rotation_score", 0.3))),
        ]
        agreement_weight = 0.0
        agreement_sum = 0.0
        for value, weight in component_values:
            if value is None:
                continue
            agreement_weight += weight
            same_side = (value >= 50 and market_regime_score >= 50) or (value < 50 and market_regime_score < 50)
            if same_side:
                agreement_sum += weight
        if agreement_weight > 0:
            component_agreement_score = 100.0 * agreement_sum / agreement_weight

    distance_from_neutral_score = None if market_regime_score is None else float(min(100.0, 2.0 * abs(market_regime_score - 50.0)))
    interval_days = {
        "rsi_window_days": (evaluation_date - rsi_start_date).days,
        "month_offset_days": (evaluation_date - month_anchor_date).days,
        "week_offset_days": (evaluation_date - week_anchor_date).days,
    }
    persistence_score, persistence_values = _compute_persistence_score(
        evaluation_date,
        interval_days,
        market_regime_score,
        _ensure_market_cache_dir(cache_dir),
    )
    confidence_weights = config.get("confidence_weights", {})
    active_confidence_weights = confidence_weights.get("full_history" if persistence_score is not None else "reduced_history", {})
    market_regime_confidence = _weighted_sum_available(
        [
            (component_agreement_score, float(active_confidence_weights.get("component_agreement_score", 0.45))),
            (distance_from_neutral_score, float(active_confidence_weights.get("distance_from_neutral_score", 0.35))),
            (persistence_score, float(active_confidence_weights.get("persistence_score", 0.2))),
        ]
    )
    if (
        market_regime_confidence is not None
        and component_agreement_score is not None
        and market_regime_confidence >= 70
        and component_agreement_score >= 70
    ):
        market_regime_status = "Confirmed"
    elif market_regime_confidence is not None and market_regime_confidence >= 50:
        market_regime_status = "Tentative"
    elif market_regime_confidence is not None:
        market_regime_status = "Mixed / Transitional"
    else:
        market_regime_status = None

    fit_weights = config.get("sector_regime_fit_weights", {})
    sector_rows["regime_preference_score"] = sector_rows.apply(
        lambda row: _lookup_preference_score(
            market_regime_label,
            str(row.get("family", "unassigned")),
            str(row.get("sector", "")),
            config,
        ),
        axis=1,
    )
    sector_rows["sector_regime_fit_score"] = sector_rows.apply(
        lambda row: _weighted_average_available(
            [
                (_to_float(row.get("T_now")), float(fit_weights.get("technical_score", 0.25))),
                (_to_float(row.get("P_now")), float(fit_weights.get("participation_score", 0.25))),
                (
                    _to_float(row.get("sector_rsi_participation_composite_score")),
                    float(fit_weights.get("sector_rsi_participation_composite_score", 0.2)),
                ),
                (_to_float(row.get("regime_preference_score")), float(fit_weights.get("regime_preference_score", 0.3))),
            ]
        ),
        axis=1,
    )
    fit_bands = config.get("score_bands", {}).get("sector_regime_fit_flag", [])
    sector_rows["sector_regime_fit_flag"] = sector_rows["sector_regime_fit_score"].map(
        lambda score: _score_to_label(_to_float(score), fit_bands)
    )

    stock_base = prepared_eval[[
        "ticker",
        "sector",
        "industry",
        "general_technical_score",
        "fundamental_total_score",
    ]].copy()
    stock_base["family"] = stock_base["sector"].map(lambda sector: family_lookup.get(str(sector), "unassigned"))
    sector_fit_df = sector_rows[["sector", "family", "sector_regime_fit_score", "sector_regime_fit_flag"]].copy()
    setup_df = stock_base.merge(
        stock_rsi_df[["ticker", "stock_rsi_regime_score", "stock_rsi_regime_label"]],
        on="ticker",
        how="left",
    ).merge(
        sector_fit_df,
        on=["sector", "family"],
        how="left",
    )
    setup_df["market_alignment_score"] = setup_df.apply(
        lambda row: _lookup_market_alignment_score(
            market_regime_label,
            str(row.get("family", "unassigned")),
            str(row.get("sector", "")),
            str(row.get("stock_rsi_regime_label")) if pd.notna(row.get("stock_rsi_regime_label")) else None,
            config,
        ),
        axis=1,
    )
    readiness_weights = config.get("setup_readiness_weights", {})
    setup_df["setup_readiness_score"] = setup_df.apply(
        lambda row: _weighted_average_available(
            [
                (_to_float(row.get("general_technical_score")), float(readiness_weights.get("general_technical_score", 0.25))),
                (_to_float(row.get("stock_rsi_regime_score")), float(readiness_weights.get("stock_rsi_regime_score", 0.2))),
                (_to_float(row.get("fundamental_total_score")), float(readiness_weights.get("fundamental_total_score", 0.2))),
                (_to_float(row.get("sector_regime_fit_score")), float(readiness_weights.get("sector_regime_fit_score", 0.2))),
                (_to_float(row.get("market_alignment_score")), float(readiness_weights.get("market_alignment_score", 0.15))),
            ]
        ),
        axis=1,
    )
    setup_df["evaluation_date"] = evaluation_date.isoformat()
    setup_df["rsi_start_date"] = rsi_start_date.isoformat()
    setup_df["month_anchor_date"] = month_anchor_date.isoformat()
    setup_df["week_anchor_date"] = week_anchor_date.isoformat()
    stock_rsi_df["month_anchor_date"] = month_anchor_date.isoformat()
    stock_rsi_df["week_anchor_date"] = week_anchor_date.isoformat()

    payload = {
        "metadata": {
            "version": str(config.get("version", "1.0")),
            "evaluation_date": evaluation_date.isoformat(),
            "rsi_start_date": rsi_start_date.isoformat(),
            "month_anchor_date": month_anchor_date.isoformat(),
            "week_anchor_date": week_anchor_date.isoformat(),
            "rsi_window_days": interval_days["rsi_window_days"],
            "month_offset_days": interval_days["month_offset_days"],
            "week_offset_days": interval_days["week_offset_days"],
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "source_files": {
                "evaluation_report_select": evaluation_source_path.name,
                "month_anchor_report_select": month_source_path.name,
                "week_anchor_report_select": week_source_path.name,
                "daily_price_caches": daily_sources,
                "weekly_price_caches": weekly_sources,
            },
        },
        "market_summary": {
            "market_regime_score": market_regime_score,
            "market_regime_label": market_regime_label,
            "market_regime_confidence": market_regime_confidence,
            "market_regime_status": market_regime_status,
            "market_sector_rotation_score": market_sector_rotation_score,
            "leading_family_classifier": leading_family,
        },
        "component_scores": {
            "market_rsi_participation_composite_score": market_participation.get("market_rsi_participation_composite_score"),
            "risk_appetite_score": risk_appetite.get("risk_appetite_score"),
            "market_sector_rotation_score": market_sector_rotation_score,
            "component_agreement_score": component_agreement_score,
            "distance_from_neutral_score": distance_from_neutral_score,
            "persistence_score": persistence_score,
            "persistence_values_used": persistence_values,
        },
        "breadth": {
            "market_rsi_breadth_pct_60": market_participation.get("market_rsi_breadth_pct_60"),
            "market_rsi_breadth_pct_75": market_participation.get("market_rsi_breadth_pct_75"),
            "market_rsi_breadth_pct_lt40": market_participation.get("market_rsi_breadth_pct_lt40"),
        },
        "risk_appetite": risk_appetite,
        "family_scores": [
            {
                "family": str(row.get("family")),
                "sector_rotation_score": _to_float(row.get("sector_rotation_score")),
                "sector_count": int(row.get("sector_count", 0)),
            }
            for row in family_scores_df.to_dict(orient="records")
        ],
        "sector_rows": [
            {
                key: value if key in {"sector", "family", "sector_regime_fit_flag"} else _to_float(value)
                for key, value in row.items()
            }
            for row in sector_rows.to_dict(orient="records")
        ],
        "cache_files": {},
    }
    return {
        "stock_rsi_regime_df": stock_rsi_df,
        "setup_readiness_df": setup_df,
        "market_snapshot_payload": payload,
    }


def _json_ready_df(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    for column in working.columns:
        if pd.api.types.is_datetime64_any_dtype(working[column]):
            working[column] = working[column].dt.strftime("%Y-%m-%d")
    return working


def save_market_bundle(
    *,
    signature: str,
    stock_rsi_regime_df: pd.DataFrame,
    setup_readiness_df: pd.DataFrame,
    market_snapshot_payload: Mapping[str, Any],
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    directory = _ensure_market_cache_dir(cache_dir)
    stock_file = stock_rsi_regime_path(signature, directory)
    setup_file = setup_readiness_path(signature, directory)
    snapshot_file = market_snapshot_path(signature, directory)
    _json_ready_df(stock_rsi_regime_df).to_json(stock_file, orient="records", lines=True, force_ascii=False)
    _json_ready_df(setup_readiness_df).to_json(setup_file, orient="records", lines=True, force_ascii=False)
    payload = json.loads(json.dumps(market_snapshot_payload, default=_to_iso))
    payload["cache_files"] = {
        "market_snapshot": snapshot_file.name,
        "stock_rsi_regime": stock_file.name,
        "setup_readiness": setup_file.name,
    }
    snapshot_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "market_snapshot": snapshot_file,
        "stock_rsi_regime": stock_file,
        "setup_readiness": setup_file,
    }


def load_market_bundle(signature: str, cache_dir: Path | None = None) -> dict[str, object]:
    directory = _ensure_market_cache_dir(cache_dir)
    snapshot_file = market_snapshot_path(signature, directory)
    stock_file = stock_rsi_regime_path(signature, directory)
    setup_file = setup_readiness_path(signature, directory)
    if not (snapshot_file.exists() and stock_file.exists() and setup_file.exists()):
        raise FileNotFoundError(f"Market cache files are incomplete for signature: {signature}")
    return {
        "market_snapshot_payload": json.loads(snapshot_file.read_text(encoding="utf-8")),
        "stock_rsi_regime_df": pd.read_json(stock_file, orient="records", lines=True),
        "setup_readiness_df": pd.read_json(setup_file, orient="records", lines=True),
        "paths": {
            "market_snapshot": snapshot_file,
            "stock_rsi_regime": stock_file,
            "setup_readiness": setup_file,
        },
    }


def compute_and_save_market_bundle(
    *,
    evaluation_df: pd.DataFrame,
    month_df: pd.DataFrame,
    week_df: pd.DataFrame,
    evaluation_date: date,
    rsi_start_date: date,
    month_anchor_date: date,
    week_anchor_date: date,
    evaluation_source_path: Path,
    month_source_path: Path,
    week_source_path: Path,
    config: Mapping[str, Any],
    sector_families: Mapping[str, Sequence[str]],
    force_recompute: bool = False,
    daily_prices_df: Optional[pd.DataFrame] = None,
    weekly_prices_df: Optional[pd.DataFrame] = None,
    cache_dir: Path | None = None,
) -> dict[str, object]:
    directory = _ensure_market_cache_dir(cache_dir)
    signature = build_market_signature(evaluation_date, rsi_start_date, month_anchor_date, week_anchor_date)
    status = market_cache_status(signature, directory)
    if not force_recompute and bool(status["ready"]):
        cached = load_market_bundle(signature, directory)
        cached["signature"] = signature
        cached["cached"] = True
        return cached
    computed = compute_market_bundle(
        evaluation_df=evaluation_df,
        month_df=month_df,
        week_df=week_df,
        evaluation_date=evaluation_date,
        rsi_start_date=rsi_start_date,
        month_anchor_date=month_anchor_date,
        week_anchor_date=week_anchor_date,
        evaluation_source_path=evaluation_source_path,
        month_source_path=month_source_path,
        week_source_path=week_source_path,
        config=config,
        sector_families=sector_families,
        daily_prices_df=daily_prices_df,
        weekly_prices_df=weekly_prices_df,
        cache_dir=directory,
    )
    paths = save_market_bundle(
        signature=signature,
        stock_rsi_regime_df=computed["stock_rsi_regime_df"],
        setup_readiness_df=computed["setup_readiness_df"],
        market_snapshot_payload=computed["market_snapshot_payload"],
        cache_dir=directory,
    )
    computed["paths"] = paths
    computed["signature"] = signature
    computed["cached"] = False
    return computed
