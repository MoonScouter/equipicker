from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from market_service import (
    DATA_DIR,
    MARKET_CACHE_DIR,
    build_market_cache_key,
    load_market_bundle,
    load_market_regime_config,
    load_stock_rsi_regime_overlay_cache,
    market_snapshot_path,
    stock_rsi_regime_overlay_path,
)
from prices_service import normalize_price_ticker, prices_cache_path


HIDDEN_MECHANICS_CACHE_DIR = DATA_DIR / "hidden_mechanics"
HIDDEN_MECHANICS_VERSION = "1.0"
ACTIONABLE_LABELS = {
    "Continuation Candidate",
    "Healthy Leadership",
    "Distribution Watch (early)",
    "Late Leadership / Distribution Risk",
}
WATCHLIST_LABELS = {"Recovery Watch", "Accumulation", "Contrarian Recovery"}
CONTEXT_ONLY_LABELS = {"Deteriorating", "Neutral / No Edge"}


def hidden_mechanics_snapshot_path(evaluation_date: date, cache_dir: Path | None = None) -> Path:
    directory = cache_dir or HIDDEN_MECHANICS_CACHE_DIR
    return directory / f"hidden_mechanics_eval_{evaluation_date.isoformat()}.json"


def _ensure_cache_dir(cache_dir: Path | None = None) -> Path:
    directory = cache_dir or HIDDEN_MECHANICS_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _to_float(value: object) -> Optional[float]:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    return float(numeric)


def _clip(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return float(np.clip(value, lower, upper))


def _safe_mean(values: Sequence[object], *, default: Optional[float] = None) -> Optional[float]:
    numeric = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").dropna()
    if numeric.empty:
        return default
    return float(numeric.mean())


def _json_float(value: object) -> object:
    numeric = _to_float(value)
    return numeric if numeric is not None and math.isfinite(numeric) else None


def _parse_date(value: object) -> Optional[date]:
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _percentile_map(values: Mapping[str, object]) -> dict[str, float]:
    clean = {key: _to_float(value) for key, value in values.items()}
    clean = {key: value for key, value in clean.items() if value is not None}
    if not clean:
        return {}
    ordered = sorted(clean.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: 50.0}
    return {key: 100.0 * index / (len(ordered) - 1) for index, (key, _) in enumerate(ordered)}


def _gated_score(headline_score: float, internal_score: float, *, gate_floor: float = 50.0, gate_span: float = 20.0) -> float:
    raw = math.sqrt(max(0.0, headline_score) * max(0.0, internal_score))
    headline_gate = np.clip((headline_score - gate_floor) / gate_span, 0.0, 1.0)
    internal_gate = np.clip((internal_score - gate_floor) / gate_span, 0.0, 1.0)
    return _clip(50.0 + (raw - 50.0) * math.sqrt(float(headline_gate) * float(internal_gate)))


def _load_market_snapshot_for_date(evaluation_date: date) -> dict[str, Any]:
    signature = build_market_cache_key(evaluation_date)
    return load_market_bundle(signature)["market_snapshot_payload"]  # type: ignore[index]


def list_market_snapshot_dates(cache_dir: Path | None = None) -> list[date]:
    directory = cache_dir or MARKET_CACHE_DIR
    dates: list[date] = []
    for path in directory.glob("market_snapshot_eval_*.json"):
        match = re.search(r"market_snapshot_eval_(\d{4}-\d{2}-\d{2})\.json$", path.name)
        if not match:
            continue
        parsed = _parse_date(match.group(1))
        if parsed is not None:
            dates.append(parsed)
    return sorted(set(dates))


def _resolve_market_snapshot_date(target_date: Optional[date]) -> Optional[date]:
    if target_date is None:
        return None
    available = [entry for entry in list_market_snapshot_dates() if entry <= target_date]
    return available[-1] if available else None


def _sector_rows_by_name(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = snapshot.get("sector_rows", [])
    if not isinstance(rows, list):
        return {}
    return {str(row.get("sector", "")).strip(): row for row in rows if str(row.get("sector", "")).strip()}


def _p_internal(row: Mapping[str, Any] | None) -> Optional[float]:
    if not row:
        return None
    return _safe_mean(
        [row.get("mk_breadth"), row.get("rel_perf_breadth"), row.get("rel_vol_breadth")],
        default=None,
    )


def _latest_rows_on_or_before(df: pd.DataFrame, evaluation_date: date) -> pd.DataFrame:
    if df is None or df.empty or "ticker" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()
    working = df.copy()
    working["ticker"] = working["ticker"].map(normalize_price_ticker)
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.date
    working = working[(working["ticker"].astype(str).str.len() > 0) & working["date"].notna()]
    working = working[working["date"] <= evaluation_date]
    if working.empty:
        return pd.DataFrame()
    return (
        working.sort_values(["ticker", "date"], kind="stable")
        .drop_duplicates(subset=["ticker"], keep="last")
        .reset_index(drop=True)
    )


@lru_cache(maxsize=8)
def _load_price_history_raw(frequency: str, year: int) -> pd.DataFrame:
    path = prices_cache_path(frequency, year)  # type: ignore[arg-type]
    if not path.exists():
        return pd.DataFrame()
    try:
        combined = pd.read_json(path, orient="records", lines=True)
    except ValueError:
        return pd.DataFrame()
    if combined.empty:
        return pd.DataFrame()
    useful_columns = [
        "ticker",
        "date",
        "rs",
        "obvm",
        "last_50_break_type",
        "last_50_break_date",
        "rsi_divergence_flag",
        "rsi_divergence_confirmed",
        "atr_pctile_200d",
    ]
    for column in useful_columns:
        if column not in combined.columns:
            combined[column] = pd.NA
    combined = combined.loc[:, useful_columns]
    if "ticker" in combined.columns and "date" in combined.columns:
        combined = combined.copy()
        combined["ticker"] = combined["ticker"].map(normalize_price_ticker)
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date
        combined["last_50_break_date"] = pd.to_datetime(combined["last_50_break_date"], errors="coerce").dt.date
        combined["last_50_break_type"] = (
            combined["last_50_break_type"].where(combined["last_50_break_type"].notna(), pd.NA).astype("string").str.lower()
        )
        combined["rsi_divergence_flag"] = (
            combined["rsi_divergence_flag"].where(combined["rsi_divergence_flag"].notna(), pd.NA).astype("string").str.lower()
        )
        combined["rsi_divergence_confirmed"] = combined["rsi_divergence_confirmed"].map(
            lambda value: bool(value) if str(value).strip().lower() in {"true", "1", "yes"} else False
        )
        for column in ["rs", "obvm", "atr_pctile_200d"]:
            combined[column] = pd.to_numeric(combined[column], errors="coerce")
        combined = combined.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"], kind="stable")
    return combined


def _load_latest_price_rows(frequency: str, evaluation_date: date) -> pd.DataFrame:
    years = (evaluation_date.year - 1, evaluation_date.year) if evaluation_date.month == 1 else (evaluation_date.year,)
    parts = [_load_price_history_raw(frequency, year) for year in years]
    parts = [part for part in parts if not part.empty]
    if not parts:
        return pd.DataFrame()
    combined = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    if "date" not in combined.columns or "ticker" not in combined.columns:
        return pd.DataFrame()
    scoped = combined[combined["date"] <= evaluation_date].copy()
    if scoped.empty:
        return pd.DataFrame()
    scoped = scoped.sort_values(["ticker", "date"], kind="stable")
    latest_parts: list[pd.Series] = []
    for _, group in scoped.groupby("ticker", sort=False):
        tail = group.tail(20)
        latest = tail.iloc[-1].copy()
        if "rs" in tail.columns:
            rs_values = pd.to_numeric(tail["rs"], errors="coerce").dropna()
            latest["rs_sma20"] = float(rs_values.mean()) if len(rs_values) >= 5 else np.nan
        if "obvm" in tail.columns:
            obvm_values = pd.to_numeric(tail["obvm"], errors="coerce").dropna()
            latest["obvm_sma20"] = float(obvm_values.mean()) if len(obvm_values) >= 5 else np.nan
        latest_parts.append(latest)
    if not latest_parts:
        return pd.DataFrame()
    return pd.DataFrame(latest_parts).reset_index(drop=True)


def _load_rsi_overlay_rows(evaluation_date: date, ticker_sector: Mapping[str, str]) -> tuple[pd.DataFrame, Optional[date]]:
    path = stock_rsi_regime_overlay_path(evaluation_date.year)
    overlay = load_stock_rsi_regime_overlay_cache(path)
    if overlay.empty and evaluation_date.month == 1:
        overlay = load_stock_rsi_regime_overlay_cache(stock_rsi_regime_overlay_path(evaluation_date.year - 1))
    latest = _latest_rows_on_or_before(overlay, evaluation_date)
    if latest.empty:
        return pd.DataFrame(), None
    latest["sector"] = latest["ticker"].map(ticker_sector)
    latest = latest.dropna(subset=["sector"])
    latest_date = _parse_date(latest["date"].max())
    return latest.reset_index(drop=True), latest_date


def _divergence_weight(flag: object, confirmed: object, frequency: str) -> float:
    text = str(flag or "").strip().lower()
    if not text or text == "none" or text == "<na>":
        return 0.0
    base = 2.0 if frequency == "weekly" else 1.0
    soft = 1.0 if frequency == "weekly" else 0.5
    is_confirmed = bool(confirmed) if not pd.isna(confirmed) else False
    if text == "positive":
        return base if is_confirmed else soft
    if text in {"potential-positive", "extension-positive"}:
        return soft
    if text == "negative":
        return -base if is_confirmed else -soft
    if text in {"potential-negative", "extension-negative"}:
        return -soft
    return 0.0


def _aggregate_rsi_by_sector(rsi_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    if rsi_df.empty:
        return {}
    rows: dict[str, dict[str, float]] = {}
    for sector, group in rsi_df.groupby("sector", sort=False):
        score20 = pd.to_numeric(group.get("stock_rsi_regime_20d_score"), errors="coerce")
        flags = group.get("stock_rsi_regime_20d_vs_50d_flag", pd.Series(index=group.index, dtype=object)).astype(str)
        valid_scores = score20.dropna()
        valid_flags = flags[flags.isin(["Positive", "Negative", "Neutral"])]
        positive_pct = float((valid_flags == "Positive").mean() * 100.0) if not valid_flags.empty else 0.0
        negative_pct = float((valid_flags == "Negative").mean() * 100.0) if not valid_flags.empty else 0.0
        score_ge_60 = float((valid_scores >= 60).mean() * 100.0) if not valid_scores.empty else 50.0
        score_lt_40 = float((valid_scores < 40).mean() * 100.0) if not valid_scores.empty else 50.0
        avg_score20 = float(valid_scores.mean()) if not valid_scores.empty else 50.0
        rsi_cross_net = positive_pct - negative_pct
        rsi_cross_score = _clip(50.0 + 1.5 * rsi_cross_net)
        rsi_breadth_score = _safe_mean([score_ge_60, 100.0 - score_lt_40, avg_score20], default=50.0) or 50.0
        rows[str(sector)] = {
            "rsi_company_count": float(len(group)),
            "rsi_valid_count": float(len(valid_scores)),
            "rsi_cross_positive_pct": positive_pct,
            "rsi_cross_negative_pct": negative_pct,
            "rsi_cross_net_breadth": rsi_cross_net,
            "rsi_cross_score": rsi_cross_score,
            "rsi_20d_avg_score": avg_score20,
            "rsi_20d_pct_ge_60": score_ge_60,
            "rsi_20d_pct_lt_40": score_lt_40,
            "rsi_breadth_score": rsi_breadth_score,
            "internal_momentum_score": 0.55 * rsi_cross_score + 0.45 * rsi_breadth_score,
        }
    return rows


def _aggregate_confirmation_by_sector(
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    ticker_sector: Mapping[str, str],
    evaluation_date: date,
) -> dict[str, dict[str, float]]:
    daily = daily_df.copy() if daily_df is not None else pd.DataFrame()
    weekly = weekly_df.copy() if weekly_df is not None else pd.DataFrame()
    for frame in (daily, weekly):
        if not frame.empty:
            frame["ticker"] = frame["ticker"].map(normalize_price_ticker)
            frame["sector"] = frame["ticker"].map(ticker_sector)
    if daily.empty:
        return {}

    weekly_weights: dict[str, float] = {}
    if not weekly.empty:
        weekly_weights = {
            str(row.ticker): _divergence_weight(row.rsi_divergence_flag, row.rsi_divergence_confirmed, "weekly")
            for row in weekly.itertuples(index=False)
            if getattr(row, "ticker", None)
        }

    rows: dict[str, dict[str, float]] = {}
    for sector, group in daily.dropna(subset=["sector"]).groupby("sector", sort=False):
        rs = pd.to_numeric(group.get("rs"), errors="coerce")
        obvm = pd.to_numeric(group.get("obvm"), errors="coerce")
        rs_sma = pd.to_numeric(group.get("rs_sma20", pd.Series(np.nan, index=group.index)), errors="coerce")
        obvm_sma = pd.to_numeric(group.get("obvm_sma20", pd.Series(np.nan, index=group.index)), errors="coerce")
        rs_mask = rs.notna() & rs_sma.notna()
        obvm_mask = obvm.notna() & obvm_sma.notna()
        rs_breadth = float((rs[rs_mask] > rs_sma[rs_mask]).mean() * 100.0) if rs_mask.any() else 50.0
        obvm_breadth = float((obvm[obvm_mask] > obvm_sma[obvm_mask]).mean() * 100.0) if obvm_mask.any() else 50.0

        break_dates = pd.to_datetime(group.get("last_50_break_date"), errors="coerce").dt.date
        break_types = group.get("last_50_break_type", pd.Series(index=group.index, dtype=object)).astype(str).str.lower()
        recent_mask = break_dates.map(lambda value: pd.notna(value) and 0 <= (evaluation_date - value).days <= 31)
        recent_count = int(recent_mask.sum())
        if recent_count:
            positive_break = float(((break_types == "up") & recent_mask).sum() / recent_count * 100.0)
            negative_break = float(((break_types == "down") & recent_mask).sum() / recent_count * 100.0)
        else:
            positive_break = 0.0
            negative_break = 0.0
        structure_breadth = positive_break - negative_break
        structure_score = _clip(50.0 + 1.5 * structure_breadth)

        weights = []
        for row in group.itertuples(index=False):
            ticker = str(getattr(row, "ticker", ""))
            daily_weight = _divergence_weight(
                getattr(row, "rsi_divergence_flag", None),
                getattr(row, "rsi_divergence_confirmed", None),
                "daily",
            )
            weights.append(daily_weight + weekly_weights.get(ticker, 0.0))
        avg_weight = float(np.nanmean(weights)) if weights else 0.0
        weighted_net_divergence_breadth = float(np.clip(100.0 * avg_weight / 3.0, -100.0, 100.0))
        divergence_tilt = _clip(50.0 + 2.0 * weighted_net_divergence_breadth)
        confirmation_score = (
            0.35 * rs_breadth
            + 0.25 * obvm_breadth
            + 0.20 * structure_score
            + 0.20 * divergence_tilt
        )
        atr = pd.to_numeric(group.get("atr_pctile_200d"), errors="coerce").dropna()
        rows[str(sector)] = {
            "confirmation_company_count": float(len(group)),
            "rs_valid_count": float(rs_mask.sum()),
            "obvm_valid_count": float(obvm_mask.sum()),
            "rs_inflection_breadth": rs_breadth,
            "obvm_confirmation_breadth": obvm_breadth,
            "positive_50d_break_breadth": positive_break,
            "negative_50d_break_breadth": negative_break,
            "structure_breadth": structure_breadth,
            "structure_score": structure_score,
            "weighted_net_divergence_breadth": weighted_net_divergence_breadth,
            "divergence_tilt": divergence_tilt,
            "confirmation_score": confirmation_score,
            "atr_pctile_200d_median": float(atr.median()) if not atr.empty else 50.0,
        }
    return rows


def _load_previous_hidden_snapshot(evaluation_date: date, cache_dir: Path | None = None) -> Optional[dict[str, Any]]:
    directory = cache_dir or HIDDEN_MECHANICS_CACHE_DIR
    if not directory.exists():
        return None
    prior_paths: list[tuple[date, Path]] = []
    for path in directory.glob("hidden_mechanics_eval_*.json"):
        match = re.search(r"hidden_mechanics_eval_(\d{4}-\d{2}-\d{2})\.json$", path.name)
        if not match:
            continue
        parsed = _parse_date(match.group(1))
        if parsed is not None and parsed < evaluation_date:
            prior_paths.append((parsed, path))
    if not prior_paths:
        return None
    _, path = sorted(prior_paths, key=lambda item: item[0])[-1]
    return json.loads(path.read_text(encoding="utf-8"))


def _confidence_band(confidence: float) -> str:
    if confidence >= 70:
        return "High"
    if confidence >= 50:
        return "Medium"
    return "Low"


def _label_score_band(score: float) -> str:
    if score >= 70:
        return "strong"
    if score >= 55:
        return "constructive"
    if score >= 45:
        return "neutral"
    if score >= 30:
        return "fragile"
    return "weak"


def _decide_label(row: Mapping[str, float], rep_prev: Optional[float]) -> tuple[str, str, str, bool, bool]:
    hlw = row["headline_weakness_score"]
    hls = row["headline_strength_score"]
    infl = row["inflection_score"]
    imom = row["internal_momentum_score"]
    conf = row["confirmation_score"]
    rep = row["internal_repair_score"]
    hr = row["hidden_recovery_score"]
    hk = row["hidden_risk_score"]
    fit = row["sector_regime_fit_score"]
    internals_strong = infl >= 55 and imom >= 50 and conf >= 55
    internals_weak = infl < 45 or imom < 40 or rep < 45
    conflicted = (imom >= 65 and infl < 40) or (imom <= 35 and infl > 70)
    recovery_persists = rep_prev is not None and rep > rep_prev
    risk_persists = rep_prev is not None and rep < rep_prev

    if hr >= 58 and recovery_persists:
        if fit < 50:
            return "Contrarian Recovery", "hidden recovery (persisted)", "low" if hr < 62 else "normal", recovery_persists, risk_persists
        if conf >= 58 and imom >= 50:
            return "Accumulation", "hidden recovery (persisted)", "low" if hr < 62 else "normal", recovery_persists, risk_persists
        return "Recovery Watch", "hidden recovery (persisted)", "low" if hr < 62 else "normal", recovery_persists, risk_persists
    if hk >= 58 and risk_persists:
        return "Late Leadership / Distribution Risk", "hidden risk (persisted)", "low" if hk < 62 else "normal", recovery_persists, risk_persists
    if hls >= 60 and imom <= 35:
        return "Distribution Watch (early)", "single-component RSI rollover", "low", recovery_persists, risk_persists
    if hlw >= 60 and imom >= 65 and infl >= 50:
        return "Recovery Watch", "single-component RSI repair", "low", recovery_persists, risk_persists
    if conflicted:
        return "Neutral / No Edge", "conflict guard", "low", recovery_persists, risk_persists
    if hls >= 70 and internals_strong:
        return "Continuation Candidate", "agreement", "normal", recovery_persists, risk_persists
    if hls >= 60 and not internals_weak:
        return "Healthy Leadership", "agreement", "normal", recovery_persists, risk_persists
    if hlw >= 60 and internals_weak:
        return "Deteriorating", "agreement", "normal", recovery_persists, risk_persists
    return "Neutral / No Edge", "no edge", "normal", recovery_persists, risk_persists


def _component_agreement(label: str, row: Mapping[str, float]) -> float:
    if label in {"Continuation Candidate", "Healthy Leadership"}:
        checks = [
            row["headline_strength_score"] >= 60,
            row["current_participation_score"] >= 50,
            row["inflection_score"] >= 50,
            row["internal_momentum_score"] >= 50,
            row["confirmation_score"] >= 50,
        ]
    elif label in WATCHLIST_LABELS:
        checks = [
            row["headline_weakness_score"] >= 55,
            row["inflection_score"] >= 50,
            row["internal_momentum_score"] >= 50,
            row["confirmation_score"] >= 45,
        ]
    elif label in {"Late Leadership / Distribution Risk", "Distribution Watch (early)"}:
        checks = [
            row["headline_strength_score"] >= 55,
            row["inflection_score"] <= 50,
            row["internal_momentum_score"] <= 45,
            row["confirmation_score"] <= 55,
        ]
    elif label == "Deteriorating":
        checks = [
            row["headline_weakness_score"] >= 55,
            row["inflection_score"] < 45,
            row["internal_momentum_score"] < 50,
            row["confirmation_score"] < 50,
        ]
    else:
        return 50.0
    return 100.0 * sum(bool(item) for item in checks) / len(checks)


def _calculate_confidence(
    label: str,
    trigger: str,
    haircut: str,
    row: Mapping[str, float],
    rep_prev: Optional[float],
    market_regime_status: Optional[str],
) -> tuple[float, str, dict[str, float]]:
    component_agreement = _component_agreement(label, row)
    distance = min(
        100.0,
        2.0
        * max(
            abs(row["hidden_recovery_score"] - 50.0),
            abs(row["hidden_risk_score"] - 50.0),
            abs(row["internal_repair_score"] - 50.0),
            abs(row["internal_deterioration_score"] - 50.0),
        ),
    )
    coverage = _safe_mean(
        [
            min(100.0, row.get("rsi_valid_count", 0.0) / max(row.get("rsi_company_count", 1.0), 1.0) * 100.0),
            min(100.0, row.get("rs_valid_count", 0.0) / max(row.get("confirmation_company_count", 1.0), 1.0) * 100.0),
            min(100.0, row.get("obvm_valid_count", 0.0) / max(row.get("confirmation_company_count", 1.0), 1.0) * 100.0),
        ],
        default=50.0,
    ) or 50.0
    if row.get("company_count", 0.0) < 6:
        coverage = min(coverage, 20.0)
    elif row.get("company_count", 0.0) < 10:
        coverage = min(coverage, 45.0)
    persistence = 75.0 if rep_prev is not None else 20.0
    if trigger in {"hidden recovery (persisted)", "hidden risk (persisted)"}:
        persistence = 100.0

    confidence = (
        0.40 * component_agreement
        + 0.25 * distance
        + 0.20 * coverage
        + 0.15 * persistence
    )
    if market_regime_status == "Mixed / Transitional":
        confidence *= 0.85
    if row.get("atr_pctile_200d_median", 50.0) >= 80 and label in {"Recovery Watch", "Accumulation", "Contrarian Recovery"}:
        confidence -= 8.0
    if row.get("rsi_20d_dispersion", 0.0) >= 25:
        confidence -= 5.0
    if haircut == "low" or trigger in {"single-component RSI rollover", "single-component RSI repair", "conflict guard"}:
        confidence = min(confidence, 45.0)
    if trigger == "hidden recovery (persisted)" and 58 <= row["hidden_recovery_score"] < 62:
        confidence = min(confidence, 55.0)
    if trigger == "hidden risk (persisted)" and 58 <= row["hidden_risk_score"] < 62:
        confidence = min(confidence, 55.0)
    if label in {"Recovery Watch", "Accumulation"}:
        confidence = min(confidence, 50.0)
    confidence = _clip(confidence)
    return confidence, _confidence_band(confidence), {
        "component_agreement": component_agreement,
        "distance_from_neutral": distance,
        "data_coverage": coverage,
        "persistence": persistence,
    }


def _build_signal_explanation(row: Mapping[str, Any]) -> str:
    sector = str(row.get("sector", "The sector"))
    label = str(row.get("forward_label", "Neutral / No Edge"))
    headline = "strong" if row["headline_strength_score"] >= 60 else "weak" if row["headline_weakness_score"] >= 60 else "mixed"
    inflection = _label_score_band(float(row["inflection_score"]))
    momentum = _label_score_band(float(row["internal_momentum_score"]))
    confirmation = _label_score_band(float(row["confirmation_score"]))
    confidence = str(row.get("confidence_band", "Low")).lower()
    if label == "Continuation Candidate":
        return (
            f"{sector} is a continuation candidate: the monthly headline is {headline}, "
            f"inflection is {inflection}, RSI regime momentum is {momentum}, and confirmation is {confirmation}. "
            f"Confidence is {confidence}."
        )
    if label == "Healthy Leadership":
        return (
            f"{sector} shows healthy leadership because the recent headline is supported by participation and internals. "
            f"Inflection is {inflection}, RSI momentum is {momentum}, and confirmation is {confirmation}."
        )
    if label == "Distribution Watch (early)":
        return (
            f"{sector} is an early distribution watch: the headline remains strong, but the RSI cross layer is rolling over. "
            f"Treat this as a low-confidence warning until broader confirmation weakens."
        )
    if label == "Late Leadership / Distribution Risk":
        return (
            f"{sector} still looks strong retrospectively, but persisted deterioration in internal repair raises hidden-risk risk. "
            f"Inflection is {inflection} and confirmation is {confirmation}."
        )
    if label in WATCHLIST_LABELS:
        return (
            f"{sector} is monitoring-only: the latest headline is {headline}, while internal repair is visible but not yet validated as actionable. "
            f"Inflection is {inflection}, RSI momentum is {momentum}, and confirmation is {confirmation}."
        )
    if label == "Deteriorating":
        return (
            f"{sector} remains a context-only deteriorating sector: weak headline evidence is not being offset by repair. "
            f"Use this to describe weakness, not as a standalone prediction."
        )
    return (
        f"{sector} has no clear forward edge: the headline and internal mechanics do not align strongly enough for a directional call."
    )


def compute_hidden_mechanics_snapshot(
    evaluation_date: date,
    *,
    save: bool = True,
    force_recompute: bool = False,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    output_path = hidden_mechanics_snapshot_path(evaluation_date, cache_dir)
    if output_path.exists() and not force_recompute:
        return json.loads(output_path.read_text(encoding="utf-8"))

    market_snapshot = _load_market_snapshot_for_date(evaluation_date)
    sector_rows = _sector_rows_by_name(market_snapshot)
    metadata = market_snapshot.get("metadata", {})
    week_date = _resolve_market_snapshot_date(_parse_date(metadata.get("week_anchor_date")))
    month_date = _resolve_market_snapshot_date(_parse_date(metadata.get("month_anchor_date")))
    week_rows = _sector_rows_by_name(_load_market_snapshot_for_date(week_date)) if week_date else {}
    month_rows = _sector_rows_by_name(_load_market_snapshot_for_date(month_date)) if month_date else {}

    bundle = load_market_bundle(build_market_cache_key(evaluation_date))
    stock_rsi_df = bundle.get("stock_rsi_regime_df")
    ticker_sector = {}
    if isinstance(stock_rsi_df, pd.DataFrame) and not stock_rsi_df.empty:
        stock_rsi_df = stock_rsi_df.copy()
        stock_rsi_df["ticker"] = stock_rsi_df["ticker"].map(normalize_price_ticker)
        ticker_sector = {
            str(row.ticker): str(row.sector)
            for row in stock_rsi_df[["ticker", "sector"]].dropna().drop_duplicates().itertuples(index=False)
        }

    rsi_rows, rsi_overlay_date = _load_rsi_overlay_rows(evaluation_date, ticker_sector)
    rsi_by_sector = _aggregate_rsi_by_sector(rsi_rows)
    daily_prices = _load_latest_price_rows("daily", evaluation_date)
    weekly_prices = _load_latest_price_rows("weekly", evaluation_date)
    confirmation_by_sector = _aggregate_confirmation_by_sector(daily_prices, weekly_prices, ticker_sector, evaluation_date)

    previous = _load_previous_hidden_snapshot(evaluation_date, cache_dir)
    previous_by_sector = {
        str(row.get("sector")): row
        for row in (previous or {}).get("sector_rows", [])
        if str(row.get("sector", "")).strip()
    }

    returns = {sector: row.get("one_month_pct") for sector, row in sector_rows.items()}
    strength_pct = _percentile_map(returns)
    hidden_rows: list[dict[str, Any]] = []
    config = load_market_regime_config()
    trend_config = config.get("trend_of_change", {})
    weekly_weight = float(trend_config.get("weekly_weight", 0.6))
    monthly_weight = float(trend_config.get("monthly_weight", 0.4))
    monthly_divisor = float(trend_config.get("monthly_divisor", 4.0))
    score_factor = float(trend_config.get("score_factor", 8.0))
    market_summary = market_snapshot.get("market_summary", {})
    market_regime_status = market_summary.get("market_regime_status")

    for sector, market_row in sector_rows.items():
        p_now = _p_internal(market_row)
        p_week = _p_internal(week_rows.get(sector))
        p_month = _p_internal(month_rows.get(sector))
        t_now = _to_float(market_row.get("T_now"))
        t_week = _to_float((week_rows.get(sector) or {}).get("T_now")) or _to_float(market_row.get("T_1w_ago"))
        t_month = _to_float((month_rows.get(sector) or {}).get("T_now")) or _to_float(market_row.get("T_1m_ago"))
        d_p_internal = None
        if p_now is not None and p_week is not None and p_month is not None:
            d_p_internal = weekly_weight * (p_now - p_week) + monthly_weight * ((p_now - p_month) / monthly_divisor)
        d_t = None
        if t_now is not None and t_week is not None and t_month is not None:
            d_t = weekly_weight * (t_now - t_week) + monthly_weight * ((t_now - t_month) / monthly_divisor)
        inflection_score = _clip(50.0 + score_factor * (_safe_mean([d_p_internal, d_t], default=0.0) or 0.0))

        rsi = rsi_by_sector.get(sector, {})
        conf = confirmation_by_sector.get(sector, {})
        current_participation_score = p_now if p_now is not None else 50.0
        technical_quality_score = t_now if t_now is not None else 50.0
        internal_momentum_score = float(rsi.get("internal_momentum_score", 50.0))
        confirmation_score = float(conf.get("confirmation_score", 50.0))
        headline_strength_score = strength_pct.get(sector, 50.0)
        headline_weakness_score = 100.0 - headline_strength_score
        internal_repair_score = (
            0.35 * inflection_score
            + 0.30 * internal_momentum_score
            + 0.25 * confirmation_score
            + 0.10 * technical_quality_score
        )
        internal_deterioration_score = (
            0.35 * (100.0 - inflection_score)
            + 0.30 * (100.0 - internal_momentum_score)
            + 0.25 * (100.0 - confirmation_score)
            + 0.10 * (100.0 - technical_quality_score)
        )
        hidden_recovery_score = _gated_score(headline_weakness_score, internal_repair_score)
        hidden_risk_score = _gated_score(headline_strength_score, internal_deterioration_score)
        base_row: dict[str, float] = {
            "headline_strength_score": headline_strength_score,
            "headline_weakness_score": headline_weakness_score,
            "current_participation_score": current_participation_score,
            "technical_quality_score": technical_quality_score,
            "inflection_score": inflection_score,
            "internal_momentum_score": internal_momentum_score,
            "confirmation_score": confirmation_score,
            "internal_repair_score": internal_repair_score,
            "internal_deterioration_score": internal_deterioration_score,
            "hidden_recovery_score": hidden_recovery_score,
            "hidden_risk_score": hidden_risk_score,
            "sector_regime_fit_score": _to_float(market_row.get("sector_regime_fit_score")) or 50.0,
            "company_count": max(float(rsi.get("rsi_company_count", 0.0)), float(conf.get("confirmation_company_count", 0.0))),
            **{key: float(value) for key, value in rsi.items()},
            **{key: float(value) for key, value in conf.items()},
        }
        rep_prev = _to_float(previous_by_sector.get(sector, {}).get("internal_repair_score"))
        label, trigger, haircut, recovery_persists, risk_persists = _decide_label(base_row, rep_prev)
        confidence, confidence_band, confidence_inputs = _calculate_confidence(
            label,
            trigger,
            haircut,
            base_row,
            rep_prev,
            str(market_regime_status) if market_regime_status is not None else None,
        )
        row: dict[str, Any] = {
            "sector": sector,
            "family": market_row.get("family"),
            "sector_1m_market_cap_variation": _json_float(market_row.get("one_month_pct")),
            "sector_rotation_score": _json_float(market_row.get("sector_rotation_score")),
            "sector_regime_fit_score": _json_float(market_row.get("sector_regime_fit_score")),
            "sector_regime_fit_flag": market_row.get("sector_regime_fit_flag"),
            "market_breadth": _json_float(market_row.get("mk_breadth")),
            "relative_performance_breadth": _json_float(market_row.get("rel_perf_breadth")),
            "relative_volume_breadth": _json_float(market_row.get("rel_vol_breadth")),
            "P_internal_1w": p_week,
            "P_internal_1m": p_month,
            "dP_internal": d_p_internal,
            "dT": d_t,
            "rep_prev": rep_prev,
            "recovery_persists": recovery_persists,
            "risk_persists": risk_persists,
            "forward_label": label,
            "trigger": trigger,
            "confidence": confidence,
            "confidence_band": confidence_band,
            "confidence_haircut": haircut,
            "confidence_inputs": confidence_inputs,
            "actionability": "watchlist" if label in WATCHLIST_LABELS else "context" if label in CONTEXT_ONLY_LABELS else "actionable",
            **base_row,
        }
        row["signal_explanation"] = _build_signal_explanation(row)
        hidden_rows.append(row)

    hidden_rows = sorted(hidden_rows, key=lambda row: (row["actionability"] == "context", -row["confidence"], row["sector"]))
    payload = {
        "metadata": {
            "version": HIDDEN_MECHANICS_VERSION,
            "evaluation_date": evaluation_date.isoformat(),
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "horizon": "next 4 weeks / next monthly board",
            "methodology": "brainstorming/hidden-mechanics.md",
            "market_snapshot": market_snapshot_path(build_market_cache_key(evaluation_date)).name,
            "week_anchor_snapshot_date": week_date.isoformat() if week_date else None,
            "month_anchor_snapshot_date": month_date.isoformat() if month_date else None,
            "rsi_overlay_date": rsi_overlay_date.isoformat() if rsi_overlay_date else None,
            "prior_hidden_mechanics_date": (previous or {}).get("metadata", {}).get("evaluation_date") if previous else None,
        },
        "market_summary": market_summary,
        "sector_rows": hidden_rows,
    }
    if save:
        path = hidden_mechanics_snapshot_path(evaluation_date, cache_dir)
        _ensure_cache_dir(cache_dir)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def load_hidden_mechanics_snapshot(evaluation_date: date, cache_dir: Path | None = None) -> dict[str, Any]:
    path = hidden_mechanics_snapshot_path(evaluation_date, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Hidden Mechanics snapshot is missing for {evaluation_date.isoformat()}")
    return json.loads(path.read_text(encoding="utf-8"))


def list_hidden_mechanics_snapshot_dates(cache_dir: Path | None = None) -> list[date]:
    directory = cache_dir or HIDDEN_MECHANICS_CACHE_DIR
    if not directory.exists():
        return []
    dates: list[date] = []
    for path in directory.glob("hidden_mechanics_eval_*.json"):
        match = re.search(r"hidden_mechanics_eval_(\d{4}-\d{2}-\d{2})\.json$", path.name)
        if match:
            parsed = _parse_date(match.group(1))
            if parsed is not None:
                dates.append(parsed)
    return sorted(set(dates))


def compute_hidden_mechanics_history(*, force_recompute: bool = False, max_snapshots: int | None = None) -> dict[str, Any]:
    dates = list_market_snapshot_dates()
    if max_snapshots is not None and max_snapshots > 0:
        dates = dates[-max_snapshots:]
    computed: list[str] = []
    errors: list[str] = []
    for entry in dates:
        try:
            compute_hidden_mechanics_snapshot(entry, force_recompute=force_recompute)
            computed.append(entry.isoformat())
        except Exception as exc:  # pragma: no cover - UI diagnostic surface
            errors.append(f"{entry.isoformat()}: {exc}")
    return {"computed_dates": computed, "errors": errors}


def build_hidden_mechanics_interpretation(snapshot: Mapping[str, Any]) -> str:
    rows = list(snapshot.get("sector_rows", []))
    if not rows:
        return "No Hidden Mechanics rows are available for this date."
    actionable = [row for row in rows if row.get("actionability") == "actionable"]
    watchlist = [row for row in rows if row.get("actionability") == "watchlist"]
    risk_rows = [row for row in actionable if row.get("forward_label") in {"Distribution Watch (early)", "Late Leadership / Distribution Risk"}]
    leadership = [row for row in actionable if row.get("forward_label") in {"Continuation Candidate", "Healthy Leadership"}]
    parts: list[str] = []
    if leadership:
        names = ", ".join(str(row["sector"]) for row in leadership[:3])
        parts.append(f"Constructive agreement is strongest in {names}, where the headline and internal mechanics broadly line up.")
    if risk_rows:
        names = ", ".join(str(row["sector"]) for row in risk_rows[:3])
        parts.append(f"Hidden-risk attention is warranted in {names}; these labels point to fading internals under a stronger recent headline.")
    if watchlist:
        names = ", ".join(str(row["sector"]) for row in watchlist[:3])
        parts.append(f"Recovery-style labels are monitoring-only for now: {names}. They should be discussed as early repair, not first-page actionable calls.")
    if not parts:
        parts.append("The board is selective this month: most sectors route to Neutral / No Edge or context-only labels.")
    parts.append("The expected horizon is roughly four weeks, and validation should be read cross-sectionally rather than as an absolute-return forecast.")
    return " ".join(parts)


def compute_hidden_mechanics_validation(snapshot_dates: Sequence[date] | None = None) -> dict[str, Any]:
    dates = sorted(snapshot_dates or list_hidden_mechanics_snapshot_dates())
    if len(dates) < 2:
        return {
            "summary": {"eligible_predictions": 0, "hit_rate": None, "partial_rate": None},
            "rows": [],
            "label_stats": [],
            "message": "Need at least two Hidden Mechanics snapshots to validate a 4-week-forward outcome.",
        }
    snapshots = {entry: load_hidden_mechanics_snapshot(entry) for entry in dates}
    rows: list[dict[str, Any]] = []
    for entry in dates:
        target = entry + timedelta(days=28)
        future_candidates = [candidate for candidate in dates if candidate >= target]
        if not future_candidates:
            continue
        future_date = future_candidates[0]
        if (future_date - entry).days > 45:
            continue
        current = {str(row.get("sector")): row for row in snapshots[entry].get("sector_rows", [])}
        future = {str(row.get("sector")): row for row in snapshots[future_date].get("sector_rows", [])}
        future_returns = {
            sector: row.get("sector_1m_market_cap_variation")
            for sector, row in future.items()
            if _to_float(row.get("sector_1m_market_cap_variation")) is not None
        }
        future_rank = _percentile_map(future_returns)
        for sector, row in current.items():
            label = str(row.get("forward_label"))
            if label in CONTEXT_ONLY_LABELS:
                continue
            future_row = future.get(sector)
            if not future_row:
                continue
            rank = future_rank.get(sector)
            p_change = (_to_float(future_row.get("current_participation_score")) or 50.0) - (
                _to_float(row.get("current_participation_score")) or 50.0
            )
            imom_change = (_to_float(future_row.get("internal_momentum_score")) or 50.0) - (
                _to_float(row.get("internal_momentum_score")) or 50.0
            )
            conf_change = (_to_float(future_row.get("confirmation_score")) or 50.0) - (
                _to_float(row.get("confirmation_score")) or 50.0
            )
            hit = False
            partial = False
            if label in {"Continuation Candidate", "Healthy Leadership"}:
                hit = rank is not None and rank >= 50
                partial = hit or p_change >= 0
            elif label in WATCHLIST_LABELS:
                partial = (p_change + imom_change + conf_change) / 3.0 > 0
                hit = partial and rank is not None and rank >= 40
            elif label in {"Late Leadership / Distribution Risk", "Distribution Watch (early)"}:
                hit = rank is not None and rank <= 50
                partial = hit or p_change < 0 or imom_change < 0
            rows.append(
                {
                    "snapshot_date": entry.isoformat(),
                    "future_date": future_date.isoformat(),
                    "sector": sector,
                    "prior_label": label,
                    "prior_hidden_recovery_score": _json_float(row.get("hidden_recovery_score")),
                    "prior_hidden_risk_score": _json_float(row.get("hidden_risk_score")),
                    "future_return": _json_float(future_row.get("sector_1m_market_cap_variation")),
                    "future_rank": rank,
                    "p_internal_change": p_change,
                    "internal_momentum_change": imom_change,
                    "confirmation_change": conf_change,
                    "result": "hit" if hit else "partial" if partial else "miss",
                }
            )
    eligible = len(rows)
    hits = sum(1 for row in rows if row["result"] == "hit")
    partials = sum(1 for row in rows if row["result"] in {"hit", "partial"})
    label_stats: list[dict[str, Any]] = []
    if rows:
        frame = pd.DataFrame(rows)
        for label, group in frame.groupby("prior_label", sort=True):
            label_stats.append(
                {
                    "label": label,
                    "count": int(len(group)),
                    "hit_rate": float((group["result"] == "hit").mean() * 100.0),
                    "partial_rate": float(group["result"].isin(["hit", "partial"]).mean() * 100.0),
                    "avg_future_rank": float(pd.to_numeric(group["future_rank"], errors="coerce").mean()),
                }
            )
    return {
        "summary": {
            "eligible_predictions": eligible,
            "hit_rate": float(hits / eligible * 100.0) if eligible else None,
            "partial_rate": float(partials / eligible * 100.0) if eligible else None,
        },
        "rows": rows,
        "label_stats": label_stats,
        "message": None if rows else "No snapshot pairs are at least 28 days apart yet.",
    }
