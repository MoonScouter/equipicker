from __future__ import annotations

from datetime import date
from pathlib import Path
from time import sleep
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd
import requests

from equipicker_connect import CACHE_DIR
from prices_service import normalize_price_ticker, normalize_price_tickers, parse_manual_price_tickers, resolve_all_price_tickers
from split_service import EODHD_API_TOKEN_ENV_ALIASES, require_config_value

ANALYST_RATINGS_DIR = CACHE_DIR / "analyst-ratings"
ANALYST_RATINGS_COLUMNS: tuple[str, ...] = (
    "Ticker",
    "Name",
    "Sector",
    "Industry",
    "Rating",
    "TargetPrice",
    "StrongBuy",
    "Buy",
    "Hold",
    "Sell",
    "StrongSell",
    "TotalRatings",
    "%StrongBuy",
    "%Buy",
    "%Hold",
    "%Sell",
    "%StrongSell",
)


def analyst_ratings_path(run_date: date | None = None) -> Path:
    resolved_date = run_date or date.today()
    return ANALYST_RATINGS_DIR / f"analyst_ratings_{resolved_date:%Y_%m_%d}.xlsx"


def list_analyst_ratings_files() -> list[Path]:
    return sorted(ANALYST_RATINGS_DIR.glob("analyst_ratings_*.xlsx"))


def latest_analyst_ratings_file() -> Path | None:
    files = list_analyst_ratings_files()
    return files[-1] if files else None


def parse_manual_analyst_tickers(raw_text: str) -> list[str]:
    return parse_manual_price_tickers(raw_text)


def resolve_analyst_rating_tickers(
    *,
    scope: str,
    manual_tickers: Sequence[object] | None = None,
    all_tickers_resolver: Callable[[], list[str]] = resolve_all_price_tickers,
) -> list[str]:
    if scope == "all":
        return all_tickers_resolver()
    if scope != "specific":
        raise ValueError(f"Unsupported analyst ratings import scope: {scope}")
    return normalize_price_tickers(manual_tickers or [])


def build_metadata_lookup(metadata_df: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if metadata_df is None or metadata_df.empty or "ticker" not in metadata_df.columns:
        return {}
    working = metadata_df.copy()
    working["ticker"] = working["ticker"].map(normalize_price_ticker)
    name_column = _first_existing_column(working, ("Name", "name", "company", "Company"))
    sector_column = _first_existing_column(working, ("Sector", "sector"))
    industry_column = _first_existing_column(working, ("Industry", "industry"))
    lookup: dict[str, dict[str, object]] = {}
    for _, row in working.iterrows():
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker or ticker in lookup:
            continue
        lookup[ticker] = {
            "Name": row.get(name_column, "") if name_column else "",
            "Sector": row.get(sector_column, "") if sector_column else "",
            "Industry": row.get(industry_column, "") if industry_column else "",
        }
    return lookup


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def fetch_analyst_ratings(
    session: requests.Session,
    *,
    eodhd_api_token: str,
    ticker: str,
    timeout: int = 15,
) -> Mapping[str, object] | None:
    response = session.get(
        f"https://eodhd.com/api/fundamentals/{ticker}",
        params={"api_token": eodhd_api_token, "fmt": "json"},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        return None
    ratings = payload.get("AnalystRatings")
    return ratings if isinstance(ratings, Mapping) else None


def analyst_rating_row(
    ticker: str,
    ratings: Mapping[str, object] | None,
    *,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata = metadata or {}
    strong_buy = _optional_number(ratings.get("StrongBuy") if ratings else None)
    buy = _optional_number(ratings.get("Buy") if ratings else None)
    hold = _optional_number(ratings.get("Hold") if ratings else None)
    sell = _optional_number(ratings.get("Sell") if ratings else None)
    strong_sell = _optional_number(ratings.get("StrongSell") if ratings else None)
    rating = ratings.get("Rating") if ratings else None
    target_price = _optional_number(ratings.get("TargetPrice") if ratings else None)
    total = sum(value for value in (strong_buy, buy, hold, sell, strong_sell) if value is not None)
    total_ratings = total if total > 0 else None

    pct_values: dict[str, float | None] = {
        "%StrongBuy": None,
        "%Buy": None,
        "%Hold": None,
        "%Sell": None,
        "%StrongSell": None,
    }
    if total >= 3:
        pct_values = {
            "%StrongBuy": _percent_of_total(strong_buy, total),
            "%Buy": _percent_of_total(buy, total),
            "%Hold": _percent_of_total(hold, total),
            "%Sell": _percent_of_total(sell, total),
            "%StrongSell": _percent_of_total(strong_sell, total),
        }

    row = {
        "Ticker": ticker,
        "Name": _clean_metadata_value(metadata.get("Name")),
        "Sector": _clean_metadata_value(metadata.get("Sector")),
        "Industry": _clean_metadata_value(metadata.get("Industry")),
        "Rating": rating,
        "TargetPrice": target_price,
        "StrongBuy": strong_buy,
        "Buy": buy,
        "Hold": hold,
        "Sell": sell,
        "StrongSell": strong_sell,
        "TotalRatings": total_ratings,
    }
    row.update(pct_values)
    return row


def _optional_number(value: object) -> float | int | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    numeric_float = float(numeric)
    return int(numeric_float) if numeric_float.is_integer() else numeric_float


def _percent_of_total(value: float | int | None, total: float | int) -> float | None:
    if value is None:
        return None
    return value / total * 100


def _clean_metadata_value(value: object) -> object:
    return "" if value is None or pd.isna(value) else value


def analyst_ratings_dataframe(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    for column in ANALYST_RATINGS_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df.loc[:, list(ANALYST_RATINGS_COLUMNS)]


def analyst_ratings_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Sector", "%StrongBuy", "%Buy", "%Hold", "%Sell", "%StrongSell"]).set_index("Sector")
    working = df.copy()
    working["TotalRatings"] = pd.to_numeric(working["TotalRatings"], errors="coerce")
    percentage_columns = ["%StrongBuy", "%Buy", "%Hold", "%Sell", "%StrongSell"]
    for column in percentage_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    eligible = working[working["TotalRatings"] >= 3]
    summary_rows: list[dict[str, object]] = []
    summary_rows.append({"Sector": "ALL", **{column: eligible[column].mean() for column in percentage_columns}})
    for sector in eligible["Sector"].dropna().unique():
        sector_text = str(sector).strip()
        if not sector_text:
            continue
        sector_df = eligible[eligible["Sector"] == sector]
        summary_rows.append({"Sector": sector_text, **{column: sector_df[column].mean() for column in percentage_columns}})
    return pd.DataFrame(summary_rows).set_index("Sector")


def save_analyst_ratings_workbook(df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_df = analyst_ratings_dataframe(df.to_dict("records") if df is not None else [])
    summary_df = analyst_ratings_summary(prepared_df)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        prepared_df.to_excel(writer, index=False, sheet_name="AnalystRatings")
        summary_df.to_excel(writer, sheet_name="Averages")
    return output_path


def load_analyst_ratings_file(path: Path) -> pd.DataFrame:
    loaded = pd.read_excel(path, sheet_name="AnalystRatings")
    return analyst_ratings_dataframe(loaded.to_dict("records"))


def import_analyst_ratings(
    *,
    scope: str,
    manual_tickers: Sequence[object] | None = None,
    run_date: date | None = None,
    metadata_df: pd.DataFrame | None = None,
    sleep_seconds: float = 1.0,
    session: requests.Session | None = None,
    eodhd_api_token: str | None = None,
    all_tickers_resolver: Callable[[], list[str]] = resolve_all_price_tickers,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    tickers = resolve_analyst_rating_tickers(
        scope=scope,
        manual_tickers=manual_tickers,
        all_tickers_resolver=all_tickers_resolver,
    )
    if not tickers:
        raise ValueError("No tickers were resolved for analyst ratings import.")

    token = eodhd_api_token or require_config_value(EODHD_API_TOKEN_ENV_ALIASES, label="EODHD API token")
    metadata_lookup = build_metadata_lookup(metadata_df)
    close_session = session is None
    active_session = session or requests.Session()
    rows: list[dict[str, object]] = []
    errors: list[str] = []
    try:
        for index, ticker in enumerate(tickers, start=1):
            if progress_callback:
                progress_callback(f"{ticker} ({index}/{len(tickers)})")
            try:
                ratings = fetch_analyst_ratings(
                    active_session,
                    eodhd_api_token=token,
                    ticker=ticker,
                )
            except Exception as exc:
                ratings = None
                errors.append(f"{ticker}: {exc}")
            rows.append(analyst_rating_row(ticker, ratings, metadata=metadata_lookup.get(ticker)))
            if sleep_seconds and index < len(tickers):
                sleep(sleep_seconds)
    finally:
        if close_session:
            active_session.close()

    output_path = analyst_ratings_path(run_date)
    result_df = analyst_ratings_dataframe(rows)
    save_analyst_ratings_workbook(result_df, output_path)
    return {
        "saved_path": output_path,
        "saved_rows": len(result_df),
        "requested_tickers_count": len(tickers),
        "errors": errors,
    }


def enrich_analyst_ratings_with_latest_prices(
    ratings_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    tickers: Sequence[object] | None = None,
) -> pd.DataFrame:
    ratings = analyst_ratings_dataframe(ratings_df.to_dict("records") if ratings_df is not None else [])
    ratings["Ticker"] = ratings["Ticker"].map(normalize_price_ticker)
    requested_tickers = set(normalize_price_tickers(tickers or []))
    if requested_tickers:
        ratings = ratings[ratings["Ticker"].isin(requested_tickers)].copy()

    latest_prices = _latest_prices_by_ticker(prices_df)
    enriched = ratings.merge(latest_prices, left_on="Ticker", right_on="ticker", how="left")
    enriched = enriched.drop(columns=["ticker"], errors="ignore")
    return enriched.rename(columns={"adjusted_close": "LastClose", "date": "CloseDate"})


def _latest_prices_by_ticker(prices_df: pd.DataFrame) -> pd.DataFrame:
    if prices_df is None or prices_df.empty:
        return pd.DataFrame(columns=["ticker", "date", "adjusted_close"])
    required_columns = {"ticker", "date", "adjusted_close"}
    if not required_columns.issubset(prices_df.columns):
        return pd.DataFrame(columns=["ticker", "date", "adjusted_close"])
    working = prices_df[["ticker", "date", "adjusted_close"]].copy()
    working["ticker"] = working["ticker"].map(normalize_price_ticker)
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.date
    working["adjusted_close"] = pd.to_numeric(working["adjusted_close"], errors="coerce")
    working = working.dropna(subset=["date"])
    if working.empty:
        return pd.DataFrame(columns=["ticker", "date", "adjusted_close"])
    latest = (
        working.sort_values(["ticker", "date"], kind="stable")
        .drop_duplicates(subset=["ticker"], keep="last")
        .reset_index(drop=True)
    )
    latest["date"] = latest["date"].map(lambda value: value.isoformat() if pd.notna(value) else None)
    return latest.loc[:, ["ticker", "date", "adjusted_close"]]
