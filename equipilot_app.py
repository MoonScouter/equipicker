"""Streamlit UI for managing Monthly Scoring Board generation and helper files."""
from __future__ import annotations

import io
import json
import logging
import base64
import calendar
import re
import time
from bisect import bisect_right
from html import escape as html_escape
from contextlib import redirect_stdout, redirect_stderr
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from streamlit.components.v1 import html

from equipicker_connect import (
    bucharest_today_str,
    report_cache_path,
    scoring_cache_path,
)
from equipicker_filters import (
    accel_down_weak,
    accel_up_weak,
    extreme_accel_down,
    extreme_accel_up,
)
from indices_service import (
    INDEX_TICKER_TO_NAME,
    INDEX_TICKERS,
    fetch_indices_ohlc_since,
    indices_cache_path,
    save_indices_cache,
)
from prices_service import (
    PRICE_CACHE_COLUMNS,
    PRICE_CACHE_REQUIRED_COLUMNS,
    import_prices_cache,
    list_prices_cache_paths,
    load_prices_cache,
    normalize_price_ticker,
    parse_manual_price_tickers,
    prices_cache_path,
)
from report_select_service import generate_report_select_cache
from report_config import DEFAULT_CONFIG_PATH, ReportConfig, load_report_config, save_report_config
from grid_layout_config import clear_grid_layout, load_grid_layout, save_grid_layout
from openai_responses_service import (
    API_TEMPLATES_DIR,
    OPENAI_API_KEY_ENV,
    PROMPT_STORE_DIR,
    RESPONSE_OUTPUT_DIR,
    build_responses_payload,
    clone_default_prompt_payload,
    clone_default_template_payload,
    ensure_api_storage_dirs,
    list_saved_names,
    load_json_document,
    load_output_text,
    parse_string_list,
    run_responses_request,
    save_json_document,
    save_output_text,
)
from market_service import (
    build_market_cache_key,
    compute_and_save_market_bundle,
    get_default_market_anchors,
    load_market_bundle,
    load_market_methodology_text,
    load_market_regime_config,
    load_sector_families,
    market_cache_status,
    resolve_anchor_on_or_before,
)
from weekly_scoring_board import generate_weekly_scoring_board_pdf
from weekly_scoring_board import compute_sector_overview_stats

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPT_PATH = BASE_DIR / "summary_prompt.txt"
REPORTS_DIR = BASE_DIR / "reports"
QUADRANTS_DIR = REPORTS_DIR / "quadrants"
CONFIG_PATH = DEFAULT_CONFIG_PATH
CONFIG_DIR = CONFIG_PATH.parent
THEMATICS_CONFIG_PATH = CONFIG_DIR / "thematics.json"
GRID_LAYOUT_CONFIG_PATH = CONFIG_DIR / "grid_layouts.json"
SUMMARY_JSON_PATH = CONFIG_DIR / "text_generated.json"
BANNER_CANDIDATES = [
    BASE_DIR / "banner_equipilot.png",
    BASE_DIR / "banner_.png",
    BASE_DIR / "banner_1.png",
    BASE_DIR / "banner_2.png",
    BASE_DIR / "banner.png",
]
BANNER_SIDE_CANDIDATES = [
    BASE_DIR / "banner_equipilot_2.png",
]
APP_VERSION = "1.0.0"
_FAVICON_PATH = BASE_DIR / "blue_wings.png"
_favicon = Image.open(_FAVICON_PATH) if _FAVICON_PATH.exists() else None

CAP_BUCKET_ORDER = ["Nano", "Micro", "Small", "Mid", "Large", "Mega", "Unknown"]
COMPANY_GRID_MAX_HEIGHT = 820
COMPANY_GRID_FAST_RENDER_THRESHOLD = 800
QUADRANT_BORDER_D_T_THRESHOLD = 5.0
TREND_SYMBOL_UP = "📈"
TREND_SYMBOL_DOWN = "📉"
TREND_FILTER_LABELS = {
    "up": f"Up ({TREND_SYMBOL_UP})",
    "flat": "No trend",
    "down": f"Down ({TREND_SYMBOL_DOWN})",
}
TREND_FILTER_OPTIONS = ["All", TREND_FILTER_LABELS["up"], TREND_FILTER_LABELS["flat"], TREND_FILTER_LABELS["down"]]
SIGN_FILTER_OPTIONS = ["All", "Positive", "Negative"]
DIVERGENCE_FILTER_OPTIONS = ["All", "Positive", "Negative", "None"]
SHORT_TERM_FLOW_FILTER_OPTIONS = ["All", "Positive", "Negative", "Neutral"]
AI_REVENUE_FILTER_OPTIONS = ["All", "direct", "indirect", "none"]
AI_DISRUPTION_FILTER_OPTIONS = ["All", "high", "medium", "low", "none"]
MARKET_TREND_THRESHOLD = 3.0

GRID_SURFACE_FUNDAMENTAL_COMPANY = "fundamental_company_grid"
GRID_SURFACE_TECHNICAL_COMPANY = "technical_company_grid"
GRID_SURFACE_THEMATICS_BASKET = "thematics_basket_table"
GRID_SURFACE_THEMATICS_COMPANY = "thematics_company_grid"


def _normalize_grid_visible_columns(
    visible_columns: Optional[Sequence[str]],
    available_columns: Sequence[str],
    *,
    locked_columns: Optional[Sequence[str]] = None,
    default_columns: Optional[Sequence[str]] = None,
) -> list[str]:
    available = [
        column_name
        for column_name in dict.fromkeys(str(column).strip() for column in available_columns if str(column).strip())
    ]
    if not available:
        return []

    locked = [
        column_name
        for column_name in dict.fromkeys(str(column).strip() for column in (locked_columns or []) if str(column).strip())
        if column_name in available
    ]
    locked_set = set(locked)

    if default_columns is None:
        default_list = list(available)
    else:
        default_list = [
            column_name
            for column_name in dict.fromkeys(
                str(column).strip() for column in default_columns if str(column).strip()
            )
            if column_name in available
        ]
        if not default_list:
            default_list = list(available)
    default_visible = locked + [column_name for column_name in default_list if column_name not in locked_set]

    if visible_columns is None:
        return default_visible

    normalized_requested = [
        column_name
        for column_name in dict.fromkeys(
            str(column).strip() for column in visible_columns if isinstance(column, str) and str(column).strip()
        )
        if column_name in available
    ]
    if not normalized_requested:
        return default_visible
    normalized_visible = locked + [
        column_name for column_name in normalized_requested if column_name not in locked_set
    ]
    minimum_visible = len(locked) if locked else min(1, len(available))
    if len(normalized_visible) < minimum_visible:
        return default_visible
    return normalized_visible


def _grid_layout_state_key(surface_id: str) -> str:
    return f"{surface_id}_grid_layout_visible_columns"


def _get_grid_visible_columns(
    surface_id: str,
    available_columns: Sequence[str],
    *,
    locked_columns: Optional[Sequence[str]] = None,
) -> list[str]:
    available = list(available_columns)
    state_key = _grid_layout_state_key(surface_id)
    state_value = st.session_state.get(state_key)
    default_columns = list(available)
    if isinstance(state_value, list):
        visible_columns = _normalize_grid_visible_columns(
            state_value,
            available,
            locked_columns=locked_columns,
            default_columns=default_columns,
        )
    else:
        saved_layout = load_grid_layout(surface_id, GRID_LAYOUT_CONFIG_PATH)
        visible_columns = _normalize_grid_visible_columns(
            saved_layout,
            available,
            locked_columns=locked_columns,
            default_columns=default_columns,
        )
    st.session_state[state_key] = list(visible_columns)
    return list(visible_columns)


def _set_grid_visible_columns(surface_id: str, visible_columns: Sequence[str]) -> None:
    st.session_state[_grid_layout_state_key(surface_id)] = list(visible_columns)


def _render_grid_column_customizer(
    surface_id: str,
    available_columns: Sequence[str],
    *,
    locked_columns: Optional[Sequence[str]] = None,
) -> list[str]:
    available = list(available_columns)
    if not available:
        return []

    locked = [column_name for column_name in (locked_columns or []) if column_name in available]
    visible_columns = _get_grid_visible_columns(surface_id, available, locked_columns=locked)
    hidden_columns = [column_name for column_name in available if column_name not in visible_columns]
    minimum_visible = len(locked) if locked else 1

    visible_select_key = f"{surface_id}_grid_layout_visible_select"
    hidden_select_key = f"{surface_id}_grid_layout_hidden_select"
    removable_columns = [column_name for column_name in visible_columns if column_name not in set(locked)]

    if removable_columns:
        if st.session_state.get(visible_select_key) not in removable_columns:
            st.session_state[visible_select_key] = removable_columns[0]
    else:
        st.session_state.pop(visible_select_key, None)

    if hidden_columns:
        if st.session_state.get(hidden_select_key) not in hidden_columns:
            st.session_state[hidden_select_key] = hidden_columns[0]
    else:
        st.session_state.pop(hidden_select_key, None)

    with st.expander(f"Columns ({len(visible_columns)}/{len(available)})", expanded=False):
        if locked:
            st.caption(f"Locked columns: {', '.join(locked)}")
        else:
            st.caption("Customize visible columns and save the layout for this grid.")

        manage_cols = st.columns([1.45, 0.75, 0.75, 0.95])
        selected_visible = st.session_state.get(visible_select_key)
        with manage_cols[0]:
            st.selectbox(
                "Visible columns",
                options=removable_columns or ["No removable columns"],
                key=visible_select_key,
                disabled=not removable_columns,
            )
        selected_index = visible_columns.index(selected_visible) if selected_visible in visible_columns else -1
        first_movable_index = len(locked)
        with manage_cols[1]:
            move_up_clicked = st.button(
                "Up",
                key=f"{surface_id}_grid_layout_move_up",
                use_container_width=True,
                disabled=selected_index < 0 or selected_index <= first_movable_index,
            )
        with manage_cols[2]:
            move_down_clicked = st.button(
                "Down",
                key=f"{surface_id}_grid_layout_move_down",
                use_container_width=True,
                disabled=selected_index < 0 or selected_index >= len(visible_columns) - 1,
            )
        with manage_cols[3]:
            remove_clicked = st.button(
                "Remove",
                key=f"{surface_id}_grid_layout_remove",
                use_container_width=True,
                disabled=selected_index < 0 or len(visible_columns) <= minimum_visible,
            )

        add_cols = st.columns([1.45, 0.95])
        with add_cols[0]:
            st.selectbox(
                "Hidden columns",
                options=hidden_columns or ["No hidden columns"],
                key=hidden_select_key,
                disabled=not hidden_columns,
            )
        with add_cols[1]:
            add_clicked = st.button(
                "Add",
                key=f"{surface_id}_grid_layout_add",
                use_container_width=True,
                disabled=not hidden_columns,
            )

        action_cols = st.columns([1, 1, 2.2])
        with action_cols[0]:
            save_clicked = st.button(
                "Save columns",
                key=f"{surface_id}_grid_layout_save",
                use_container_width=True,
            )
        with action_cols[1]:
            reset_clicked = st.button(
                "Reset saved layout",
                key=f"{surface_id}_grid_layout_reset",
                use_container_width=True,
            )
        with action_cols[2]:
            st.caption("Saved layouts are app-wide for this grid surface.")

        if move_up_clicked and selected_index > first_movable_index:
            updated = list(visible_columns)
            updated[selected_index - 1], updated[selected_index] = updated[selected_index], updated[selected_index - 1]
            _set_grid_visible_columns(surface_id, updated)
            st.rerun()

        if move_down_clicked and 0 <= selected_index < len(visible_columns) - 1:
            updated = list(visible_columns)
            updated[selected_index + 1], updated[selected_index] = updated[selected_index], updated[selected_index + 1]
            _set_grid_visible_columns(surface_id, updated)
            st.rerun()

        if remove_clicked and 0 <= selected_index and len(visible_columns) > minimum_visible:
            updated = [column_name for column_name in visible_columns if column_name != selected_visible]
            updated = _normalize_grid_visible_columns(updated, available, locked_columns=locked, default_columns=available)
            _set_grid_visible_columns(surface_id, updated)
            st.rerun()

        if add_clicked:
            selected_hidden = st.session_state.get(hidden_select_key)
            if isinstance(selected_hidden, str) and selected_hidden in hidden_columns:
                updated = list(visible_columns) + [selected_hidden]
                updated = _normalize_grid_visible_columns(updated, available, locked_columns=locked, default_columns=available)
                _set_grid_visible_columns(surface_id, updated)
                st.rerun()

        if save_clicked:
            save_grid_layout(surface_id, visible_columns, GRID_LAYOUT_CONFIG_PATH)
            st.success(f"Saved columns to {GRID_LAYOUT_CONFIG_PATH.name}.")

        if reset_clicked:
            clear_grid_layout(surface_id, GRID_LAYOUT_CONFIG_PATH)
            default_columns = _normalize_grid_visible_columns(
                None,
                available,
                locked_columns=locked,
                default_columns=available,
            )
            _set_grid_visible_columns(surface_id, default_columns)
            st.rerun()

    return list(st.session_state.get(_grid_layout_state_key(surface_id), visible_columns))


def _apply_grid_column_layout(
    display_df: pd.DataFrame,
    surface_id: str,
    *,
    locked_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    visible_columns = _render_grid_column_customizer(
        surface_id,
        display_df.columns.tolist(),
        locked_columns=locked_columns,
    )
    return display_df.loc[:, visible_columns].copy()


def _trend_symbol_for_delta(delta_value: Optional[float], threshold: float) -> str:
    if delta_value is None or pd.isna(delta_value):
        return ""
    if delta_value > threshold:
        return TREND_SYMBOL_UP
    if delta_value < -threshold:
        return TREND_SYMBOL_DOWN
    return ""


def _trend_direction_for_delta(delta_value: Optional[float], threshold: float) -> str:
    if delta_value is None or pd.isna(delta_value):
        return "none"
    if delta_value > threshold:
        return "up"
    if delta_value < -threshold:
        return "down"
    return "flat"

def _format_ts(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return "(not found)"
    ts = datetime.fromtimestamp(path.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M")


def get_latest_sector_tables_file() -> Optional[Path]:
    candidates = sorted(DATA_DIR.glob("sector_data_tables_*.txt"))
    return candidates[-1] if candidates else None


def read_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _parse_report_select_date(path: Path) -> Optional[date]:
    stem = path.stem
    if not stem.startswith("report_select_"):
        return None
    date_str = stem.replace("report_select_", "", 1)
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None


def list_report_select_dates(data_dir: Path | None = None) -> list[date]:
    data_dir = data_dir or DATA_DIR
    dates: list[date] = []
    for pattern in ("report_select_*.xlsx", "report_select_*.csv"):
        for path in data_dir.glob(pattern):
            parsed = _parse_report_select_date(path)
            if parsed:
                dates.append(parsed)
    return sorted(set(dates))


def _report_select_directory_signature(data_dir: Path | None = None) -> int:
    data_dir = data_dir or DATA_DIR
    try:
        return data_dir.stat().st_mtime_ns
    except OSError:
        return 0


def _path_cache_signature(path_value: str | Path | None) -> str:
    if path_value is None:
        return ""
    path = Path(path_value)
    try:
        stats = path.stat()
    except OSError:
        return f"{path}|missing"
    return f"{path.resolve()}|{stats.st_size}|{stats.st_mtime_ns}"


def _paths_cache_signature(paths: list[Path]) -> str:
    return "|".join(_path_cache_signature(path) for path in sorted(paths, key=lambda entry: str(entry)))


@st.cache_data(show_spinner=False)
def _cached_available_report_select_dates(directory_signature: int) -> tuple[date, ...]:
    _ = directory_signature
    return tuple(list_report_select_dates())


def get_available_report_select_dates() -> tuple[date, ...]:
    return _cached_available_report_select_dates(_report_select_directory_signature())


def _ordinal_day_number(day_number: int) -> str:
    if 10 <= day_number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_number % 10, "th")
    return f"{day_number}{suffix}"


def _report_select_calendar_label_fragments(entry: date) -> tuple[str, ...]:
    weekday_name = calendar.day_name[entry.weekday()]
    month_name = calendar.month_name[entry.month]
    ordinal_day = _ordinal_day_number(entry.day)
    plain_day = str(entry.day)
    return (
        f"{weekday_name}, {month_name} {ordinal_day} {entry.year}",
        f"{weekday_name}, {month_name} {ordinal_day}, {entry.year}",
        f"{weekday_name}, {month_name} {plain_day} {entry.year}",
        f"{weekday_name}, {month_name} {plain_day}, {entry.year}",
    )


def _render_report_select_calendar_highlight_layer() -> None:
    available_dates = get_available_report_select_dates()
    if not available_dates:
        return

    rules: list[str] = ["<style>"]
    for entry in available_dates:
        selectors = [
            (
                '[role="gridcell"][aria-roledescription="button"]'
                f'[aria-label*={json.dumps(fragment)}]'
            )
            for fragment in _report_select_calendar_label_fragments(entry)
        ]
        selector_list = ",\n".join(selectors)
        rules.extend(
            [
                f"{selector_list}::after {{",
                "  border-color: #1D74F5 !important;",
                "  border-width: 2px !important;",
                "  border-radius: 999px !important;",
                "  box-shadow: 0 0 0 1px rgba(29, 116, 245, 0.2) !important;",
                "}",
                f"{selector_list} > div {{",
                "  color: #1D4ED8 !important;",
                "  font-weight: 600 !important;",
                "}",
            ]
        )
    rules.extend(
        [
            '[role="gridcell"][aria-roledescription="button"][aria-selected="true"]::after {',
            "  border-width: 2px !important;",
            "}",
            "</style>",
        ]
    )
    st.markdown("\n".join(rules), unsafe_allow_html=True)


def render_report_select_date_input(*args, **kwargs):
    _render_report_select_calendar_highlight_layer()
    return st.date_input(*args, **kwargs)


def resolve_report_select_path(date_value: date) -> Tuple[Optional[Path], Tuple[Path, Path]]:
    base = f"report_select_{date_value.isoformat()}"
    xlsx_path = DATA_DIR / f"{base}.xlsx"
    csv_path = DATA_DIR / f"{base}.csv"
    if xlsx_path.exists():
        return xlsx_path, (xlsx_path, csv_path)
    if csv_path.exists():
        return csv_path, (xlsx_path, csv_path)
    return None, (xlsx_path, csv_path)


def normalize_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for original, target in {
        "Sector": "sector",
        "General_Technical_Score": "general_technical_score",
        "General Technical Score": "general_technical_score",
    }.items():
        if original in df.columns and target not in df.columns:
            rename_map[original] = target
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def ensure_required_columns(df: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            f"Missing columns in {path}: {', '.join(missing)}"
        )


@st.cache_data(show_spinner=False)
def load_report_select(path_str: str, cache_signature: str = "") -> pd.DataFrame:
    _ = cache_signature
    path = Path(path_str)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


@st.cache_data(show_spinner=False)
def load_indices_cache_file(path_str: str) -> pd.DataFrame:
    return pd.read_excel(path_str)


@st.cache_data(show_spinner=False)
def load_prices_cache_file(path_str: str, cache_signature: str = "") -> pd.DataFrame:
    _ = cache_signature
    return load_prices_cache(Path(path_str))


@st.cache_data(show_spinner=False)
def load_thematics_config(path_str: str, cache_signature: str = "") -> dict:
    _ = cache_signature
    path = Path(path_str)
    return json.loads(path.read_text(encoding="utf-8"))


def _base_ticker_symbol(ticker: object) -> str:
    raw = str(ticker or "").strip().upper()
    if not raw:
        return ""
    if "." in raw:
        raw = raw.split(".", 1)[0]
    return raw


@st.cache_data(show_spinner=False)
def build_ai_exposure_lookup(path_str: str, cache_signature: str = "") -> dict[str, dict[str, str]]:
    payload = load_thematics_config(path_str, cache_signature)
    tickers = payload.get("ai_exposure", {}).get("tickers", {})
    lookup: dict[str, dict[str, str]] = {}
    if not isinstance(tickers, dict):
        return lookup
    for raw_ticker, exposure_data in tickers.items():
        normalized = normalize_price_ticker(raw_ticker)
        if not normalized:
            continue
        exposure_payload = exposure_data if isinstance(exposure_data, dict) else {}
        lookup[normalized] = {
            "ai_revenue_exposure": str(exposure_payload.get("ai_revenue_exposure", "none") or "none").strip().lower(),
            "ai_disruption_risk": str(exposure_payload.get("ai_disruption_risk", "none") or "none").strip().lower(),
        }
    return lookup


def _thematics_sort_key(item: dict) -> tuple[float, str]:
    tier_value = pd.to_numeric(item.get("tier"), errors="coerce")
    layer_value = pd.to_numeric(item.get("value_chain_layer"), errors="coerce")
    tier_num = float(tier_value) if pd.notna(tier_value) else 999.0
    layer_num = float(layer_value) if pd.notna(layer_value) else 999.0
    return tier_num, layer_num, str(item.get("name", "")).lower()


@st.cache_data(show_spinner=False)
def build_thematics_catalog(path_str: str, cache_signature: str = "") -> dict[str, object]:
    payload = load_thematics_config(path_str, cache_signature)
    raw_thematics = payload.get("thematics", {})
    items: dict[str, dict[str, object]] = {}
    if not isinstance(raw_thematics, dict):
        return {"items": items, "roots": []}

    for basket_name, raw_value in raw_thematics.items():
        basket_payload = raw_value if isinstance(raw_value, dict) else {}
        normalized_tickers = [
            normalize_price_ticker(ticker_value)
            for ticker_value in basket_payload.get("tickers", [])
            if normalize_price_ticker(ticker_value)
        ]
        items[str(basket_name)] = {
            "name": str(basket_name),
            "description": str(basket_payload.get("description", "") or ""),
            "article_narrative": str(basket_payload.get("article_narrative", "") or ""),
            "tier": basket_payload.get("tier"),
            "tier_label": "AI" if pd.to_numeric(basket_payload.get("tier"), errors="coerce") == 1 else f"Tier {basket_payload.get('tier')}",
            "value_chain_layer": basket_payload.get("value_chain_layer"),
            "parent": str(basket_payload.get("parent", "") or ""),
            "is_parent": bool(basket_payload.get("is_parent", False)),
            "sub_baskets": [str(entry) for entry in basket_payload.get("sub_baskets", [])],
            "tickers": normalized_tickers,
            "ticker_count": len(normalized_tickers),
            "is_ai_super_parent": False,
            "is_ai_group_child": False,
        }

    ai_child_names = [
        item_name
        for item_name, item in items.items()
        if not str(item.get("parent", "") or "").strip()
        and pd.to_numeric(item.get("tier"), errors="coerce") == 1
        and item_name != "AI"
    ]
    if ai_child_names:
        ai_unique_tickers = sorted(
            {
                ticker_value
                for child_name in ai_child_names
                for ticker_value in items[child_name].get("tickers", [])
            }
        )
        items["AI"] = {
            "name": "AI",
            "description": "Umbrella basket across all tier-1 AI groups, counting repeated companies only once.",
            "article_narrative": "High-level aggregate across AI infrastructure, silicon, software, applications, services, security, robotics, and adjacent AI basket groups.",
            "tier": 0,
            "tier_label": "AI umbrella",
            "value_chain_layer": 0,
            "parent": "",
            "is_parent": True,
            "sub_baskets": sorted(ai_child_names, key=lambda child_name: _thematics_sort_key(items[child_name])),
            "tickers": ai_unique_tickers,
            "ticker_count": len(ai_unique_tickers),
            "is_ai_super_parent": True,
            "is_ai_group_child": False,
        }
        for child_name in ai_child_names:
            items[child_name]["parent"] = "AI"
            items[child_name]["is_ai_group_child"] = True

    for item in items.values():
        child_names = [child_name for child_name in item["sub_baskets"] if child_name in items]
        if item["is_parent"] and child_names:
            child_names = sorted(child_names, key=lambda child_name: _thematics_sort_key(items[child_name]))
        item["children"] = child_names

    roots = [
        item["name"]
        for item in items.values()
        if not str(item.get("parent", "") or "").strip()
    ]
    roots = sorted(roots, key=lambda root_name: _thematics_sort_key(items[root_name]))
    return {"items": items, "roots": roots}


@st.cache_data(show_spinner=False)
def build_price_history_lookup(path_str: str, cache_signature: str = "") -> dict[str, dict[str, list[object]]]:
    loaded_prices = load_prices_cache_file(path_str, cache_signature)
    if loaded_prices.empty:
        return {}
    normalized_prices = loaded_prices[["ticker", "date", "adjusted_close"]].copy()
    normalized_prices["ticker"] = normalized_prices["ticker"].fillna("").astype(str).str.strip().str.upper()
    normalized_prices["date"] = pd.to_datetime(normalized_prices["date"], errors="coerce").dt.date
    normalized_prices["adjusted_close"] = pd.to_numeric(normalized_prices["adjusted_close"], errors="coerce")
    normalized_prices = normalized_prices.dropna(subset=["date"]).drop_duplicates(
        subset=["ticker", "date"],
        keep="last",
    )
    lookup: dict[str, dict[str, list[object]]] = {}
    if normalized_prices.empty:
        return lookup
    grouped = normalized_prices.groupby("ticker", sort=False)
    for ticker_value, group in grouped:
        sorted_group = group.sort_values("date", kind="stable")
        lookup[str(ticker_value)] = {
            "dates": sorted_group["date"].tolist(),
            "closes": pd.to_numeric(sorted_group["adjusted_close"], errors="coerce").tolist(),
        }
    return lookup


def invalidate_prices_cache_views() -> None:
    load_prices_cache_file.clear()
    build_price_history_lookup.clear()
    _load_company_divergence_metrics_for_date.clear()


def _queue_show_all_company_reset(prefix: str) -> None:
    st.session_state[f"{prefix}_pending_show_all_companies_reset"] = True


def _apply_pending_show_all_company_reset(prefix: str) -> None:
    pending_key = f"{prefix}_pending_show_all_companies_reset"
    if st.session_state.pop(pending_key, False):
        st.session_state[f"{prefix}_show_all_companies"] = False


def _bump_selection_nonce(state_key: str) -> None:
    st.session_state[state_key] = int(st.session_state.get(state_key, 0) or 0) + 1


def _price_close_for_target(
    price_entry: Optional[dict[str, list[object]]],
    target_date: date,
    *,
    exact: bool = False,
    allow_after_fallback: bool = False,
) -> Optional[float]:
    if not price_entry:
        return None
    dates = price_entry.get("dates", [])
    closes = price_entry.get("closes", [])
    if not dates or not closes:
        return None
    if exact:
        idx = bisect_right(dates, target_date) - 1
        if idx < 0 or idx >= len(dates) or dates[idx] != target_date:
            return None
        close_value = pd.to_numeric(closes[idx], errors="coerce")
        return None if pd.isna(close_value) else float(close_value)
    idx = bisect_right(dates, target_date) - 1
    if idx < 0 or idx >= len(dates):
        if not allow_after_fallback:
            return None
        after_idx = bisect_right(dates, target_date - timedelta(days=1))
        if after_idx < 0 or after_idx >= len(dates):
            return None
        close_value = pd.to_numeric(closes[after_idx], errors="coerce")
        return None if pd.isna(close_value) else float(close_value)
    close_value = pd.to_numeric(closes[idx], errors="coerce")
    return None if pd.isna(close_value) else float(close_value)


def _compute_company_return_metrics(
    tickers: list[str],
    price_lookup: dict[str, dict[str, list[object]]],
    reference_date: date,
) -> tuple[pd.DataFrame, bool]:
    anchor_targets = {
        "1w_perf": reference_date - timedelta(days=7),
        "1m_perf": reference_date - timedelta(days=30),
        "3m_perf": reference_date - timedelta(days=90),
        "ytd_perf": date(reference_date.year, 1, 1),
    }
    rows: list[dict[str, object]] = []
    exact_anchor_missing = False
    for ticker_value in tickers:
        price_entry = price_lookup.get(ticker_value)
        anchor_close = _price_close_for_target(price_entry, reference_date, exact=True)
        if anchor_close is None:
            exact_anchor_missing = True
        row: dict[str, object] = {
            "ticker": ticker_value,
            "anchor_close": anchor_close,
        }
        for metric_name, target_date in anchor_targets.items():
            base_close = _price_close_for_target(
                price_entry,
                target_date,
                allow_after_fallback=(metric_name == "ytd_perf"),
            )
            if anchor_close is None or base_close is None or base_close == 0:
                row[metric_name] = np.nan
            else:
                row[metric_name] = 100.0 * (anchor_close / base_close - 1.0)
        rows.append(row)
    return pd.DataFrame(rows), exact_anchor_missing


def _basket_average(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def _format_percent_value(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "N/A"
    return f"{float(numeric):.1f}%"


def _format_numeric_value(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "N/A"
    return f"{float(numeric):.1f}"


def _render_score_with_symbol(value: object, symbol: str) -> str:
    formatted = _format_numeric_value(value)
    if formatted == "N/A" or not symbol:
        return formatted
    return f"{formatted} {symbol}"


def _render_percent_with_symbol(value: object, symbol: str) -> str:
    formatted = _format_percent_value(value)
    if formatted == "N/A" or not symbol:
        return formatted
    return f"{formatted} {symbol}"


def _render_thematics_score_with_symbol(value: object, symbol: str) -> str:
    rendered = _render_score_with_symbol(value, symbol)
    if rendered == "N/A":
        return rendered
    # Streamlit's grid keeps text columns left-aligned in this view, so pad the
    # score text itself to visually center the value-plus-symbol string.
    return f"\u2007{rendered}\u2007"


def normalize_indices_cache_for_comparison(cache_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ticker", "date", "adjusted_close"}
    missing_columns = sorted(required_columns.difference(cache_df.columns))
    if missing_columns:
        raise ValueError(
            "Indices cache missing required columns: "
            f"{', '.join(missing_columns)}"
        )

    working_df = cache_df[["ticker", "date", "adjusted_close"]].copy()
    working_df["ticker"] = working_df["ticker"].fillna("").astype(str).str.strip()
    working_df = working_df[working_df["ticker"].isin(INDEX_TICKERS)]
    working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce").dt.date
    working_df["adjusted_close"] = pd.to_numeric(working_df["adjusted_close"], errors="coerce")
    working_df = working_df.dropna(subset=["date"]).drop_duplicates(
        subset=["ticker", "date"], keep="first"
    )
    return working_df


def normalize_prices_cache_for_check(cache_df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(PRICE_CACHE_REQUIRED_COLUMNS.difference(cache_df.columns))
    if "rsi_divergence_flag" in cache_df.columns:
        divergence_raw = cache_df["rsi_divergence_flag"]
        if divergence_raw.isna().all():
            missing_columns.append("rsi_divergence_flag")
    if missing_columns:
        raise ValueError(
            "Prices cache missing required columns: "
            f"{', '.join(sorted(set(missing_columns)))}"
        )

    working_df = cache_df[list(PRICE_CACHE_COLUMNS)].copy()
    working_df["ticker"] = working_df["ticker"].fillna("").astype(str).str.strip().str.upper()
    working_df["date"] = pd.to_datetime(working_df["date"], errors="coerce").dt.date
    for column in ["adjusted_close", "adjusted_high", "adjusted_low", "rs", "obvm", "rsi_14"]:
        working_df[column] = pd.to_numeric(working_df[column], errors="coerce")
    working_df["rsi_divergence_flag"] = (
        working_df["rsi_divergence_flag"]
        .where(working_df["rsi_divergence_flag"].notna(), pd.NA)
        .astype("string")
        .str.strip()
        .str.lower()
    )
    working_df.loc[
        ~working_df["rsi_divergence_flag"].isin({"positive", "negative", "none"}),
        "rsi_divergence_flag",
    ] = pd.NA
    working_df = working_df[working_df["ticker"].astype(str).str.len() > 0]
    working_df = working_df.dropna(subset=["date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )
    return working_df


def build_indices_comparison_table(
    normalized_cache_df: pd.DataFrame,
    date_1: date,
    date_2: date,
) -> pd.DataFrame:
    lookup = normalized_cache_df.set_index(["ticker", "date"])["adjusted_close"]
    rows: list[dict[str, object]] = []
    for ticker in INDEX_TICKERS:
        close_date_1 = lookup.get((ticker, date_1), np.nan)
        close_date_2 = lookup.get((ticker, date_2), np.nan)
        variation_pct = np.nan
        if pd.notna(close_date_1) and pd.notna(close_date_2) and float(close_date_1) != 0.0:
            variation_pct = round(((float(close_date_2) - float(close_date_1)) / float(close_date_1)) * 100.0, 2)
        rows.append(
            {
                "Index": INDEX_TICKER_TO_NAME[ticker],
                "Close Date 1": close_date_1,
                "Close Date 2": close_date_2,
                "Variation %": variation_pct,
            }
        )
    return pd.DataFrame(rows)


def _compute_variation_pct(close_date_1: object, close_date_2: object) -> float:
    close_1 = pd.to_numeric(close_date_1, errors="coerce")
    close_2 = pd.to_numeric(close_date_2, errors="coerce")
    if pd.isna(close_1) or pd.isna(close_2) or float(close_1) == 0.0:
        return np.nan
    return round(((float(close_2) - float(close_1)) / float(close_1)) * 100.0, 2)


def _upsert_indices_close(
    cache_df: pd.DataFrame,
    *,
    ticker: str,
    target_date: date,
    adjusted_close: float,
) -> pd.DataFrame:
    updated = cache_df.copy()
    if "ticker" not in updated.columns or "date" not in updated.columns:
        raise ValueError("Indices cache missing required columns: ticker/date")
    if "adjusted_close" not in updated.columns:
        updated["adjusted_close"] = np.nan

    ticker_series = updated["ticker"].fillna("").astype(str).str.strip()
    date_series = pd.to_datetime(updated["date"], errors="coerce").dt.date
    mask = (ticker_series == ticker) & (date_series == target_date)

    if mask.any():
        updated.loc[mask, "adjusted_close"] = float(adjusted_close)
        return updated

    new_row = {col: pd.NA for col in updated.columns}
    new_row["ticker"] = ticker
    new_row["date"] = pd.Timestamp(target_date)
    new_row["adjusted_close"] = float(adjusted_close)
    return pd.concat([updated, pd.DataFrame([new_row])], ignore_index=True)


@st.cache_data(show_spinner=False)
def compute_sector_T(
    path_str: str,
    max_tilt: float,
    extreme_coverage_target: float,
) -> pd.DataFrame:
    path = Path(path_str)
    df = normalize_report_columns(load_report_select(path_str)).copy()
    required = {
        "sector",
        "intermediate_trend",
        "long_term_trend",
        "momentum",
        "relative_performance",
        "relative_volume",
        "rs_daily",
        "rs_sma20",
        "obvm_daily",
        "obvm_sma20",
        "near_1y_high_5pct",
        "near_1y_low_5pct",
    }
    ensure_required_columns(df, required, path)
    df["sector"] = df["sector"].fillna("Unspecified")
    numeric_cols = [
        "intermediate_trend",
        "long_term_trend",
        "momentum",
        "relative_performance",
        "relative_volume",
        "rs_daily",
        "rs_sma20",
        "obvm_daily",
        "obvm_sma20",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def is_yes(value: object) -> bool:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        return str(value).strip().lower() == "yes"

    results = []
    for sector, group in df.groupby("sector", dropna=False):
        sector_count = int(len(group))
        avg_intermediate = group["intermediate_trend"].mean()
        avg_long_term = group["long_term_trend"].mean()
        avg_momentum = group["momentum"].mean()
        rs_score_avg = group["relative_performance"].mean()
        obvm_score_avg = group["relative_volume"].mean()

        rs_breadth = np.nan
        rs_mask = group["rs_daily"].notna() & group["rs_sma20"].notna()
        if rs_mask.any():
            rs_breadth = (group.loc[rs_mask, "rs_daily"] > group.loc[rs_mask, "rs_sma20"]).mean() * 100.0

        obvm_breadth = np.nan
        obvm_mask = group["obvm_daily"].notna() & group["obvm_sma20"].notna()
        if obvm_mask.any():
            obvm_breadth = (group.loc[obvm_mask, "obvm_daily"] > group.loc[obvm_mask, "obvm_sma20"]).mean() * 100.0

        rs_components = [rs_breadth, rs_score_avg]
        rs_pillar = np.nanmean(rs_components) if any(pd.notna(rs_components)) else np.nan

        obvm_components = [obvm_breadth, obvm_score_avg]
        obvm_pillar = np.nanmean(obvm_components) if any(pd.notna(obvm_components)) else np.nan

        raw_components = [avg_intermediate, avg_long_term, avg_momentum, rs_pillar, obvm_pillar]
        raw_t = np.nanmean(raw_components) if any(pd.notna(raw_components)) else np.nan

        high_count = int(group["near_1y_high_5pct"].apply(is_yes).sum())
        low_count = int(group["near_1y_low_5pct"].apply(is_yes).sum())
        extreme_total = high_count + low_count
        extreme_coverage = (extreme_total / sector_count) if sector_count else 0.0

        if extreme_total > 0:
            share_high = high_count / extreme_total
            share_low = low_count / extreme_total
            tilt = share_high - share_low
            if extreme_coverage_target > 0:
                strength = min(1.0, extreme_coverage / extreme_coverage_target)
            else:
                strength = 1.0
            multiplier = 1.0 + max_tilt * tilt * strength
        else:
            multiplier = 1.0

        t_value = raw_t * multiplier if pd.notna(raw_t) else np.nan
        if pd.notna(t_value):
            t_value = float(np.clip(t_value, 0.0, 100.0))

        results.append({
            "sector": sector,
            "T": t_value,
            "raw_T": raw_t,
            "t_multiplier": multiplier,
            "avg_intermediate_trend": avg_intermediate,
            "avg_long_term_trend": avg_long_term,
            "avg_momentum": avg_momentum,
            "rs_breadth_daily": rs_breadth,
            "obvm_breadth_daily": obvm_breadth,
            "rs_score_avg": rs_score_avg,
            "obvm_score_avg": obvm_score_avg,
            "rs_pillar_blended": rs_pillar,
            "obvm_pillar_blended": obvm_pillar,
            "near_high_count": high_count,
            "near_low_count": low_count,
            "extreme_coverage": extreme_coverage,
            "sector_stock_count": sector_count,
        })

    return pd.DataFrame(results)


@st.cache_data(show_spinner=False)
def compute_sector_participation_components(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    df = normalize_report_columns(load_report_select(path_str)).copy()
    required = {
        "sector",
        "1m_close",
        "eod_price_used",
        "ic_eod_price_used",
        "market_cap",
        "rs_monthly",
        "obvm_monthly",
    }
    ensure_required_columns(df, required, path)
    df["sector"] = df["sector"].fillna("Unspecified")
    stats = compute_sector_overview_stats(df)
    if stats.empty:
        return pd.DataFrame(columns=[
            "sector",
            "one_month_pct",
            "mk_breadth",
            "rel_perf_breadth",
            "rel_vol_breadth",
        ])
    components = stats[[
        "sector",
        "sector_1m_var_pct_num",
        "market_breadth_num",
        "rs_breadth_num",
        "obvm_breadth_num",
    ]].copy()
    components = components.rename(columns={
        "sector_1m_var_pct_num": "one_month_pct",
        "market_breadth_num": "mk_breadth",
        "rs_breadth_num": "rel_perf_breadth",
        "obvm_breadth_num": "rel_vol_breadth",
    })
    for col in ["one_month_pct", "mk_breadth", "rel_perf_breadth", "rel_vol_breadth"]:
        components[col] = pd.to_numeric(components[col], errors="coerce")
    return components


@st.cache_data(show_spinner=False)
def compute_sector_P(
    path_str: str,
    strong_pos_threshold: float,
    weak_pos_threshold: float,
    weak_neg_threshold: float,
    mult_strong_pos: float,
    mult_weak_pos: float,
    mult_weak_neg: float,
    mult_strong_neg: float,
) -> Tuple[pd.DataFrame, bool]:
    components = compute_sector_participation_components(path_str).copy()
    if components.empty:
        empty = components.assign(
            P=pd.Series(dtype=float),
            mc_var_score=pd.Series(dtype=float),
            mc_var_score_discounted=pd.Series(dtype=float),
        )
        return empty, False

    components["mc_var_score"] = components["one_month_pct"].rank(pct=True) * 100.0

    def apply_discount(value: float, score: float) -> float:
        if pd.isna(value) or pd.isna(score):
            return score
        if value > strong_pos_threshold:
            mult = mult_strong_pos
        elif value > weak_pos_threshold:
            mult = mult_weak_pos
        elif value > weak_neg_threshold:
            mult = mult_weak_neg
        else:
            mult = mult_strong_neg
        discounted = score * mult
        return float(np.clip(discounted, 0.0, 100.0))

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
    mc_var_all_negative = bool((components["one_month_pct"] < 0).all())
    return components, mc_var_all_negative


def build_quadrant_df(prev_metrics: pd.DataFrame, curr_metrics: pd.DataFrame) -> pd.DataFrame:
    merged = prev_metrics.merge(curr_metrics, on="sector", suffixes=("_prev", "_curr"), how="inner")
    merged["dT"] = merged["T_curr"] - merged["T_prev"]
    merged["dP"] = merged["P_curr"] - merged["P_prev"]
    merged["P_pct_curr"] = merged["P_curr"].rank(pct=True) * 100.0
    merged["T_pct_curr"] = merged["T_curr"].rank(pct=True) * 100.0
    merged["P_pct_prev"] = merged["P_prev"].rank(pct=True) * 100.0
    merged["T_pct_prev"] = merged["T_prev"].rank(pct=True) * 100.0
    return merged


def save_quadrant_json(
    df: pd.DataFrame,
    output_path: Path,
    date_prev: date,
    date_curr: date,
    curr_mc_var_all_negative: bool,
    prev_mc_var_all_negative: bool,
    curr_p_median: float,
    curr_t_median: float,
    settings: Dict[str, object],
) -> Path:
    def to_float(value: object) -> Optional[float]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        if pd.isna(value):
            return None
        return float(value)

    payload = {
        "date_prev": date_prev.isoformat(),
        "date_curr": date_curr.isoformat(),
        "mode_defaults": {"threshold": 50},
        "settings": settings,
        "global_flags": {
            "curr_mc_var_all_negative": bool(curr_mc_var_all_negative),
            "prev_mc_var_all_negative": bool(prev_mc_var_all_negative),
            "curr_P_median": to_float(curr_p_median),
            "curr_T_median": to_float(curr_t_median),
        },
        "sectors": [],
    }
    for _, row in df.iterrows():
        payload["sectors"].append({
            "sector": row["sector"],
            "T_prev": to_float(row.get("T_prev")),
            "T_curr": to_float(row.get("T_curr")),
            "dT": to_float(row.get("dT")),
            "P_prev": to_float(row.get("P_prev")),
            "P_curr": to_float(row.get("P_curr")),
            "dP": to_float(row.get("dP")),
            "raw_T_prev": to_float(row.get("raw_T_prev")),
            "raw_T_curr": to_float(row.get("raw_T_curr")),
            "rs_score_avg_prev": to_float(row.get("rs_score_avg_prev")),
            "rs_score_avg_curr": to_float(row.get("rs_score_avg_curr")),
            "obvm_score_avg_prev": to_float(row.get("obvm_score_avg_prev")),
            "obvm_score_avg_curr": to_float(row.get("obvm_score_avg_curr")),
            "rs_pillar_blended_prev": to_float(row.get("rs_pillar_blended_prev")),
            "rs_pillar_blended_curr": to_float(row.get("rs_pillar_blended_curr")),
            "obvm_pillar_blended_prev": to_float(row.get("obvm_pillar_blended_prev")),
            "obvm_pillar_blended_curr": to_float(row.get("obvm_pillar_blended_curr")),
            "t_multiplier_prev": to_float(row.get("t_multiplier_prev")),
            "t_multiplier_curr": to_float(row.get("t_multiplier_curr")),
            "near_high_count_prev": to_float(row.get("near_high_count_prev")),
            "near_high_count_curr": to_float(row.get("near_high_count_curr")),
            "near_low_count_prev": to_float(row.get("near_low_count_prev")),
            "near_low_count_curr": to_float(row.get("near_low_count_curr")),
            "extreme_coverage_prev": to_float(row.get("extreme_coverage_prev")),
            "extreme_coverage_curr": to_float(row.get("extreme_coverage_curr")),
            "participation_components_curr": {
                "mc_var_raw": to_float(row.get("one_month_pct_curr")),
                "mc_var_score": to_float(row.get("mc_var_score_curr")),
                "mc_var_score_discounted": to_float(row.get("mc_var_score_discounted_curr")),
                "mk_breadth": to_float(row.get("mk_breadth_curr")),
                "rel_perf_breadth": to_float(row.get("rel_perf_breadth_curr")),
                "rel_vol_breadth": to_float(row.get("rel_vol_breadth_curr")),
            },
            "participation_components_prev": {
                "mc_var_raw": to_float(row.get("one_month_pct_prev")),
                "mc_var_score": to_float(row.get("mc_var_score_prev")),
                "mc_var_score_discounted": to_float(row.get("mc_var_score_discounted_prev")),
                "mk_breadth": to_float(row.get("mk_breadth_prev")),
                "rel_perf_breadth": to_float(row.get("rel_perf_breadth_prev")),
                "rel_vol_breadth": to_float(row.get("rel_vol_breadth_prev")),
            },
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def plot_quadrants(df: pd.DataFrame, mode: str) -> go.Figure:
    mode = mode.lower()
    if mode == "percentile":
        x_col = "P_pct_curr"
        y_col = "T_pct_curr"
        x_label = "Participation (P) percentile"
        y_label = "Technical strength (T) percentile"
    else:
        x_col = "P_curr"
        y_col = "T_curr"
        x_label = "Participation (P)"
        y_label = "Technical strength (T)"

    fig = go.Figure()

    quadrant_colors = {
        "ll": "rgba(235, 87, 87, 0.06)",
        "lh": "rgba(255, 204, 102, 0.06)",
        "hl": "rgba(255, 204, 102, 0.06)",
        "hh": "rgba(11, 163, 96, 0.06)",
    }
    fig.add_shape(type="rect", x0=0, y0=0, x1=50, y1=50,
                  fillcolor=quadrant_colors["ll"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, y0=50, x1=50, y1=100,
                  fillcolor=quadrant_colors["lh"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=50, y0=0, x1=100, y1=50,
                  fillcolor=quadrant_colors["hl"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=50, y0=50, x1=100, y1=100,
                  fillcolor=quadrant_colors["hh"], line_width=0, layer="below")

    quadrant_labels = [
        ("A", 0.75, 0.75),
        ("B", 0.25, 0.75),
        ("C", 0.75, 0.25),
        ("D", 0.25, 0.25),
    ]
    for label, x_pos, y_pos in quadrant_labels:
        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=112, color="rgba(90, 90, 90, 0.10)"),
            align="center",
            xanchor="center",
            yanchor="middle",
        )

    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100,
                  line=dict(color="rgba(90, 90, 90, 0.2)", width=1))
    fig.add_shape(type="line", x0=0, y0=50, x1=100, y1=50,
                  line=dict(color="rgba(90, 90, 90, 0.2)", width=1))

    df_plot = df.sort_values("sector")
    outline_specs: list[tuple[int, str]] = []
    for _, row in df_plot.iterrows():
        d_t = row.get("dT")
        if pd.isna(d_t):
            outline_specs.append((0, "rgba(0,0,0,0)"))
        elif d_t > QUADRANT_BORDER_D_T_THRESHOLD:
            outline_specs.append((5, "#0BA360"))
        elif d_t < -QUADRANT_BORDER_D_T_THRESHOLD:
            outline_specs.append((5, "#EB5757"))
        else:
            outline_specs.append((0, "rgba(0,0,0,0)"))

    palette = [
        "#0072B2",  # blue
        "#E69F00",  # orange
        "#56B4E9",  # sky
        "#F0E442",  # yellow
        "#CC79A7",  # purple
        "#8B4513",  # brown
        "#999999",  # gray
        "#3B3EAC",  # indigo
        "#00A6D6",  # azure
        "#FFB000",  # amber
        "#6A3D9A",  # violet
        "#1B3A57",  # deep blue
    ]
    for idx, (_, row) in enumerate(df_plot.iterrows()):
        color = palette[idx % len(palette)]
        line_width, line_color = outline_specs[idx]
        fig.add_trace(go.Scatter(
            x=[row.get(x_col)],
            y=[row.get(y_col)],
            mode="markers",
            marker=dict(
                size=28,
                color=color,
                line=dict(width=line_width, color=line_color),
            ),
            name=str(row.get("sector", "")),
            hovertemplate="<b>%{text}</b><br>P: %{x:.1f}<br>T: %{y:.1f}<extra></extra>",
            text=[row.get("sector", "")],
            cliponaxis=False,
            showlegend=True,
        ))

    fig.update_layout(
        height=660,
        margin=dict(l=40, r=40, t=40, b=110),
        xaxis=dict(
            range=[0, 102],
            title=x_label,
            dtick=5,
            showgrid=True,
            gridcolor="rgba(90, 90, 90, 0.12)",
            gridwidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            range=[0, 102],
            title=y_label,
            dtick=5,
            showgrid=True,
            gridcolor="rgba(90, 90, 90, 0.12)",
            gridwidth=1,
            zeroline=False,
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.3,
            xanchor="center",
            x=0.5,
            title=None,
            font=dict(size=11),
        ),
    )
    return fig


def clear_quadrant_caches() -> None:
    load_report_select.clear()
    compute_sector_T.clear()
    compute_sector_participation_components.clear()
    compute_sector_P.clear()
    build_fundamental_tables_for_path.clear()
    build_technical_tables_for_path.clear()
    build_trended_table_for_paths.clear()

def load_all_texts() -> Dict[str, str]:
    latest_sector = get_latest_sector_tables_file()
    return {
        "prompt": read_text(PROMPT_PATH),
        "sector_path": str(latest_sector) if latest_sector else "",
        "sector_text": read_text(latest_sector),
        "summary_json": read_text(SUMMARY_JSON_PATH),
    }


def sync_editors(force: bool = False) -> None:
    bundle = st.session_state.setdefault("file_bundle", load_all_texts())
    if force or "prompt_content" not in st.session_state:
        st.session_state["prompt_content"] = bundle.get("prompt", "")
    if force or "sector_content" not in st.session_state:
        st.session_state["sector_content"] = bundle.get("sector_text", "")
    if force or "summary_content" not in st.session_state:
        st.session_state["summary_content"] = bundle.get("summary_json", "")
    st.session_state.setdefault("logs", "")
    st.session_state.setdefault("home_logs", "")


def refresh_files() -> None:
    load_report_select.clear()
    build_fundamental_tables_for_path.clear()
    build_technical_tables_for_path.clear()
    build_trended_table_for_paths.clear()
    st.session_state["file_bundle"] = load_all_texts()
    sync_editors(force=True)


def clear_runtime_caches() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["force_sync"] = True
    st.session_state["header_notice"] = "Cache cleared."
    st.rerun()


def append_log(
    message: str,
    *,
    logs_key: str = "logs",
    placeholder_key: str = "log_placeholder",
) -> None:
    current = st.session_state.get(logs_key, "")
    st.session_state[logs_key] = (current + ("\n" if current else "") + message)
    placeholder = st.session_state.get(placeholder_key)
    if placeholder is not None:
        placeholder.code(st.session_state[logs_key] or "(no logs yet)")


class StreamlitLogHandler(logging.Handler):
    """Logging handler that feeds messages into the Streamlit log panel."""

    def __init__(self, *, logs_key: str = "logs", placeholder_key: str = "log_placeholder"):
        super().__init__()
        self.logs_key = logs_key
        self.placeholder_key = placeholder_key

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI side-effects
        append_log(
            self.format(record),
            logs_key=self.logs_key,
            placeholder_key=self.placeholder_key,
        )


def run_generation(
    report_date: date,
    run_sql: bool,
    eod_as_of_date: Optional[date],
) -> None:
    output = REPORTS_DIR / f"Monthly_Scoring_Board_{report_date.isoformat()}.pdf"
    cache_anchor = eod_as_of_date or report_date
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    handler = StreamlitLogHandler()
    handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            generate_weekly_scoring_board_pdf(
                output,
                report_date=report_date,
                run_sql=run_sql,
                use_cache=True,
                eod_as_of_date=cache_anchor,
                cache_date=cache_anchor,
                use_config=False,
            )
        append_log(f"PDF generated successfully: {output}")
        st.success(f"PDF generated: {output}")
    except Exception as exc:  # pragma: no cover - show in UI
        append_log(f"Generation failed: {exc}")
        st.error(f"Generation failed: {exc}")
    finally:
        if stdout_buffer.getvalue():
            append_log(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            append_log(stderr_buffer.getvalue())
        refresh_files()
        root_logger.removeHandler(handler)


def summary_counts(summary_text: str) -> str:
    try:
        parsed = json.loads(summary_text)
    except json.JSONDecodeError:
        return "JSON invalid ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ cannot compute counts."
    counts = {
        key: len(value) if isinstance(value, list) else 0
        for key, value in parsed.items()
    }
    return ", ".join(f"{k}: {v}" for k, v in counts.items())


def copy_button(text: str, key: str) -> None:
    """Render a copy-to-clipboard button using Streamlit's HTML component."""
    html(
        f"""
        <button onclick='navigator.clipboard.writeText({json.dumps(text)})'
                style='padding:6px 12px; margin-top:18px;' id='{key}'>
            Copy to clipboard
        </button>
        """,
        height=60,
    )


def _deep_merge_dict(base: dict[str, object], incoming: dict[str, object]) -> dict[str, object]:
    for key, value in incoming.items():
        if isinstance(base.get(key), dict) and isinstance(value, dict):
            _deep_merge_dict(base[key], value)  # type: ignore[index]
        else:
            base[key] = value
    return base


def _rows_to_records(rows: object, columns: list[str]) -> list[dict[str, object]]:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    else:
        frame = pd.DataFrame(rows or [], columns=columns)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    frame = frame.reindex(columns=columns, fill_value="")
    return frame.replace({np.nan: ""}).to_dict("records")


def _render_rows_editor(
    label: str,
    rows: list[dict[str, object]],
    columns: list[str],
    *,
    key: str,
    help_text: str = "",
    height: int = 180,
) -> list[dict[str, object]]:
    st.caption(label)
    if help_text:
        st.caption(help_text)
    edited = st.data_editor(
        pd.DataFrame(rows or [], columns=columns),
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        height=height,
        key=key,
    )
    normalized = _rows_to_records(edited, columns)
    return normalized


def apply_api_template_state(template_payload: Optional[dict[str, object]] = None) -> None:
    template_state = clone_default_template_payload()
    if template_payload:
        _deep_merge_dict(template_state, template_payload)

    prompt_payload = template_state["prompt"]
    web_search_payload = template_state["web_search"]
    file_search_payload = template_state["file_search"]
    ranking_payload = file_search_payload["ranking_options"]
    filters_payload = file_search_payload["filters"]
    user_location_payload = web_search_payload["user_location"]
    reasoning_payload = template_state.get("reasoning", {})
    text_payload = template_state.get("text", {})

    st.session_state["api_template_name"] = str(template_state.get("name", "") or "")
    st.session_state["api_template_model"] = str(template_state.get("model", "") or "")
    st.session_state["api_template_developer_text"] = str(template_state.get("developer_text", "") or "")
    st.session_state["api_template_user_text"] = str(template_state.get("user_text", "") or "")
    st.session_state["api_template_prompt_id"] = str(prompt_payload.get("id", "") or "")
    st.session_state["api_template_prompt_version"] = str(prompt_payload.get("version", "") or "")
    st.session_state["api_template_prompt_variables_rows"] = _rows_to_records(
        prompt_payload.get("variables", []), ["key", "value"]
    )
    st.session_state["api_template_tool_choice"] = str(template_state.get("tool_choice", "auto") or "auto")
    st.session_state["api_template_temperature"] = float(template_state.get("temperature", 1.0) or 1.0)
    st.session_state["api_template_top_p"] = float(template_state.get("top_p", 1.0) or 1.0)
    max_output_tokens = template_state.get("max_output_tokens", "")
    st.session_state["api_template_max_output_tokens"] = (
        "" if max_output_tokens in ("", None) else str(max_output_tokens)
    )
    if isinstance(reasoning_payload, dict):
        reasoning_effort = reasoning_payload.get("effort", "")
    else:
        reasoning_effort = ""
    st.session_state["api_template_reasoning_effort"] = str(reasoning_effort or "")
    if isinstance(text_payload, dict):
        verbosity = text_payload.get("verbosity", "")
    else:
        verbosity = ""
    st.session_state["api_template_text_verbosity"] = str(verbosity or "")
    st.session_state["api_template_store"] = bool(template_state.get("store", True))
    st.session_state["api_template_parallel_tool_calls"] = bool(
        template_state.get("parallel_tool_calls", True)
    )
    st.session_state["api_template_metadata_rows"] = _rows_to_records(
        template_state.get("metadata", []), ["key", "value"]
    )
    st.session_state["api_template_default_output_name"] = str(
        template_state.get("default_output_name", "") or ""
    )

    st.session_state["api_template_web_enabled"] = bool(web_search_payload.get("enabled", False))
    st.session_state["api_template_web_allowed_domains"] = "\n".join(
        [str(item) for item in web_search_payload.get("allowed_domains", []) if str(item).strip()]
    )
    st.session_state["api_template_web_external_access"] = bool(
        web_search_payload.get("external_web_access", True)
    )
    st.session_state["api_template_web_country"] = str(user_location_payload.get("country", "") or "")
    st.session_state["api_template_web_city"] = str(user_location_payload.get("city", "") or "")
    st.session_state["api_template_web_region"] = str(user_location_payload.get("region", "") or "")
    st.session_state["api_template_web_timezone"] = str(user_location_payload.get("timezone", "") or "")

    st.session_state["api_template_file_enabled"] = bool(file_search_payload.get("enabled", False))
    st.session_state["api_template_file_vector_store_ids"] = ", ".join(
        parse_string_list(file_search_payload.get("vector_store_ids", []))
    )
    st.session_state["api_template_file_max_num_results"] = int(
        file_search_payload.get("max_num_results", 8) or 8
    )
    st.session_state["api_template_file_include_results"] = bool(
        file_search_payload.get("include_results", False)
    )
    st.session_state["api_template_file_ranker"] = str(ranking_payload.get("ranker", "") or "")
    st.session_state["api_template_file_score_threshold"] = str(
        ranking_payload.get("score_threshold", "") or ""
    )
    st.session_state["api_template_file_filter_type"] = str(filters_payload.get("type", "and") or "and")
    st.session_state["api_template_file_filter_rows"] = _rows_to_records(
        filters_payload.get("rows", []), ["key", "type", "value_type", "value"]
    )


def apply_prompt_store_state(prompt_payload: Optional[dict[str, object]] = None) -> None:
    prompt_state = clone_default_prompt_payload()
    if prompt_payload:
        prompt_state.update(prompt_payload)
    st.session_state["api_prompt_name"] = str(prompt_state.get("name", "") or "")
    st.session_state["api_prompt_developer_text"] = str(prompt_state.get("developer_text", "") or "")
    st.session_state["api_prompt_user_text"] = str(prompt_state.get("user_text", "") or "")


def ensure_api_state() -> None:
    ensure_api_storage_dirs()
    if "api_template_model" not in st.session_state:
        apply_api_template_state()
    if "api_prompt_name" not in st.session_state:
        apply_prompt_store_state()
    st.session_state.setdefault("api_last_output_text", "")
    st.session_state.setdefault("api_last_output_path", "")
    st.session_state.setdefault("api_last_payload", {})


def collect_api_template_from_state(
    prompt_variable_rows: list[dict[str, object]],
    metadata_rows: list[dict[str, object]],
    filter_rows: list[dict[str, object]],
) -> dict[str, object]:
    score_threshold_value = str(st.session_state.get("api_template_file_score_threshold", "") or "").strip()
    max_output_tokens_value = str(st.session_state.get("api_template_max_output_tokens", "") or "").strip()
    reasoning_effort_value = str(st.session_state.get("api_template_reasoning_effort", "") or "").strip()
    text_verbosity_value = str(st.session_state.get("api_template_text_verbosity", "") or "").strip()
    return {
        "name": str(st.session_state.get("api_template_name", "") or "").strip(),
        "model": str(st.session_state.get("api_template_model", "") or "").strip(),
        "developer_text": st.session_state.get("api_template_developer_text", "") or "",
        "user_text": st.session_state.get("api_template_user_text", "") or "",
        "prompt": {
            "id": str(st.session_state.get("api_template_prompt_id", "") or "").strip(),
            "version": str(st.session_state.get("api_template_prompt_version", "") or "").strip(),
            "variables": prompt_variable_rows,
        },
        "tool_choice": str(st.session_state.get("api_template_tool_choice", "auto") or "auto"),
        "temperature": float(st.session_state.get("api_template_temperature", 1.0) or 1.0),
        "top_p": float(st.session_state.get("api_template_top_p", 1.0) or 1.0),
        "max_output_tokens": max_output_tokens_value,
        "reasoning": {
            "effort": reasoning_effort_value,
        },
        "text": {
            "verbosity": text_verbosity_value,
        },
        "store": bool(st.session_state.get("api_template_store", True)),
        "parallel_tool_calls": bool(st.session_state.get("api_template_parallel_tool_calls", True)),
        "metadata": metadata_rows,
        "web_search": {
            "enabled": bool(st.session_state.get("api_template_web_enabled", False)),
            "allowed_domains": st.session_state.get("api_template_web_allowed_domains", "") or "",
            "external_web_access": bool(st.session_state.get("api_template_web_external_access", True)),
            "user_location": {
                "country": st.session_state.get("api_template_web_country", "") or "",
                "city": st.session_state.get("api_template_web_city", "") or "",
                "region": st.session_state.get("api_template_web_region", "") or "",
                "timezone": st.session_state.get("api_template_web_timezone", "") or "",
            },
        },
        "file_search": {
            "enabled": bool(st.session_state.get("api_template_file_enabled", False)),
            "vector_store_ids": st.session_state.get("api_template_file_vector_store_ids", "") or "",
            "max_num_results": int(st.session_state.get("api_template_file_max_num_results", 8) or 8),
            "include_results": bool(st.session_state.get("api_template_file_include_results", False)),
            "ranking_options": {
                "ranker": str(st.session_state.get("api_template_file_ranker", "") or "").strip(),
                "score_threshold": score_threshold_value,
            },
            "filters": {
                "type": str(st.session_state.get("api_template_file_filter_type", "and") or "and"),
                "rows": filter_rows,
            },
        },
        "default_output_name": str(st.session_state.get("api_template_default_output_name", "") or "").strip(),
    }


def collect_prompt_store_from_state() -> dict[str, str]:
    return {
        "name": str(st.session_state.get("api_prompt_name", "") or "").strip(),
        "developer_text": st.session_state.get("api_prompt_developer_text", "") or "",
        "user_text": st.session_state.get("api_prompt_user_text", "") or "",
    }


def render_api_templates_subtab() -> None:
    template_names = list_saved_names(API_TEMPLATES_DIR, ".json")
    template_options = [""] + template_names
    default_index = 0
    selected_name = st.selectbox(
        "Saved template",
        template_options,
        index=default_index,
        key="api_template_selected_name",
        format_func=lambda value: value or "(new template)",
    )
    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        if st.button("Load template", use_container_width=True, key="api_load_template_button"):
            if not selected_name:
                st.info("Select a saved template first.")
            else:
                apply_api_template_state(load_json_document(API_TEMPLATES_DIR, selected_name))
                st.success(f"Loaded template: {selected_name}")
    with action_cols[1]:
        if st.button("Clear template", use_container_width=True, key="api_clear_template_button"):
            apply_api_template_state()
            st.success("Template editor reset.")

    st.info(
        f"Set `{OPENAI_API_KEY_ENV}` before triggering requests. "
        f"PowerShell: `$env:{OPENAI_API_KEY_ENV}=\"your-key\"` or `setx {OPENAI_API_KEY_ENV} \"your-key\"`."
    )

    save_clicked = False
    run_clicked = False
    with st.form(key="api_template_form", clear_on_submit=False):
        top_cols = st.columns([1.4, 1.2, 1.0])
        with top_cols[0]:
            st.text_input("Template file name", key="api_template_name", help="Saved under data/api_templates/")
        with top_cols[1]:
            st.text_input("Model", key="api_template_model")
        with top_cols[2]:
            st.text_input(
                "Output file name",
                key="api_template_default_output_name",
                help="Blank uses the request trigger timestamp.",
            )

        prompt_col, request_col = st.columns(2)
        with prompt_col:
            st.markdown("**Prompt Reference**")
            st.text_input("Prompt ID", key="api_template_prompt_id", help="Sent as prompt.id.")
            st.text_input("Prompt version", key="api_template_prompt_version")
            prompt_variable_rows = _render_rows_editor(
                "Prompt variables",
                st.session_state.get("api_template_prompt_variables_rows", []),
                ["key", "value"],
                key="api_prompt_variables_editor",
                help_text="Optional prompt variables saved with the template.",
            )
            st.session_state["api_template_prompt_variables_rows"] = prompt_variable_rows
        with request_col:
            st.markdown("**Request Controls**")
            st.selectbox("Tool choice", ["auto", "none", "required"], key="api_template_tool_choice")
            numeric_cols = st.columns(2)
            with numeric_cols[0]:
                st.number_input(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    step=0.1,
                    key="api_template_temperature",
                )
                st.checkbox("Store response server-side", key="api_template_store")
            with numeric_cols[1]:
                st.number_input(
                    "Top P",
                    min_value=0.0,
                    max_value=1.0,
                    step=0.05,
                    key="api_template_top_p",
                )
                st.checkbox("Parallel tool calls", key="api_template_parallel_tool_calls")
            st.selectbox(
                "Reasoning effort",
                ["", "none", "low", "medium", "high", "xhigh"],
                key="api_template_reasoning_effort",
                format_func=lambda value: value or "(model default)",
            )
            st.selectbox(
                "Text verbosity",
                ["", "low", "medium", "high"],
                key="api_template_text_verbosity",
                format_func=lambda value: value or "(model default)",
            )
            st.text_input(
                "Max output tokens",
                key="api_template_max_output_tokens",
                help="Optional. Leave blank to use the model default.",
            )
            metadata_rows = _render_rows_editor(
                "Request metadata",
                st.session_state.get("api_template_metadata_rows", []),
                ["key", "value"],
                key="api_request_metadata_editor",
                help_text="Optional metadata sent on the response request.",
            )
            st.session_state["api_template_metadata_rows"] = metadata_rows

        text_cols = st.columns(2)
        with text_cols[0]:
            st.text_area(
                "Developer instructions override",
                key="api_template_developer_text",
                height=220,
            )
        with text_cols[1]:
            st.text_area(
                "User input override",
                key="api_template_user_text",
                height=220,
            )

        web_tab, file_tab = st.tabs(["Web Search", "File Search"])
        with web_tab:
            st.checkbox("Enable web search", key="api_template_web_enabled")
            st.checkbox("External web access", key="api_template_web_external_access")
            st.text_area(
                "Allowed domains",
                key="api_template_web_allowed_domains",
                height=120,
                help="One domain per line or comma-separated.",
            )
            location_cols = st.columns(4)
            with location_cols[0]:
                st.text_input("Country", key="api_template_web_country")
            with location_cols[1]:
                st.text_input("City", key="api_template_web_city")
            with location_cols[2]:
                st.text_input("Region", key="api_template_web_region")
            with location_cols[3]:
                st.text_input("Timezone", key="api_template_web_timezone")

        with file_tab:
            st.checkbox("Enable file search", key="api_template_file_enabled")
            st.text_input(
                "Vector store IDs",
                key="api_template_file_vector_store_ids",
                help="Comma-separated IDs. You can also paste newline-separated IDs.",
            )
            file_cols = st.columns(4)
            with file_cols[0]:
                st.number_input(
                    "Max results",
                    min_value=1,
                    step=1,
                    key="api_template_file_max_num_results",
                )
            with file_cols[1]:
                st.checkbox("Include search results", key="api_template_file_include_results")
            with file_cols[2]:
                st.text_input("Ranker", key="api_template_file_ranker")
            with file_cols[3]:
                st.text_input("Score threshold", key="api_template_file_score_threshold")
            st.selectbox("Filter combine mode", ["and", "or"], key="api_template_file_filter_type")
            filter_rows = _render_rows_editor(
                "File metadata filters",
                st.session_state.get("api_template_file_filter_rows", []),
                ["key", "type", "value_type", "value"],
                key="api_file_metadata_filters_editor",
                help_text="Operators: eq, ne, gt, gte, lt, lte, in, nin. Metadata names stay fully editable.",
                height=220,
            )
            st.session_state["api_template_file_filter_rows"] = filter_rows

        template_payload = collect_api_template_from_state(prompt_variable_rows, metadata_rows, filter_rows)

        action_cols = st.columns([1, 1, 2])
        with action_cols[0]:
            save_clicked = st.form_submit_button("Save template", use_container_width=True)
        with action_cols[1]:
            run_clicked = st.form_submit_button("Run request", use_container_width=True)

    if save_clicked:
        template_name = str(template_payload.get("name", "") or "").strip()
        if not template_name:
            st.error("Template file name is required before saving.")
        else:
            saved_path = save_json_document(API_TEMPLATES_DIR, template_name, template_payload)
            st.success(f"Template saved: {saved_path.name}")
    if run_clicked:
        try:
            payload, _response, output_text = run_responses_request(template_payload)
            output_path = save_output_text(
                output_text,
                str(template_payload.get("default_output_name", "") or "").strip(),
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Request failed: {exc}")
        else:
            st.session_state["api_last_payload"] = payload
            st.session_state["api_last_output_text"] = output_text
            st.session_state["api_last_output_path"] = str(output_path)
            st.success(f"Response saved: {output_path.name}")

    preview_payload: dict[str, object]
    try:
        preview_payload = build_responses_payload(template_payload)
    except Exception as exc:
        st.warning(f"Payload preview unavailable: {exc}")
    else:
        with st.expander("Payload preview", expanded=False):
            st.json(preview_payload)

    last_output_path = st.session_state.get("api_last_output_path", "")
    if last_output_path:
        st.caption(f"Latest saved output: {Path(last_output_path).name}")


def render_api_outputs_subtab() -> None:
    output_names = list_saved_names(RESPONSE_OUTPUT_DIR, ".txt")
    if not output_names:
        st.info("No output files saved yet. Trigger a request from the Templates sub-tab first.")
        return

    selected_output = st.selectbox("Saved output file", output_names, key="api_output_selected_name")
    selected_text = load_output_text(selected_output)
    st.caption(f"Folder: {RESPONSE_OUTPUT_DIR}")
    st.text_area(
        "Saved output text",
        value=selected_text,
        height=360,
        disabled=True,
        key=f"api_output_viewer_{selected_output}",
    )
    copy_button(selected_text, "api_output_copy_button")


def render_api_prompts_subtab() -> None:
    prompt_names = list_saved_names(PROMPT_STORE_DIR, ".json")
    prompt_options = [""] + prompt_names
    selected_name = st.selectbox(
        "Saved prompt file",
        prompt_options,
        key="api_prompt_selected_name",
        format_func=lambda value: value or "(new prompt file)",
    )
    action_cols = st.columns([1, 1, 2])
    with action_cols[0]:
        if st.button("Load prompt", use_container_width=True, key="api_load_prompt_button"):
            if not selected_name:
                st.info("Select a saved prompt first.")
            else:
                apply_prompt_store_state(load_json_document(PROMPT_STORE_DIR, selected_name))
                st.success(f"Loaded prompt: {selected_name}")
    with action_cols[1]:
        if st.button("Clear prompt", use_container_width=True, key="api_clear_prompt_button"):
            apply_prompt_store_state()
            st.success("Prompt editor reset.")

    st.text_input("Prompt file name", key="api_prompt_name", help="Saved under data/prompt_store/")
    prompt_cols = st.columns(2)
    with prompt_cols[0]:
        st.text_area(
            "Developer prompt text",
            key="api_prompt_developer_text",
            height=260,
        )
    with prompt_cols[1]:
        st.text_area(
            "User prompt text",
            key="api_prompt_user_text",
            height=260,
        )

    if st.button("Save prompt", use_container_width=False, key="api_save_prompt_button"):
        prompt_payload = collect_prompt_store_from_state()
        prompt_name = prompt_payload["name"].strip()
        if not prompt_name:
            st.error("Prompt file name is required before saving.")
        else:
            saved_path = save_json_document(PROMPT_STORE_DIR, prompt_name, prompt_payload)
            st.success(f"Prompt saved: {saved_path.name}")


def render_api_tab() -> None:
    render_subtab_group_intro(
        "API cockpit",
        "Save reusable OpenAI Responses API templates, manage prompt files, and load saved output text files.",
    )
    template_tab, outputs_tab, prompts_tab = st.tabs(["Templates", "Outputs", "Prompts"])
    with template_tab:
        render_api_templates_subtab()
    with outputs_tab:
        render_api_outputs_subtab()
    with prompts_tab:
        render_api_prompts_subtab()


def get_banner_path() -> Optional[Path]:
    for banner_path in BANNER_CANDIDATES:
        if banner_path.exists():
            return banner_path
    return None


def get_side_banner_path() -> Optional[Path]:
    for banner_path in BANNER_SIDE_CANDIDATES:
        if banner_path.exists():
            return banner_path
    return None


@st.cache_data(show_spinner=False)
def _banner_data_uri(path_str: str) -> str:
    path = Path(path_str)
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def compose_summary_prompt_final(prompt_text: str, sector_text: str, summary_json_text: str) -> str:
    final_prompt = prompt_text.replace("<<TABELE_EQUIPICKER_ANTERIOARE>>", sector_text)
    return final_prompt.replace("<<OUTPUT_ANTERIOR_JSON>>", summary_json_text)


def apply_theme_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --ep-font:'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --ep-navy:#0F2747;
    --ep-sky:#2E90FA;
    --ep-sky-soft:#DAECFF;
    --ep-mint:#22C55E;
    --ep-amber:#F59E0B;
    --ep-rose:#EF4444;
    --ep-ink:#1F2A44;
    --ep-muted:#64748B;
    --ep-surface:#FFFFFF;
    --ep-surface-soft:#F8FCFF;
    --ep-border:#D9E4EE;
}
/* Hide Streamlit chrome — keep sidebar toggle visible */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
div[data-testid="stToolbar"] {
    visibility: visible !important;
}
header[data-testid="stHeader"] [data-testid="stStatusWidget"],
header[data-testid="stHeader"] [data-testid="stToolbarActions"],
header[data-testid="stHeader"] [data-testid="stMainMenu"],
header[data-testid="stHeader"] a {
    display: none !important;
}
div[data-testid="stDecoration"] {display: none;}
header[data-testid="stHeader"] {
    background: linear-gradient(180deg, rgba(251, 253, 248, 0.96) 0%, rgba(247, 251, 255, 0.92) 100%) !important;
    border-bottom: 1px solid rgba(217, 228, 238, 0.9) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    z-index: 1000 !important;
}
/* Sidebar toggle button in both expanded and collapsed states */
header[data-testid="stHeader"] button[kind="headerNoPadding"][aria-label*="sidebar" i],
header[data-testid="stHeader"] [data-testid="collapsedControl"],
header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"] button[kind="headerNoPadding"][aria-label*="sidebar" i],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button,
section[data-testid="stSidebar"] button[aria-label*="sidebar" i] {
    visibility: visible !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: linear-gradient(180deg, #EF4444 0%, #C62828 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255, 235, 238, 0.55) !important;
    border-radius: 8px !important;
    box-shadow: 0 6px 16px rgba(198, 40, 40, 0.35) !important;
    width: 36px !important;
    height: 36px !important;
    margin: 0.5rem !important;
    opacity: 1 !important;
    position: relative !important;
    z-index: 1001 !important;
}
header[data-testid="stHeader"] button[kind="headerNoPadding"][aria-label*="sidebar" i]:hover,
header[data-testid="stHeader"] [data-testid="collapsedControl"]:hover,
header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"]:hover,
section[data-testid="stSidebar"] button[kind="headerNoPadding"][aria-label*="sidebar" i]:hover,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover,
section[data-testid="stSidebar"] button[aria-label*="sidebar" i]:hover {
    background: linear-gradient(180deg, #F87171 0%, #DC2626 100%) !important;
    border-color: rgba(255, 255, 255, 0.8) !important;
    box-shadow: 0 8px 18px rgba(220, 38, 38, 0.42) !important;
}
header[data-testid="stHeader"] button[kind="headerNoPadding"][aria-label*="sidebar" i] svg,
header[data-testid="stHeader"] [data-testid="collapsedControl"] svg,
header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] svg,
section[data-testid="stSidebar"] button[kind="headerNoPadding"][aria-label*="sidebar" i] svg,
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button svg,
section[data-testid="stSidebar"] button[aria-label*="sidebar" i] svg {
    fill: #FFFFFF !important;
    stroke: #FFFFFF !important;
}
/* Professional font */
html, body, [class*="css"], .stApp, .stMarkdown,
.stSelectbox, .stMultiSelect, .stDateInput, .stTextInput,
.stTextArea, .stNumberInput, button, input, textarea, select {
    font-family: var(--ep-font) !important;
}
h1, h2, h3, h4, h5, h6 { font-family: var(--ep-font) !important; }
.stApp {
    color: var(--ep-ink);
    background:
        radial-gradient(circle at 10% 6%, rgba(34, 197, 94, 0.11) 0%, rgba(34, 197, 94, 0.00) 36%),
        radial-gradient(circle at 88% 8%, rgba(46, 144, 250, 0.13) 0%, rgba(46, 144, 250, 0.00) 42%),
        linear-gradient(180deg, #FBFDF8 0%, #F7FBFF 60%, #F6FBF8 100%);
}
.block-container {
    max-width: 1580px;
    padding-top: 2.3rem;
    padding-bottom: 1.8rem;
}
h1, h2, h3 {
    color: var(--ep-ink);
    letter-spacing: -0.02em;
}
div[data-baseweb="tab-list"] {
    gap: 0.32rem;
    margin-top: 0.2rem;
    margin-bottom: 0.7rem;
    border-bottom: 0;
}
button[role="tab"] {
    border: 1px solid var(--ep-border);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.78);
    padding: 0.38rem 0.9rem;
    font-weight: 600;
    color: #3D4D66;
    transition: all 0.2s ease;
}
button[role="tab"]:hover {
    border-color: var(--ep-sky);
    color: var(--ep-sky);
}
button[role="tab"][aria-selected="true"] {
    background: linear-gradient(110deg, #ECF4FF 0%, #E6F5EC 100%);
    border-color: rgba(46, 144, 250, 0.42);
    color: #123159;
    box-shadow: 0 6px 14px rgba(17, 42, 78, 0.08);
}
.stButton > button {
    border-radius: 12px;
    border: 1px solid var(--ep-border);
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    border-color: var(--ep-sky);
    color: var(--ep-sky);
}
.stDateInput > div,
.stSelectbox > div > div,
.stTextArea textarea {
    border-radius: 12px !important;
}
.ep-appbar {
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:1rem;
    padding:0.75rem 0 0.6rem 0;
}
.ep-brand {
    font-size:1.72rem;
    font-weight:800;
    color: var(--ep-navy);
    line-height:1.15;
}
.ep-tagline {
    margin-top:0.1rem;
    font-size:0.93rem;
    color: #4C617E;
}
.ep-time-wrap {
    text-align:right;
}
.ep-time-label {
    color: var(--ep-muted);
    font-size:0.76rem;
    font-weight:600;
}
.ep-time-value {
    color: var(--ep-navy);
    font-size:0.95rem;
    font-weight:700;
    margin-top:0.12rem;
}
.ep-banner-wrap {
    margin:0.35rem 0 0.95rem 0;
    border-radius:14px;
    overflow:hidden;
    border:1px solid #D4E3EF;
    box-shadow:0 12px 28px rgba(17, 42, 78, 0.08);
}
.ep-section-shell {
    background: rgba(255, 255, 255, 0.72);
    border: 1px solid #D9E7F1;
    border-radius: 14px;
    padding: 0.86rem 0.95rem;
    margin-bottom: 0.72rem;
    animation: ep-fade-in 0.35s ease-out both;
}
.ep-breadcrumb {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #5F7694;
}
.ep-page-title {
    font-size: 1.34rem;
    font-weight: 780;
    color: var(--ep-navy);
    margin-top: 0.17rem;
}
.ep-page-subtitle {
    margin-top: 0.18rem;
    color: #4D627F;
    font-size: 0.93rem;
}
.ep-kpi-card {
    border-radius: 14px;
    border: 1px solid #D9E6F0;
    background: linear-gradient(180deg, #FFFFFF 0%, #F9FCFF 100%);
    padding: 0.72rem 0.86rem;
    min-height: 108px;
    box-shadow: 0 8px 20px rgba(15, 39, 71, 0.06);
    animation: ep-fade-in 0.4s ease-out both;
}
.ep-kpi-label {
    font-size: 0.74rem;
    color: #607691;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.ep-kpi-value {
    margin-top: 0.3rem;
    font-size: 1.26rem;
    font-weight: 800;
    color: var(--ep-navy);
    line-height: 1.2;
}
.ep-kpi-note {
    margin-top: 0.32rem;
    color: #5A708C;
    font-size: 0.82rem;
}
.ep-tone-positive .ep-kpi-value {
    color: #0E9B56;
}
.ep-tone-warn .ep-kpi-value {
    color: #D97706;
}
.ep-tone-neutral .ep-kpi-value {
    color: var(--ep-navy);
}
.ep-log-timeline {
    margin-top: 0.45rem;
    border: 1px solid var(--ep-border);
    border-radius: 12px;
    background: #FBFDFF;
    max-height: 260px;
    overflow-y: auto;
}
.ep-log-timeline ul {
    margin: 0;
    padding: 0.5rem 0.68rem;
    list-style: none;
}
.ep-log-item {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.77rem;
    line-height: 1.35;
    padding: 0.16rem 0;
    color: #3D4E66;
    border-bottom: 1px dashed rgba(151, 171, 196, 0.35);
}
.ep-log-item:last-child {
    border-bottom: 0;
}
.ep-log-dot {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    margin-top: 0.31rem;
    background: #93A8C3;
    flex: none;
}
.ep-log-item.positive .ep-log-dot { background: var(--ep-mint); }
.ep-log-item.warning .ep-log-dot { background: var(--ep-amber); }
.ep-log-item.error .ep-log-dot { background: var(--ep-rose); }
.ep-chip-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-top: 0.4rem;
}
.ep-chip {
    border-radius: 999px;
    border: 1px solid #D2E2ED;
    padding: 0.2rem 0.56rem;
    font-size: 0.75rem;
    color: #3A567B;
    background: rgba(255, 255, 255, 0.84);
}
.ep-subtab-shell {
    border: 1px solid #D9E7F1;
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(247,251,255,0.92) 100%);
    padding: 0.8rem 0.95rem;
    margin: 0.15rem 0 0.8rem 0;
    box-shadow: 0 8px 22px rgba(15, 39, 71, 0.05);
    animation: ep-fade-in 0.3s ease-out both;
}
.ep-subtab-title {
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #496685;
}
.ep-subtab-note {
    margin-top: 0.2rem;
    color: #4D627F;
    font-size: 0.9rem;
}
/* Fade-in animation */
@keyframes ep-fade-in {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
div[data-baseweb="tab-panel"] { animation: ep-fade-in 0.32s ease-out both; }
.ep-kpi-card:nth-child(2) { animation-delay: 0.06s; }
.ep-kpi-card:nth-child(3) { animation-delay: 0.12s; }
.ep-kpi-card:nth-child(4) { animation-delay: 0.18s; }
.ep-kpi-card:nth-child(5) { animation-delay: 0.24s; }
/* Custom scrollbars - Webkit */
.ep-log-timeline::-webkit-scrollbar,
[data-testid="stDataFrame"]::-webkit-scrollbar {
    width: 6px; height: 6px;
}
.ep-log-timeline::-webkit-scrollbar-track,
[data-testid="stDataFrame"]::-webkit-scrollbar-track {
    background: var(--ep-surface-soft); border-radius: 8px;
}
.ep-log-timeline::-webkit-scrollbar-thumb,
[data-testid="stDataFrame"]::-webkit-scrollbar-thumb {
    background: var(--ep-sky-soft); border-radius: 8px; border: 1px solid var(--ep-border);
}
.ep-log-timeline::-webkit-scrollbar-thumb:hover,
[data-testid="stDataFrame"]::-webkit-scrollbar-thumb:hover {
    background: var(--ep-sky);
}
/* Firefox scrollbar */
.ep-log-timeline, [data-testid="stDataFrame"] {
    scrollbar-width: thin;
    scrollbar-color: var(--ep-sky-soft) var(--ep-surface-soft);
}
/* Styled dataframes */
[data-testid="stDataFrame"] table thead th {
    background: linear-gradient(180deg, #0F2747 0%, #1A3A5C 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    border-bottom: 2px solid var(--ep-sky) !important;
}
[data-testid="stDataFrame"] table tbody tr:nth-child(even) {
    background-color: var(--ep-surface-soft) !important;
}
[data-testid="stDataFrame"] table tbody tr:nth-child(odd) {
    background-color: var(--ep-surface) !important;
}
[data-testid="stDataFrame"] table tbody tr:hover {
    background-color: var(--ep-sky-soft) !important;
    transition: background-color 0.15s ease;
}
[data-testid="stDataFrame"] table td {
    border-bottom: 1px solid var(--ep-border) !important;
    font-size: 0.82rem;
}
/* Footer */
.ep-footer {
    margin-top: 2rem;
    padding: 0.75rem 0;
    border-top: 1px solid var(--ep-border);
}
.ep-footer-inner {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
}
.ep-footer-item {
    font-size: 0.75rem;
    color: var(--ep-muted);
    font-weight: 500;
}
.ep-footer-disclaimer {
    font-style: italic;
    color: #8899AB;
}
.ep-footer-sep {
    color: var(--ep-border);
    font-size: 0.7rem;
}
/* Sidebar branding */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1D38 0%, #0F2747 40%, #132E52 100%);
    border-right: 1px solid rgba(46, 144, 250, 0.15);
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] p {
    color: #C8D8EB !important;
}
section[data-testid="stSidebar"] .stMarkdown strong {
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(46, 144, 250, 0.2);
}
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(110deg, var(--ep-sky) 0%, #1A6ED8 100%);
    color: #FFFFFF;
    border: none;
    font-weight: 700;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(110deg, #1A6ED8 0%, var(--ep-sky) 100%);
    color: #FFFFFF;
}
section[data-testid="stSidebar"] .stDateInput > div,
section[data-testid="stSidebar"] .stDateInput > div > div,
section[data-testid="stSidebar"] .stDateInput [data-baseweb="input"],
section[data-testid="stSidebar"] .stDateInput [data-baseweb="base-input"] {
    background: rgba(255, 255, 255, 0.22) !important;
    border: 1px solid rgba(160, 205, 255, 0.65) !important;
    border-radius: 10px !important;
    box-shadow: inset 0 0 0 1px rgba(15, 39, 71, 0.14) !important;
}
section[data-testid="stSidebar"] .stDateInput:hover > div,
section[data-testid="stSidebar"] .stDateInput:hover > div > div,
section[data-testid="stSidebar"] .stDateInput:hover [data-baseweb="input"],
section[data-testid="stSidebar"] .stDateInput:hover [data-baseweb="base-input"] {
    background: rgba(255, 255, 255, 0.27) !important;
    border-color: rgba(194, 224, 255, 0.82) !important;
}
section[data-testid="stSidebar"] .stDateInput:focus-within > div,
section[data-testid="stSidebar"] .stDateInput:focus-within > div > div,
section[data-testid="stSidebar"] .stDateInput:focus-within [data-baseweb="input"],
section[data-testid="stSidebar"] .stDateInput:focus-within [data-baseweb="base-input"] {
    background: rgba(255, 255, 255, 0.32) !important;
    border-color: #D6EBFF !important;
    box-shadow: 0 0 0 3px rgba(46, 144, 250, 0.18) !important;
}
section[data-testid="stSidebar"] .stDateInput input {
    color: #F8FBFF !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
section[data-testid="stSidebar"] .stDateInput input::placeholder {
    color: #DCEBFA !important;
    opacity: 1 !important;
}
section[data-testid="stSidebar"] .stDateInput button {
    color: #F8FBFF !important;
}
section[data-testid="stSidebar"] .stDateInput svg {
    fill: #DCEBFA !important;
    stroke: #DCEBFA !important;
}</style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    now_bucharest = datetime.now(ZoneInfo("Europe/Bucharest"))
    left_col, right_col = st.columns([3.6, 1.1])

    with left_col:
        st.markdown(
            """
<div class="ep-appbar">
  <div>
    <div class="ep-brand">Equipilot</div>
    <div class="ep-tagline">Professional cockpit for monthly scoring and market monitoring.</div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        button_spacer, button_col = st.columns([0.22, 0.78])
        with button_col:
            st.markdown('<div style="height:0.38rem;"></div>', unsafe_allow_html=True)
            if st.button("Clear cache", key="header_clear_cache", use_container_width=True):
                clear_runtime_caches()
        st.markdown(
            f"""
<div class="ep-time-wrap">
  <div class="ep-time-label">Current time (Europe/Bucharest)</div>
  <div class="ep-time-value">{now_bucharest:%Y-%m-%d %H:%M:%S}</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    notice = st.session_state.pop("header_notice", None)
    if notice:
        st.success(notice)

    banner_path = get_banner_path()
    if banner_path:
        banner_uri = _banner_data_uri(str(banner_path))
        side_banner_path = get_side_banner_path() or banner_path
        side_banner_uri = _banner_data_uri(str(side_banner_path))
        st.markdown(
            f"""
<div class="ep-banner-wrap">
  <div style="
      width:100%;
      height:150px;
      overflow:hidden;
      background-image:url('{side_banner_uri}');
      background-size:cover;
      background-position:center center;
  ">
    <img src="{banner_uri}"
         style="
             display:block;
             width:100%;
             height:100%;
             object-fit:contain;
             object-position:center center;
         " />
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.empty()


def render_page_intro(title: str, subtitle: str, breadcrumb: str) -> None:
    st.markdown(
        f"""
<div class="ep-section-shell">
  <div class="ep-breadcrumb">{breadcrumb}</div>
  <div class="ep-page-title">{title}</div>
  <div class="ep-page-subtitle">{subtitle}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
    if get_available_report_select_dates():
        st.caption("Blue ring in the calendar means a `report_select` file exists for that date.")


def render_kpi_card(label: str, value: str, note: str = "", tone: str = "neutral") -> None:
    safe_label = html_escape(label)
    safe_value = html_escape(value)
    safe_note = html_escape(note)
    st.markdown(
        f"""
<div class="ep-kpi-card ep-tone-{tone}">
  <div class="ep-kpi-label">{safe_label}</div>
  <div class="ep-kpi-value">{safe_value}</div>
  <div class="ep-kpi-note">{safe_note}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_chip_row(chips: list[str]) -> None:
    if not chips:
        return
    payload = "".join(f'<span class="ep-chip">{html_escape(chip)}</span>' for chip in chips)
    st.markdown(f'<div class="ep-chip-row">{payload}</div>', unsafe_allow_html=True)



def render_subtab_group_intro(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
<div class="ep-subtab-shell">
  <div class="ep-subtab-title">{html_escape(title)}</div>
  <div class="ep-subtab-note">{html_escape(subtitle)}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

def render_log_timeline(log_text: str, empty_message: str = "No log events yet.") -> None:
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    if not lines:
        st.caption(empty_message)
        return

    items: list[str] = []
    for line in lines[-80:]:
        line_lower = line.lower()
        css_class = "info"
        if any(token in line_lower for token in ("error", "failed", "traceback", "exception")):
            css_class = "error"
        elif any(token in line_lower for token in ("warning", "warn")):
            css_class = "warning"
        elif any(token in line_lower for token in ("success", "generated", "completed", "updated")):
            css_class = "positive"
        items.append(
            f'<li class="ep-log-item {css_class}">'
            f'<span class="ep-log-dot"></span><span>{html_escape(line)}</span></li>'
        )

    st.markdown(
        f'<div class="ep-log-timeline"><ul>{"".join(items)}</ul></div>',
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render professional footer bar."""
    now_bucharest = datetime.now(ZoneInfo("Europe/Bucharest"))
    st.markdown(
        f"""
<div class="ep-footer">
  <div class="ep-footer-inner">
    <span class="ep-footer-item">Equipilot v{APP_VERSION}</span>
    <span class="ep-footer-sep">&middot;</span>
    <span class="ep-footer-item ep-footer-disclaimer">For internal analytical use only. Not investment advice.</span>
    <span class="ep-footer-sep">&middot;</span>
    <span class="ep-footer-item">Last sync: {now_bucharest:%Y-%m-%d %H:%M} EET</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(config: ReportConfig) -> ReportConfig:
    """Render branded sidebar with logo and global config controls."""
    with st.sidebar:
        logo_path = BASE_DIR / "logo_1.jpg"
        if logo_path.exists():
            logo_uri = _banner_data_uri(str(logo_path))
            st.markdown(
                f"""
<div style="text-align:center; padding:0.8rem 0 0.5rem 0;">
    <img src="{logo_uri}" style="width:128px; border-radius:18px; box-shadow:0 10px 28px rgba(15,39,71,0.24);" />
    <div style="font-size:1.28rem; font-weight:800; color:#FFFFFF; margin-top:0.65rem;">Equipilot</div>
    <div style="font-size:0.8rem; color:#8BA3C1; margin-top:0.16rem;">v{APP_VERSION}</div>
</div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("---")

        st.markdown("**Global Config**")
        st.caption("These dates are used as defaults across all tabs.")

        report_date_value = render_report_select_date_input(
            "Report date",
            value=config.report_date,
            key="sidebar_report_date",
        )
        st.markdown(
            '<div style="margin-top:-0.35rem; margin-bottom:0.55rem; font-size:0.72rem; font-style:italic; color:#9FB3CA;">Used only for sector pulse.</div>',
            unsafe_allow_html=True,
        )
        eod_as_of_value = render_report_select_date_input(
            "EOD as-of date",
            value=config.eod_as_of_date or report_date_value,
            key="sidebar_eod_as_of_date",
        )

        if st.button("Save config", key="sidebar_save_config", use_container_width=True):
            save_report_config(
                ReportConfig(report_date=report_date_value, eod_as_of_date=eod_as_of_value),
                CONFIG_PATH,
            )
            st.success("Config saved.")
            st.rerun()

        st.markdown("---")
        st.caption(f"Equipilot v{APP_VERSION}")

    return ReportConfig(report_date=report_date_value, eod_as_of_date=eod_as_of_value)


def run_report_select_export(anchor_date: date, run_sql: bool) -> None:
    cache_file = report_cache_path(cache_date=anchor_date)
    existed_before = cache_file.exists()
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    handler = StreamlitLogHandler(logs_key="home_logs", placeholder_key="home_log_placeholder")
    handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    try:
        append_log(
            f"Starting report_select generation for {anchor_date.isoformat()} (run_sql={run_sql})",
            logs_key="home_logs",
            placeholder_key="home_log_placeholder",
        )
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            generated_path = generate_report_select_cache(anchor_date=anchor_date, run_sql=run_sql)
            load_report_select.clear()
            build_fundamental_tables_for_path.clear()
            build_technical_tables_for_path.clear()
            build_trended_table_for_paths.clear()
            loaded_rows = len(load_report_select(str(generated_path)))
        append_log(
            f"Rows loaded: {loaded_rows}",
            logs_key="home_logs",
            placeholder_key="home_log_placeholder",
        )
    except Exception as exc:  # pragma: no cover - UI feedback
        append_log(
            f"report_select generation failed: {exc}",
            logs_key="home_logs",
            placeholder_key="home_log_placeholder",
        )
        st.error(f"Report-select generation failed: {exc}")
    else:
        if run_sql:
            append_log(
                f"SQL executed. report_select file updated: {cache_file}",
                logs_key="home_logs",
                placeholder_key="home_log_placeholder",
            )
            st.success(f"SQL executed. report_select file updated: {cache_file}")
        elif existed_before:
            append_log(
                f"Reused existing report_select cache: {cache_file}",
                logs_key="home_logs",
                placeholder_key="home_log_placeholder",
            )
            st.info(f"Reused existing report_select cache: {cache_file}")
        else:
            append_log(
                f"Generated report_select cache: {cache_file}",
                logs_key="home_logs",
                placeholder_key="home_log_placeholder",
            )
            st.success(f"Generated report_select cache: {cache_file}")
        _cached_available_report_select_dates.clear()
        st.caption(f"Rows available: {loaded_rows}")
    finally:
        if stdout_buffer.getvalue():
            append_log(
                stdout_buffer.getvalue(),
                logs_key="home_logs",
                placeholder_key="home_log_placeholder",
            )
        if stderr_buffer.getvalue():
            append_log(
                stderr_buffer.getvalue(),
                logs_key="home_logs",
                placeholder_key="home_log_placeholder",
            )
        root_logger.removeHandler(handler)


def get_default_board_eod(config: ReportConfig) -> date:
    available_dates = get_available_report_select_dates()
    if config.eod_as_of_date:
        selected, _ = resolve_report_select_path(config.eod_as_of_date)
        if selected is not None:
            return config.eod_as_of_date
    if available_dates:
        return available_dates[-1]
    return date.fromisoformat(bucharest_today_str())


def get_default_previous_board_eod(current_eod: date) -> date:
    previous_dates = [entry for entry in get_available_report_select_dates() if entry < current_eod]
    if previous_dates:
        return previous_dates[-1]
    return current_eod - timedelta(days=30)


def load_report_select_for_eod(
    eod_date: date,
) -> Tuple[Optional[pd.DataFrame], Optional[Path], Tuple[Path, Path], Optional[str]]:
    resolved_path, expected_candidates = resolve_report_select_path(eod_date)
    if resolved_path is None:
        return None, None, expected_candidates, None
    try:
        loaded_df = normalize_report_columns(
            load_report_select(str(resolved_path), _path_cache_signature(resolved_path)).copy()
        )
    except Exception as exc:  # pragma: no cover - UI feedback
        return None, resolved_path, expected_candidates, str(exc)
    return loaded_df, resolved_path, expected_candidates, None


def _perf_mark(timings: list[tuple[str, float]], label: str, started_at: float) -> None:
    timings.append((label, (time.perf_counter() - started_at) * 1000.0))


def _render_perf_timings(show_perf: bool, timings: list[tuple[str, float]]) -> None:
    if not show_perf or not timings:
        return
    joined = " | ".join(f"{label}: {duration:.1f}ms" for label, duration in timings)
    st.caption(f"Perf timings: {joined}")


def _company_divergence_cache_signature() -> str:
    daily_signature = _paths_cache_signature(list_prices_cache_paths("daily"))
    weekly_signature = _paths_cache_signature(list_prices_cache_paths("weekly"))
    return f"daily:{daily_signature}|weekly:{weekly_signature}"


def _company_grid_height(row_count: int, *, row_height: int, min_height: int) -> int:
    return min(COMPANY_GRID_MAX_HEIGHT, max(min_height, 72 + row_count * row_height))


def _use_fast_company_grid_render(row_count: int) -> bool:
    return row_count >= COMPANY_GRID_FAST_RENDER_THRESHOLD


def validate_required_columns(
    df: pd.DataFrame,
    required: set[str],
    source_path: Path,
    board_name: str,
) -> bool:
    missing = sorted(required.difference(df.columns))
    if missing:
        st.error(
            f"{board_name}: missing columns in {source_path}: {', '.join(missing)}"
        )
        return False
    return True


def render_missing_report_select(eod_date: date, candidates: Tuple[Path, Path]) -> None:
    st.error(
        f"Missing report_select file for {eod_date.isoformat()}. "
        f"Expected {candidates[0]} (or {candidates[1]})."
    )
    st.caption("Generate it from Home > Report Excel Import using `Generate report_select Excel`.")


def render_board_title_band(title: str) -> None:
    st.markdown(
        f"""
<div style="
  background:linear-gradient(92deg, #0F2747 0%, #1A4B7A 70%, #2169A8 100%);
  color:#F7FBFF;
  font-weight:700;
  padding:9px 12px;
  border-radius:10px;
  margin:8px 0 10px 0;
  border:1px solid rgba(255,255,255,0.18);
  box-shadow:0 7px 14px rgba(15,39,71,0.16);
">
  {title}
</div>
        """,
        unsafe_allow_html=True,
    )


def _parse_number(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and np.isnan(value):
            return None
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw.replace("%", "").replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _score_color_css(value: object) -> str:
    numeric = _parse_number(value)
    if numeric is None:
        return "color:#A0A8B5; text-align:center; white-space:nowrap;"
    if numeric >= 80:
        return "color:#0BA360; font-weight:700; text-align:center; white-space:nowrap;"
    if numeric >= 60:
        return "color:#3BAE72; font-weight:700; text-align:center; white-space:nowrap;"
    if numeric >= 40:
        return "color:#F2994A; font-weight:700; text-align:center; white-space:nowrap;"
    return "color:#EB5757; font-weight:700; text-align:center; white-space:nowrap;"


def _variation_color_css(value: object) -> str:
    numeric = _parse_number(value)
    if numeric is None:
        return "color:#A0A8B5;"
    if numeric > 0:
        return "color:#0BA360; font-weight:700;"
    if numeric < 0:
        return "color:#EB5757; font-weight:700;"
    return "color:#425466;"


def _breadth_color_css(value: object) -> str:
    numeric = _parse_number(value)
    if numeric is None:
        return "color:#A0A8B5;"
    if numeric > 50:
        return "color:#0BA360; font-weight:700;"
    return "color:#EB5757; font-weight:700;"


def _signal_color_css(value: object) -> str:
    text = str(value).strip().lower()
    if text == "positive":
        return "color:#0BA360; font-weight:700;"
    if text == "negative":
        return "color:#EB5757; font-weight:700;"
    return "color:#425466; font-weight:700;"


def _highlight_max_style(series: pd.Series) -> list[str]:
    numeric = pd.to_numeric(series.map(_parse_number), errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return ["" for _ in range(len(series))]
    max_value = valid.max()
    return [
        "border:2px solid #0C97FF; border-radius:999px; font-weight:700;"
        if pd.notna(item) and item == max_value else ""
        for item in numeric
    ]


def render_pdf_like_table(
    df: pd.DataFrame,
    *,
    center_all_except_first: bool = False,
    center_cols: Optional[list[str]] = None,
    highlight_max_cols: Optional[list[str]] = None,
    value_color_rules: Optional[Dict[str, Callable[[object], str]]] = None,
    format_map: Optional[Dict[str, str]] = None,
    selectable: bool = False,
    selection_key: Optional[str] = None,
) -> Optional[int]:
    if df.empty:
        st.info("No data available for current selection.")
        return None

    style_frame = df.copy()
    estimated_height = max(220, 72 + len(style_frame) * 38)

    if selectable:
        display_frame = style_frame.copy()
        needs_light_styling = bool(
            value_color_rules or highlight_max_cols or center_all_except_first or center_cols
        )
        if needs_light_styling:
            selectable_view = display_frame.style.hide(axis="index")
            first_col = display_frame.columns[0]
            selectable_view = selectable_view.set_properties(
                subset=[first_col], **{"text-align": "left", "font-weight": "600"}
            )
            if center_all_except_first:
                other_cols = display_frame.columns[1:].tolist()
                if other_cols:
                    selectable_view = selectable_view.set_properties(
                        subset=other_cols, **{"text-align": "center"}
                    )
            if center_cols:
                usable_center_cols = [col for col in center_cols if col in display_frame.columns]
                if usable_center_cols:
                    selectable_view = selectable_view.set_properties(
                        subset=usable_center_cols, **{"text-align": "center"}
                    )
            if value_color_rules:
                for col, rule_fn in value_color_rules.items():
                    if col in display_frame.columns:
                        selectable_view = selectable_view.applymap(rule_fn, subset=[col])
            if highlight_max_cols:
                for col in highlight_max_cols:
                    if col in display_frame.columns:
                        selectable_view = selectable_view.apply(_highlight_max_style, axis=0, subset=[col])
            if format_map:
                valid_formats = {col: fmt for col, fmt in format_map.items() if col in display_frame.columns}
                if valid_formats:
                    selectable_view = selectable_view.format(valid_formats, na_rep="N/A")
        else:
            if format_map:
                valid_formats = {col: fmt for col, fmt in format_map.items() if col in display_frame.columns}
                for col, fmt in valid_formats.items():
                    display_frame[col] = display_frame[col].map(
                        lambda value, f=fmt: (f.format(value) if pd.notna(value) else "N/A")
                    )
            selectable_view = display_frame
        try:
            selection_event = st.dataframe(
                selectable_view,
                use_container_width=True,
                height=estimated_height,
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                key=selection_key,
            )
        except TypeError:
            st.dataframe(selectable_view, use_container_width=True, height=estimated_height, hide_index=True)
            st.warning("Row click selection is unavailable on this Streamlit version.")
            return None

        selected_rows: list[int] = []
        if hasattr(selection_event, "selection") and hasattr(selection_event.selection, "rows"):
            selected_rows = list(selection_event.selection.rows or [])
        elif isinstance(selection_event, dict):
            selected_rows = list(
                selection_event.get("selection", {}).get("rows", [])
            )
        if not selected_rows:
            return None
        selected_index = selected_rows[0]
        if selected_index < 0 or selected_index >= len(style_frame):
            return None
        return int(selected_index)

    styler = style_frame.style.hide(axis="index")
    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#0B2D5C"),
                    ("color", "#FFFFFF"),
                    ("font-weight", "700"),
                    ("text-align", "center"),
                    ("border", "1px solid #D3E1EA"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border", "1px solid #D3E1EA"),
                    ("color", "#425466"),
                    ("vertical-align", "middle"),
                ],
            },
        ],
        overwrite=False,
    )

    def stripe_rows(row: pd.Series) -> list[str]:
        background = "#F6FBFF" if row.name % 2 else "#FFFFFF"
        return [f"background-color:{background}" for _ in row]

    styler = styler.apply(stripe_rows, axis=1)

    first_col = style_frame.columns[0]
    styler = styler.set_properties(subset=[first_col], **{"text-align": "left", "font-weight": "600"})
    if center_all_except_first:
        other_cols = style_frame.columns[1:].tolist()
        if other_cols:
            styler = styler.set_properties(subset=other_cols, **{"text-align": "center"})
    if center_cols:
        usable_center_cols = [col for col in center_cols if col in style_frame.columns]
        if usable_center_cols:
            styler = styler.set_properties(subset=usable_center_cols, **{"text-align": "center"})

    if value_color_rules:
        for col, rule_fn in value_color_rules.items():
            if col in style_frame.columns:
                styler = styler.applymap(rule_fn, subset=[col])

    if highlight_max_cols:
        for col in highlight_max_cols:
            if col in style_frame.columns:
                styler = styler.apply(_highlight_max_style, axis=0, subset=[col])

    if format_map:
        valid_formats = {col: fmt for col, fmt in format_map.items() if col in style_frame.columns}
        if valid_formats:
            styler = styler.format(valid_formats, na_rep="N/A")
    st.dataframe(styler, use_container_width=True, height=estimated_height)
    return None


def _clear_drilldown_selection(prefix: str) -> None:
    st.session_state.pop(f"{prefix}_selected_key", None)
    st.session_state.pop(f"{prefix}_selected_mode", None)
    _queue_show_all_company_reset(prefix)
    for suffix in (
        "signature",
        "default_sector",
        "default_industry",
        "default_fund_range",
        "default_tech_range",
        "default_rsi_regime_range",
        "default_sector_regime_fit_range",
        "default_fund_momentum_range",
        "default_tech_trend_dir",
        "default_daily_rsi_divergence",
        "default_weekly_rsi_divergence",
        "pending_reset",
        "sector",
        "industry",
        "cap",
        "fund_range",
        "tech_range",
        "rsi_regime_range",
        "sector_regime_fit_range",
        "fund_momentum_range",
        "tech_trend_dir",
        "daily_rsi_divergence",
        "weekly_rsi_divergence",
        "ticker",
    ):
        st.session_state.pop(f"{prefix}_drilldown_filter_{suffix}", None)


def _handle_show_all_company_toggle(prefix: str) -> None:
    if bool(st.session_state.get(f"{prefix}_show_all_companies", False)):
        st.session_state.pop(f"{prefix}_selected_key", None)
        st.session_state.pop(f"{prefix}_selected_mode", None)
        _bump_selection_nonce(f"{prefix}_drilldown_nonce")


def _default_sector_regime_fit_range_for_company_scope(scope_kind: str) -> tuple[float, float]:
    if scope_kind == "show_all":
        return (60.0, 100.0)
    return (0.0, 100.0)


def _sync_drilldown_signature(prefix: str, signature: tuple[str, ...]) -> None:
    signature_key = f"{prefix}_drilldown_signature"
    previous_signature = st.session_state.get(signature_key)
    if previous_signature != signature:
        _clear_drilldown_selection(prefix)
        st.session_state[signature_key] = signature


def _sort_table_by_total_desc(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Total" not in df.columns:
        return df
    sorted_df = df.copy()
    sorted_df["__total_sort"] = pd.to_numeric(sorted_df["Total"].map(_parse_number), errors="coerce")
    return (
        sorted_df.sort_values("__total_sort", ascending=False, na_position="last", kind="mergesort")
        .drop(columns=["__total_sort"])
        .reset_index(drop=True)
    )


def _format_market_cap_display(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    raw = str(value).strip()
    if not raw:
        return "N/A"

    if raw[-1:].upper() in {"B", "M"}:
        return raw

    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return raw
    if numeric_value >= 1_000_000_000:
        return f"{numeric_value / 1_000_000_000:.2f}B"
    return f"{numeric_value / 1_000_000:.2f}M"


def _empty_market_regime_company_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ticker", "stock_rsi_regime_score", "sector_regime_fit_score"]
    )


def _empty_company_divergence_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ticker", "rsi_divergence_daily_flag", "rsi_divergence_weekly_flag"]
    )


def _latest_divergence_flags_for_frequency(
    frequency: str,
    evaluation_date: date,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for cache_path in list_prices_cache_paths(frequency):
        loaded = load_prices_cache(cache_path)
        if loaded.empty:
            continue
        parts.append(loaded)
    if not parts:
        return pd.DataFrame(columns=["ticker", "rsi_divergence_flag"])

    working = pd.concat(parts, ignore_index=True)
    if working.empty:
        return pd.DataFrame(columns=["ticker", "rsi_divergence_flag"])

    working["ticker"] = working["ticker"].map(normalize_price_ticker)
    working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.date
    working["rsi_divergence_flag"] = (
        working.get("rsi_divergence_flag", pd.Series(index=working.index, dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    working.loc[
        ~working["rsi_divergence_flag"].isin({"positive", "negative", "none"}),
        "rsi_divergence_flag",
    ] = pd.NA
    working = working[
        (working["ticker"].astype(str).str.len() > 0)
        & working["date"].notna()
        & (working["date"] <= evaluation_date)
    ]
    if working.empty:
        return pd.DataFrame(columns=["ticker", "rsi_divergence_flag"])
    return (
        working.sort_values(["ticker", "date"], kind="stable")
        .drop_duplicates(subset=["ticker"], keep="last")
        .loc[:, ["ticker", "rsi_divergence_flag"]]
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def _load_company_divergence_metrics_for_date(
    evaluation_date: date,
) -> pd.DataFrame:
    daily_df = _latest_divergence_flags_for_frequency("daily", evaluation_date).rename(
        columns={"rsi_divergence_flag": "rsi_divergence_daily_flag"}
    )
    weekly_df = _latest_divergence_flags_for_frequency("weekly", evaluation_date).rename(
        columns={"rsi_divergence_flag": "rsi_divergence_weekly_flag"}
    )
    if daily_df.empty and weekly_df.empty:
        return _empty_company_divergence_metrics()
    if daily_df.empty:
        daily_df = pd.DataFrame(columns=["ticker", "rsi_divergence_daily_flag"])
    if weekly_df.empty:
        weekly_df = pd.DataFrame(columns=["ticker", "rsi_divergence_weekly_flag"])
    merged = daily_df.merge(weekly_df, on="ticker", how="outer")
    for column in ["rsi_divergence_daily_flag", "rsi_divergence_weekly_flag"]:
        if column not in merged.columns:
            merged[column] = pd.NA
    return merged.drop_duplicates(subset=["ticker"], keep="last").reset_index(drop=True)


def _market_regime_company_metrics_cache_signature(evaluation_date: date) -> str:
    cache_key = build_market_cache_key(evaluation_date)
    cache_state = market_cache_status(cache_key)
    parts = [
        cache_key,
        "ready" if bool(cache_state.get("ready")) else "missing",
    ]
    for path_key in ["market_snapshot", "stock_rsi_regime", "setup_readiness"]:
        path_value = cache_state.get(path_key)
        path = Path(path_value) if path_value is not None else None
        if path is None:
            parts.extend([path_key, "", "missing"])
            continue
        try:
            mtime_ns = str(path.stat().st_mtime_ns)
        except OSError:
            mtime_ns = "missing"
        parts.extend([path_key, str(path), mtime_ns])
    return "|".join(parts)


@st.cache_data(show_spinner=False)
def _load_market_regime_company_metrics_for_date_cached(
    evaluation_date: date,
    cache_signature: str,
) -> tuple[pd.DataFrame, Optional[str]]:
    del cache_signature
    cache_key = build_market_cache_key(evaluation_date)
    cache_state = market_cache_status(cache_key)
    missing_warning = (
        f"Market regime cache is missing for {evaluation_date.isoformat()}. "
        "RSI Regime Score and Sector Regime Fit are shown as N/A and the default regime filters are relaxed until "
        "Market > Values is computed for this EOD."
    )
    if not bool(cache_state.get("ready")):
        return _empty_market_regime_company_metrics(), missing_warning

    try:
        bundle = load_market_bundle(cache_key)
    except Exception as exc:  # pragma: no cover - UI feedback
        return (
            _empty_market_regime_company_metrics(),
            f"Market regime cache for {evaluation_date.isoformat()} could not be loaded ({exc}). "
            "RSI Regime Score and Sector Regime Fit are shown as N/A and the default regime filters are relaxed.",
        )

    setup_df = bundle.get("setup_readiness_df", pd.DataFrame())
    if not isinstance(setup_df, pd.DataFrame) or setup_df.empty:
        return (
            _empty_market_regime_company_metrics(),
            f"Market regime cache for {evaluation_date.isoformat()} is incomplete. "
            "RSI Regime Score and Sector Regime Fit are shown as N/A and the default regime filters are relaxed.",
        )

    available_columns = [
        column
        for column in ["ticker", "stock_rsi_regime_score", "sector_regime_fit_score"]
        if column in setup_df.columns
    ]
    if "ticker" not in available_columns:
        return (
            _empty_market_regime_company_metrics(),
            f"Market regime cache for {evaluation_date.isoformat()} is missing ticker data. "
            "RSI Regime Score and Sector Regime Fit are shown as N/A and the default regime filters are relaxed.",
        )

    working = setup_df[available_columns].copy()
    working["ticker"] = working["ticker"].map(normalize_price_ticker)
    if "stock_rsi_regime_score" not in working.columns:
        working["stock_rsi_regime_score"] = np.nan
    if "sector_regime_fit_score" not in working.columns:
        working["sector_regime_fit_score"] = np.nan
    working["stock_rsi_regime_score"] = pd.to_numeric(
        working["stock_rsi_regime_score"], errors="coerce"
    )
    working["sector_regime_fit_score"] = pd.to_numeric(
        working["sector_regime_fit_score"], errors="coerce"
    )

    warning_message: Optional[str] = None
    missing_columns = [
        column
        for column in ["stock_rsi_regime_score", "sector_regime_fit_score"]
        if column not in available_columns
    ]
    if missing_columns:
        warning_message = (
            f"Market regime cache for {evaluation_date.isoformat()} is missing {', '.join(missing_columns)}. "
            "Unavailable values are shown as N/A and the default regime filters are relaxed."
        )

    return (
        working[
            ["ticker", "stock_rsi_regime_score", "sector_regime_fit_score"]
        ].drop_duplicates(subset=["ticker"], keep="first"),
        warning_message,
    )


def _load_market_regime_company_metrics_for_date(
    evaluation_date: date,
) -> tuple[pd.DataFrame, Optional[str]]:
    return _load_market_regime_company_metrics_for_date_cached(
        evaluation_date,
        _market_regime_company_metrics_cache_signature(evaluation_date),
    )


def _clear_market_regime_company_metrics_cache() -> None:
    _load_market_regime_company_metrics_for_date_cached.clear()


_load_market_regime_company_metrics_for_date.clear = _clear_market_regime_company_metrics_cache  # type: ignore[attr-defined]


def _enrich_company_universe_with_market_regime(
    company_df: pd.DataFrame,
    evaluation_date: date,
) -> tuple[pd.DataFrame, Optional[str]]:
    enriched = company_df.copy()
    for column in ["stock_rsi_regime_score", "sector_regime_fit_score"]:
        if column not in enriched.columns:
            enriched[column] = np.nan
    if enriched.empty:
        return enriched, None

    regime_df, warning_message = _load_market_regime_company_metrics_for_date(
        evaluation_date
    )
    if regime_df.empty:
        return enriched, warning_message

    merged = enriched.drop(
        columns=[
            column
            for column in ["stock_rsi_regime_score", "sector_regime_fit_score"]
            if column in enriched.columns
        ]
    ).merge(regime_df, on="ticker", how="left")
    return merged, warning_message


def _enrich_company_universe_with_rsi_divergence(
    company_df: pd.DataFrame,
    evaluation_date: date,
) -> pd.DataFrame:
    enriched = company_df.copy()
    for column in ["rsi_divergence_daily_flag", "rsi_divergence_weekly_flag"]:
        if column not in enriched.columns:
            enriched[column] = pd.NA
    if enriched.empty:
        return enriched

    divergence_df = _load_company_divergence_metrics_for_date(evaluation_date)
    if divergence_df.empty:
        return enriched

    merged = enriched.drop(
        columns=[
            column
            for column in ["rsi_divergence_daily_flag", "rsi_divergence_weekly_flag"]
            if column in enriched.columns
        ]
    ).merge(divergence_df, on="ticker", how="left")
    return merged


def _market_cap_bucket_from_usd(value: object) -> str:
    market_cap_num = pd.to_numeric(value, errors="coerce")
    if pd.isna(market_cap_num) or float(market_cap_num) < 0:
        return "Unknown"
    market_cap_b = float(market_cap_num) / 1_000_000_000.0
    if market_cap_b < 0.05:
        return "Nano"
    if market_cap_b < 0.3:
        return "Micro"
    if market_cap_b < 2:
        return "Small"
    if market_cap_b < 10:
        return "Mid"
    if market_cap_b < 200:
        return "Large"
    return "Mega"


def _sign_label(value: object) -> str:
    numeric_value = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric_value):
        return "N/A"
    return "Positive" if float(numeric_value) > 0 else "Negative"


def _format_divergence_flag(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    label = str(value).strip().lower()
    if label == "positive":
        return "Positive"
    if label == "negative":
        return "Negative"
    if label == "none":
        return "None"
    return "N/A"


def _format_short_term_flow_flag(value: object) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    label = str(value).strip().lower()
    if label == "positive":
        return "Positive"
    if label == "negative":
        return "Negative"
    if label == "neutral":
        return "Neutral"
    return "N/A"


def _filter_by_optional_label_value(
    df: pd.DataFrame,
    *,
    column: str,
    selected_value: str,
    label_map: dict[str, str],
) -> tuple[pd.DataFrame, bool]:
    normalized_selected = str(selected_value or "").strip()
    if normalized_selected == "All" or column not in df.columns:
        return df.copy(), False
    target_value = label_map.get(normalized_selected)
    if target_value is None:
        return df.copy(), False
    filtered = df[df[column].fillna("__na__").astype(str).str.lower() == target_value.lower()]
    return filtered.copy(), True


def _filter_by_optional_numeric_range(
    df: pd.DataFrame,
    *,
    column: str,
    range_value: tuple[float, float],
    min_value: float = 0.0,
    max_value: float = 100.0,
) -> tuple[pd.DataFrame, bool]:
    if column not in df.columns:
        return df.copy(), False
    numeric_series = pd.to_numeric(df[column], errors="coerce")
    if not numeric_series.notna().any():
        return df.copy(), False
    if (
        abs(float(range_value[0]) - float(min_value)) < 1e-9
        and abs(float(range_value[1]) - float(max_value)) < 1e-9
    ):
        return df.copy(), False
    filtered = df[numeric_series.between(float(range_value[0]), float(range_value[1]), inclusive="both")]
    return filtered.copy(), True


def _prepare_company_drilldown_universe(
    report_df: pd.DataFrame,
    *,
    include_beta: bool = False,
    thematic_memberships: Optional[dict[str, list[str]]] = None,
    thematics_config_signature: str = "",
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    required_columns = {
        "ticker",
        "sector",
        "industry",
        "market_cap",
        "fundamental_total_score",
        "general_technical_score",
    }
    missing = sorted(required_columns.difference(report_df.columns))
    if missing:
        return None, f"Missing required columns for company list: {', '.join(missing)}"

    selected_columns = [
        "ticker",
        "sector",
        "industry",
        "market_cap",
        "fundamental_total_score",
        "general_technical_score",
    ]
    if "company" in report_df.columns:
        selected_columns.insert(1, "company")
    if "fundamental_momentum" in report_df.columns:
        selected_columns.append("fundamental_momentum")
    for pillar_column in [
        "fundamental_growth",
        "fundamental_value",
        "fundamental_quality",
        "fundamental_risk",
    ]:
        if pillar_column in report_df.columns:
            selected_columns.append(pillar_column)
    if "rs_monthly" in report_df.columns:
        selected_columns.append("rs_monthly")
    if "obvm_monthly" in report_df.columns:
        selected_columns.append("obvm_monthly")
    for short_term_column in ["rs_daily", "rs_sma20", "obvm_daily", "obvm_sma20"]:
        if short_term_column in report_df.columns:
            selected_columns.append(short_term_column)
    if include_beta and "beta" in report_df.columns:
        selected_columns.append("beta")

    working = report_df[selected_columns].copy()
    working["ticker"] = working["ticker"].map(normalize_price_ticker)
    if "company" in working.columns:
        working["company"] = working["company"].fillna("").astype(str).str.strip()
    else:
        working["company"] = ""
    working["company"] = working["company"].where(working["company"].str.len() > 0, working["ticker"])
    working["sector"] = working["sector"].fillna("Unspecified")
    working["industry"] = working["industry"].fillna("Unspecified")
    working["market_cap"] = pd.to_numeric(working["market_cap"], errors="coerce")
    working["fundamental_total_score"] = pd.to_numeric(working["fundamental_total_score"], errors="coerce")
    working["general_technical_score"] = pd.to_numeric(working["general_technical_score"], errors="coerce")
    if "fundamental_momentum" in working.columns:
        working["fundamental_momentum"] = pd.to_numeric(working["fundamental_momentum"], errors="coerce")
    else:
        working["fundamental_momentum"] = np.nan
    for pillar_column in [
        "fundamental_growth",
        "fundamental_value",
        "fundamental_quality",
        "fundamental_risk",
    ]:
        if pillar_column in working.columns:
            working[pillar_column] = pd.to_numeric(working[pillar_column], errors="coerce")
        else:
            working[pillar_column] = np.nan
    if "rs_monthly" in working.columns:
        working["rs_monthly"] = pd.to_numeric(working["rs_monthly"], errors="coerce")
    else:
        working["rs_monthly"] = np.nan
    if "obvm_monthly" in working.columns:
        working["obvm_monthly"] = pd.to_numeric(working["obvm_monthly"], errors="coerce")
    else:
        working["obvm_monthly"] = np.nan
    for short_term_column in ["rs_daily", "rs_sma20", "obvm_daily", "obvm_sma20"]:
        if short_term_column in working.columns:
            working[short_term_column] = pd.to_numeric(working[short_term_column], errors="coerce")
        else:
            working[short_term_column] = np.nan
    if "beta" in working.columns:
        working["beta"] = pd.to_numeric(working["beta"], errors="coerce")
    else:
        working["beta"] = np.nan
    working["market_cap_bucket"] = working["market_cap"].apply(_market_cap_bucket_from_usd)
    working["rel_strength"] = working["rs_monthly"].map(_sign_label)
    working["rel_volume"] = working["obvm_monthly"].map(_sign_label)
    working["short_term_flow"] = pd.NA
    short_term_valid_mask = working[["rs_daily", "rs_sma20", "obvm_daily", "obvm_sma20"]].notna().all(axis=1)
    short_term_positive_mask = (
        short_term_valid_mask
        & (working["rs_daily"] >= working["rs_sma20"])
        & (working["obvm_daily"] >= working["obvm_sma20"])
    )
    short_term_negative_mask = (
        short_term_valid_mask
        & (working["rs_daily"] < working["rs_sma20"])
        & (working["obvm_daily"] < working["obvm_sma20"])
    )
    working.loc[short_term_valid_mask, "short_term_flow"] = "neutral"
    working.loc[short_term_positive_mask, "short_term_flow"] = "positive"
    working.loc[short_term_negative_mask, "short_term_flow"] = "negative"

    ai_lookup = (
        build_ai_exposure_lookup(str(THEMATICS_CONFIG_PATH), thematics_config_signature)
        if THEMATICS_CONFIG_PATH.exists()
        else {}
    )
    ai_revenue_exposure: list[str] = []
    ai_disruption_risk: list[str] = []
    thematic_display: list[str] = []
    thematic_membership_values: list[list[str]] = []
    membership_lookup = thematic_memberships or {}
    for ticker_value in working["ticker"].tolist():
        ai_entry = ai_lookup.get(ticker_value, {})
        ai_revenue_exposure.append(str(ai_entry.get("ai_revenue_exposure", "none")))
        ai_disruption_risk.append(str(ai_entry.get("ai_disruption_risk", "none")))
        memberships = sorted(set(membership_lookup.get(ticker_value, [])))
        thematic_membership_values.append(memberships)
        thematic_display.append(" | ".join(memberships) if memberships else "Unassigned")
    working["ai_revenue_exposure"] = ai_revenue_exposure
    working["ai_disruption_risk"] = ai_disruption_risk
    working["thematic"] = thematic_display
    working["thematic_memberships"] = thematic_membership_values
    working["stock_rsi_regime_score"] = np.nan
    working["sector_regime_fit_score"] = np.nan
    working["rsi_divergence_daily_flag"] = pd.NA
    working["rsi_divergence_weekly_flag"] = pd.NA
    return working, None


def build_company_drilldown_context(
    report_df: pd.DataFrame,
    *,
    evaluation_date: date,
    selected_sector: str,
    selected_mode: str,
    selected_key: str,
) -> tuple[str, Optional[pd.DataFrame], list[str], list[str], Optional[str], Optional[str]]:
    company_universe, error_message = _prepare_company_drilldown_universe(report_df)
    if error_message:
        return "", None, [], [], error_message, None
    assert company_universe is not None
    company_universe, regime_warning = _enrich_company_universe_with_market_regime(
        company_universe,
        evaluation_date,
    )
    company_universe = _enrich_company_universe_with_rsi_divergence(
        company_universe,
        evaluation_date,
    )
    if selected_mode == "all":
        return "All companies", company_universe, [], [], None, regime_warning
    if selected_mode == "sector":
        title = f"Companies in Sector: {selected_key}"
        default_sectors = [selected_key]
        default_industries: list[str] = []
    else:
        title = f"Companies in {selected_sector} / {selected_key}"
        default_sectors = [selected_sector]
        default_industries = [selected_key]
    return title, company_universe, default_sectors, default_industries, None, regime_warning


@st.cache_data(show_spinner=False)
def _load_prepared_company_universe_for_report_path(
    report_path_str: str,
    report_cache_signature: str,
    evaluation_date: date,
    thematics_config_signature: str,
    market_cache_signature: str,
    divergence_cache_signature: str,
) -> tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    _ = market_cache_signature, divergence_cache_signature
    try:
        report_df = normalize_report_columns(load_report_select(report_path_str, report_cache_signature).copy())
    except Exception as exc:  # pragma: no cover - defensive cache wrapper
        return None, f"Failed reading {report_path_str}: {exc}", None
    company_universe, error_message = _prepare_company_drilldown_universe(
        report_df,
        thematics_config_signature=thematics_config_signature,
    )
    if error_message:
        return None, error_message, None
    assert company_universe is not None
    company_universe, regime_warning = _enrich_company_universe_with_market_regime(
        company_universe,
        evaluation_date,
    )
    company_universe = _enrich_company_universe_with_rsi_divergence(
        company_universe,
        evaluation_date,
    )
    return company_universe, None, regime_warning


def build_company_drilldown_context_from_path(
    report_path: Path,
    *,
    evaluation_date: date,
    selected_sector: str,
    selected_mode: str,
    selected_key: str,
) -> tuple[str, Optional[pd.DataFrame], list[str], list[str], Optional[str], Optional[str]]:
    company_universe, error_message, regime_warning = _load_prepared_company_universe_for_report_path(
        str(report_path),
        _path_cache_signature(report_path),
        evaluation_date,
        _path_cache_signature(THEMATICS_CONFIG_PATH),
        _market_regime_company_metrics_cache_signature(evaluation_date),
        _company_divergence_cache_signature(),
    )
    if error_message:
        return "", None, [], [], error_message, None
    assert company_universe is not None
    if selected_mode == "all":
        return "All companies", company_universe, [], [], None, regime_warning
    if selected_mode == "sector":
        title = f"Companies in Sector: {selected_key}"
        default_sectors = [selected_key]
        default_industries: list[str] = []
    else:
        title = f"Companies in {selected_sector} / {selected_key}"
        default_sectors = [selected_sector]
        default_industries = [selected_key]
    return title, company_universe, default_sectors, default_industries, None, regime_warning


def _annotate_company_technical_trend(
    company_df: pd.DataFrame,
    previous_report_df: pd.DataFrame,
    *,
    threshold: float = 5.0,
) -> pd.DataFrame:
    annotated = company_df.copy()
    annotated["technical_trend_delta"] = np.nan
    annotated["technical_trend_symbol"] = ""
    annotated["technical_trend_direction"] = "none"
    if annotated.empty:
        return annotated
    if (
        "ticker" not in previous_report_df.columns
        or "general_technical_score" not in previous_report_df.columns
    ):
        return annotated

    previous_scores = previous_report_df[["ticker", "general_technical_score"]].copy()
    previous_scores["ticker"] = previous_scores["ticker"].fillna("").astype(str).str.strip()
    previous_scores["general_technical_score_prev"] = pd.to_numeric(
        previous_scores["general_technical_score"], errors="coerce"
    )
    previous_scores = previous_scores.drop(columns=["general_technical_score"]).drop_duplicates(
        subset=["ticker"], keep="first"
    )

    merged = annotated.merge(previous_scores, on="ticker", how="left")
    delta = merged["general_technical_score"] - merged["general_technical_score_prev"]
    merged["technical_trend_delta"] = delta
    merged["technical_trend_symbol"] = [
        _trend_symbol_for_delta(delta_value, threshold) for delta_value in delta.tolist()
    ]
    merged["technical_trend_direction"] = [
        _trend_direction_for_delta(delta_value, threshold) for delta_value in delta.tolist()
    ]
    return merged.drop(columns=["general_technical_score_prev"])


def _annotate_company_score_trends(
    company_df: pd.DataFrame,
    previous_report_df: pd.DataFrame,
    *,
    threshold: float = 5.0,
) -> pd.DataFrame:
    annotated = _annotate_company_technical_trend(company_df, previous_report_df, threshold=threshold)
    if annotated.empty or "ticker" not in previous_report_df.columns:
        annotated["fundamental_trend_symbol"] = ""
        annotated["fundamental_trend_direction"] = "none"
        annotated["fundamental_momentum_trend_symbol"] = ""
        annotated["fundamental_momentum_trend_direction"] = "none"
        annotated["fundamental_growth_trend_symbol"] = ""
        annotated["fundamental_growth_trend_direction"] = "none"
        annotated["fundamental_value_trend_symbol"] = ""
        annotated["fundamental_value_trend_direction"] = "none"
        annotated["fundamental_quality_trend_symbol"] = ""
        annotated["fundamental_quality_trend_direction"] = "none"
        annotated["fundamental_risk_trend_symbol"] = ""
        annotated["fundamental_risk_trend_direction"] = "none"
        return annotated

    previous_scores = previous_report_df.copy()
    previous_scores["ticker"] = previous_scores["ticker"].map(normalize_price_ticker)
    previous_scores["fundamental_total_score_prev"] = pd.to_numeric(
        previous_scores.get("fundamental_total_score"), errors="coerce"
    )
    previous_scores["fundamental_momentum_prev"] = pd.to_numeric(
        previous_scores.get("fundamental_momentum"), errors="coerce"
    )
    previous_scores["fundamental_growth_prev"] = pd.to_numeric(
        previous_scores.get("fundamental_growth"), errors="coerce"
    )
    previous_scores["fundamental_value_prev"] = pd.to_numeric(
        previous_scores.get("fundamental_value"), errors="coerce"
    )
    previous_scores["fundamental_quality_prev"] = pd.to_numeric(
        previous_scores.get("fundamental_quality"), errors="coerce"
    )
    previous_scores["fundamental_risk_prev"] = pd.to_numeric(
        previous_scores.get("fundamental_risk"), errors="coerce"
    )
    previous_subset = previous_scores[
        [
            "ticker",
            "fundamental_total_score_prev",
            "fundamental_momentum_prev",
            "fundamental_growth_prev",
            "fundamental_value_prev",
            "fundamental_quality_prev",
            "fundamental_risk_prev",
        ]
    ].drop_duplicates(subset=["ticker"], keep="first")

    merged = annotated.merge(previous_subset, on="ticker", how="left")
    for metric_name, current_col, prev_col, symbol_col, direction_col in [
        (
            "fundamental",
            "fundamental_total_score",
            "fundamental_total_score_prev",
            "fundamental_trend_symbol",
            "fundamental_trend_direction",
        ),
        (
            "fundamental_momentum",
            "fundamental_momentum",
            "fundamental_momentum_prev",
            "fundamental_momentum_trend_symbol",
            "fundamental_momentum_trend_direction",
        ),
        (
            "fundamental_growth",
            "fundamental_growth",
            "fundamental_growth_prev",
            "fundamental_growth_trend_symbol",
            "fundamental_growth_trend_direction",
        ),
        (
            "fundamental_value",
            "fundamental_value",
            "fundamental_value_prev",
            "fundamental_value_trend_symbol",
            "fundamental_value_trend_direction",
        ),
        (
            "fundamental_quality",
            "fundamental_quality",
            "fundamental_quality_prev",
            "fundamental_quality_trend_symbol",
            "fundamental_quality_trend_direction",
        ),
        (
            "fundamental_risk",
            "fundamental_risk",
            "fundamental_risk_prev",
            "fundamental_risk_trend_symbol",
            "fundamental_risk_trend_direction",
        ),
    ]:
        current_values = pd.to_numeric(merged.get(current_col), errors="coerce")
        previous_values = pd.to_numeric(merged.get(prev_col), errors="coerce")
        deltas = current_values - previous_values
        merged[symbol_col] = [
            _trend_symbol_for_delta(delta_value, threshold) for delta_value in deltas.tolist()
        ]
        merged[direction_col] = [
            _trend_direction_for_delta(delta_value, threshold) for delta_value in deltas.tolist()
        ]
    return merged.drop(
        columns=[
            "fundamental_total_score_prev",
            "fundamental_momentum_prev",
            "fundamental_growth_prev",
            "fundamental_value_prev",
            "fundamental_quality_prev",
            "fundamental_risk_prev",
        ]
    )

def _build_company_filter_state(
    *,
    thematics: Optional[list[str]] = None,
    sectors: Optional[list[str]] = None,
    industries: Optional[list[str]] = None,
    caps: Optional[list[str]] = None,
    fund_range: tuple[float, float] = (0.0, 100.0),
    tech_range: tuple[float, float] = (0.0, 100.0),
    rsi_regime_range: tuple[float, float] = (0.0, 100.0),
    sector_regime_fit_range: tuple[float, float] = (0.0, 100.0),
    fund_momentum_range: tuple[float, float] = (0.0, 100.0),
    tech_trend_dir: str = "All",
    daily_rsi_divergence: str = "All",
    weekly_rsi_divergence: str = "All",
    short_term_flow: str = "All",
    rel_strength: str = "All",
    rel_volume: str = "All",
    ai_revenue_exposure: str = "All",
    ai_disruption_risk: str = "All",
    beta_range: tuple[float, float] = (0.0, 5.0),
    ticker: str = "",
) -> dict[str, object]:
    return {
        "thematic": list(thematics or []),
        "sector": list(sectors or []),
        "industry": list(industries or []),
        "cap": list(caps or []),
        "fund_range": tuple(fund_range),
        "tech_range": tuple(tech_range),
        "rsi_regime_range": tuple(rsi_regime_range),
        "sector_regime_fit_range": tuple(sector_regime_fit_range),
        "fund_momentum_range": tuple(fund_momentum_range),
        "tech_trend_dir": tech_trend_dir,
        "daily_rsi_divergence": daily_rsi_divergence,
        "weekly_rsi_divergence": weekly_rsi_divergence,
        "short_term_flow": short_term_flow,
        "rel_strength": rel_strength,
        "rel_volume": rel_volume,
        "ai_revenue_exposure": ai_revenue_exposure,
        "ai_disruption_risk": ai_disruption_risk,
        "beta_range": tuple(beta_range),
        "ticker": ticker,
    }


def _build_all_companies_filter_state() -> dict[str, object]:
    return _build_company_filter_state()


def _load_company_filter_default_state(prefix: str) -> dict[str, object]:
    return _build_company_filter_state(
        thematics=st.session_state.get(f"{prefix}_drilldown_filter_default_thematic", []),
        sectors=st.session_state.get(f"{prefix}_drilldown_filter_default_sector", []),
        industries=st.session_state.get(f"{prefix}_drilldown_filter_default_industry", []),
        caps=st.session_state.get(f"{prefix}_drilldown_filter_default_cap", []),
        fund_range=tuple(st.session_state.get(f"{prefix}_drilldown_filter_default_fund_range", (0.0, 100.0))),
        tech_range=tuple(st.session_state.get(f"{prefix}_drilldown_filter_default_tech_range", (0.0, 100.0))),
        rsi_regime_range=tuple(
            st.session_state.get(f"{prefix}_drilldown_filter_default_rsi_regime_range", (0.0, 100.0))
        ),
        sector_regime_fit_range=tuple(
            st.session_state.get(f"{prefix}_drilldown_filter_default_sector_regime_fit_range", (0.0, 100.0))
        ),
        fund_momentum_range=tuple(
            st.session_state.get(f"{prefix}_drilldown_filter_default_fund_momentum_range", (0.0, 100.0))
        ),
        tech_trend_dir=st.session_state.get(f"{prefix}_drilldown_filter_default_tech_trend_dir", "All"),
        daily_rsi_divergence=st.session_state.get(
            f"{prefix}_drilldown_filter_default_daily_rsi_divergence",
            "All",
        ),
        weekly_rsi_divergence=st.session_state.get(
            f"{prefix}_drilldown_filter_default_weekly_rsi_divergence",
            "All",
        ),
        short_term_flow=st.session_state.get(f"{prefix}_drilldown_filter_default_short_term_flow", "All"),
        rel_strength=st.session_state.get(f"{prefix}_drilldown_filter_default_rel_strength", "All"),
        rel_volume=st.session_state.get(f"{prefix}_drilldown_filter_default_rel_volume", "All"),
        ai_revenue_exposure=st.session_state.get(
            f"{prefix}_drilldown_filter_default_ai_revenue_exposure",
            "All",
        ),
        ai_disruption_risk=st.session_state.get(
            f"{prefix}_drilldown_filter_default_ai_disruption_risk",
            "All",
        ),
        beta_range=tuple(st.session_state.get(f"{prefix}_drilldown_filter_default_beta_range", (0.0, 5.0))),
        ticker="",
    )


def _apply_company_filter_state(prefix: str, state: dict[str, object]) -> None:
    for suffix, value in state.items():
        key = f"{prefix}_drilldown_filter_{suffix}"
        if isinstance(value, list):
            st.session_state[key] = list(value)
        elif isinstance(value, tuple):
            st.session_state[key] = tuple(value)
        else:
            st.session_state[key] = value


def _filter_company_grid_by_ticker_list(company_df: pd.DataFrame, raw_ticker_query: str) -> pd.DataFrame:
    normalized_tickers = parse_manual_price_tickers(raw_ticker_query)
    if not normalized_tickers or "ticker" not in company_df.columns:
        return company_df.copy()
    normalized_query_set = set(normalized_tickers)
    normalized_company_tickers = company_df["ticker"].map(normalize_price_ticker)
    return company_df[normalized_company_tickers.isin(normalized_query_set)].copy()


def _sync_drilldown_filter_defaults(
    prefix: str,
    signature: tuple[str, ...],
    *,
    default_sectors: list[str],
    default_industries: list[str],
    default_thematics: Optional[list[str]] = None,
    default_cap_buckets: Optional[list[str]] = None,
    default_fund_range: tuple[float, float] = (0.0, 100.0),
    default_tech_range: tuple[float, float] = (0.0, 100.0),
    default_rsi_regime_range: tuple[float, float] = (0.0, 100.0),
    default_sector_regime_fit_range: tuple[float, float] = (0.0, 100.0),
    default_fund_momentum_range: tuple[float, float] = (0.0, 100.0),
    default_tech_trend_dir: str = "All",
    default_daily_rsi_divergence: str = "All",
    default_weekly_rsi_divergence: str = "All",
    default_short_term_flow: str = "All",
    default_rel_strength: str = "All",
    default_rel_volume: str = "All",
    default_ai_revenue_exposure: str = "All",
    default_ai_disruption_risk: str = "All",
    default_beta_range: tuple[float, float] = (0.0, 5.0),
) -> None:
    signature_key = f"{prefix}_drilldown_filter_signature"
    if st.session_state.get(signature_key) == signature:
        return
    default_thematics_list = list(default_thematics or [])
    default_cap_list = list(default_cap_buckets or [])
    st.session_state[f"{prefix}_drilldown_filter_default_thematic"] = default_thematics_list
    st.session_state[f"{prefix}_drilldown_filter_default_sector"] = list(default_sectors)
    st.session_state[f"{prefix}_drilldown_filter_default_industry"] = list(default_industries)
    st.session_state[f"{prefix}_drilldown_filter_default_cap"] = default_cap_list
    st.session_state[f"{prefix}_drilldown_filter_default_fund_range"] = tuple(default_fund_range)
    st.session_state[f"{prefix}_drilldown_filter_default_tech_range"] = tuple(default_tech_range)
    st.session_state[f"{prefix}_drilldown_filter_default_rsi_regime_range"] = tuple(default_rsi_regime_range)
    st.session_state[f"{prefix}_drilldown_filter_default_sector_regime_fit_range"] = tuple(default_sector_regime_fit_range)
    st.session_state[f"{prefix}_drilldown_filter_default_fund_momentum_range"] = tuple(default_fund_momentum_range)
    st.session_state[f"{prefix}_drilldown_filter_default_tech_trend_dir"] = default_tech_trend_dir
    st.session_state[f"{prefix}_drilldown_filter_default_daily_rsi_divergence"] = default_daily_rsi_divergence
    st.session_state[f"{prefix}_drilldown_filter_default_weekly_rsi_divergence"] = default_weekly_rsi_divergence
    st.session_state[f"{prefix}_drilldown_filter_default_short_term_flow"] = default_short_term_flow
    st.session_state[f"{prefix}_drilldown_filter_default_rel_strength"] = default_rel_strength
    st.session_state[f"{prefix}_drilldown_filter_default_rel_volume"] = default_rel_volume
    st.session_state[f"{prefix}_drilldown_filter_default_ai_revenue_exposure"] = default_ai_revenue_exposure
    st.session_state[f"{prefix}_drilldown_filter_default_ai_disruption_risk"] = default_ai_disruption_risk
    st.session_state[f"{prefix}_drilldown_filter_default_beta_range"] = tuple(default_beta_range)
    st.session_state[f"{prefix}_drilldown_filter_thematic"] = default_thematics_list
    st.session_state[f"{prefix}_drilldown_filter_sector"] = list(default_sectors)
    st.session_state[f"{prefix}_drilldown_filter_industry"] = list(default_industries)
    st.session_state[f"{prefix}_drilldown_filter_cap"] = default_cap_list
    st.session_state[f"{prefix}_drilldown_filter_fund_range"] = tuple(default_fund_range)
    st.session_state[f"{prefix}_drilldown_filter_tech_range"] = tuple(default_tech_range)
    st.session_state[f"{prefix}_drilldown_filter_rsi_regime_range"] = tuple(default_rsi_regime_range)
    st.session_state[f"{prefix}_drilldown_filter_sector_regime_fit_range"] = tuple(default_sector_regime_fit_range)
    st.session_state[f"{prefix}_drilldown_filter_fund_momentum_range"] = tuple(default_fund_momentum_range)
    st.session_state[f"{prefix}_drilldown_filter_tech_trend_dir"] = default_tech_trend_dir
    st.session_state[f"{prefix}_drilldown_filter_daily_rsi_divergence"] = default_daily_rsi_divergence
    st.session_state[f"{prefix}_drilldown_filter_weekly_rsi_divergence"] = default_weekly_rsi_divergence
    st.session_state[f"{prefix}_drilldown_filter_short_term_flow"] = default_short_term_flow
    st.session_state[f"{prefix}_drilldown_filter_rel_strength"] = default_rel_strength
    st.session_state[f"{prefix}_drilldown_filter_rel_volume"] = default_rel_volume
    st.session_state[f"{prefix}_drilldown_filter_ai_revenue_exposure"] = default_ai_revenue_exposure
    st.session_state[f"{prefix}_drilldown_filter_ai_disruption_risk"] = default_ai_disruption_risk
    st.session_state[f"{prefix}_drilldown_filter_beta_range"] = tuple(default_beta_range)
    st.session_state[f"{prefix}_drilldown_filter_ticker"] = ""
    st.session_state[signature_key] = signature


def _bind_ctrl_f_to_ticker_filter(input_label: str, scope_key: str) -> None:
    html(
        f"""
<script>
(function() {{
  const scopeKey = {json.dumps(scope_key)};
  const targetLabel = {json.dumps(input_label)};
  try {{
    const parentWindow = window.parent;
    const parentDoc = parentWindow.document;
    parentWindow.__equipilotCtrlFHandlers = parentWindow.__equipilotCtrlFHandlers || {{}};
    if (parentWindow.__equipilotCtrlFHandlers[scopeKey]) {{
      return;
    }}
    const isVisible = (el) => Boolean(el && el.offsetParent !== null);
    const resolveTarget = () => {{
      const allInputs = parentDoc.querySelectorAll("input[aria-label]");
      for (const input of allInputs) {{
        if ((input.getAttribute("aria-label") || "").trim() === targetLabel && isVisible(input)) {{
          return input;
        }}
      }}
      return null;
    }};
    const handler = (event) => {{
      const key = String(event.key || "").toLowerCase();
      if (!(event.ctrlKey || event.metaKey) || key !== "f") {{
        return;
      }}
      const target = resolveTarget();
      if (!target) {{
        return;
      }}
      event.preventDefault();
      target.focus();
      target.select();
    }};
    parentDoc.addEventListener("keydown", handler, true);
    parentWindow.__equipilotCtrlFHandlers[scopeKey] = handler;
  }} catch (err) {{
    // Manual focus still works if parent document access is restricted.
  }}
}})();
</script>
        """,
        height=0,
    )


def render_company_drilldown_filters(
    company_df: pd.DataFrame,
    *,
    prefix: str,
    ticker_label: str,
    include_fundamental_momentum_filter: bool = False,
    include_technical_trend_filter: bool = False,
    include_thematic_filter: bool = False,
    include_rel_strength_filter: bool = False,
    include_rel_volume_filter: bool = False,
    include_ai_exposure_filters: bool = False,
    include_beta_filter: bool = False,
) -> pd.DataFrame:
    def _is_full_numeric_range(
        range_value: tuple[float, float],
        *,
        min_value: float,
        max_value: float,
    ) -> bool:
        return (
            abs(float(range_value[0]) - float(min_value)) < 1e-9
            and abs(float(range_value[1]) - float(max_value)) < 1e-9
        )

    thematic_key = f"{prefix}_drilldown_filter_thematic"
    sector_key = f"{prefix}_drilldown_filter_sector"
    industry_key = f"{prefix}_drilldown_filter_industry"
    cap_key = f"{prefix}_drilldown_filter_cap"
    fund_range_key = f"{prefix}_drilldown_filter_fund_range"
    tech_range_key = f"{prefix}_drilldown_filter_tech_range"
    rsi_regime_range_key = f"{prefix}_drilldown_filter_rsi_regime_range"
    sector_regime_fit_range_key = f"{prefix}_drilldown_filter_sector_regime_fit_range"
    fund_momentum_range_key = f"{prefix}_drilldown_filter_fund_momentum_range"
    tech_trend_dir_key = f"{prefix}_drilldown_filter_tech_trend_dir"
    daily_rsi_divergence_key = f"{prefix}_drilldown_filter_daily_rsi_divergence"
    weekly_rsi_divergence_key = f"{prefix}_drilldown_filter_weekly_rsi_divergence"
    short_term_flow_key = f"{prefix}_drilldown_filter_short_term_flow"
    rel_strength_key = f"{prefix}_drilldown_filter_rel_strength"
    rel_volume_key = f"{prefix}_drilldown_filter_rel_volume"
    ai_revenue_exposure_key = f"{prefix}_drilldown_filter_ai_revenue_exposure"
    ai_disruption_risk_key = f"{prefix}_drilldown_filter_ai_disruption_risk"
    beta_range_key = f"{prefix}_drilldown_filter_beta_range"
    ticker_key = f"{prefix}_drilldown_filter_ticker"
    pending_reset_key = f"{prefix}_drilldown_filter_pending_reset"
    default_thematic_key = f"{prefix}_drilldown_filter_default_thematic"
    default_sector_key = f"{prefix}_drilldown_filter_default_sector"
    default_industry_key = f"{prefix}_drilldown_filter_default_industry"
    default_cap_key = f"{prefix}_drilldown_filter_default_cap"
    default_fund_range_key = f"{prefix}_drilldown_filter_default_fund_range"
    default_tech_range_key = f"{prefix}_drilldown_filter_default_tech_range"
    default_rsi_regime_range_key = f"{prefix}_drilldown_filter_default_rsi_regime_range"
    default_sector_regime_fit_range_key = f"{prefix}_drilldown_filter_default_sector_regime_fit_range"
    default_fund_momentum_range_key = f"{prefix}_drilldown_filter_default_fund_momentum_range"
    default_tech_trend_dir_key = f"{prefix}_drilldown_filter_default_tech_trend_dir"
    default_daily_rsi_divergence_key = f"{prefix}_drilldown_filter_default_daily_rsi_divergence"
    default_weekly_rsi_divergence_key = f"{prefix}_drilldown_filter_default_weekly_rsi_divergence"
    default_short_term_flow_key = f"{prefix}_drilldown_filter_default_short_term_flow"
    default_rel_strength_key = f"{prefix}_drilldown_filter_default_rel_strength"
    default_rel_volume_key = f"{prefix}_drilldown_filter_default_rel_volume"
    default_ai_revenue_exposure_key = f"{prefix}_drilldown_filter_default_ai_revenue_exposure"
    default_ai_disruption_risk_key = f"{prefix}_drilldown_filter_default_ai_disruption_risk"
    default_beta_range_key = f"{prefix}_drilldown_filter_default_beta_range"

    # Streamlit forbids changing widget-bound session keys after widget instantiation.
    # Apply reset defaults at the top of a rerun before creating widgets.
    pending_filter_action = str(st.session_state.pop(pending_reset_key, "") or "")
    if pending_filter_action == "reset":
        _apply_company_filter_state(prefix, _load_company_filter_default_state(prefix))
    elif pending_filter_action == "clear":
        _apply_company_filter_state(prefix, _build_all_companies_filter_state())

    st.session_state.setdefault(
        thematic_key, list(st.session_state.get(default_thematic_key, []))
    )
    st.session_state.setdefault(
        sector_key, list(st.session_state.get(default_sector_key, []))
    )
    st.session_state.setdefault(
        industry_key, list(st.session_state.get(default_industry_key, []))
    )
    st.session_state.setdefault(cap_key, list(st.session_state.get(default_cap_key, [])))
    st.session_state.setdefault(fund_range_key, tuple(st.session_state.get(default_fund_range_key, (0.0, 100.0))))
    st.session_state.setdefault(tech_range_key, tuple(st.session_state.get(default_tech_range_key, (0.0, 100.0))))
    st.session_state.setdefault(
        rsi_regime_range_key,
        tuple(st.session_state.get(default_rsi_regime_range_key, (0.0, 100.0))),
    )
    st.session_state.setdefault(
        sector_regime_fit_range_key,
        tuple(st.session_state.get(default_sector_regime_fit_range_key, (0.0, 100.0))),
    )
    st.session_state.setdefault(
        fund_momentum_range_key,
        tuple(st.session_state.get(default_fund_momentum_range_key, (0.0, 100.0))),
    )
    st.session_state.setdefault(
        tech_trend_dir_key,
        st.session_state.get(default_tech_trend_dir_key, "All"),
    )
    st.session_state.setdefault(
        daily_rsi_divergence_key,
        st.session_state.get(default_daily_rsi_divergence_key, "All"),
    )
    st.session_state.setdefault(
        weekly_rsi_divergence_key,
        st.session_state.get(default_weekly_rsi_divergence_key, "All"),
    )
    st.session_state.setdefault(
        short_term_flow_key,
        st.session_state.get(default_short_term_flow_key, "All"),
    )
    st.session_state.setdefault(
        rel_strength_key,
        st.session_state.get(default_rel_strength_key, "All"),
    )
    st.session_state.setdefault(
        rel_volume_key,
        st.session_state.get(default_rel_volume_key, "All"),
    )
    st.session_state.setdefault(
        ai_revenue_exposure_key,
        st.session_state.get(default_ai_revenue_exposure_key, "All"),
    )
    st.session_state.setdefault(
        ai_disruption_risk_key,
        st.session_state.get(default_ai_disruption_risk_key, "All"),
    )
    st.session_state.setdefault(
        beta_range_key,
        tuple(st.session_state.get(default_beta_range_key, (0.0, 5.0))),
    )
    st.session_state.setdefault(ticker_key, "")
    st.markdown("**Company filters**")

    with st.form(key=f"{prefix}_drilldown_filters_form", clear_on_submit=False):
        top_layout: list[float] = []
        if include_thematic_filter:
            top_layout.append(1.1)
        top_layout.extend([1, 1, 1])
        filter_cols_top = st.columns(top_layout)
        next_top_col = 0
        selected_thematics: list[str] = []
        if include_thematic_filter:
            with filter_cols_top[next_top_col]:
                thematic_options = sorted(
                    {
                        thematic_name
                        for memberships in company_df.get("thematic_memberships", pd.Series([], dtype=object)).tolist()
                        if isinstance(memberships, list)
                        for thematic_name in memberships
                    }
                )
                selected_thematics = st.multiselect("Thematic filter", options=thematic_options, key=thematic_key)
            next_top_col += 1
        with filter_cols_top[next_top_col]:
            sector_options = sorted(company_df["sector"].dropna().unique().tolist())
            selected_sectors = st.multiselect("Sector filter", options=sector_options, key=sector_key)
        next_top_col += 1
        with filter_cols_top[next_top_col]:
            if selected_sectors:
                industry_source = company_df[company_df["sector"].isin(selected_sectors)]
            else:
                industry_source = company_df
            industry_options = sorted(industry_source["industry"].dropna().unique().tolist())
            selected_industries_state = st.session_state.get(industry_key, [])
            selected_industries_state = [entry for entry in selected_industries_state if entry in industry_options]
            if st.session_state.get(industry_key, []) != selected_industries_state:
                st.session_state[industry_key] = selected_industries_state
            selected_industries = st.multiselect("Industry filter", options=industry_options, key=industry_key)
        next_top_col += 1
        with filter_cols_top[next_top_col]:
            cap_options = list(CAP_BUCKET_ORDER)
            selected_caps = st.multiselect("Market cap bucket", options=cap_options, key=cap_key)

        filter_cols_middle = st.columns([1, 1, 1, 1])
        trend_filter_value = "All"
        with filter_cols_middle[0]:
            fundamental_range = st.slider(
                "Fundamental score range",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                key=fund_range_key,
            )
        with filter_cols_middle[1]:
            technical_range = st.slider(
                "Technical score range",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                key=tech_range_key,
            )
        with filter_cols_middle[2]:
            rsi_regime_range = st.slider(
                "RSI Regime Score range",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                key=rsi_regime_range_key,
            )
        with filter_cols_middle[3]:
            sector_regime_fit_range = st.slider(
                "Sector Regime Fit range",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                key=sector_regime_fit_range_key,
            )

        detail_layout: list[float] = []
        if include_fundamental_momentum_filter:
            detail_layout.append(1)
        if include_technical_trend_filter:
            detail_layout.append(1)
        if include_beta_filter:
            detail_layout.append(1)
        filter_cols_detail = st.columns(detail_layout) if detail_layout else []
        next_detail_col = 0
        if include_fundamental_momentum_filter:
            with filter_cols_detail[next_detail_col]:
                fundamental_momentum_range = st.slider(
                    "Fundamental momentum range",
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    key=fund_momentum_range_key,
                )
            next_detail_col += 1
        else:
            fundamental_momentum_range = (0.0, 100.0)
        if include_technical_trend_filter:
            with filter_cols_detail[next_detail_col]:
                trend_filter_value = st.selectbox(
                    "Technical trend filter",
                    options=TREND_FILTER_OPTIONS,
                    key=tech_trend_dir_key,
                )
            next_detail_col += 1
        if include_beta_filter:
            with filter_cols_detail[next_detail_col]:
                beta_range = st.slider(
                    "Beta range",
                    min_value=0.0,
                    max_value=5.0,
                    step=0.1,
                    key=beta_range_key,
                )
        else:
            beta_range = (0.0, 5.0)

        bottom_layout: list[float] = [1, 1, 1]
        if include_rel_strength_filter:
            bottom_layout.append(1)
        if include_rel_volume_filter:
            bottom_layout.append(1)
        if include_ai_exposure_filters:
            bottom_layout.extend([1, 1])
        bottom_layout.append(1.4)
        filter_cols_bottom = st.columns(bottom_layout)
        next_bottom_col = 0
        daily_rsi_divergence_value = "All"
        weekly_rsi_divergence_value = "All"
        short_term_flow_value = "All"
        rel_strength_value = "All"
        rel_volume_value = "All"
        ai_revenue_exposure_value = "All"
        ai_disruption_risk_value = "All"
        with filter_cols_bottom[next_bottom_col]:
            daily_rsi_divergence_value = st.selectbox(
                "Daily RSI Divergence",
                options=DIVERGENCE_FILTER_OPTIONS,
                key=daily_rsi_divergence_key,
            )
        next_bottom_col += 1
        with filter_cols_bottom[next_bottom_col]:
            weekly_rsi_divergence_value = st.selectbox(
                "Weekly RSI Divergence",
                options=DIVERGENCE_FILTER_OPTIONS,
                key=weekly_rsi_divergence_key,
            )
        next_bottom_col += 1
        with filter_cols_bottom[next_bottom_col]:
            short_term_flow_value = st.selectbox(
                "Short Term Flow",
                options=SHORT_TERM_FLOW_FILTER_OPTIONS,
                key=short_term_flow_key,
            )
        next_bottom_col += 1
        if include_rel_strength_filter:
            with filter_cols_bottom[next_bottom_col]:
                rel_strength_value = st.selectbox(
                    "Rel Strength",
                    options=SIGN_FILTER_OPTIONS,
                    key=rel_strength_key,
                )
            next_bottom_col += 1
        if include_rel_volume_filter:
            with filter_cols_bottom[next_bottom_col]:
                rel_volume_value = st.selectbox(
                    "Rel Volume",
                    options=SIGN_FILTER_OPTIONS,
                    key=rel_volume_key,
                )
            next_bottom_col += 1
        if include_ai_exposure_filters:
            with filter_cols_bottom[next_bottom_col]:
                ai_revenue_exposure_value = st.selectbox(
                    "AI revenue exposure",
                    options=AI_REVENUE_FILTER_OPTIONS,
                    key=ai_revenue_exposure_key,
                )
            next_bottom_col += 1
            with filter_cols_bottom[next_bottom_col]:
                ai_disruption_risk_value = st.selectbox(
                    "AI disruption risk",
                    options=AI_DISRUPTION_FILTER_OPTIONS,
                    key=ai_disruption_risk_key,
                )
            next_bottom_col += 1
        with filter_cols_bottom[next_bottom_col]:
            ticker_query = st.text_input(
                ticker_label,
                key=ticker_key,
                placeholder="Type one or more tickers, e.g. MSFT, AMN, GOOG",
            ).strip()

        action_cols = st.columns([1, 1, 1, 2])
        with action_cols[0]:
            apply_clicked = st.form_submit_button("Apply filters", use_container_width=True)
        with action_cols[1]:
            reset_clicked = st.form_submit_button("Reset filters", use_container_width=True)
        with action_cols[2]:
            clear_clicked = st.form_submit_button("Remove filters", use_container_width=True)
        if not apply_clicked and not reset_clicked and not clear_clicked:
            st.caption("Adjust filters, then click Apply filters.")

    _bind_ctrl_f_to_ticker_filter(ticker_label, f"{prefix}_drilldown_ctrlf")

    if reset_clicked:
        st.session_state[pending_reset_key] = "reset"
        st.rerun()

    if clear_clicked:
        st.session_state[pending_reset_key] = "clear"
        st.rerun()

    filtered = company_df.copy()
    if include_thematic_filter and selected_thematics:
        filtered = filtered[
            filtered["thematic_memberships"].apply(
                lambda memberships: bool(set(selected_thematics).intersection(memberships or []))
            )
        ]
    if selected_sectors:
        filtered = filtered[filtered["sector"].isin(selected_sectors)]
    if selected_industries:
        filtered = filtered[filtered["industry"].isin(selected_industries)]
    if selected_caps:
        filtered = filtered[filtered["market_cap_bucket"].isin(selected_caps)]

    fund_min, fund_max = fundamental_range
    tech_min, tech_max = technical_range
    if not _is_full_numeric_range(tuple(fundamental_range), min_value=0.0, max_value=100.0):
        filtered = filtered[filtered["fundamental_total_score"].between(fund_min, fund_max, inclusive="both")]
    if not _is_full_numeric_range(tuple(technical_range), min_value=0.0, max_value=100.0):
        filtered = filtered[filtered["general_technical_score"].between(tech_min, tech_max, inclusive="both")]
    filtered, _ = _filter_by_optional_numeric_range(
        filtered,
        column="stock_rsi_regime_score",
        range_value=tuple(rsi_regime_range),
    )
    filtered, _ = _filter_by_optional_numeric_range(
        filtered,
        column="sector_regime_fit_score",
        range_value=tuple(sector_regime_fit_range),
    )
    filtered, _ = _filter_by_optional_label_value(
        filtered,
        column="rsi_divergence_daily_flag",
        selected_value=daily_rsi_divergence_value,
        label_map={"Positive": "positive", "Negative": "negative", "None": "none"},
    )
    filtered, _ = _filter_by_optional_label_value(
        filtered,
        column="rsi_divergence_weekly_flag",
        selected_value=weekly_rsi_divergence_value,
        label_map={"Positive": "positive", "Negative": "negative", "None": "none"},
    )
    filtered, _ = _filter_by_optional_label_value(
        filtered,
        column="short_term_flow",
        selected_value=short_term_flow_value,
        label_map={"Positive": "positive", "Negative": "negative", "Neutral": "neutral"},
    )
    if include_fundamental_momentum_filter and "fundamental_momentum" in filtered.columns:
        if (
            filtered["fundamental_momentum"].notna().any()
            and not _is_full_numeric_range(tuple(fundamental_momentum_range), min_value=0.0, max_value=100.0)
        ):
            momentum_min, momentum_max = fundamental_momentum_range
            filtered = filtered[
                filtered["fundamental_momentum"].between(momentum_min, momentum_max, inclusive="both")
            ]
    if include_technical_trend_filter and "technical_trend_direction" in filtered.columns:
        if trend_filter_value == TREND_FILTER_LABELS["up"]:
            filtered = filtered[filtered["technical_trend_direction"] == "up"]
        elif trend_filter_value == TREND_FILTER_LABELS["flat"]:
            filtered = filtered[filtered["technical_trend_direction"] == "flat"]
        elif trend_filter_value == TREND_FILTER_LABELS["down"]:
            filtered = filtered[filtered["technical_trend_direction"] == "down"]
    if include_beta_filter and "beta" in filtered.columns:
        beta_min, beta_max = beta_range
        if filtered["beta"].notna().any() and not _is_full_numeric_range(tuple(beta_range), min_value=0.0, max_value=5.0):
            filtered = filtered[filtered["beta"].between(beta_min, beta_max, inclusive="both")]
    if include_rel_strength_filter and rel_strength_value != "All" and "rel_strength" in filtered.columns:
        filtered = filtered[filtered["rel_strength"] == rel_strength_value]
    if include_rel_volume_filter and rel_volume_value != "All" and "rel_volume" in filtered.columns:
        filtered = filtered[filtered["rel_volume"] == rel_volume_value]
    if include_ai_exposure_filters and ai_revenue_exposure_value != "All" and "ai_revenue_exposure" in filtered.columns:
        filtered = filtered[filtered["ai_revenue_exposure"] == ai_revenue_exposure_value]
    if include_ai_exposure_filters and ai_disruption_risk_value != "All" and "ai_disruption_risk" in filtered.columns:
        filtered = filtered[filtered["ai_disruption_risk"] == ai_disruption_risk_value]

    if ticker_query:
        filtered = _filter_company_grid_by_ticker_list(filtered, ticker_query)
    return filtered.copy()


def format_company_drilldown_display(company_df: pd.DataFrame, *, sort_by: str) -> pd.DataFrame:
    if company_df.empty:
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Market Cap",
                "Fundamental Score",
                "Fundamental Momentum",
                "Technical Score",
                "RSI Regime Score",
                "Sector Regime Fit",
                "Short Term Flow",
                "RSI Divergence (D)",
                "RSI Divergence (W)",
                "Rel Strength",
                "Rel Volume",
                "AI Revenue Exposure",
                "AI Disruption Risk",
            ]
        )

    if sort_by == "fundamental":
        sort_columns = ["fundamental_total_score", "general_technical_score"]
    else:
        sort_columns = ["general_technical_score", "fundamental_total_score"]
    sorted_df = company_df.sort_values(by=sort_columns, ascending=[False, False], na_position="last").copy()
    if "company" in sorted_df.columns:
        sorted_df["company"] = sorted_df["company"].fillna("").astype(str).str.strip()
        sorted_df["company"] = sorted_df["company"].where(sorted_df["company"].str.len() > 0, sorted_df["ticker"])
    else:
        sorted_df["company"] = sorted_df["ticker"]
    if "fundamental_momentum" not in sorted_df.columns:
        sorted_df["fundamental_momentum"] = np.nan
    if "rel_strength" not in sorted_df.columns:
        sorted_df["rel_strength"] = "N/A"
    if "rel_volume" not in sorted_df.columns:
        sorted_df["rel_volume"] = "N/A"
    if "stock_rsi_regime_score" not in sorted_df.columns:
        sorted_df["stock_rsi_regime_score"] = np.nan
    if "sector_regime_fit_score" not in sorted_df.columns:
        sorted_df["sector_regime_fit_score"] = np.nan
    if "rsi_divergence_daily_flag" not in sorted_df.columns:
        sorted_df["rsi_divergence_daily_flag"] = pd.NA
    if "rsi_divergence_weekly_flag" not in sorted_df.columns:
        sorted_df["rsi_divergence_weekly_flag"] = pd.NA
    if "short_term_flow" not in sorted_df.columns:
        sorted_df["short_term_flow"] = pd.NA
    if "ai_revenue_exposure" not in sorted_df.columns:
        sorted_df["ai_revenue_exposure"] = "none"
    if "ai_disruption_risk" not in sorted_df.columns:
        sorted_df["ai_disruption_risk"] = "none"

    display_columns = [
        "ticker",
        "company",
        "sector",
        "industry",
        "market_cap",
        "fundamental_total_score",
        "fundamental_momentum",
        "general_technical_score",
        "stock_rsi_regime_score",
        "sector_regime_fit_score",
        "short_term_flow",
        "rsi_divergence_daily_flag",
        "rsi_divergence_weekly_flag",
        "rel_strength",
        "rel_volume",
        "ai_revenue_exposure",
        "ai_disruption_risk",
    ]
    rename_map = {
        "ticker": "Ticker",
        "company": "Company",
        "sector": "Sector",
        "industry": "Industry",
        "market_cap": "Market Cap",
        "fundamental_total_score": "Fundamental Score",
        "fundamental_momentum": "Fundamental Momentum",
        "general_technical_score": "Technical Score",
        "stock_rsi_regime_score": "RSI Regime Score",
        "sector_regime_fit_score": "Sector Regime Fit",
        "short_term_flow": "Short Term Flow",
        "rsi_divergence_daily_flag": "RSI Divergence (D)",
        "rsi_divergence_weekly_flag": "RSI Divergence (W)",
        "rel_strength": "Rel Strength",
        "rel_volume": "Rel Volume",
        "ai_revenue_exposure": "AI Revenue Exposure",
        "ai_disruption_risk": "AI Disruption Risk",
    }

    display_df = sorted_df[display_columns].rename(columns=rename_map)
    display_df["Market Cap"] = display_df["Market Cap"].map(_format_market_cap_display)
    display_df["Fundamental Score"] = display_df["Fundamental Score"].map(_format_numeric_value)
    display_df["Fundamental Momentum"] = display_df["Fundamental Momentum"].map(_format_numeric_value)
    technical_trend_symbols = (
        sorted_df.get("technical_trend_symbol", pd.Series(index=sorted_df.index, dtype=object))
        .fillna("")
        .astype(str)
        .tolist()
    )
    rendered_technical_scores: list[str] = []
    for numeric_value, trend_symbol in zip(display_df["Technical Score"].tolist(), technical_trend_symbols):
        if trend_symbol:
            rendered_technical_scores.append(_render_thematics_score_with_symbol(numeric_value, trend_symbol))
        else:
            rendered_technical_scores.append(_format_numeric_value(numeric_value))
    display_df["Technical Score"] = rendered_technical_scores
    display_df["RSI Regime Score"] = display_df["RSI Regime Score"].map(_format_numeric_value)
    display_df["Sector Regime Fit"] = display_df["Sector Regime Fit"].map(_format_numeric_value)
    display_df["Short Term Flow"] = display_df["Short Term Flow"].map(_format_short_term_flow_flag)
    display_df["RSI Divergence (D)"] = display_df["RSI Divergence (D)"].map(_format_divergence_flag)
    display_df["RSI Divergence (W)"] = display_df["RSI Divergence (W)"].map(_format_divergence_flag)
    display_df["Rel Strength"] = display_df["Rel Strength"].fillna("N/A")
    display_df["Rel Volume"] = display_df["Rel Volume"].fillna("N/A")
    display_df["AI Revenue Exposure"] = display_df["AI Revenue Exposure"].fillna("none")
    display_df["AI Disruption Risk"] = display_df["AI Disruption Risk"].fillna("none")
    return display_df.reset_index(drop=True)


def _style_ai_exposure_value(value: object) -> str:
    text = str(value).strip().lower()
    if text == "direct":
        return "color:#15803D; font-weight:700;"
    if text == "indirect":
        return "color:#B45309; font-weight:700;"
    if text == "none":
        return "color:#7B8BA0;"
    return "color:#334E68;"


def _style_ai_risk_value(value: object) -> str:
    text = str(value).strip().lower()
    if text == "high":
        return "color:#B42318; font-weight:700;"
    if text == "medium":
        return "color:#B45309; font-weight:700;"
    if text == "low":
        return "color:#15803D; font-weight:700;"
    if text == "none":
        return "color:#7B8BA0;"
    return "color:#334E68;"


def _build_company_drilldown_styler(display_df: pd.DataFrame) -> "Styler":
    styler = display_df.style.hide(axis="index")
    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#F4F8FC"),
                    ("color", "#173A5E"),
                    ("font-weight", "800"),
                    ("text-align", "center"),
                    ("border-bottom", "1px solid #D9E6F2"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid #E7EFF7"),
                    ("vertical-align", "middle"),
                    ("color", "#334E68"),
                ],
            },
        ],
        overwrite=False,
    )

    left_columns = [
        "Ticker",
        "Company",
        "Sector",
        "Industry",
        "Short Term Flow",
        "Rel Strength",
        "Rel Volume",
        "AI Revenue Exposure",
        "AI Disruption Risk",
    ]
    centered_columns = [column for column in display_df.columns if column not in left_columns]
    usable_left_columns = [column for column in left_columns if column in display_df.columns]
    if usable_left_columns:
        styler = styler.set_properties(subset=usable_left_columns, **{"text-align": "left"})
    if centered_columns:
        styler = styler.set_properties(subset=centered_columns, **{"text-align": "center"})

    score_columns = [
        "Fundamental Score",
        "Fundamental Momentum",
        "Technical Score",
        "RSI Regime Score",
        "Sector Regime Fit",
    ]
    usable_score_columns = [column for column in score_columns if column in display_df.columns]
    if usable_score_columns:
        styler = styler.applymap(_score_color_css, subset=usable_score_columns)
    usable_sign_columns = [
        column for column in ["Short Term Flow", "Rel Strength", "Rel Volume"] if column in display_df.columns
    ]
    if usable_sign_columns:
        styler = styler.applymap(_style_sign_label_value, subset=usable_sign_columns)
    usable_divergence_columns = [
        column for column in ["RSI Divergence (D)", "RSI Divergence (W)"] if column in display_df.columns
    ]
    if usable_divergence_columns:
        styler = styler.applymap(_style_sign_label_value, subset=usable_divergence_columns)
    if "AI Revenue Exposure" in display_df.columns:
        styler = styler.applymap(_style_ai_exposure_value, subset=["AI Revenue Exposure"])
    if "AI Disruption Risk" in display_df.columns:
        styler = styler.applymap(_style_ai_risk_value, subset=["AI Disruption Risk"])

    return styler


def apply_trend_symbols_to_table(
    current_table: pd.DataFrame,
    previous_table: pd.DataFrame,
    score_columns: list[str],
    *,
    threshold: float = 5.0,
) -> pd.DataFrame:
    if current_table.empty:
        return current_table

    key_col = current_table.columns[0]
    annotated = current_table.copy()
    if key_col not in previous_table.columns:
        return annotated

    current_subset = current_table[[key_col] + [col for col in score_columns if col in current_table.columns]].copy()
    previous_subset = previous_table[[key_col] + [col for col in score_columns if col in previous_table.columns]].copy()
    merged = current_subset.merge(previous_subset, on=key_col, how="left", suffixes=("_curr", "_prev"))

    for col in score_columns:
        if col not in annotated.columns:
            continue
        curr_col = f"{col}_curr"
        prev_col = f"{col}_prev"
        current_numeric = merged[curr_col].map(_parse_number) if curr_col in merged.columns else pd.Series([None] * len(merged))
        previous_numeric = merged[prev_col].map(_parse_number) if prev_col in merged.columns else pd.Series([None] * len(merged))

        deltas = []
        for curr_value, prev_value in zip(current_numeric, previous_numeric):
            if curr_value is None or prev_value is None:
                deltas.append(None)
            else:
                deltas.append(curr_value - prev_value)

        rendered_values: list[str] = []
        for curr_value, delta_value in zip(current_numeric, deltas):
            if curr_value is None:
                rendered_values.append("N/A")
                continue
            trend_symbol = _trend_symbol_for_delta(delta_value, threshold)
            suffix = f" {trend_symbol}" if trend_symbol else ""
            rendered_values.append(f"{curr_value:.1f}{suffix}")
        annotated[col] = rendered_values
    return annotated


def _metric_trend_symbol(
    current_value: object,
    previous_value: object,
    *,
    threshold: float = MARKET_TREND_THRESHOLD,
) -> str:
    current_numeric = _parse_number(current_value)
    previous_numeric = _parse_number(previous_value)
    if current_numeric is None or previous_numeric is None:
        return ""
    return _trend_symbol_for_delta(current_numeric - previous_numeric, threshold)


def _build_market_component_rows(
    component_scores: dict[str, object],
    breadth: dict[str, object],
    risk_appetite: dict[str, object],
    previous_payload: Optional[dict[str, object]],
) -> pd.DataFrame:
    previous_component_scores = dict(previous_payload.get("component_scores", {})) if previous_payload else {}
    previous_breadth = dict(previous_payload.get("breadth", {})) if previous_payload else {}
    previous_risk_appetite = dict(previous_payload.get("risk_appetite", {})) if previous_payload else {}
    row_definitions = [
        (
            "Market RSI participation composite",
            component_scores.get("market_rsi_participation_composite_score"),
            previous_component_scores.get("market_rsi_participation_composite_score"),
            "numeric",
        ),
        (
            "Risk appetite score",
            risk_appetite.get("risk_appetite_score"),
            previous_risk_appetite.get("risk_appetite_score"),
            "numeric",
        ),
        (
            "Market sector rotation score",
            component_scores.get("market_sector_rotation_score"),
            previous_component_scores.get("market_sector_rotation_score"),
            "numeric",
        ),
        (
            "Component agreement score",
            component_scores.get("component_agreement_score"),
            previous_component_scores.get("component_agreement_score"),
            "numeric",
        ),
        (
            "Distance from neutral score",
            component_scores.get("distance_from_neutral_score"),
            previous_component_scores.get("distance_from_neutral_score"),
            "numeric",
        ),
        (
            "Persistence score",
            component_scores.get("persistence_score"),
            previous_component_scores.get("persistence_score"),
            "numeric",
        ),
        (
            "Market breadth < 40",
            breadth.get("market_rsi_breadth_pct_lt40"),
            previous_breadth.get("market_rsi_breadth_pct_lt40"),
            "percent",
        ),
        (
            "Quality / Defensive count",
            risk_appetite.get("quality_defensive_count"),
            previous_risk_appetite.get("quality_defensive_count"),
            "integer",
        ),
        (
            "Speculative count",
            risk_appetite.get("speculative_count"),
            previous_risk_appetite.get("speculative_count"),
            "integer",
        ),
    ]

    rows: list[dict[str, str]] = []
    for label, current_value, previous_value, value_kind in row_definitions:
        symbol = _metric_trend_symbol(current_value, previous_value)
        if value_kind == "percent":
            display_value = _render_percent_with_symbol(current_value, symbol)
        elif value_kind == "integer":
            parsed_value = _parse_number(current_value)
            if parsed_value is None:
                display_value = "N/A"
            else:
                display_value = f"{int(round(parsed_value))}{f' {symbol}' if symbol else ''}"
        else:
            display_value = _render_score_with_symbol(current_value, symbol)
        rows.append({"Component": label, "Value": display_value})
    return pd.DataFrame(rows)


def build_sector_pulse_display(df: pd.DataFrame) -> pd.DataFrame:
    stats = compute_sector_overview_stats(df)
    if stats.empty:
        return pd.DataFrame(columns=[
            "Sector",
            "1-month %",
            "Mk Breadth",
            "Rel. Perf. Breadth",
            "Rel Vol. Breadth",
            "Signal",
        ])

    def resolve_signal(variation: object, market: object, rs: object, obvm: object) -> str:
        variation_num = _parse_number(variation)
        market_num = _parse_number(market)
        rs_num = _parse_number(rs)
        obvm_num = _parse_number(obvm)
        if variation_num is None or market_num is None or rs_num is None or obvm_num is None:
            return "Mixed"
        variation_state = "green" if variation_num > 0 else ("red" if variation_num < 0 else None)
        if variation_state is None:
            return "Mixed"
        breadth_states = ["green" if metric > 50 else "red" for metric in [market_num, rs_num, obvm_num]]
        states = [variation_state] + breadth_states
        if all(state == "green" for state in states):
            return "Positive"
        if all(state == "red" for state in states):
            return "Negative"
        return "Mixed"

    table = pd.DataFrame({
        "Sector": stats["sector"].fillna("Unspecified"),
        "1-month %": stats["sector_1m_var_pct"],
        "Mk Breadth": stats["market_breadth"],
        "Rel. Perf. Breadth": stats["rs_breadth"],
        "Rel Vol. Breadth": stats["obvm_breadth"],
    })
    table["Signal"] = table.apply(
        lambda row: resolve_signal(row["1-month %"], row["Mk Breadth"], row["Rel. Perf. Breadth"], row["Rel Vol. Breadth"]),
        axis=1,
    )
    return table


@st.cache_data(show_spinner=False)
def build_fundamental_tables_for_path(path_str: str) -> dict[str, pd.DataFrame]:
    path = Path(path_str)
    df = normalize_report_columns(load_report_select(path_str)).copy()
    required = {
        "sector",
        "industry",
        "fundamental_total_score",
        "fundamental_value",
        "fundamental_growth",
        "fundamental_quality",
        "fundamental_risk",
        "fundamental_momentum",
    }
    ensure_required_columns(df, required, path)
    df["sector"] = df["sector"].fillna("Unspecified")
    df["industry"] = df.get("industry", pd.Series(index=df.index)).fillna("Unspecified")

    tables: dict[str, pd.DataFrame] = {}
    stats = compute_sector_overview_stats(df)
    if stats.empty:
        tables["All sectors"] = pd.DataFrame(columns=["Sector", "Total", "P1", "P2", "P3", "P4", "P5"])
    else:
        tables["All sectors"] = stats[
            [
                "sector",
                "avg_total_score",
                "avg_value",
                "avg_growth",
                "avg_quality",
                "avg_risk",
                "avg_momentum",
            ]
        ].rename(
            columns={
                "sector": "Sector",
                "avg_total_score": "Total",
                "avg_value": "P1",
                "avg_growth": "P2",
                "avg_quality": "P3",
                "avg_risk": "P4",
                "avg_momentum": "P5",
            }
        ).sort_values("Total", ascending=False, na_position="last").reset_index(drop=True)

    for sector in sorted(df["sector"].dropna().unique().tolist()):
        filtered = df[df["sector"] == sector]
        grouped = (
            filtered.groupby("industry", dropna=False)[
                [
                    "fundamental_total_score",
                    "fundamental_value",
                    "fundamental_growth",
                    "fundamental_quality",
                    "fundamental_risk",
                    "fundamental_momentum",
                ]
            ]
            .mean(numeric_only=True)
            .reset_index()
            .rename(
                columns={
                    "industry": "Industry",
                    "fundamental_total_score": "Total",
                    "fundamental_value": "P1",
                    "fundamental_growth": "P2",
                    "fundamental_quality": "P3",
                    "fundamental_risk": "P4",
                    "fundamental_momentum": "P5",
                }
            )
            .sort_values("Total", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        tables[str(sector)] = grouped
    return tables


@st.cache_data(show_spinner=False)
def build_technical_tables_for_path(path_str: str) -> dict[str, pd.DataFrame]:
    path = Path(path_str)
    df = normalize_report_columns(load_report_select(path_str)).copy()
    required = {
        "sector",
        "industry",
        "general_technical_score",
        "relative_performance",
        "relative_volume",
        "momentum",
        "intermediate_trend",
        "long_term_trend",
    }
    ensure_required_columns(df, required, path)
    df["sector"] = df["sector"].fillna("Unspecified")
    df["industry"] = df.get("industry", pd.Series(index=df.index)).fillna("Unspecified")
    score_columns = [
        "general_technical_score",
        "relative_performance",
        "relative_volume",
        "momentum",
        "intermediate_trend",
        "long_term_trend",
    ]

    tables: dict[str, pd.DataFrame] = {}
    tables["All sectors"] = (
        df.groupby("sector", dropna=False)[score_columns]
        .mean(numeric_only=True)
        .reset_index()
        .rename(
            columns={
                "sector": "Sector",
                "general_technical_score": "Total",
                "relative_performance": "Relative Performance",
                "relative_volume": "Relative Volume",
                "momentum": "Momentum",
                "intermediate_trend": "Intermediate Trend",
                "long_term_trend": "Long-term Trend",
            }
        )
        .sort_values("Total", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    for sector in sorted(df["sector"].dropna().unique().tolist()):
        filtered = df[df["sector"] == sector]
        grouped = (
            filtered.groupby("industry", dropna=False)[score_columns]
            .mean(numeric_only=True)
            .reset_index()
            .rename(
                columns={
                    "industry": "Industry",
                    "general_technical_score": "Total",
                    "relative_performance": "Relative Performance",
                    "relative_volume": "Relative Volume",
                    "momentum": "Momentum",
                    "intermediate_trend": "Intermediate Trend",
                    "long_term_trend": "Long-term Trend",
                }
            )
            .sort_values("Total", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        tables[str(sector)] = grouped
    return tables


def build_fundamental_table(df: pd.DataFrame, selected_sector: str) -> pd.DataFrame:
    if selected_sector == "All sectors":
        stats = compute_sector_overview_stats(df)
        if stats.empty:
            return pd.DataFrame(columns=["Sector", "Total", "P1", "P2", "P3", "P4", "P5"])
        table = stats[
            [
                "sector",
                "avg_total_score",
                "avg_value",
                "avg_growth",
                "avg_quality",
                "avg_risk",
                "avg_momentum",
            ]
        ].rename(
            columns={
                "sector": "Sector",
                "avg_total_score": "Total",
                "avg_value": "P1",
                "avg_growth": "P2",
                "avg_quality": "P3",
                "avg_risk": "P4",
                "avg_momentum": "P5",
            }
        ).sort_values("Total", ascending=False, na_position="last")
        return table.reset_index(drop=True)

    filtered = df.copy()
    filtered["sector"] = filtered["sector"].fillna("Unspecified")
    filtered["industry"] = filtered.get("industry", pd.Series(index=filtered.index)).fillna("Unspecified")
    filtered = filtered[filtered["sector"] == selected_sector]
    if filtered.empty:
        return pd.DataFrame(columns=["Industry", "Total", "P1", "P2", "P3", "P4", "P5"])

    grouped = (
        filtered.groupby("industry", dropna=False)[
            [
                "fundamental_total_score",
                "fundamental_value",
                "fundamental_growth",
                "fundamental_quality",
                "fundamental_risk",
                "fundamental_momentum",
            ]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .rename(
            columns={
                "industry": "Industry",
                "fundamental_total_score": "Total",
                "fundamental_value": "P1",
                "fundamental_growth": "P2",
                "fundamental_quality": "P3",
                "fundamental_risk": "P4",
                "fundamental_momentum": "P5",
            }
        )
        .sort_values("Total", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    return grouped


def build_technical_table(df: pd.DataFrame, selected_sector: str) -> pd.DataFrame:
    work = df.copy()
    work["sector"] = work["sector"].fillna("Unspecified")
    work["industry"] = work.get("industry", pd.Series(index=work.index)).fillna("Unspecified")
    score_columns = [
        "general_technical_score",
        "relative_performance",
        "relative_volume",
        "momentum",
        "intermediate_trend",
        "long_term_trend",
    ]

    if selected_sector == "All sectors":
        grouped = (
            work.groupby("sector", dropna=False)[score_columns]
            .mean(numeric_only=True)
            .reset_index()
            .rename(
                columns={
                    "sector": "Sector",
                    "general_technical_score": "Total",
                    "relative_performance": "Relative Performance",
                    "relative_volume": "Relative Volume",
                    "momentum": "Momentum",
                    "intermediate_trend": "Intermediate Trend",
                    "long_term_trend": "Long-term Trend",
                }
            )
            .sort_values("Total", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        return grouped

    filtered = work[work["sector"] == selected_sector]
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "Industry",
                "Total",
                "Relative Performance",
                "Relative Volume",
                "Momentum",
                "Intermediate Trend",
                "Long-term Trend",
            ]
        )
    grouped = (
        filtered.groupby("industry", dropna=False)[score_columns]
        .mean(numeric_only=True)
        .reset_index()
        .rename(
            columns={
                "industry": "Industry",
                "general_technical_score": "Total",
                "relative_performance": "Relative Performance",
                "relative_volume": "Relative Volume",
                "momentum": "Momentum",
                "intermediate_trend": "Intermediate Trend",
                "long_term_trend": "Long-term Trend",
            }
        )
        .sort_values("Total", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    return grouped


def get_fundamental_table_for_sector(path_str: str, selected_sector: str) -> pd.DataFrame:
    tables = build_fundamental_tables_for_path(path_str)
    if selected_sector == "All sectors":
        return tables.get("All sectors", pd.DataFrame(columns=["Sector", "Total", "P1", "P2", "P3", "P4", "P5"])).copy()
    return tables.get(
        selected_sector,
        pd.DataFrame(columns=["Industry", "Total", "P1", "P2", "P3", "P4", "P5"]),
    ).copy()


def get_technical_table_for_sector(path_str: str, selected_sector: str) -> pd.DataFrame:
    tables = build_technical_tables_for_path(path_str)
    if selected_sector == "All sectors":
        return tables.get(
            "All sectors",
            pd.DataFrame(
                columns=[
                    "Sector",
                    "Total",
                    "Relative Performance",
                    "Relative Volume",
                    "Momentum",
                    "Intermediate Trend",
                    "Long-term Trend",
                ]
            ),
        ).copy()
    return tables.get(
        selected_sector,
        pd.DataFrame(
            columns=[
                "Industry",
                "Total",
                "Relative Performance",
                "Relative Volume",
                "Momentum",
                "Intermediate Trend",
                "Long-term Trend",
            ]
        ),
    ).copy()


@st.cache_data(show_spinner=False)
def build_trended_table_for_paths(
    board_kind: str,
    current_path_str: str,
    previous_path_str: str,
    selected_sector: str,
    threshold: float,
    cache_revision: str = "2026-03-15-trend-symbol-fix-v2",
) -> pd.DataFrame:
    _ = cache_revision  # Cache-buster so old trended tables do not survive rendering logic changes.
    if board_kind == "fundamental":
        score_columns = ["Total", "P1", "P2", "P3", "P4", "P5"]
        current_table = get_fundamental_table_for_sector(current_path_str, selected_sector)
        previous_table = get_fundamental_table_for_sector(previous_path_str, selected_sector)
    else:
        score_columns = [
            "Total",
            "Relative Performance",
            "Relative Volume",
            "Momentum",
            "Intermediate Trend",
            "Long-term Trend",
        ]
        current_table = get_technical_table_for_sector(current_path_str, selected_sector)
        previous_table = get_technical_table_for_sector(previous_path_str, selected_sector)
    return apply_trend_symbols_to_table(current_table, previous_table, score_columns, threshold=threshold)


def render_sector_pulse_board(config: ReportConfig) -> None:
    render_page_intro(
        "Sector Pulse",
        "Sector breadth, monthly variation, and signal parity with PDF logic.",
        "Equipilot / Sector Pulse",
    )
    selected_eod = render_report_select_date_input(
        "EOD date",
        value=get_default_board_eod(config),
        key="sector_pulse_eod",
    )
    with st.spinner("Loading sector pulse data..."):
        report_df, source_path, candidates, load_error = load_report_select_for_eod(selected_eod)
    if source_path is None:
        render_missing_report_select(selected_eod, candidates)
        return
    if load_error:
        st.error(f"Failed reading {source_path}: {load_error}")
        return
    render_chip_row([f"Source file: {source_path}", f"EOD: {selected_eod.isoformat()}"])
    if not validate_required_columns(
        report_df,
        {
            "sector",
            "1m_close",
            "eod_price_used",
            "ic_eod_price_used",
            "market_cap",
            "rs_monthly",
            "obvm_monthly",
        },
        source_path,
        "Sector Pulse",
    ):
        return

    pulse_df = build_sector_pulse_display(report_df)
    st.caption(f"Rows displayed: {len(pulse_df)}")
    render_board_title_band("Sector Pulse")
    render_pdf_like_table(
        pulse_df,
        center_all_except_first=True,
        value_color_rules={
            "1-month %": _variation_color_css,
            "Mk Breadth": _breadth_color_css,
            "Rel. Perf. Breadth": _breadth_color_css,
            "Rel Vol. Breadth": _breadth_color_css,
            "Signal": _signal_color_css,
        },
    )


def render_fundamental_scoring_board(config: ReportConfig) -> None:
    render_page_intro(
        "Fundamental Scoring",
        "Cross-sector and sector-to-industry drill-down view using report_select snapshots.",
        "Equipilot / Fundamental Scoring",
    )
    default_eod = get_default_board_eod(config)
    selected_eod = render_report_select_date_input("EOD date", value=default_eod, key="fundamental_scoring_eod")
    previous_eod = render_report_select_date_input(
        "EOD date (previous)",
        value=get_default_previous_board_eod(selected_eod),
        key="fundamental_scoring_prev_eod",
    )
    show_trend = st.checkbox(
        "Show trend arrows vs previous EOD",
        value=True,
        key="fundamental_scoring_show_trends",
    )
    show_perf = st.checkbox(
        "Show perf timings",
        value=False,
        key="fundamental_scoring_perf_toggle",
    )
    timings: list[tuple[str, float]] = []

    t_start = time.perf_counter()
    with st.spinner("Loading fundamental scoring data..."):
        report_df, source_path, candidates, load_error = load_report_select_for_eod(selected_eod)
    _perf_mark(timings, "load current", t_start)
    if source_path is None:
        render_missing_report_select(selected_eod, candidates)
        return
    if load_error:
        st.error(f"Failed reading {source_path}: {load_error}")
        return

    previous_report_df: Optional[pd.DataFrame] = None
    previous_path: Optional[Path] = None
    previous_ready = False
    if show_trend:
        t_start = time.perf_counter()
        previous_df, previous_path, previous_candidates, previous_error = load_report_select_for_eod(previous_eod)
        _perf_mark(timings, "load previous", t_start)
        if previous_path is None:
            render_missing_report_select(previous_eod, previous_candidates)
        elif previous_error:
            st.error(f"Failed reading {previous_path}: {previous_error}")
        else:
            previous_report_df = previous_df
            previous_ready = True

    chips = [f"Current file: {source_path}", f"EOD: {selected_eod.isoformat()}"]
    if show_trend:
        if previous_path is not None:
            chips.append(f"Previous file: {previous_path}")
        chips.append(f"EOD previous: {previous_eod.isoformat()}")
    else:
        chips.append("Trend mode: Off")
    render_chip_row(chips)

    if not validate_required_columns(
        report_df,
        {
            "ticker",
            "sector",
            "industry",
            "1m_close",
            "eod_price_used",
            "ic_eod_price_used",
            "market_cap",
            "rs_monthly",
            "obvm_monthly",
            "fundamental_total_score",
            "general_technical_score",
            "fundamental_value",
            "fundamental_growth",
            "fundamental_quality",
            "fundamental_risk",
            "fundamental_momentum",
        },
        source_path,
        "Fundamental Scoring",
    ):
        return
    if show_trend and previous_ready and previous_report_df is not None and previous_path is not None:
        previous_ready = validate_required_columns(
            previous_report_df,
            {
                "ticker",
                "sector",
                "industry",
                "1m_close",
                "eod_price_used",
                "ic_eod_price_used",
                "market_cap",
                "rs_monthly",
                "obvm_monthly",
                "fundamental_total_score",
                "general_technical_score",
                "fundamental_value",
                "fundamental_growth",
                "fundamental_quality",
                "fundamental_risk",
                "fundamental_momentum",
            },
            previous_path,
            "Fundamental Scoring (previous EOD)",
        )

    sector_options = ["All sectors"] + sorted(report_df["sector"].fillna("Unspecified").unique().tolist())
    sector_col, show_all_col = st.columns([1.4, 1.0])
    with sector_col:
        selected_sector = st.selectbox(
            "Sector view",
            options=sector_options,
            index=0,
            key="fundamental_scoring_sector_select",
        )
    with show_all_col:
        _apply_pending_show_all_company_reset("fundamental")
        st.checkbox(
            "Show all companies",
            key="fundamental_show_all_companies",
            on_change=_handle_show_all_company_toggle,
            args=("fundamental",),
        )
    t_start = time.perf_counter()
    table_df = get_fundamental_table_for_sector(str(source_path), selected_sector)
    _perf_mark(timings, "build table", t_start)

    score_columns = ["Total", "P1", "P2", "P3", "P4", "P5"]
    render_df = table_df
    format_map = {
        "Total": "{:.1f}",
        "P1": "{:.1f}",
        "P2": "{:.1f}",
        "P3": "{:.1f}",
        "P4": "{:.1f}",
        "P5": "{:.1f}",
    }
    if show_trend and previous_ready and previous_path is not None:
        t_start = time.perf_counter()
        render_df = build_trended_table_for_paths(
            "fundamental",
            str(source_path),
            str(previous_path),
            selected_sector,
            5.0,
        )
        _perf_mark(timings, "trend annotate", t_start)
        format_map = None
    render_df = _sort_table_by_total_desc(render_df)

    drilldown_signature = (
        selected_eod.isoformat(),
        previous_eod.isoformat(),
        selected_sector,
        "trend_on" if show_trend else "trend_off",
    )
    _sync_drilldown_signature("fundamental", drilldown_signature)
    drilldown_nonce_key = "fundamental_drilldown_nonce"
    drilldown_nonce = st.session_state.setdefault(drilldown_nonce_key, 0)
    table_widget_key = (
        f"fundamental_scoring_select_{selected_eod.isoformat()}_"
        f"{previous_eod.isoformat()}_{selected_sector.replace(' ', '_')}_"
        f"{'trend' if show_trend else 'notrend'}_{drilldown_nonce}_sortv2"
    )
    if selected_sector == "All sectors":
        render_board_title_band("Cross-Sector Fundamental Scoring")
    else:
        render_board_title_band(f"{selected_sector} - Industry Fundamental Scoring")

    t_start = time.perf_counter()
    selected_row_index = render_pdf_like_table(
        render_df,
        center_all_except_first=True,
        highlight_max_cols=["Total", "P1", "P2", "P3", "P4", "P5"],
        value_color_rules={
            "Total": _score_color_css,
            "P1": _score_color_css,
            "P2": _score_color_css,
            "P3": _score_color_css,
            "P4": _score_color_css,
            "P5": _score_color_css,
        },
        format_map=format_map,
        selectable=True,
        selection_key=table_widget_key,
    )
    _perf_mark(timings, "render table", t_start)
    if selected_row_index is not None:
        next_selected_key = str(render_df.iloc[selected_row_index, 0]).strip()
        next_selected_mode = "sector" if selected_sector == "All sectors" else "industry"
        show_all_active = bool(st.session_state.get("fundamental_show_all_companies", False))
        selection_changed = (
            st.session_state.get("fundamental_selected_key") != next_selected_key
            or st.session_state.get("fundamental_selected_mode") != next_selected_mode
        )
        if selection_changed or show_all_active:
            if show_all_active:
                _queue_show_all_company_reset("fundamental")
            st.session_state["fundamental_selected_key"] = next_selected_key
            st.session_state["fundamental_selected_mode"] = next_selected_mode
            if show_all_active:
                st.rerun()

    show_all_companies = bool(st.session_state.get("fundamental_show_all_companies", False))
    selected_key = st.session_state.get("fundamental_selected_key")
    selected_mode = st.session_state.get("fundamental_selected_mode")
    scope_mode = "all" if show_all_companies else str(selected_mode) if selected_key and selected_mode else ""
    scope_key = "__all__" if show_all_companies else str(selected_key or "")
    if scope_mode:
        company_trend_enabled = bool(show_trend and previous_ready and previous_report_df is not None)
        t_start = time.perf_counter()
        details_title, company_universe, default_sectors, default_industries, details_error, regime_warning = build_company_drilldown_context_from_path(
            source_path,
            evaluation_date=selected_eod,
            selected_sector=selected_sector,
            selected_mode=scope_mode,
            selected_key=scope_key,
        )
        _perf_mark(timings, "company universe", t_start)
        if details_error:
            st.warning(details_error)
        elif company_universe is not None:
            if company_trend_enabled and previous_report_df is not None:
                t_start = time.perf_counter()
                company_universe = _annotate_company_technical_trend(
                    company_universe,
                    previous_report_df,
                    threshold=5.0,
                )
                _perf_mark(timings, "company trend", t_start)
            filter_signature = (
                selected_eod.isoformat(),
                previous_eod.isoformat(),
                selected_sector,
                scope_mode,
                scope_key,
            )
            _sync_drilldown_filter_defaults(
                "fundamental",
                filter_signature,
                default_sectors=default_sectors,
                default_industries=default_industries,
                default_cap_buckets=["Large", "Mega"],
                default_fund_range=(50.0, 100.0),
                default_tech_range=(60.0, 100.0),
                default_rsi_regime_range=(70.0, 100.0),
                default_sector_regime_fit_range=_default_sector_regime_fit_range_for_company_scope(
                    "show_all" if scope_mode == "all" else "selected"
                ),
                default_fund_momentum_range=(60.0, 100.0),
                default_tech_trend_dir="All",
            )
            st.markdown("---")
            st.caption(details_title)
            if regime_warning:
                st.caption(regime_warning)
            t_start = time.perf_counter()
            filtered_companies = render_company_drilldown_filters(
                company_universe,
                prefix="fundamental",
                ticker_label="Ticker filter (Fundamental drilldown)",
                include_fundamental_momentum_filter=True,
                include_technical_trend_filter=company_trend_enabled,
                include_rel_strength_filter=True,
                include_rel_volume_filter=True,
                include_ai_exposure_filters=True,
            )
            _perf_mark(timings, "company filters", t_start)
            st.caption(f"Companies after filters: {len(filtered_companies)}")
            t_start = time.perf_counter()
            details_display = format_company_drilldown_display(filtered_companies, sort_by="fundamental")
            details_display = _apply_grid_column_layout(
                details_display,
                GRID_SURFACE_FUNDAMENTAL_COMPANY,
            )
            _perf_mark(timings, "company display", t_start)
            if details_display.empty:
                st.info("No companies found for the selected row.")
            else:
                details_height = _company_grid_height(len(details_display), row_height=34, min_height=220)
                use_fast_grid = _use_fast_company_grid_render(len(details_display))
                if use_fast_grid:
                    st.caption("Large result set: using fast grid rendering.")
                else:
                    t_start = time.perf_counter()
                    details_styler = _build_company_drilldown_styler(details_display)
                    _perf_mark(timings, "company styler", t_start)
                t_start = time.perf_counter()
                st.dataframe(
                    details_display if use_fast_grid else details_styler,
                    use_container_width=True,
                    height=details_height,
                    hide_index=True,
                )
                _perf_mark(timings, "company render", t_start)
            if st.button("Hide company list", key="fundamental_hide_company_list"):
                _clear_drilldown_selection("fundamental")
                st.session_state[drilldown_nonce_key] = drilldown_nonce + 1
                st.rerun()
    _render_perf_timings(show_perf, timings)


def render_technical_scoring_board(config: ReportConfig) -> None:
    render_page_intro(
        "Technical Scoring",
        "Technical pillar scoring with sector view and industry drill-down by selected sector.",
        "Equipilot / Technical Scoring",
    )
    default_eod = get_default_board_eod(config)
    selected_eod = render_report_select_date_input("EOD date", value=default_eod, key="technical_scoring_eod")
    previous_eod = render_report_select_date_input(
        "EOD date (previous)",
        value=get_default_previous_board_eod(selected_eod),
        key="technical_scoring_prev_eod",
    )
    show_trend = st.checkbox(
        "Show trend arrows vs previous EOD",
        value=True,
        key="technical_scoring_show_trends",
    )
    show_perf = st.checkbox(
        "Show perf timings",
        value=False,
        key="technical_scoring_perf_toggle",
    )
    timings: list[tuple[str, float]] = []

    t_start = time.perf_counter()
    with st.spinner("Loading technical scoring data..."):
        report_df, source_path, candidates, load_error = load_report_select_for_eod(selected_eod)
    _perf_mark(timings, "load current", t_start)
    if source_path is None:
        render_missing_report_select(selected_eod, candidates)
        return
    if load_error:
        st.error(f"Failed reading {source_path}: {load_error}")
        return

    previous_report_df: Optional[pd.DataFrame] = None
    previous_path: Optional[Path] = None
    previous_ready = False
    if show_trend:
        t_start = time.perf_counter()
        previous_df, previous_path, previous_candidates, previous_error = load_report_select_for_eod(previous_eod)
        _perf_mark(timings, "load previous", t_start)
        if previous_path is None:
            render_missing_report_select(previous_eod, previous_candidates)
        elif previous_error:
            st.error(f"Failed reading {previous_path}: {previous_error}")
        else:
            previous_report_df = previous_df
            previous_ready = True

    chips = [f"Current file: {source_path}", f"EOD: {selected_eod.isoformat()}"]
    if show_trend:
        if previous_path is not None:
            chips.append(f"Previous file: {previous_path}")
        chips.append(f"EOD previous: {previous_eod.isoformat()}")
    else:
        chips.append("Trend mode: Off")
    render_chip_row(chips)

    if not validate_required_columns(
        report_df,
        {
            "sector",
            "industry",
            "general_technical_score",
            "relative_performance",
            "relative_volume",
            "momentum",
            "intermediate_trend",
            "long_term_trend",
        },
        source_path,
        "Technical Scoring",
    ):
        return
    if show_trend and previous_ready and previous_report_df is not None and previous_path is not None:
        previous_ready = validate_required_columns(
            previous_report_df,
            {
                "sector",
                "industry",
                "general_technical_score",
                "relative_performance",
                "relative_volume",
                "momentum",
                "intermediate_trend",
                "long_term_trend",
            },
            previous_path,
            "Technical Scoring (previous EOD)",
        )

    sector_options = ["All sectors"] + sorted(report_df["sector"].fillna("Unspecified").unique().tolist())
    sector_col, show_all_col = st.columns([1.4, 1.0])
    with sector_col:
        selected_sector = st.selectbox(
            "Sector view",
            options=sector_options,
            index=0,
            key="technical_scoring_sector_select",
        )
    with show_all_col:
        _apply_pending_show_all_company_reset("technical")
        st.checkbox(
            "Show all companies",
            key="technical_show_all_companies",
            on_change=_handle_show_all_company_toggle,
            args=("technical",),
        )
    t_start = time.perf_counter()
    table_df = get_technical_table_for_sector(str(source_path), selected_sector)
    _perf_mark(timings, "build table", t_start)

    if selected_sector == "All sectors":
        render_board_title_band("Cross-Sector Technical Scoring")
    else:
        render_board_title_band(f"{selected_sector} - Industry Technical Scoring")
    score_columns = [
        "Total",
        "Relative Performance",
        "Relative Volume",
        "Momentum",
        "Intermediate Trend",
        "Long-term Trend",
    ]
    render_df = table_df
    format_map = {column: "{:.1f}" for column in score_columns}
    if show_trend and previous_ready and previous_path is not None:
        t_start = time.perf_counter()
        render_df = build_trended_table_for_paths(
            "technical",
            str(source_path),
            str(previous_path),
            selected_sector,
            5.0,
        )
        _perf_mark(timings, "trend annotate", t_start)
        format_map = None

    drilldown_signature = (
        selected_eod.isoformat(),
        previous_eod.isoformat(),
        selected_sector,
        "trend_on" if show_trend else "trend_off",
    )
    _sync_drilldown_signature("technical", drilldown_signature)
    drilldown_nonce_key = "technical_drilldown_nonce"
    drilldown_nonce = st.session_state.setdefault(drilldown_nonce_key, 0)
    table_widget_key = (
        f"technical_scoring_select_{selected_eod.isoformat()}_"
        f"{previous_eod.isoformat()}_{selected_sector.replace(' ', '_')}_"
        f"{'trend' if show_trend else 'notrend'}_{drilldown_nonce}"
    )

    t_start = time.perf_counter()
    selected_row_index = render_pdf_like_table(
        render_df,
        center_all_except_first=True,
        highlight_max_cols=score_columns,
        value_color_rules={column: _score_color_css for column in score_columns},
        format_map=format_map,
        selectable=True,
        selection_key=table_widget_key,
    )
    _perf_mark(timings, "render table", t_start)
    if selected_row_index is not None:
        next_selected_key = str(render_df.iloc[selected_row_index, 0]).strip()
        next_selected_mode = "sector" if selected_sector == "All sectors" else "industry"
        show_all_active = bool(st.session_state.get("technical_show_all_companies", False))
        selection_changed = (
            st.session_state.get("technical_selected_key") != next_selected_key
            or st.session_state.get("technical_selected_mode") != next_selected_mode
        )
        if selection_changed or show_all_active:
            if show_all_active:
                _queue_show_all_company_reset("technical")
            st.session_state["technical_selected_key"] = next_selected_key
            st.session_state["technical_selected_mode"] = next_selected_mode
            if show_all_active:
                st.rerun()

    show_all_companies = bool(st.session_state.get("technical_show_all_companies", False))
    selected_key = st.session_state.get("technical_selected_key")
    selected_mode = st.session_state.get("technical_selected_mode")
    scope_mode = "all" if show_all_companies else str(selected_mode) if selected_key and selected_mode else ""
    scope_key = "__all__" if show_all_companies else str(selected_key or "")
    if scope_mode:
        company_trend_enabled = bool(show_trend and previous_ready and previous_report_df is not None)
        t_start = time.perf_counter()
        details_title, company_universe, default_sectors, default_industries, details_error, regime_warning = build_company_drilldown_context_from_path(
            source_path,
            evaluation_date=selected_eod,
            selected_sector=selected_sector,
            selected_mode=scope_mode,
            selected_key=scope_key,
        )
        _perf_mark(timings, "company universe", t_start)
        if details_error:
            st.warning(details_error)
        elif company_universe is not None:
            if company_trend_enabled and previous_report_df is not None:
                t_start = time.perf_counter()
                company_universe = _annotate_company_technical_trend(
                    company_universe,
                    previous_report_df,
                    threshold=5.0,
                )
                _perf_mark(timings, "company trend", t_start)
            filter_signature = (
                selected_eod.isoformat(),
                previous_eod.isoformat(),
                selected_sector,
                scope_mode,
                scope_key,
            )
            _sync_drilldown_filter_defaults(
                "technical",
                filter_signature,
                default_sectors=default_sectors,
                default_industries=default_industries,
                default_cap_buckets=["Large", "Mega"],
                default_fund_range=(50.0, 100.0),
                default_tech_range=(60.0, 100.0),
                default_rsi_regime_range=(70.0, 100.0),
                default_sector_regime_fit_range=_default_sector_regime_fit_range_for_company_scope(
                    "show_all" if scope_mode == "all" else "selected"
                ),
                default_fund_momentum_range=(60.0, 100.0),
                default_tech_trend_dir="All",
            )
            st.markdown("---")
            st.caption(details_title)
            if regime_warning:
                st.caption(regime_warning)
            t_start = time.perf_counter()
            filtered_companies = render_company_drilldown_filters(
                company_universe,
                prefix="technical",
                ticker_label="Ticker filter (Technical drilldown)",
                include_fundamental_momentum_filter=True,
                include_technical_trend_filter=company_trend_enabled,
                include_rel_strength_filter=True,
                include_rel_volume_filter=True,
                include_ai_exposure_filters=True,
            )
            _perf_mark(timings, "company filters", t_start)
            st.caption(f"Companies after filters: {len(filtered_companies)}")
            t_start = time.perf_counter()
            details_display = format_company_drilldown_display(filtered_companies, sort_by="technical")
            details_display = _apply_grid_column_layout(
                details_display,
                GRID_SURFACE_TECHNICAL_COMPANY,
            )
            _perf_mark(timings, "company display", t_start)
            if details_display.empty:
                st.info("No companies found for the selected row.")
            else:
                details_height = _company_grid_height(len(details_display), row_height=34, min_height=220)
                use_fast_grid = _use_fast_company_grid_render(len(details_display))
                if use_fast_grid:
                    st.caption("Large result set: using fast grid rendering.")
                else:
                    t_start = time.perf_counter()
                    details_styler = _build_company_drilldown_styler(details_display)
                    _perf_mark(timings, "company styler", t_start)
                t_start = time.perf_counter()
                st.dataframe(
                    details_display if use_fast_grid else details_styler,
                    use_container_width=True,
                    height=details_height,
                    hide_index=True,
                )
                _perf_mark(timings, "company render", t_start)
            if st.button("Hide company list", key="technical_hide_company_list"):
                _clear_drilldown_selection("technical")
                st.session_state[drilldown_nonce_key] = drilldown_nonce + 1
                st.rerun()
    _render_perf_timings(show_perf, timings)


def _run_trade_idea_filter(strategy_name: str, report_df: pd.DataFrame) -> pd.DataFrame:
    working_df = report_df.copy()
    if strategy_name == "extreme_accel_up":
        return extreme_accel_up(working_df, DATA_DIR, save_output=False)
    if strategy_name == "accel_up_weak":
        return accel_up_weak(working_df, DATA_DIR, save_output=False)
    if strategy_name == "extreme_accel_down":
        return extreme_accel_down(working_df, DATA_DIR, save_output=False)
    if strategy_name == "accel_down_weak":
        return accel_down_weak(working_df, DATA_DIR, save_output=False)
    raise ValueError(f"Unsupported trade-idea strategy: {strategy_name}")


def _trade_idea_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    preferred_columns = [
        "ticker",
        "company",
        "sector",
        "industry",
        "market_cap",
        "general_technical_score",
        "relative_performance",
        "relative_volume",
        "momentum",
        "intermediate_trend",
        "long_term_trend",
        "fundamental_total_score",
        "eod_price_used",
        "near_1y_high_5pct",
        "near_1y_low_5pct",
        "near_ma20_5pct",
        "near_ma50_5pct",
        "near_ma200_5pct",
        "eod_price_date",
    ]
    available = [col for col in preferred_columns if col in df.columns]
    if available:
        return df.loc[:, available]
    return df


def _render_trade_idea_strategy(
    *,
    selected_eod: date,
    strategy_name: str,
    board_title: str,
    strategy_subtitle: str,
    required_columns: set[str],
) -> None:
    st.caption(strategy_subtitle)
    with st.spinner(f"Loading {strategy_name} candidates..."):
        report_df, source_path, candidates, load_error = load_report_select_for_eod(selected_eod)
    if source_path is None:
        render_missing_report_select(selected_eod, candidates)
        return
    if load_error:
        st.error(f"Failed reading {source_path}: {load_error}")
        return
    render_chip_row([f"Source file: {source_path}", f"EOD: {selected_eod.isoformat()}"])
    if not validate_required_columns(report_df, required_columns, source_path, board_title):
        return

    try:
        ideas_df = _run_trade_idea_filter(strategy_name, report_df)
    except Exception as exc:  # pragma: no cover - UI feedback
        st.error(f"{board_title}: filter execution failed: {exc}")
        return

    st.caption(f"Stocks passing filter: {len(ideas_df)}")
    display_df = _trade_idea_display_frame(ideas_df)
    if display_df.empty:
        st.info("No stocks matched this filter for the selected EOD.")
        return

    ticker_filter_query = st.text_input(
        "Ticker filter (Trade ideas)",
        key=f"{strategy_name}_trade_ideas_ticker_filter",
        placeholder="Type ticker symbol...",
    ).strip()
    grid_df = display_df.copy()
    if ticker_filter_query and "ticker" in grid_df.columns:
        ticker_mask = grid_df["ticker"].fillna("").astype(str).str.contains(
            ticker_filter_query,
            case=False,
            na=False,
            regex=False,
        )
        grid_df = grid_df[ticker_mask].copy()

    if grid_df.empty:
        st.info("No stocks matched this filter/ticker selection for the selected EOD.")
        return

    if "sector" in grid_df.columns:
        sector_counts = (
            grid_df["sector"]
            .fillna("Unspecified")
            .astype(str)
            .str.strip()
            .replace("", "Unspecified")
            .value_counts()
        )
        if not sector_counts.empty:
            sector_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=sector_counts.index.tolist(),
                        values=sector_counts.values.tolist(),
                        textinfo="label+percent",
                        hole=0.35,
                    )
                ]
            )
            sector_pie.update_layout(
                margin=dict(l=10, r=10, t=10, b=10),
                height=320,
            )
            st.plotly_chart(
                sector_pie,
                use_container_width=True,
                key=f"{strategy_name}_trade_ideas_sector_pie",
            )

    export_cols = st.columns([1, 1, 4])
    csv_data = grid_df.to_csv(index=False).encode("utf-8")
    with export_cols[0]:
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{strategy_name}_{selected_eod.isoformat()}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{strategy_name}_trade_ideas_download_csv",
        )
    with export_cols[1]:
        st.download_button(
            "Download JSON",
            data=grid_df.to_json(orient="records", force_ascii=False, indent=2),
            file_name=f"{strategy_name}_{selected_eod.isoformat()}.json",
            mime="application/json",
            use_container_width=True,
            key=f"{strategy_name}_trade_ideas_download_json",
        )

    estimated_height = max(260, 72 + len(grid_df) * 38)
    st.dataframe(grid_df, use_container_width=True, height=estimated_height, hide_index=True)


def render_trade_ideas(config: ReportConfig) -> None:
    render_page_intro(
        "Trade Ideas",
        "Actionable candidates from curated acceleration filters.",
        "Equipilot / Trade Ideas",
    )
    render_chip_row(
        [
            "Strategies currently active: extreme_accel_up, accel_up_weak, accel_down_weak, extreme_accel_down",
            "Data source: report_select_<EOD> file",
        ]
    )
    selected_eod = render_report_select_date_input(
        "EOD date",
        value=get_default_board_eod(config),
        key="trade_ideas_eod",
    )
    strategy_specs = [
        {
            "name": "extreme_accel_up",
            "subtitle": "Highest-conviction bullish acceleration setup with strict technical and thrust constraints.",
            "required_columns": {
                "rs_daily",
                "rs_sma20",
                "rs_monthly",
                "obvm_daily",
                "obvm_sma20",
                "obvm_monthly",
                "obvm_weekly",
                "rsi_weekly",
                "rsi_daily",
                "eod_price_used",
                "sma_daily_20",
                "sma_daily_50",
                "sma_daily_200",
            },
            "style": ("#16A34A", "#FFFFFF", "#15803D"),
        },
        {
            "name": "accel_up_weak",
            "subtitle": "Moderate bullish acceleration profile with constructive but cooling momentum behavior.",
            "required_columns": {
                "rs_daily",
                "rs_sma20",
                "rs_monthly",
                "obvm_daily",
                "obvm_sma20",
                "obvm_monthly",
                "obvm_weekly",
                "rsi_weekly",
                "rsi_daily",
                "eod_price_used",
                "sma_daily_50",
                "sma_daily_200",
            },
            "style": ("#BBF7D0", "#14532D", "#4ADE80"),
        },
        {
            "name": "accel_down_weak",
            "subtitle": "Moderate bearish acceleration profile with early downside follow-through behavior.",
            "required_columns": {
                "rs_daily",
                "rs_sma20",
                "rs_monthly",
                "obvm_daily",
                "obvm_sma20",
                "obvm_monthly",
                "obvm_weekly",
                "rsi_weekly",
                "rsi_daily",
                "eod_price_used",
                "sma_daily_50",
                "sma_daily_200",
            },
            "style": ("#FECACA", "#7F1D1D", "#FCA5A5"),
        },
        {
            "name": "extreme_accel_down",
            "subtitle": "Highest-conviction bearish acceleration setup with strict downside momentum and trend constraints.",
            "required_columns": {
                "rs_daily",
                "rs_sma20",
                "rs_monthly",
                "obvm_daily",
                "obvm_sma20",
                "obvm_monthly",
                "obvm_weekly",
                "rsi_weekly",
                "rsi_daily",
                "eod_price_used",
                "sma_daily_20",
                "sma_daily_50",
                "sma_daily_200",
            },
            "style": ("#DC2626", "#FFFFFF", "#B91C1C"),
        },
    ]
    strategy_names = [entry["name"] for entry in strategy_specs]
    selected_strategy_key = "trade_ideas_selected_strategy"
    if st.session_state.get(selected_strategy_key) not in strategy_names:
        st.session_state[selected_strategy_key] = "extreme_accel_up"
    selected_strategy = str(st.session_state.get(selected_strategy_key))

    strategy_css_rules: list[str] = []
    for spec in strategy_specs:
        bg_color, text_color, border_color = spec["style"]
        button_key = f"trade_ideas_btn_{spec['name']}"
        strategy_css_rules.append(
            f"""
div.st-key-{button_key} button {{
  background-color: {bg_color} !important;
  color: {text_color} !important;
  border: 1px solid {border_color} !important;
}}
div.st-key-{button_key} button p {{
  color: {text_color} !important;
}}
div.st-key-{button_key} button:hover {{
  filter: brightness(0.96);
}}
            """
        )
        if spec["name"] == selected_strategy:
            strategy_css_rules.append(
                f"""
div.st-key-{button_key} button {{
  border-width: 3px !important;
  box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.22) !important;
}}
                """
            )
    st.markdown(
        f"""
<style>
{''.join(strategy_css_rules)}
</style>
        """,
        unsafe_allow_html=True,
    )

    selector_cols = st.columns(4)
    for idx, spec in enumerate(strategy_specs):
        with selector_cols[idx]:
            if st.button(spec["name"], key=f"trade_ideas_btn_{spec['name']}", use_container_width=True):
                st.session_state[selected_strategy_key] = spec["name"]
                st.rerun()

    selected_spec = next(entry for entry in strategy_specs if entry["name"] == selected_strategy)
    render_board_title_band(f"Trade Ideas - {selected_spec['name']}")
    _render_trade_idea_strategy(
        selected_eod=selected_eod,
        strategy_name=selected_spec["name"],
        board_title=f"Trade Ideas / {selected_spec['name']}",
        strategy_subtitle=str(selected_spec["subtitle"]),
        required_columns=set(selected_spec["required_columns"]),
    )


def list_indices_cache_paths() -> list[Path]:
    return sorted(DATA_DIR.glob("indices-prices-*.xlsx"))


def get_latest_indices_cache_date(cache_paths: Optional[list[Path]] = None) -> Tuple[Optional[date], list[str]]:
    latest_date: Optional[date] = None
    errors: list[str] = []
    paths = cache_paths if cache_paths is not None else list_indices_cache_paths()
    for path in paths:
        try:
            cache_df = load_indices_cache_file(str(path))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if cache_df.empty:
            continue
        try:
            normalized_cache_df = normalize_indices_cache_for_comparison(cache_df)
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if normalized_cache_df.empty:
            continue
        available_dates = normalized_cache_df["date"].dropna().tolist()
        if not available_dates:
            continue
        path_latest_date = max(available_dates)
        if latest_date is None or path_latest_date > latest_date:
            latest_date = path_latest_date
    return latest_date, errors


def get_latest_prices_cache_date(
    frequency: str,
    cache_paths: Optional[list[Path]] = None,
    *,
    on_or_before: Optional[date] = None,
) -> Tuple[Optional[date], list[str]]:
    latest_date: Optional[date] = None
    errors: list[str] = []
    paths = cache_paths if cache_paths is not None else list_prices_cache_paths(frequency)
    for path in paths:
        try:
            cache_df = load_prices_cache_file(str(path))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if cache_df.empty:
            continue
        try:
            normalized_cache_df = normalize_prices_cache_for_check(cache_df)
        except ValueError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        if normalized_cache_df.empty:
            continue
        if on_or_before is not None:
            normalized_cache_df = normalized_cache_df[normalized_cache_df["date"] <= on_or_before]
            if normalized_cache_df.empty:
                continue
        available_dates = normalized_cache_df["date"].dropna().tolist()
        if not available_dates:
            continue
        path_latest_date = max(available_dates)
        if latest_date is None or path_latest_date > latest_date:
            latest_date = path_latest_date
    return latest_date, errors


def get_report_select_import_state(selected_date: date) -> Dict[str, object]:
    resolved_report_path, expected_candidates = resolve_report_select_path(selected_date)
    return {
        "selected_date": selected_date,
        "report_select_exists": resolved_report_path is not None,
        "report_select_path": resolved_report_path,
        "report_select_candidates": expected_candidates,
    }


def evaluate_home_import_checks(
    selected_date: date,
    cache_paths: Optional[list[Path]] = None,
    daily_price_cache_paths: Optional[list[Path]] = None,
    weekly_price_cache_paths: Optional[list[Path]] = None,
) -> Dict[str, object]:
    report_select_state = get_report_select_import_state(selected_date)
    active_cache_paths = cache_paths if cache_paths is not None else list_indices_cache_paths()
    latest_indices_date, indices_cache_errors = get_latest_indices_cache_date(active_cache_paths)
    active_daily_price_cache_paths = (
        daily_price_cache_paths if daily_price_cache_paths is not None else list_prices_cache_paths("daily")
    )
    active_weekly_price_cache_paths = (
        weekly_price_cache_paths if weekly_price_cache_paths is not None else list_prices_cache_paths("weekly")
    )
    latest_daily_prices_date, daily_prices_cache_errors = get_latest_prices_cache_date(
        "daily",
        active_daily_price_cache_paths,
    )
    latest_weekly_prices_date, weekly_prices_cache_errors = get_latest_prices_cache_date(
        "weekly",
        active_weekly_price_cache_paths,
    )
    indices_check_passed = latest_indices_date is not None and latest_indices_date >= selected_date
    daily_prices_check_passed = latest_daily_prices_date is not None and latest_daily_prices_date >= selected_date
    weekly_prices_check_passed = (
        latest_weekly_prices_date is not None
        and latest_weekly_prices_date + timedelta(days=4) >= selected_date
    )
    overall_ready = bool(report_select_state["report_select_exists"]) and all(
        [indices_check_passed, daily_prices_check_passed, weekly_prices_check_passed]
    )
    return {
        **report_select_state,
        "latest_indices_date": latest_indices_date,
        "indices_check_passed": indices_check_passed,
        "indices_cache_paths": active_cache_paths,
        "indices_cache_errors": indices_cache_errors,
        "latest_daily_prices_date": latest_daily_prices_date,
        "daily_prices_check_passed": daily_prices_check_passed,
        "daily_prices_cache_paths": active_daily_price_cache_paths,
        "daily_prices_cache_errors": daily_prices_cache_errors,
        "latest_weekly_prices_date": latest_weekly_prices_date,
        "weekly_prices_check_passed": weekly_prices_check_passed,
        "weekly_prices_cache_paths": active_weekly_price_cache_paths,
        "weekly_prices_cache_errors": weekly_prices_cache_errors,
        "overall_ready": overall_ready,
    }


def load_indices_cache_state(cache_year: Optional[int] = None) -> Dict[str, object]:
    resolved_year = cache_year or date.today().year
    cache_file = indices_cache_path(resolved_year)
    state: Dict[str, object] = {
        "cache_year": resolved_year,
        "cache_file": cache_file,
        "cache_df": None,
        "normalized_cache_df": None,
        "available_dates": [],
        "warning_message": None,
        "error_message": None,
    }
    if not cache_file.exists():
        state["warning_message"] = "No indices cache found for current year. Use Home > Indices Import to create it."
        return state

    try:
        cache_df = load_indices_cache_file(str(cache_file))
    except Exception as exc:  # pragma: no cover - UI feedback
        state["error_message"] = f"Failed reading indices cache: {exc}"
        return state

    state["cache_df"] = cache_df
    if cache_df.empty:
        state["warning_message"] = "Indices cache is empty. Refresh it from Home > Indices Import."
        return state

    try:
        normalized_cache_df = normalize_indices_cache_for_comparison(cache_df)
    except ValueError as exc:
        state["error_message"] = str(exc)
        return state

    state["normalized_cache_df"] = normalized_cache_df
    if normalized_cache_df.empty:
        state["warning_message"] = "No matching index rows found in cache after ticker/date normalization."
        return state

    available_dates = sorted(normalized_cache_df["date"].dropna().unique().tolist())
    state["available_dates"] = available_dates
    if not available_dates:
        state["warning_message"] = "No valid dates available in cache."
    return state


def load_prices_cache_state(frequency: str, cache_year: Optional[int] = None) -> Dict[str, object]:
    resolved_year = cache_year or date.today().year
    cache_file = prices_cache_path(frequency, resolved_year)
    state: Dict[str, object] = {
        "frequency": frequency,
        "cache_year": resolved_year,
        "cache_file": cache_file,
        "cache_df": None,
        "normalized_cache_df": None,
        "available_dates": [],
        "latest_date": None,
        "row_count": 0,
        "warning_message": None,
        "error_message": None,
    }
    if not cache_file.exists():
        state["warning_message"] = (
            f"No {frequency} prices cache found for current year. Use Home > Prices Import to create it."
        )
        return state

    try:
        cache_df = load_prices_cache_file(str(cache_file))
    except Exception as exc:  # pragma: no cover - UI feedback
        state["error_message"] = f"Failed reading {frequency} prices cache: {exc}"
        return state

    state["cache_df"] = cache_df
    if cache_df.empty:
        state["warning_message"] = (
            f"{frequency.capitalize()} prices cache is empty. Refresh it from Home > Prices Import."
        )
        return state

    try:
        normalized_cache_df = normalize_prices_cache_for_check(cache_df)
    except ValueError as exc:
        state["error_message"] = str(exc)
        return state

    state["normalized_cache_df"] = normalized_cache_df
    if normalized_cache_df.empty:
        state["warning_message"] = (
            f"No valid {frequency} price rows found in cache after ticker/date normalization."
        )
        return state

    available_dates = sorted(normalized_cache_df["date"].dropna().unique().tolist())
    state["available_dates"] = available_dates
    state["row_count"] = int(len(normalized_cache_df))
    state["latest_date"] = available_dates[-1] if available_dates else None
    if not available_dates:
        state["warning_message"] = f"No valid dates available in {frequency} prices cache."
    return state


def render_indices_cache_state_feedback(cache_state: Dict[str, object], *, hint: Optional[str] = None) -> bool:
    error_message = cache_state.get("error_message")
    if error_message:
        st.error(str(error_message))
        if hint:
            st.caption(hint)
        return False

    warning_message = cache_state.get("warning_message")
    if warning_message:
        st.warning(str(warning_message))
        if hint:
            st.caption(hint)
        return False
    return True


def render_home_check_subtab(config: ReportConfig) -> None:
    render_subtab_group_intro(
        "Check",
        "Select a date and validate whether the main import files are already prepared.",
    )
    selected_date = render_report_select_date_input(
        "Check date",
        value=get_default_board_eod(config),
        key="home_check_date",
    )
    check_state = evaluate_home_import_checks(selected_date)
    report_select_path = check_state.get("report_select_path")
    report_candidates = check_state.get("report_select_candidates", (None, None))
    latest_indices_date = check_state.get("latest_indices_date")
    latest_daily_prices_date = check_state.get("latest_daily_prices_date")
    latest_weekly_prices_date = check_state.get("latest_weekly_prices_date")
    indices_check_passed = bool(check_state.get("indices_check_passed"))
    daily_prices_check_passed = bool(check_state.get("daily_prices_check_passed"))
    weekly_prices_check_passed = bool(check_state.get("weekly_prices_check_passed"))
    overall_ready = bool(check_state.get("overall_ready"))

    report_note = (
        report_select_path.name
        if isinstance(report_select_path, Path)
        else f"Expected {report_candidates[0].name} or {report_candidates[1].name}"
    )
    indices_note = (
        f"Latest cached date: {latest_indices_date.isoformat()}"
        if isinstance(latest_indices_date, date)
        else "No valid indices cache date found"
    )
    daily_prices_note = (
        f"Latest cached date: {latest_daily_prices_date.isoformat()}"
        if isinstance(latest_daily_prices_date, date)
        else "No valid daily prices cache date found"
    )
    weekly_prices_note = (
        f"Latest cached date: {latest_weekly_prices_date.isoformat()} (covers through {(latest_weekly_prices_date + timedelta(days=4)).isoformat()})"
        if isinstance(latest_weekly_prices_date, date)
        else "No valid weekly prices cache date found"
    )

    status_cols = st.columns(5)
    with status_cols[0]:
        render_kpi_card(
            "Report Excel imported",
            "PASS" if check_state.get("report_select_exists") else "MISSING",
            report_note,
            "positive" if check_state.get("report_select_exists") else "warn",
        )
    with status_cols[1]:
        render_kpi_card(
            "Indices cache >= check date",
            "PASS" if indices_check_passed else "CHECK",
            indices_note,
            "positive" if indices_check_passed else "warn",
        )
    with status_cols[2]:
        render_kpi_card(
            "Daily prices >= check date",
            "PASS" if daily_prices_check_passed else "CHECK",
            daily_prices_note,
            "positive" if daily_prices_check_passed else "warn",
        )
    with status_cols[3]:
        render_kpi_card(
            "Weekly date + 4d >= check date",
            "PASS" if weekly_prices_check_passed else "CHECK",
            weekly_prices_note,
            "positive" if weekly_prices_check_passed else "warn",
        )
    with status_cols[4]:
        render_kpi_card(
            "Overall import readiness",
            "READY" if overall_ready else "INCOMPLETE",
            selected_date.isoformat(),
            "positive" if overall_ready else "warn",
        )

    details_df = pd.DataFrame(
        [
            {
                "Check": "Report Excel imported",
                "Status": "PASS" if check_state.get("report_select_exists") else "MISSING",
                "Details": report_note,
            },
            {
                "Check": "Indices latest cache date >= selected date",
                "Status": "PASS" if indices_check_passed else "STALE / MISSING",
                "Details": indices_note,
            },
            {
                "Check": "Daily prices latest cache date >= selected date",
                "Status": "PASS" if daily_prices_check_passed else "STALE / MISSING",
                "Details": daily_prices_note,
            },
            {
                "Check": "Weekly prices latest cache date + 4 days >= selected date",
                "Status": "PASS" if weekly_prices_check_passed else "STALE / MISSING",
                "Details": weekly_prices_note,
            },
        ]
    )
    st.dataframe(details_df, use_container_width=True, hide_index=True)

    render_chip_row([
        f"Indices cache files scanned: {len(check_state.get('indices_cache_paths', []))}",
        f"Daily prices cache files scanned: {len(check_state.get('daily_prices_cache_paths', []))}",
        f"Weekly prices cache files scanned: {len(check_state.get('weekly_prices_cache_paths', []))}",
        f"Selected date: {selected_date.isoformat()}",
    ])

    indices_cache_errors = check_state.get("indices_cache_errors", [])
    if indices_cache_errors:
        st.warning("Some indices cache files could not be used for the check.")
        st.code("\n".join(str(entry) for entry in indices_cache_errors))

    daily_prices_cache_errors = check_state.get("daily_prices_cache_errors", [])
    if daily_prices_cache_errors:
        st.warning("Some daily prices cache files could not be used for the check.")
        st.code("\n".join(str(entry) for entry in daily_prices_cache_errors))

    weekly_prices_cache_errors = check_state.get("weekly_prices_cache_errors", [])
    if weekly_prices_cache_errors:
        st.warning("Some weekly prices cache files could not be used for the check.")
        st.code("\n".join(str(entry) for entry in weekly_prices_cache_errors))


def render_home_report_excel_import(config: ReportConfig) -> None:
    render_subtab_group_intro(
        "Report Excel Import",
        "Generate report_select Excel only and review the current EOD cache status.",
    )
    default_anchor = config.eod_as_of_date or date.fromisoformat(bucharest_today_str())
    controls_col, toggles_col = st.columns([1.15, 1])
    with controls_col:
        home_anchor_date = render_report_select_date_input(
            "Report-select date (EOD anchor)",
            value=default_anchor,
            key="home_report_select_date",
        )
    with toggles_col:
        run_sql_home = st.checkbox(
            "Run SQL (force overwrite for selected date)",
            value=False,
            key="home_run_sql_toggle",
        )
    available_dates = get_available_report_select_dates()
    latest_date = available_dates[-1] if available_dates else None
    latest_source, _ = resolve_report_select_path(latest_date) if latest_date else (None, (None, None))
    latest_rows = "n/a"
    if latest_source:
        try:
            latest_rows = str(len(load_report_select(str(latest_source))))
        except Exception:
            latest_rows = "Unavailable"

    selected_source, _ = resolve_report_select_path(home_anchor_date)
    selected_status = "ready" if selected_source else "missing"

    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        render_kpi_card(
            "Latest report_select",
            latest_date.isoformat() if latest_date else "None",
            "Most recent EOD cache available",
            "neutral" if latest_date else "warn",
        )
    with kpi_cols[1]:
        render_kpi_card(
            "Rows in latest file",
            latest_rows,
            latest_source.name if latest_source else "No report_select file yet",
            "positive" if latest_source else "warn",
        )
    with kpi_cols[2]:
        render_kpi_card(
            "Selected EOD status",
            selected_status.upper(),
            home_anchor_date.isoformat(),
            "positive" if selected_source else "warn",
        )

    render_chip_row([
        f"Output target: {report_cache_path(cache_date=home_anchor_date).name}",
    ])
    st.caption(f"Output: {report_cache_path(cache_date=home_anchor_date)}")
    with st.expander("Home logs", expanded=True):
        home_placeholder = st.empty()
        st.session_state["home_log_placeholder"] = home_placeholder
        home_placeholder.code(st.session_state.get("home_logs", "") or "(no logs yet)")
        if st.button("Clear home logs", key="home_clear_logs"):
            st.session_state["home_logs"] = ""
            home_placeholder.code("(no logs yet)")
        render_log_timeline(
            st.session_state.get("home_logs", ""),
            "No runtime events yet. Trigger generation to populate logs.",
        )
    if st.button("Generate report_select Excel", use_container_width=True, key="home_generate_report_select"):
        with st.spinner("Generating report_select..."):
            run_report_select_export(home_anchor_date, run_sql_home)


def render_home_indices_import_subtab() -> None:
    render_subtab_group_intro(
        "Indices Import",
        "Refresh the yearly indices cache used by the comparison tab and by the Home checks.",
    )
    today_local = date.today()
    cache_year = today_local.year
    default_cutoff_date = date(cache_year - 1, 12, 31)
    cutoff_date = render_report_select_date_input(
        "SQL start date (exclusive)",
        value=default_cutoff_date,
        key="home_indices_cutoff_date",
        help="Query uses date > selected day at 00:00:00.",
    )
    cache_file = indices_cache_path(cache_year)
    render_chip_row(
        [
            f"Tickers in scope: {len(INDEX_TICKERS)}",
            f"Yearly cache file: {cache_file.name}",
            f"Cache year: {cache_year}",
        ]
    )
    if st.button("Update indices cache", use_container_width=True, key="home_indices_update_cache"):
        with st.spinner("Fetching indices OHLC data from database..."):
            try:
                fetched_df = fetch_indices_ohlc_since(cutoff_date)
                saved_path = save_indices_cache(fetched_df, cache_year)
                load_indices_cache_file.clear()
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"Indices cache update failed: {exc}")
            else:
                st.success(f"Indices cache updated: {saved_path} ({len(fetched_df)} rows)")

    cache_state = load_indices_cache_state(cache_year)
    st.caption(f"Cache path: {cache_file}")
    st.caption(f"Last updated: {_format_ts(cache_file)}")
    if not render_indices_cache_state_feedback(
        cache_state,
        hint="Open the Indices tab after the cache is ready to compare and edit closing prices.",
    ):
        return

    available_dates = cache_state.get("available_dates", [])
    if available_dates:
        st.caption(
            "Available date range: "
            f"{available_dates[0].isoformat()} -> {available_dates[-1].isoformat()} "
            f"({len(available_dates)} dates)"
        )


def render_home_prices_import_subtab() -> None:
    render_subtab_group_intro(
        "Prices Import",
        "Import yearly daily or weekly ticker price history used by thematic and market-regime workflows.",
    )
    st.caption(
        "Each prices import also recomputes `rsi_14` and `rsi_divergence_flag` for that frequency. Specific-ticker imports refresh both indicators only for the selected tickers."
    )
    today_local = date.today()
    cache_year = today_local.year
    default_cutoff_date = date(cache_year - 1, 12, 31)
    cutoff_date = render_report_select_date_input(
        "SQL start date (exclusive)",
        value=default_cutoff_date,
        key="home_prices_cutoff_date",
        help="Query uses date > selected day at 00:00:00.",
    )
    scope_label = st.radio(
        "Import scope",
        ["All tickers", "Specific tickers"],
        horizontal=True,
        key="home_prices_scope",
    )
    manual_tickers_text = ""
    normalized_manual_tickers: list[str] = []
    if scope_label == "Specific tickers":
        manual_tickers_text = st.text_area(
            "Tickers",
            key="home_prices_manual_tickers",
            placeholder="AAPL\nMSFT.US\nNVDA",
            help="Separate tickers with commas, spaces, or new lines. Missing .US will be added automatically.",
        )
        normalized_manual_tickers = parse_manual_price_tickers(manual_tickers_text)
        if normalized_manual_tickers:
            preview = ", ".join(normalized_manual_tickers[:12])
            if len(normalized_manual_tickers) > 12:
                preview = f"{preview}, ..."
            render_chip_row([
                f"Normalized tickers: {len(normalized_manual_tickers)}",
                f"Preview: {preview}",
            ])
        else:
            st.caption("Enter one or more tickers to run a specific-ticker import.")
    else:
        st.caption(
            "All tickers refreshes the live company overview endpoint and intersects it with local screener-eligible tickers when you run an import."
        )

    daily_cache_file = prices_cache_path("daily", cache_year)
    weekly_cache_file = prices_cache_path("weekly", cache_year)
    render_chip_row(
        [
            f"Daily cache file: {daily_cache_file.name}",
            f"Weekly cache file: {weekly_cache_file.name}",
            f"Cache year: {cache_year}",
        ]
    )

    def _run_prices_import(frequency: str) -> None:
        scope_key = "all" if scope_label == "All tickers" else "specific"
        requested_tickers = normalized_manual_tickers if scope_key == "specific" else []
        if scope_key == "specific" and not requested_tickers:
            st.error("Enter at least one ticker before running a specific-ticker prices import.")
            return
        with st.spinner(f"Importing {frequency} prices..."):
            try:
                result = import_prices_cache(
                    frequency,
                    cutoff_date,
                    scope=scope_key,
                    manual_tickers=requested_tickers,
                )
                invalidate_prices_cache_views()
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"{frequency.capitalize()} prices import failed: {exc}")
            else:
                latest_date = result.get("latest_date")
                latest_date_note = latest_date.isoformat() if isinstance(latest_date, date) else "n/a"
                st.success(
                    f"{frequency.capitalize()} prices cache updated: {result['saved_path']} "
                    f"({result['saved_rows']} rows, latest date {latest_date_note}, "
                    f"tickers requested {result['requested_tickers_count']})."
                )

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("Import daily prices", use_container_width=True, key="home_prices_import_daily"):
            _run_prices_import("daily")
    with action_cols[1]:
        if st.button("Import weekly prices", use_container_width=True, key="home_prices_import_weekly"):
            _run_prices_import("weekly")

    daily_state = load_prices_cache_state("daily", cache_year)
    weekly_state = load_prices_cache_state("weekly", cache_year)
    summary_cols = st.columns(2)
    with summary_cols[0]:
        st.markdown("**Daily cache**")
        st.caption(f"Cache path: {daily_cache_file}")
        st.caption(f"Last updated: {_format_ts(daily_cache_file)}")
        if render_indices_cache_state_feedback(
            daily_state,
            hint="Run Home > Prices Import > Import daily prices to populate the cache.",
        ):
            render_chip_row([
                f"Rows: {daily_state.get('row_count', 0)}",
                f"Latest date: {daily_state.get('latest_date').isoformat() if isinstance(daily_state.get('latest_date'), date) else 'n/a'}",
            ])
    with summary_cols[1]:
        st.markdown("**Weekly cache**")
        st.caption(f"Cache path: {weekly_cache_file}")
        st.caption(f"Last updated: {_format_ts(weekly_cache_file)}")
        if render_indices_cache_state_feedback(
            weekly_state,
            hint="Run Home > Prices Import > Import weekly prices to populate the cache.",
        ):
            render_chip_row([
                f"Rows: {weekly_state.get('row_count', 0)}",
                f"Latest date: {weekly_state.get('latest_date').isoformat() if isinstance(weekly_state.get('latest_date'), date) else 'n/a'}",
            ])


def _bands_to_df(bands: list[dict[str, object]], score_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Score": score_name,
                "Min": band.get("min"),
                "Max": band.get("max"),
                "Label": band.get("label"),
            }
            for band in bands
        ]
    )


def _weights_to_df(config_payload: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for block_name, value in config_payload.items():
        if not isinstance(value, dict):
            continue
        scalar_values = {
            str(key): scalar
            for key, scalar in value.items()
            if not isinstance(scalar, (dict, list))
        }
        if not scalar_values:
            continue
        for key, scalar in scalar_values.items():
            rows.append({"Block": block_name, "Key": key, "Value": scalar})
    return pd.DataFrame(rows)


def _preference_scores_to_df(config_payload: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime_label, regime_payload in config_payload.get("regime_preference_scores", {}).items():
        if not isinstance(regime_payload, dict):
            continue
        for family, score in regime_payload.get("family_defaults", {}).items():
            rows.append(
                {
                    "Regime": regime_label,
                    "Scope": f"family:{family}",
                    "Preference Score": score,
                }
            )
        for sector_name, score in regime_payload.get("sector_overrides", {}).items():
            rows.append(
                {
                    "Regime": regime_label,
                    "Scope": f"sector:{sector_name}",
                    "Preference Score": score,
                }
            )
    return pd.DataFrame(rows)


def _alignment_scores_to_df(config_payload: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for regime_label, regime_payload in config_payload.get("market_alignment_scores", {}).items():
        if not isinstance(regime_payload, dict):
            continue
        for family, stock_map in regime_payload.get("family_defaults", {}).items():
            if not isinstance(stock_map, dict):
                continue
            for stock_label, score in stock_map.items():
                rows.append(
                    {
                        "Regime": regime_label,
                        "Scope": f"family:{family}",
                        "Stock Regime": stock_label,
                        "Alignment Score": score,
                    }
                )
        for sector_name, stock_map in regime_payload.get("sector_overrides", {}).items():
            if not isinstance(stock_map, dict):
                continue
            for stock_label, score in stock_map.items():
                rows.append(
                    {
                        "Regime": regime_label,
                        "Scope": f"sector:{sector_name}",
                        "Stock Regime": stock_label,
                        "Alignment Score": score,
                    }
                )
    return pd.DataFrame(rows)


def render_market_methodology_subtab() -> None:
    render_subtab_group_intro(
        "Market - Methodology",
        "Deterministic formulas, mappings, and interpretation rules that drive the Market layer.",
    )
    market_config = load_market_regime_config()
    sector_families = load_sector_families()
    st.markdown(load_market_methodology_text())

    bands_df = pd.concat(
        [
            _bands_to_df(bands, score_name)
            for score_name, bands in market_config.get("score_bands", {}).items()
            if isinstance(bands, list)
        ],
        ignore_index=True,
    )
    if not bands_df.empty:
        st.markdown("**Label bands**")
        st.dataframe(bands_df, width="stretch", hide_index=True)

    weights_df = _weights_to_df(market_config)
    if not weights_df.empty:
        st.markdown("**Weights and mapping anchors**")
        st.dataframe(weights_df, width="stretch", hide_index=True)

    family_rows = [
        {"Family": family, "Sectors": ", ".join(sectors)}
        for family, sectors in sector_families.items()
    ]
    st.markdown("**Sector families**")
    st.dataframe(pd.DataFrame(family_rows), width="stretch", hide_index=True)

    preference_df = _preference_scores_to_df(market_config)
    if not preference_df.empty:
        st.markdown("**Regime preference scores**")
        st.dataframe(preference_df, width="stretch", hide_index=True)

    alignment_df = _alignment_scores_to_df(market_config)
    if not alignment_df.empty:
        st.markdown("**Market alignment scores**")
        st.dataframe(alignment_df, width="stretch", hide_index=True)


def render_market_ai_commentary_subtab() -> None:
    render_subtab_group_intro(
        "ai-commentary",
        "Reserved for future OpenAI Responses API market commentary workflows.",
    )
    st.info(
        "This area is scaffold-only in the current phase. Deterministic market scores are computed and cached here first; AI commentary will be layered on later."
    )
    render_chip_row(
        [
            "Future input: market_snapshot cache",
            "Future input: stock_rsi_regime cache",
            "Future input: setup_readiness cache",
            "Future input: current market news context",
        ]
    )
    st.caption("Use the existing API tab for manual Responses API experimentation until this workflow is implemented.")


def render_market_values_subtab(config: ReportConfig) -> None:
    render_subtab_group_intro(
        "Market - Values",
        "Operational market regime outputs, sector fit, participation, risk appetite, and cache visibility for the selected anchors.",
    )
    market_config = load_market_regime_config()
    sector_families = load_sector_families()
    available_dates = list(get_available_report_select_dates())
    if not available_dates:
        st.warning("No report_select files are available yet. Generate them from Home > Report Excel Import first.")
        return

    defaults = get_default_market_anchors(available_dates, market_config)
    interval_config = market_config.get("default_intervals_days", {})
    eval_col, prev_eval_col, rsi_col = st.columns(3)
    with eval_col:
        evaluation_date = render_report_select_date_input(
            "Evaluation date",
            value=defaults["evaluation_date"],
            key="market_eval_date",
        )
    previous_evaluation_candidates = sorted(entry for entry in available_dates if entry < evaluation_date)
    previous_evaluation_default = (
        previous_evaluation_candidates[-1] if previous_evaluation_candidates else evaluation_date
    )
    with prev_eval_col:
        previous_evaluation_date = render_report_select_date_input(
            "vs prev evaluation date",
            value=previous_evaluation_default,
            key="market_prev_eval_date",
        )
    rsi_start_default = evaluation_date - timedelta(days=int(interval_config.get("rsi_window_days", 90)))
    month_default = resolve_anchor_on_or_before(
        available_dates,
        evaluation_date - timedelta(days=int(interval_config.get("month_offset_days", 30))),
    )
    week_default = resolve_anchor_on_or_before(
        available_dates,
        evaluation_date - timedelta(days=int(interval_config.get("week_offset_days", 7))),
    )
    with rsi_col:
        rsi_start_date = render_report_select_date_input(
            "RSI regime start date",
            value=rsi_start_default,
            key="market_rsi_start_date",
        )
    month_col, week_col = st.columns(2)
    with month_col:
        month_anchor_date = render_report_select_date_input(
            "1 month ago date",
            value=month_default,
            key="market_month_anchor_date",
        )
    with week_col:
        week_anchor_date = render_report_select_date_input(
            "1 week ago date",
            value=week_default,
            key="market_week_anchor_date",
        )

    cache_key = build_market_cache_key(evaluation_date)
    previous_cache_key = build_market_cache_key(previous_evaluation_date)
    cache_state = market_cache_status(cache_key)
    previous_cache_state = market_cache_status(previous_cache_key)
    render_chip_row(
        [
            f"Cache key: {cache_key}",
            f"Current cache: {'ready' if cache_state.get('ready') else 'missing'}",
            f"Comparison cache: {'ready' if previous_cache_state.get('ready') else 'missing'}",
            f"Evaluation: {evaluation_date.isoformat()}",
            f"vs prev: {previous_evaluation_date.isoformat()}",
            f"RSI start: {rsi_start_date.isoformat()}",
            f"1M anchor: {month_anchor_date.isoformat()}",
            f"1W anchor: {week_anchor_date.isoformat()}",
        ]
    )
    compute_button_label = "Recompute market caches" if cache_state.get("ready") else "Compute market caches"
    run_market_compute = st.button(
        compute_button_label,
        use_container_width=False,
        key="market_recompute_button",
        help="Create or overwrite the market snapshot, stock RSI regime cache, and setup readiness cache for the selected evaluation date.",
    )

    bundle: Optional[dict[str, object]] = None
    if run_market_compute:
        evaluation_df, evaluation_path, evaluation_candidates, evaluation_error = load_report_select_for_eod(evaluation_date)
        if evaluation_path is None:
            render_missing_report_select(evaluation_date, evaluation_candidates)
            return
        if evaluation_error:
            st.error(f"Failed reading {evaluation_path}: {evaluation_error}")
            return

        month_df, month_path, month_candidates, month_error = load_report_select_for_eod(month_anchor_date)
        if month_path is None:
            render_missing_report_select(month_anchor_date, month_candidates)
            return
        if month_error:
            st.error(f"Failed reading {month_path}: {month_error}")
            return

        week_df, week_path, week_candidates, week_error = load_report_select_for_eod(week_anchor_date)
        if week_path is None:
            render_missing_report_select(week_anchor_date, week_candidates)
            return
        if week_error:
            st.error(f"Failed reading {week_path}: {week_error}")
            return

        with st.spinner("Computing Market layer..."):
            bundle = compute_and_save_market_bundle(
                evaluation_df=evaluation_df,
                month_df=month_df,
                week_df=week_df,
                evaluation_date=evaluation_date,
                rsi_start_date=rsi_start_date,
                month_anchor_date=month_anchor_date,
                week_anchor_date=week_anchor_date,
                evaluation_source_path=evaluation_path,
                month_source_path=month_path,
                week_source_path=week_path,
                config=market_config,
                sector_families=sector_families,
                force_recompute=True,
            )
        _load_market_regime_company_metrics_for_date.clear()
    elif cache_state.get("ready"):
        with st.spinner("Loading Market cache..."):
            bundle = load_market_bundle(cache_key)
            bundle["signature"] = cache_key
            bundle["cached"] = True
    else:
        st.info("No market cache exists for the selected evaluation date yet. Click `Compute market caches` to create it.")
        return

    payload = bundle["market_snapshot_payload"]
    metadata = dict(payload.get("metadata", {}))
    market_summary = payload.get("market_summary", {})
    component_scores = payload.get("component_scores", {})
    breadth = payload.get("breadth", {})
    risk_appetite = payload.get("risk_appetite", {})
    family_scores_df = pd.DataFrame(payload.get("family_scores", []))
    sector_df = pd.DataFrame(payload.get("sector_rows", []))
    paths = bundle.get("paths", {})
    stock_rsi_df = bundle.get("stock_rsi_regime_df", pd.DataFrame())
    setup_df = bundle.get("setup_readiness_df", pd.DataFrame())
    source_files = dict(metadata.get("source_files", {}))
    source_names = [
        Path(str(source_files.get("evaluation", ""))).name,
        Path(str(source_files.get("month_anchor", ""))).name,
        Path(str(source_files.get("week_anchor", ""))).name,
    ]
    st.caption(
        f"{'Loaded from cache' if bundle.get('cached') else 'Recomputed and saved'} | "
        f"Source files: {', '.join(name for name in source_names if name)}"
    )
    if bundle.get("cached") and any(
        [
            str(metadata.get("rsi_start_date")) != rsi_start_date.isoformat(),
            str(metadata.get("month_anchor_date")) != month_anchor_date.isoformat(),
            str(metadata.get("week_anchor_date")) != week_anchor_date.isoformat(),
        ]
    ):
        st.info(
            "The loaded cache for this evaluation date was computed with different anchors. "
            "Click `Recompute market caches` to overwrite it with the currently selected anchors."
        )

    previous_payload: Optional[dict[str, object]] = None
    if previous_evaluation_date == evaluation_date:
        previous_payload = payload
    elif previous_cache_state.get("ready"):
        previous_payload = load_market_bundle(previous_cache_key)["market_snapshot_payload"]
    else:
        st.info(
            f"No comparison cache exists for {previous_evaluation_date.isoformat()}. "
            "Trend symbols are hidden until that date is computed."
        )

    previous_market_summary = dict(previous_payload.get("market_summary", {})) if previous_payload else {}
    previous_breadth = dict(previous_payload.get("breadth", {})) if previous_payload else {}
    previous_family_scores_df = pd.DataFrame(previous_payload.get("family_scores", [])) if previous_payload else pd.DataFrame()
    previous_sector_df = pd.DataFrame(previous_payload.get("sector_rows", [])) if previous_payload else pd.DataFrame()

    summary_cols = st.columns(4)
    with summary_cols[0]:
        render_kpi_card(
            "Market Regime",
            _render_score_with_symbol(
                market_summary.get("market_regime_score"),
                _metric_trend_symbol(
                    market_summary.get("market_regime_score"),
                    previous_market_summary.get("market_regime_score"),
                ),
            ),
            str(market_summary.get("market_regime_label") or "N/A"),
            tone="neutral",
        )
    with summary_cols[1]:
        render_kpi_card(
            "Confidence",
            _render_score_with_symbol(
                market_summary.get("market_regime_confidence"),
                _metric_trend_symbol(
                    market_summary.get("market_regime_confidence"),
                    previous_market_summary.get("market_regime_confidence"),
                ),
            ),
            str(market_summary.get("market_regime_status") or "N/A"),
            tone="neutral",
        )
    with summary_cols[2]:
        render_kpi_card(
            "Sector Rotation",
            _render_score_with_symbol(
                market_summary.get("market_sector_rotation_score"),
                _metric_trend_symbol(
                    market_summary.get("market_sector_rotation_score"),
                    previous_market_summary.get("market_sector_rotation_score"),
                ),
            ),
            str(market_summary.get("leading_family_classifier") or "N/A"),
            tone="neutral",
        )
    with summary_cols[3]:
        render_kpi_card(
            "Stock RSI Breadth >= 60",
            _render_percent_with_symbol(
                breadth.get("market_rsi_breadth_pct_60"),
                _metric_trend_symbol(
                    breadth.get("market_rsi_breadth_pct_60"),
                    previous_breadth.get("market_rsi_breadth_pct_60"),
                ),
            ),
            ">=75: "
            + _render_percent_with_symbol(
                breadth.get("market_rsi_breadth_pct_75"),
                _metric_trend_symbol(
                    breadth.get("market_rsi_breadth_pct_75"),
                    previous_breadth.get("market_rsi_breadth_pct_75"),
                ),
            ),
            tone="neutral",
        )

    st.markdown("**Underlying components**")
    component_rows = _build_market_component_rows(component_scores, breadth, risk_appetite, previous_payload)
    st.dataframe(component_rows, width="stretch", hide_index=True)
    if risk_appetite.get("warning"):
        st.warning(str(risk_appetite.get("warning")))

    st.markdown("**Family leadership**")
    if family_scores_df.empty:
        st.info("No family scores are available for the selected anchors.")
    else:
        display_family_df = family_scores_df.rename(
            columns={
                "family": "Family",
                "sector_rotation_score": "Sector Rotation Score",
                "sector_count": "Sector Count",
            }
        )
        previous_family_display_df = previous_family_scores_df.rename(
            columns={
                "family": "Family",
                "sector_rotation_score": "Sector Rotation Score",
                "sector_count": "Sector Count",
            }
        )
        if not previous_family_display_df.empty:
            display_family_df = apply_trend_symbols_to_table(
                display_family_df,
                previous_family_display_df,
                ["Sector Rotation Score"],
                threshold=MARKET_TREND_THRESHOLD,
            )
        st.dataframe(display_family_df, width="stretch", hide_index=True)

    st.markdown("**Sector table**")
    if sector_df.empty:
        st.info("No sector rows are available for the selected anchors.")
    else:
        sector_display_df = sector_df.rename(
            columns={
                "sector": "Sector",
                "family": "Family",
                "P_now": "P",
                "T_now": "T",
                "dP": "dP",
                "dT": "dT",
                "trend_of_change_score": "Trend of Change",
                "sector_rsi_breadth_pct_60": "RSI Breadth >= 60",
                "sector_rsi_breadth_pct_75": "RSI Breadth >= 75",
                "sector_rsi_breadth_pct_lt40": "RSI Breadth < 40",
                "sector_rsi_participation_composite_score": "RSI Participation Composite",
                "sector_rotation_score": "Sector Rotation Score",
                "sector_regime_fit_score": "Sector Regime Fit Score",
                "sector_regime_fit_flag": "Regime Fit Flag",
            }
        )
        previous_sector_display_df = previous_sector_df.rename(
            columns={
                "sector": "Sector",
                "family": "Family",
                "P_now": "P",
                "T_now": "T",
                "dP": "dP",
                "dT": "dT",
                "trend_of_change_score": "Trend of Change",
                "sector_rsi_breadth_pct_60": "RSI Breadth >= 60",
                "sector_rsi_breadth_pct_75": "RSI Breadth >= 75",
                "sector_rsi_breadth_pct_lt40": "RSI Breadth < 40",
                "sector_rsi_participation_composite_score": "RSI Participation Composite",
                "sector_rotation_score": "Sector Rotation Score",
                "sector_regime_fit_score": "Sector Regime Fit Score",
                "sector_regime_fit_flag": "Regime Fit Flag",
            }
        )
        ordered_columns = [
            "Sector",
            "Family",
            "P",
            "T",
            "dP",
            "dT",
            "Trend of Change",
            "RSI Breadth >= 60",
            "RSI Breadth >= 75",
            "RSI Breadth < 40",
            "RSI Participation Composite",
            "Sector Rotation Score",
            "Sector Regime Fit Score",
            "Regime Fit Flag",
        ]
        ordered_columns = [column for column in ordered_columns if column in sector_display_df.columns]
        sector_display_df = sector_display_df.loc[:, ordered_columns].sort_values(
            by="Sector Rotation Score" if "Sector Rotation Score" in ordered_columns else ordered_columns[0],
            ascending=False,
            kind="stable",
        )
        sector_rsi_lt40_numeric = (
            sector_display_df["RSI Breadth < 40"].copy() if "RSI Breadth < 40" in sector_display_df.columns else None
        )
        if not previous_sector_display_df.empty:
            sector_display_df = apply_trend_symbols_to_table(
                sector_display_df,
                previous_sector_display_df.loc[:, [column for column in ordered_columns if column in previous_sector_display_df.columns]],
                [
                    "RSI Breadth >= 60",
                    "RSI Breadth >= 75",
                    "RSI Breadth < 40",
                    "RSI Participation Composite",
                    "Sector Rotation Score",
                    "Sector Regime Fit Score",
                ],
                threshold=MARKET_TREND_THRESHOLD,
            )
            if sector_rsi_lt40_numeric is not None:
                sector_display_df["RSI Breadth < 40"] = sector_rsi_lt40_numeric
        def _style_sector_fit_row(row: pd.Series) -> list[str]:
            fit_flag = str(row.get("Regime Fit Flag", "")).strip().lower()
            if fit_flag == "favored":
                color = "background-color: #eefbf1"
            elif fit_flag == "avoid":
                color = "background-color: #fff1f1"
            else:
                color = "background-color: #f8fafc"
            return [color] * len(row)

        sector_styler = sector_display_df.style.apply(_style_sector_fit_row, axis=1)
        try:
            sector_styler = sector_styler.hide(axis="index")
        except Exception:
            pass
        st.dataframe(sector_styler, width="stretch")

    st.markdown("**Cache visibility**")
    render_chip_row(
        [
            f"Market snapshot: {Path(paths.get('market_snapshot', cache_state.get('market_snapshot'))).name}",
            f"Stock RSI cache rows: {len(stock_rsi_df)}",
            f"Setup readiness rows: {len(setup_df)}",
        ]
    )
    st.caption(f"Market snapshot path: {paths.get('market_snapshot', cache_state.get('market_snapshot'))}")
    st.caption(f"Stock RSI cache path: {paths.get('stock_rsi_regime', cache_state.get('stock_rsi_regime'))}")
    st.caption(f"Setup readiness path: {paths.get('setup_readiness', cache_state.get('setup_readiness'))}")


def render_market_tab(config: ReportConfig) -> None:
    render_page_intro(
        "Market",
        "Deterministic market regime, sector fit, and setup-readiness layer built on report_select snapshots plus cached RSI histories.",
        "Equipilot / Market",
    )
    values_tab, methodology_tab, ai_commentary_tab = st.tabs(["Values", "Methodology", "AI commentary"])
    with values_tab:
        render_market_values_subtab(config)
    with methodology_tab:
        render_market_methodology_subtab()
    with ai_commentary_tab:
        render_market_ai_commentary_subtab()


def render_indices_tab() -> None:
    render_page_intro(
        "Indices",
        "Compare close values across two dates and manually adjust cached closes when needed.",
        "Equipilot / Indices",
    )
    cache_state = load_indices_cache_state()
    if not render_indices_cache_state_feedback(
        cache_state,
        hint="Populate or refresh the cache from Home > Indices Import.",
    ):
        return

    cache_df = cache_state.get("cache_df")
    normalized_cache_df = cache_state.get("normalized_cache_df")
    available_dates = cache_state.get("available_dates", [])
    cache_year = cache_state.get("cache_year")
    if cache_df is None or normalized_cache_df is None or not available_dates or cache_year is None:
        st.warning("Indices cache is not ready for comparison yet.")
        st.caption("Populate or refresh the cache from Home > Indices Import.")
        return

    date_col_1, date_col_2 = st.columns(2)
    with date_col_1:
        selected_date_1 = render_report_select_date_input(
            "Date 1",
            value=available_dates[0],
            key="indices_date_1",
        )
    with date_col_2:
        selected_date_2 = render_report_select_date_input(
            "Date 2",
            value=available_dates[-1],
            key="indices_date_2",
        )

    if selected_date_1 > selected_date_2:
        st.warning("Date 1 is after Date 2. Variation will be computed with selected order.")

    comparison_df = build_indices_comparison_table(
        normalized_cache_df,
        selected_date_1,
        selected_date_2,
    )
    date_1_col = f"{selected_date_1.isoformat()} Close"
    date_2_col = f"{selected_date_2.isoformat()} Close"
    comparison_df = comparison_df.rename(
        columns={
            "Close Date 1": date_1_col,
            "Close Date 2": date_2_col,
        }
    )
    comparison_df[date_1_col] = pd.to_numeric(comparison_df[date_1_col], errors="coerce")
    comparison_df[date_2_col] = pd.to_numeric(comparison_df[date_2_col], errors="coerce")
    editor_key = (
        f"indices_editor_{selected_date_1.isoformat()}_{selected_date_2.isoformat()}"
    )
    working_df = comparison_df.copy()

    working_df["Variation %"] = working_df.apply(
        lambda row: _compute_variation_pct(row.get(date_1_col), row.get(date_2_col)),
        axis=1,
    )

    st.caption("N/A means the selected date is not available for that index.")
    st.caption("You can edit close prices below; Variation % is recalculated automatically.")

    estimated_height = max(260, 72 + len(working_df) * 38)
    edited_df = st.data_editor(
        working_df,
        hide_index=True,
        use_container_width=True,
        height=estimated_height,
        key=editor_key,
        num_rows="fixed",
        disabled=["Index", "Variation %"],
        column_config={
            "Index": st.column_config.TextColumn("Index"),
            date_1_col: st.column_config.NumberColumn(date_1_col, format="%.2f"),
            date_2_col: st.column_config.NumberColumn(date_2_col, format="%.2f"),
            "Variation %": st.column_config.NumberColumn("Variation %", format="%.2f"),
        },
    )

    edited_df = edited_df.copy()
    edited_df[date_1_col] = pd.to_numeric(edited_df[date_1_col], errors="coerce")
    edited_df[date_2_col] = pd.to_numeric(edited_df[date_2_col], errors="coerce")
    edited_df["Variation %"] = edited_df.apply(
        lambda row: _compute_variation_pct(row.get(date_1_col), row.get(date_2_col)),
        axis=1,
    )

    edited_close_df = edited_df[["Index", date_1_col, date_2_col]].copy()
    current_close_df = working_df[["Index", date_1_col, date_2_col]].copy()
    edits_changed = not edited_close_df["Index"].equals(current_close_df["Index"])
    for close_col in (date_1_col, date_2_col):
        edited_series = pd.to_numeric(edited_close_df[close_col], errors="coerce")
        current_series = pd.to_numeric(current_close_df[close_col], errors="coerce")
        same_missing = edited_series.isna().equals(current_series.isna())
        same_values = np.allclose(
            edited_series.fillna(0.0).to_numpy(dtype=float),
            current_series.fillna(0.0).to_numpy(dtype=float),
        )
        if not (same_missing and same_values):
            edits_changed = True
            break

    if edits_changed:
        index_name_to_ticker = {name: ticker for ticker, name in INDEX_TICKER_TO_NAME.items()}
        updated_cache_df = cache_df.copy()
        for _, row in edited_df.iterrows():
            index_name = str(row.get("Index", "")).strip()
            ticker = index_name_to_ticker.get(index_name)
            if not ticker:
                continue
            close_1 = pd.to_numeric(row.get(date_1_col), errors="coerce")
            close_2 = pd.to_numeric(row.get(date_2_col), errors="coerce")
            if pd.notna(close_1):
                updated_cache_df = _upsert_indices_close(
                    updated_cache_df,
                    ticker=ticker,
                    target_date=selected_date_1,
                    adjusted_close=float(close_1),
                )
            if pd.notna(close_2):
                updated_cache_df = _upsert_indices_close(
                    updated_cache_df,
                    ticker=ticker,
                    target_date=selected_date_2,
                    adjusted_close=float(close_2),
                )
        save_indices_cache(updated_cache_df, int(cache_year))
        load_indices_cache_file.clear()
        st.rerun()


def render_home(config: ReportConfig) -> None:
    render_page_intro(
        "Home",
        "Database imports and readiness checks for report Excel, indices, and prices data.",
        "Equipilot / Home",
    )
    render_subtab_group_intro(
        "Home sections",
        "Use the sub-tabs below to validate imports, generate report Excel files, refresh index data, and import ticker price history.",
    )
    check_tab, report_tab, indices_import_tab, prices_import_tab = st.tabs(
        ["Check", "Report Excel Import", "Indices Import", "Prices Import"]
    )
    with check_tab:
        render_home_check_subtab(config)
    with report_tab:
        render_home_report_excel_import(config)
    with indices_import_tab:
        render_home_indices_import_subtab()
    with prices_import_tab:
        render_home_prices_import_subtab()


def render_monthly_board(config: ReportConfig) -> None:
    render_page_intro(
        "Monthly Sector Report",
        "Configure report dates, run generation, and manage prompt/source files.",
        "Equipilot / Sector / Monthly Sector Report",
    )
    report_date_value = render_report_select_date_input("Report date", value=config.report_date, key="monthly_report_date")
    eod_as_of_value = render_report_select_date_input(
        "EOD as-of date (30-day window anchor)",
        value=config.eod_as_of_date or report_date_value,
        key="monthly_eod_as_of_date",
    )
    if st.button("Save config", key="monthly_save_config"):
        save_report_config(ReportConfig(report_date=report_date_value, eod_as_of_date=eod_as_of_value), CONFIG_PATH)
        st.success(f"Saved config: {CONFIG_PATH}")
        st.rerun()

    effective_cache_date = eod_as_of_value.isoformat()
    output_hint = REPORTS_DIR / f"Monthly_Scoring_Board_{report_date_value.isoformat()}.pdf"
    meta_cols = st.columns(4)
    with meta_cols[0]:
        render_kpi_card("Report date", report_date_value.isoformat(), report_date_value.strftime("%B %d, %Y"))
    with meta_cols[1]:
        render_kpi_card("EOD anchor", eod_as_of_value.isoformat(), "Controls report_select/scoring cache date")
    with meta_cols[2]:
        render_kpi_card("Report cache", report_cache_path(cache_date=eod_as_of_value).name, effective_cache_date)
    with meta_cols[3]:
        render_kpi_card("PDF output", output_hint.name, "Filename remains unchanged")

    render_chip_row(
        [
            f"Report cache path: {report_cache_path(cache_date=eod_as_of_value)}",
            f"Scoring cache path: {scoring_cache_path(cache_date=eod_as_of_value)}",
            f"Cache date in use: {effective_cache_date}",
        ]
    )

    controls_col, _ = st.columns([1, 3])
    with controls_col:
        run_sql_toggle = st.checkbox("Run SQL (ignore cache)", value=False, key="monthly_run_sql_toggle")
        if st.button("Generate PDF", use_container_width=True, key="monthly_generate_pdf"):
            run_generation(report_date_value, run_sql_toggle, eod_as_of_value)
        if st.button("Refresh Files", use_container_width=True, key="monthly_refresh_files"):
            refresh_files()
            st.session_state["force_sync"] = True
            st.rerun()

    logs_tab, prompt_tab, sector_tab, summary_tab, final_prompt_tab = st.tabs(
        ["Logs", "Summary Prompt", "Sector Data Tables", "Summary Text (JSON)", "Summary Prompt Final"]
    )

    with logs_tab:
        placeholder = st.empty()
        st.session_state["log_placeholder"] = placeholder
        placeholder.code(st.session_state.get("logs", "") or "(no logs yet)")
        if st.button("Clear logs", key="monthly_clear_logs"):
            st.session_state["logs"] = ""
            placeholder.code("(no logs yet)")
        render_log_timeline(st.session_state.get("logs", ""), "PDF generation logs will appear here.")

    bundle = st.session_state["file_bundle"]

    with prompt_tab:
        st.caption(f"File: {PROMPT_PATH} (last updated: {_format_ts(PROMPT_PATH)})")
        prompt_text = st.text_area(
            "Prompt contents",
            value=st.session_state.get("prompt_content", ""),
            height=300,
            key="monthly_prompt_content_area",
        )
        st.session_state["prompt_content"] = prompt_text
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save prompt", key="monthly_save_prompt"):
                write_text(PROMPT_PATH, prompt_text)
                refresh_files()
                st.session_state["force_sync"] = True
                st.rerun()
        with c2:
            copy_button(prompt_text, "monthly_prompt_copy")

    with sector_tab:
        sector_path_str = bundle.get("sector_path")
        if not sector_path_str:
            st.warning("No sector_data_tables file found. Generate the PDF first.")
        else:
            sector_path = Path(sector_path_str)
            st.caption(f"Latest file: {sector_path.name} (last updated: {_format_ts(sector_path)})")
            sector_text = st.text_area(
                "Sector tables",
                value=st.session_state.get("sector_content", ""),
                height=300,
                key="monthly_sector_content_area",
            )
            st.session_state["sector_content"] = sector_text
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save sector tables", key="monthly_save_sector"):
                    write_text(sector_path, sector_text)
                    refresh_files()
                    st.session_state["force_sync"] = True
                    st.rerun()
            with c2:
                copy_button(sector_text, "monthly_sector_copy")

    with summary_tab:
        st.caption(f"File: {SUMMARY_JSON_PATH} (last updated: {_format_ts(SUMMARY_JSON_PATH)})")
        summary_text = st.text_area(
            "Summary JSON",
            value=st.session_state.get("summary_content", ""),
            height=300,
            key="monthly_summary_content_area",
        )
        st.session_state["summary_content"] = summary_text
        st.caption(f"Paragraph counts: {summary_counts(summary_text)}")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save summary JSON", key="monthly_save_summary_json"):
                try:
                    json.loads(summary_text)
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid JSON: {exc}")
                else:
                    write_text(SUMMARY_JSON_PATH, summary_text)
                    refresh_files()
                    st.session_state["force_sync"] = True
                    st.rerun()
        with c2:
            copy_button(summary_text, "monthly_summary_copy")

    with final_prompt_tab:
        final_prompt = compose_summary_prompt_final(
            st.session_state.get("prompt_content", ""),
            st.session_state.get("sector_content", ""),
            st.session_state.get("summary_content", ""),
        )
        st.caption("Dynamic preview based on Summary Prompt + Sector Data Tables + Summary Text (JSON).")
        st.text_area(
            "Final prompt (read-only)",
            value=final_prompt,
            height=340,
            disabled=True,
            key="monthly_final_prompt_preview",
        )
        copy_button(final_prompt, "monthly_final_prompt_copy")

def render_sector_tab(config: ReportConfig) -> None:
    render_subtab_group_intro(
        "Sector sections",
        "Switch between the monthly sector report, sector pulse, and sector scoring boards.",
    )
    monthly_tab, sector_pulse_tab, fundamental_tab, technical_tab = st.tabs(
        ["Monthly Sector Report", "Sector Pulse", "Fundamental Scoring", "Technical Scoring"]
    )
    with monthly_tab:
        render_monthly_board(config)
    with sector_pulse_tab:
        render_sector_pulse_board(config)
    with fundamental_tab:
        render_fundamental_scoring_board(config)
    with technical_tab:
        render_technical_scoring_board(config)


def _prepare_thematics_report_frame(report_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if report_df is None or report_df.empty:
        return pd.DataFrame(columns=["ticker"])
    working = report_df.copy()
    if "ticker" not in working.columns:
        return pd.DataFrame(columns=["ticker"])
    working["ticker"] = working["ticker"].map(normalize_price_ticker)
    for column in [
        "market_cap",
        "beta",
        "fundamental_total_score",
        "fundamental_growth",
        "fundamental_value",
        "fundamental_quality",
        "fundamental_risk",
        "general_technical_score",
        "stock_rsi_regime_score",
        "sector_regime_fit_score",
        "fundamental_momentum",
        "rs_daily",
        "rs_sma20",
        "obvm_daily",
        "obvm_sma20",
        "rs_monthly",
        "obvm_monthly",
    ]:
        if column in working.columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
    return working.drop_duplicates(subset=["ticker"], keep="first")


def _thematic_memberships_for_scope(
    basket_name: str,
    catalog: dict[str, object],
) -> dict[str, list[str]]:
    items = catalog.get("items", {})
    if not isinstance(items, dict) or basket_name not in items:
        return {}
    basket = items[basket_name]
    memberships: dict[str, list[str]] = {}
    children = basket.get("children", [])
    if basket.get("is_parent") and isinstance(children, list) and children:
        for child_name in children:
            child_item = items.get(child_name)
            if not isinstance(child_item, dict):
                continue
            for ticker_value in child_item.get("tickers", []):
                ticker = normalize_price_ticker(ticker_value)
                if not ticker:
                    continue
                memberships.setdefault(ticker, []).append(str(child_name))
        for ticker_value in basket.get("tickers", []):
            ticker_str = normalize_price_ticker(ticker_value)
            if not ticker_str:
                continue
            memberships.setdefault(ticker_str, [basket_name])
    else:
        for ticker_value in basket.get("tickers", []):
            ticker = normalize_price_ticker(ticker_value)
            if not ticker:
                continue
            memberships[ticker] = [basket_name]
    return memberships


def _all_thematic_memberships(catalog: dict[str, object]) -> dict[str, list[str]]:
    items = catalog.get("items", {})
    if not isinstance(items, dict):
        return {}

    memberships: dict[str, list[str]] = {}
    for basket_name, basket in items.items():
        if not isinstance(basket, dict):
            continue
        if bool(basket.get("is_ai_super_parent", False)):
            continue
        for ticker_value in basket.get("tickers", []):
            ticker_str = normalize_price_ticker(ticker_value)
            if not ticker_str:
                continue
            membership_list = memberships.setdefault(ticker_str, [])
            if basket_name not in membership_list:
                membership_list.append(str(basket_name))
    return memberships


def _build_thematics_company_universe_from_scope(
    scope_tickers: list[str],
    thematic_memberships: dict[str, list[str]],
    report_df: Optional[pd.DataFrame],
    price_lookup: dict[str, dict[str, list[object]]],
    reference_date: date,
    *,
    thematics_config_signature: str = "",
) -> tuple[pd.DataFrame, bool]:
    normalized_scope_tickers = [
        ticker for ticker in dict.fromkeys(normalize_price_ticker(ticker_value) for ticker_value in scope_tickers) if ticker
    ]
    base_df = pd.DataFrame({"ticker": normalized_scope_tickers})
    current_report = _prepare_thematics_report_frame(report_df)
    if not current_report.empty:
        available_columns = [
            column
            for column in [
                "ticker",
                "company",
                "sector",
                "industry",
                "market_cap",
                "beta",
                "fundamental_total_score",
                "fundamental_growth",
                "fundamental_value",
                "fundamental_quality",
                "fundamental_risk",
                "general_technical_score",
                "stock_rsi_regime_score",
                "sector_regime_fit_score",
                "fundamental_momentum",
                "rs_daily",
                "rs_sma20",
                "obvm_daily",
                "obvm_sma20",
                "rs_monthly",
                "obvm_monthly",
            ]
            if column in current_report.columns
        ]
        report_subset = current_report[current_report["ticker"].isin(normalized_scope_tickers)][available_columns].copy()
        base_df = base_df.merge(report_subset, on="ticker", how="left")

    if "company" not in base_df.columns:
        base_df["company"] = ""
    base_df["company"] = base_df["company"].fillna("").astype(str).str.strip()
    base_df["company"] = base_df["company"].where(base_df["company"].str.len() > 0, base_df["ticker"].map(_base_ticker_symbol))
    base_df["sector"] = base_df.get("sector", pd.Series(index=base_df.index)).fillna("Unspecified")
    base_df["industry"] = base_df.get("industry", pd.Series(index=base_df.index)).fillna("Unspecified")
    for required_numeric in [
        "market_cap",
        "beta",
        "fundamental_total_score",
        "fundamental_growth",
        "fundamental_value",
        "fundamental_quality",
        "fundamental_risk",
        "general_technical_score",
        "fundamental_momentum",
        "rs_monthly",
        "obvm_monthly",
    ]:
        if required_numeric not in base_df.columns:
            base_df[required_numeric] = np.nan

    company_universe, error_message = _prepare_company_drilldown_universe(
        base_df,
        include_beta=True,
        thematic_memberships=thematic_memberships,
        thematics_config_signature=thematics_config_signature,
    )
    if error_message or company_universe is None:
        return pd.DataFrame(), False
    company_universe, _ = _enrich_company_universe_with_market_regime(
        company_universe,
        reference_date,
    )
    company_universe = _enrich_company_universe_with_rsi_divergence(
        company_universe,
        reference_date,
    )

    performance_df, anchor_missing = _compute_company_return_metrics(scope_tickers, price_lookup, reference_date)
    if not performance_df.empty:
        company_universe = company_universe.merge(performance_df, on="ticker", how="left")
    else:
        for metric_name in ["anchor_close", "1w_perf", "1m_perf", "3m_perf", "ytd_perf"]:
            company_universe[metric_name] = np.nan
    return company_universe, anchor_missing


def _build_thematics_company_universe(
    basket_name: str,
    catalog: dict[str, object],
    report_df: Optional[pd.DataFrame],
    price_lookup: dict[str, dict[str, list[object]]],
    reference_date: date,
    *,
    thematics_config_signature: str = "",
) -> tuple[pd.DataFrame, bool]:
    items = catalog.get("items", {})
    if not isinstance(items, dict) or basket_name not in items:
        return pd.DataFrame(), False

    basket = items[basket_name]
    scope_tickers = [
        ticker for ticker in dict.fromkeys(normalize_price_ticker(ticker_value) for ticker_value in basket.get("tickers", []))
        if ticker
    ]
    memberships = _thematic_memberships_for_scope(basket_name, catalog)
    return _build_thematics_company_universe_from_scope(
        scope_tickers,
        memberships,
        report_df,
        price_lookup,
        reference_date,
        thematics_config_signature=thematics_config_signature,
    )


def _build_all_thematics_company_universe(
    catalog: dict[str, object],
    report_df: Optional[pd.DataFrame],
    price_lookup: dict[str, dict[str, list[object]]],
    reference_date: date,
    *,
    thematics_config_signature: str = "",
) -> tuple[pd.DataFrame, bool]:
    memberships = _all_thematic_memberships(catalog)
    scope_tickers = sorted(memberships.keys())
    if not scope_tickers:
        return pd.DataFrame(), False
    return _build_thematics_company_universe_from_scope(
        scope_tickers,
        memberships,
        report_df,
        price_lookup,
        reference_date,
        thematics_config_signature=thematics_config_signature,
    )


@st.cache_data(show_spinner=False)
def _build_thematics_company_universe_for_scope_cached(
    scope_mode: str,
    scope_name: str,
    catalog_path_str: str,
    catalog_cache_signature: str,
    current_report_path_str: str,
    current_report_cache_signature: str,
    prices_cache_path_str: str,
    prices_cache_signature: str,
    reference_date: date,
    market_cache_signature: str,
    divergence_cache_signature: str,
) -> tuple[pd.DataFrame, bool]:
    _ = market_cache_signature, divergence_cache_signature
    catalog = build_thematics_catalog(catalog_path_str, catalog_cache_signature)
    current_report_df: Optional[pd.DataFrame] = None
    if current_report_path_str:
        current_report_df = normalize_report_columns(
            load_report_select(current_report_path_str, current_report_cache_signature).copy()
        )
    price_lookup: dict[str, dict[str, list[object]]] = {}
    if prices_cache_path_str:
        price_lookup = build_price_history_lookup(prices_cache_path_str, prices_cache_signature)
    if scope_mode == "show_all":
        return _build_all_thematics_company_universe(
            catalog,
            current_report_df,
            price_lookup,
            reference_date,
            thematics_config_signature=catalog_cache_signature,
        )
    return _build_thematics_company_universe(
        scope_name,
        catalog,
        current_report_df,
        price_lookup,
        reference_date,
        thematics_config_signature=catalog_cache_signature,
    )


def _build_thematics_basket_metrics(
    catalog: dict[str, object],
    report_df: Optional[pd.DataFrame],
    previous_report_df: Optional[pd.DataFrame],
    price_lookup: dict[str, dict[str, list[object]]],
    reference_date: date,
) -> tuple[pd.DataFrame, bool]:
    items = catalog.get("items", {})
    if not isinstance(items, dict):
        return pd.DataFrame(), False

    current_report = _prepare_thematics_report_frame(report_df)
    previous_report = _prepare_thematics_report_frame(previous_report_df)
    all_unique_tickers = sorted(
        {
            ticker
            for item in items.values()
            if isinstance(item, dict)
            for ticker in [normalize_price_ticker(ticker_value) for ticker_value in item.get("tickers", [])]
            if ticker
        }
    )
    performance_df, anchor_missing = _compute_company_return_metrics(all_unique_tickers, price_lookup, reference_date)
    rows: list[dict[str, object]] = []

    for basket_name, basket in items.items():
        if not isinstance(basket, dict):
            continue
        scope_tickers = [
            ticker
            for ticker in dict.fromkeys(normalize_price_ticker(ticker_value) for ticker_value in basket.get("tickers", []))
            if ticker
        ]
        perf_scope = performance_df[performance_df["ticker"].isin(scope_tickers)]
        report_scope = current_report[current_report["ticker"].isin(scope_tickers)] if not current_report.empty else pd.DataFrame()
        previous_scope = previous_report[previous_report["ticker"].isin(scope_tickers)] if not previous_report.empty else pd.DataFrame()

        technical_score = _basket_average(report_scope.get("general_technical_score", pd.Series(dtype=float)))
        previous_technical_score = _basket_average(previous_scope.get("general_technical_score", pd.Series(dtype=float)))
        fundamental_score = _basket_average(report_scope.get("fundamental_total_score", pd.Series(dtype=float)))
        previous_fundamental_score = _basket_average(previous_scope.get("fundamental_total_score", pd.Series(dtype=float)))
        fundamental_momentum_score = _basket_average(report_scope.get("fundamental_momentum", pd.Series(dtype=float)))
        previous_fundamental_momentum_score = _basket_average(previous_scope.get("fundamental_momentum", pd.Series(dtype=float)))
        fundamental_growth_score = _basket_average(report_scope.get("fundamental_growth", pd.Series(dtype=float)))
        previous_fundamental_growth_score = _basket_average(previous_scope.get("fundamental_growth", pd.Series(dtype=float)))
        fundamental_value_score = _basket_average(report_scope.get("fundamental_value", pd.Series(dtype=float)))
        previous_fundamental_value_score = _basket_average(previous_scope.get("fundamental_value", pd.Series(dtype=float)))
        fundamental_quality_score = _basket_average(report_scope.get("fundamental_quality", pd.Series(dtype=float)))
        previous_fundamental_quality_score = _basket_average(previous_scope.get("fundamental_quality", pd.Series(dtype=float)))
        fundamental_risk_score = _basket_average(report_scope.get("fundamental_risk", pd.Series(dtype=float)))
        previous_fundamental_risk_score = _basket_average(previous_scope.get("fundamental_risk", pd.Series(dtype=float)))

        rows.append(
            {
                "name": basket_name,
                "description": basket.get("description", ""),
                "article_narrative": basket.get("article_narrative", ""),
                "tier": basket.get("tier"),
                "tier_label": basket.get("tier_label", ""),
                "value_chain_layer": basket.get("value_chain_layer"),
                "parent": basket.get("parent", ""),
                "is_parent": basket.get("is_parent", False),
                "children": basket.get("children", []),
                "ticker_count": len(scope_tickers),
                "beta": _basket_average(report_scope.get("beta", pd.Series(dtype=float))),
                "1w_perf": _basket_average(perf_scope.get("1w_perf", pd.Series(dtype=float))),
                "1m_perf": _basket_average(perf_scope.get("1m_perf", pd.Series(dtype=float))),
                "3m_perf": _basket_average(perf_scope.get("3m_perf", pd.Series(dtype=float))),
                "ytd_perf": _basket_average(perf_scope.get("ytd_perf", pd.Series(dtype=float))),
                "technical_scoring": technical_score,
                "rel_strength_breadth": (
                    float((report_scope["rs_monthly"] > 0).mean() * 100.0)
                    if "rs_monthly" in report_scope.columns and report_scope["rs_monthly"].notna().any()
                    else float("nan")
                ),
                "rel_volume_breadth": (
                    float((report_scope["obvm_monthly"] > 0).mean() * 100.0)
                    if "obvm_monthly" in report_scope.columns and report_scope["obvm_monthly"].notna().any()
                    else float("nan")
                ),
                "fundamental_scoring": fundamental_score,
                "fundamental_momentum_scoring": fundamental_momentum_score,
                "fundamental_growth_scoring": fundamental_growth_score,
                "fundamental_value_scoring": fundamental_value_score,
                "fundamental_quality_scoring": fundamental_quality_score,
                "fundamental_risk_scoring": fundamental_risk_score,
                "technical_trend_symbol": _trend_symbol_for_delta(
                    technical_score - previous_technical_score
                    if pd.notna(technical_score) and pd.notna(previous_technical_score)
                    else None,
                    5.0,
                ),
                "fundamental_trend_symbol": _trend_symbol_for_delta(
                    fundamental_score - previous_fundamental_score
                    if pd.notna(fundamental_score) and pd.notna(previous_fundamental_score)
                    else None,
                    5.0,
                ),
                "fundamental_momentum_trend_symbol": _trend_symbol_for_delta(
                    fundamental_momentum_score - previous_fundamental_momentum_score
                    if pd.notna(fundamental_momentum_score) and pd.notna(previous_fundamental_momentum_score)
                    else None,
                    5.0,
                ),
                "fundamental_growth_trend_symbol": _trend_symbol_for_delta(
                    fundamental_growth_score - previous_fundamental_growth_score
                    if pd.notna(fundamental_growth_score) and pd.notna(previous_fundamental_growth_score)
                    else None,
                    5.0,
                ),
                "fundamental_value_trend_symbol": _trend_symbol_for_delta(
                    fundamental_value_score - previous_fundamental_value_score
                    if pd.notna(fundamental_value_score) and pd.notna(previous_fundamental_value_score)
                    else None,
                    5.0,
                ),
                "fundamental_quality_trend_symbol": _trend_symbol_for_delta(
                    fundamental_quality_score - previous_fundamental_quality_score
                    if pd.notna(fundamental_quality_score) and pd.notna(previous_fundamental_quality_score)
                    else None,
                    5.0,
                ),
                "fundamental_risk_trend_symbol": _trend_symbol_for_delta(
                    fundamental_risk_score - previous_fundamental_risk_score
                    if pd.notna(fundamental_risk_score) and pd.notna(previous_fundamental_risk_score)
                    else None,
                    5.0,
                ),
            }
        )
    return pd.DataFrame(rows), anchor_missing


def format_thematics_company_display(company_df: pd.DataFrame) -> pd.DataFrame:
    if company_df.empty:
        return pd.DataFrame(
            columns=[
                "Thematic",
                "Ticker",
                "Company",
                "Sector",
                "Industry",
                "Market Cap",
                "Beta",
                "1W",
                "1M",
                "3M",
                "YTD",
                "TS",
                "RSI Regime",
                "Sector Regime Fit",
                "Short Term Flow",
                "RSI Divergence (D)",
                "RSI Divergence (W)",
                "FS",
                "Mom. FS",
                "Growth FS",
                "Value FS",
                "Quality FS",
                "Risk FS",
                "Rel Strength",
                "Rel Volume",
                "AI Revenue Exposure",
                "AI Disruption Risk",
            ]
        )

    sorted_df = company_df.sort_values(
        by=["general_technical_score", "fundamental_total_score", "ticker"],
        ascending=[False, False, True],
        na_position="last",
    ).copy()
    sorted_df["company"] = sorted_df["company"].fillna("").astype(str).str.strip()
    sorted_df["company"] = sorted_df["company"].where(sorted_df["company"].str.len() > 0, sorted_df["ticker"].map(_base_ticker_symbol))
    if "stock_rsi_regime_score" not in sorted_df.columns:
        sorted_df["stock_rsi_regime_score"] = np.nan
    if "sector_regime_fit_score" not in sorted_df.columns:
        sorted_df["sector_regime_fit_score"] = np.nan
    if "short_term_flow" not in sorted_df.columns:
        sorted_df["short_term_flow"] = pd.NA
    if "rsi_divergence_daily_flag" not in sorted_df.columns:
        sorted_df["rsi_divergence_daily_flag"] = pd.NA
    if "rsi_divergence_weekly_flag" not in sorted_df.columns:
        sorted_df["rsi_divergence_weekly_flag"] = pd.NA
    if "fundamental_growth" not in sorted_df.columns:
        sorted_df["fundamental_growth"] = np.nan
    if "fundamental_value" not in sorted_df.columns:
        sorted_df["fundamental_value"] = np.nan
    if "fundamental_quality" not in sorted_df.columns:
        sorted_df["fundamental_quality"] = np.nan
    if "fundamental_risk" not in sorted_df.columns:
        sorted_df["fundamental_risk"] = np.nan
    if "rel_strength" not in sorted_df.columns:
        sorted_df["rel_strength"] = "N/A"
    if "rel_volume" not in sorted_df.columns:
        sorted_df["rel_volume"] = "N/A"
    if "ai_revenue_exposure" not in sorted_df.columns:
        sorted_df["ai_revenue_exposure"] = "none"
    if "ai_disruption_risk" not in sorted_df.columns:
        sorted_df["ai_disruption_risk"] = "none"

    display_df = pd.DataFrame(
        {
            "Thematic": sorted_df["thematic"].fillna("Unassigned"),
            "Ticker": sorted_df["ticker"],
            "Company": sorted_df["company"],
            "Sector": sorted_df["sector"].fillna("Unspecified"),
            "Industry": sorted_df["industry"].fillna("Unspecified"),
            "Market Cap": pd.to_numeric(sorted_df["market_cap"], errors="coerce"),
            "Beta": pd.to_numeric(sorted_df["beta"], errors="coerce"),
            "1W": pd.to_numeric(sorted_df["1w_perf"], errors="coerce"),
            "1M": pd.to_numeric(sorted_df["1m_perf"], errors="coerce"),
            "3M": pd.to_numeric(sorted_df["3m_perf"], errors="coerce"),
            "YTD": pd.to_numeric(sorted_df["ytd_perf"], errors="coerce"),
            "TS": pd.to_numeric(sorted_df["general_technical_score"], errors="coerce"),
            "RSI Regime": pd.to_numeric(sorted_df["stock_rsi_regime_score"], errors="coerce"),
            "Sector Regime Fit": pd.to_numeric(sorted_df["sector_regime_fit_score"], errors="coerce"),
            "Short Term Flow": sorted_df["short_term_flow"].map(_format_short_term_flow_flag),
            "RSI Divergence (D)": sorted_df["rsi_divergence_daily_flag"].map(_format_divergence_flag),
            "RSI Divergence (W)": sorted_df["rsi_divergence_weekly_flag"].map(_format_divergence_flag),
            "FS": pd.to_numeric(sorted_df["fundamental_total_score"], errors="coerce"),
            "Mom. FS": pd.to_numeric(sorted_df["fundamental_momentum"], errors="coerce"),
            "Growth FS": pd.to_numeric(sorted_df["fundamental_growth"], errors="coerce"),
            "Value FS": pd.to_numeric(sorted_df["fundamental_value"], errors="coerce"),
            "Quality FS": pd.to_numeric(sorted_df["fundamental_quality"], errors="coerce"),
            "Risk FS": pd.to_numeric(sorted_df["fundamental_risk"], errors="coerce"),
            "Rel Strength": sorted_df["rel_strength"].fillna("N/A"),
            "Rel Volume": sorted_df["rel_volume"].fillna("N/A"),
            "AI Revenue Exposure": sorted_df["ai_revenue_exposure"].fillna("none"),
            "AI Disruption Risk": sorted_df["ai_disruption_risk"].fillna("none"),
        }
    )
    display_df["Market Cap"] = display_df["Market Cap"].map(_format_market_cap_display)
    display_df["Beta"] = display_df["Beta"].map(_format_numeric_value)
    for perf_column in ["1W", "1M", "3M", "YTD"]:
        display_df[perf_column] = display_df[perf_column].map(_format_percent_value)
    display_df["RSI Regime"] = display_df["RSI Regime"].map(_format_numeric_value)
    display_df["Sector Regime Fit"] = display_df["Sector Regime Fit"].map(_format_numeric_value)
    for display_column, symbol_column in [
        ("TS", "technical_trend_symbol"),
        ("FS", "fundamental_trend_symbol"),
        ("Mom. FS", "fundamental_momentum_trend_symbol"),
        ("Growth FS", "fundamental_growth_trend_symbol"),
        ("Value FS", "fundamental_value_trend_symbol"),
        ("Quality FS", "fundamental_quality_trend_symbol"),
        ("Risk FS", "fundamental_risk_trend_symbol"),
    ]:
        if symbol_column in sorted_df.columns and display_column in display_df.columns:
            trend_symbols = sorted_df[symbol_column].fillna("").astype(str)
            rendered_scores: list[str] = []
            for numeric_value, trend_symbol in zip(display_df[display_column], trend_symbols):
                rendered_scores.append(_render_thematics_score_with_symbol(numeric_value, str(trend_symbol)))
            display_df[display_column] = rendered_scores
    for display_column in ["TS", "FS", "Mom. FS", "Growth FS", "Value FS", "Quality FS", "Risk FS"]:
        if display_column in display_df.columns:
            display_df[display_column] = display_df[display_column].map(
                lambda value: value if isinstance(value, str) and value.strip() else _format_numeric_value(value)
            )
    return display_df.reset_index(drop=True)


def _build_thematics_company_trend_meta(company_df: pd.DataFrame) -> pd.DataFrame:
    if company_df.empty:
        return pd.DataFrame(
            columns=[
                "ts_trend",
                "fs_trend",
                "mom_fs_trend",
                "growth_fs_trend",
                "value_fs_trend",
                "quality_fs_trend",
                "risk_fs_trend",
            ]
        )

    sorted_df = company_df.sort_values(
        by=["general_technical_score", "fundamental_total_score", "ticker"],
        ascending=[False, False, True],
        na_position="last",
    ).copy()
    return pd.DataFrame(
        {
            "ts_trend": sorted_df.get("technical_trend_symbol", pd.Series([""] * len(sorted_df))).fillna("").astype(str),
            "fs_trend": sorted_df.get("fundamental_trend_symbol", pd.Series([""] * len(sorted_df))).fillna("").astype(str),
            "mom_fs_trend": sorted_df.get(
                "fundamental_momentum_trend_symbol",
                pd.Series([""] * len(sorted_df)),
            ).fillna("").astype(str),
            "growth_fs_trend": sorted_df.get(
                "fundamental_growth_trend_symbol",
                pd.Series([""] * len(sorted_df)),
            ).fillna("").astype(str),
            "value_fs_trend": sorted_df.get(
                "fundamental_value_trend_symbol",
                pd.Series([""] * len(sorted_df)),
            ).fillna("").astype(str),
            "quality_fs_trend": sorted_df.get(
                "fundamental_quality_trend_symbol",
                pd.Series([""] * len(sorted_df)),
            ).fillna("").astype(str),
            "risk_fs_trend": sorted_df.get(
                "fundamental_risk_trend_symbol",
                pd.Series([""] * len(sorted_df)),
            ).fillna("").astype(str),
        }
    ).reset_index(drop=True)


def _render_thematics_context_card(catalog: dict[str, object], focus_basket: Optional[str]) -> None:
    items = catalog.get("items", {})
    if not isinstance(items, dict) or not items:
        st.info("No thematic baskets available.")
        return
    basket_name = focus_basket if focus_basket in items else next(iter(items))
    basket = items[basket_name]
    description = str(basket.get("description", "") or "No description available yet.")
    narrative = str(basket.get("article_narrative", "") or "")
    parent_name = str(basket.get("parent", "") or "Root basket")
    child_names = basket.get("children", [])
    child_summary = ", ".join(child_names[:4]) if isinstance(child_names, list) and child_names else "No child baskets"
    st.markdown(
        f"""
<div style="
  border:1px solid #D9E7F1;
  border-radius:16px;
  padding:1rem 1.05rem;
  background:linear-gradient(160deg, rgba(255,255,255,0.96) 0%, rgba(236,244,255,0.96) 100%);
  box-shadow:0 10px 24px rgba(15,39,71,0.06);
">
  <div style="font-size:0.76rem;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:#5F7694;">
    Thematic Lens
  </div>
  <div style="margin-top:0.28rem;font-size:1.2rem;font-weight:800;color:#123159;">
    {html_escape(str(basket_name))}
  </div>
  <div style="margin-top:0.2rem;color:#496685;font-size:0.84rem;">
    {html_escape(str(basket.get("tier_label", "")))} | Layer {html_escape(str(basket.get("value_chain_layer", "n/a")))} | Parent: {html_escape(parent_name)}
  </div>
  <div style="margin-top:0.75rem;color:#334E68;font-size:0.92rem;line-height:1.45;">
    {html_escape(description)}
  </div>
  <div style="margin-top:0.75rem;padding:0.72rem;border-radius:12px;background:rgba(18,49,89,0.05);color:#29415D;font-size:0.88rem;line-height:1.45;">
    {html_escape(narrative or child_summary)}
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _build_thematics_constituents_frame(
    catalog: dict[str, object],
    focus_basket: Optional[str],
    report_df: Optional[pd.DataFrame],
) -> tuple[str, pd.DataFrame]:
    items = catalog.get("items", {})
    if not isinstance(items, dict) or not items:
        return "", pd.DataFrame(columns=["Ticker", "Company"])

    basket_name = focus_basket if focus_basket in items else next(iter(items))
    basket = items.get(basket_name, {})
    if not isinstance(basket, dict):
        return basket_name, pd.DataFrame(columns=["Ticker", "Company"])

    raw_tickers = basket.get("tickers", [])
    tickers: list[str] = []
    seen: set[str] = set()
    for ticker_value in raw_tickers if isinstance(raw_tickers, list) else []:
        ticker = normalize_price_ticker(ticker_value)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)

    company_lookup: dict[str, str] = {}
    prepared_report = _prepare_thematics_report_frame(report_df)
    if not prepared_report.empty:
        working = prepared_report.copy()
        if "company" in working.columns:
            working["company"] = working["company"].fillna("").astype(str).str.strip()
            company_lookup = (
                working.assign(
                    company=working["company"].where(
                        working["company"].str.len() > 0,
                        working["ticker"].map(_base_ticker_symbol),
                    )
                )
                .drop_duplicates(subset=["ticker"], keep="first")
                .set_index("ticker")["company"]
                .to_dict()
            )

    rows = [
        {
            "Ticker": ticker,
            "Company": str(company_lookup.get(ticker, _base_ticker_symbol(ticker)) or _base_ticker_symbol(ticker)),
        }
        for ticker in tickers
    ]
    return basket_name, pd.DataFrame(rows, columns=["Ticker", "Company"])


def _render_thematics_constituents_card(
    catalog: dict[str, object],
    focus_basket: Optional[str],
    report_df: Optional[pd.DataFrame],
) -> None:
    basket_name, constituents_df = _build_thematics_constituents_frame(catalog, focus_basket, report_df)
    if not basket_name:
        return

    st.markdown(
        f"""
<div style="
  margin-top:0.95rem;
  border:1px solid #D9E7F1;
  border-radius:16px;
  padding:1rem 1.05rem;
  background:linear-gradient(160deg, rgba(255,255,255,0.98) 0%, rgba(248,251,255,0.98) 100%);
  box-shadow:0 10px 24px rgba(15,39,71,0.05);
">
  <div style="font-size:0.76rem;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:#5F7694;">
    Constituents
  </div>
  <div style="margin-top:0.28rem;font-size:1rem;font-weight:800;color:#123159;">
    {html_escape(str(basket_name))} | {len(constituents_df)} names
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if constituents_df.empty:
        st.caption("No tickers are configured for this thematic.")
        return

    display_height = min(max(180, 68 + len(constituents_df) * 34), 420)
    styler = constituents_df.style.hide(axis="index")
    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#F4F8FC"),
                    ("color", "#173A5E"),
                    ("font-weight", "800"),
                    ("text-align", "left"),
                    ("border-bottom", "1px solid #D9E6F2"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid #E7EFF7"),
                    ("vertical-align", "middle"),
                    ("color", "#334E68"),
                ],
            },
        ],
        overwrite=False,
    )
    styler = styler.set_properties(subset=["Ticker", "Company"], **{"text-align": "left"})
    st.dataframe(
        styler,
        use_container_width=True,
        height=display_height,
        hide_index=True,
        key="thematics_lens_constituents",
    )


def _handle_thematics_basket_checkbox_change(state_prefix: str, basket_name: str) -> None:
    checkbox_state_key = f"{state_prefix}_check_{basket_name}"
    selected_key = f"{state_prefix}_selected_basket"
    focus_key = f"{state_prefix}_focus_basket"
    checked = bool(st.session_state.get(checkbox_state_key))
    if checked:
        st.session_state[selected_key] = basket_name
        _queue_show_all_company_reset(state_prefix)
    elif st.session_state.get(selected_key) == basket_name:
        st.session_state[selected_key] = None
    st.session_state[focus_key] = basket_name


def _handle_thematics_show_all_company_toggle(state_prefix: str) -> None:
    if bool(st.session_state.get(f"{state_prefix}_show_all_companies", False)):
        st.session_state[f"{state_prefix}_selected_basket"] = None
        _bump_selection_nonce(f"{state_prefix}_basket_table_nonce")


def _render_thematics_basket_table(
    basket_metrics_df: pd.DataFrame,
    catalog: dict[str, object],
    *,
    view_mode: str = "all",
) -> None:
    _render_thematics_basket_table_v2(basket_metrics_df, catalog, view_mode=view_mode)
    return
    if basket_metrics_df.empty:
        st.info("No basket metrics available for the selected dates.")
        return

    items = catalog.get("items", {})
    roots = catalog.get("roots", [])
    if not isinstance(items, dict) or not isinstance(roots, list):
        st.info("No thematic catalog could be built.")
        return

    state_prefix = "thematics_impl"
    expanded_key = f"{state_prefix}_expanded_parents"
    selected_key = f"{state_prefix}_selected_basket"
    focus_key = f"{state_prefix}_focus_basket"
    expanded_parents = set(st.session_state.get(expanded_key, []))
    selected_basket = st.session_state.get(selected_key)

    header_cols = st.columns([0.55, 2.75, 0.75, 0.8, 0.8, 0.8, 0.8, 0.9, 0.95, 0.95, 0.95, 1.05])
    header_labels = [
        "",
        "Name",
        "Beta",
        "1W",
        "1M",
        "3M",
        "YTD",
        "TS",
        "RS %",
        "Vol %",
        "FS",
        "Mom. FS",
    ]
    for col, label in zip(header_cols, header_labels):
        with col:
            st.markdown(f"**{label}**")

    metrics_by_name = basket_metrics_df.set_index("name").to_dict(orient="index")
    visible_rows: list[str] = []
    for root_name in roots:
        visible_rows.append(root_name)
        root_item = items.get(root_name, {})
        if root_name in expanded_parents:
            for child_name in root_item.get("children", []):
                visible_rows.append(child_name)

    for basket_name in visible_rows:
        row = metrics_by_name.get(basket_name, {})
        basket = items.get(basket_name, {})
        is_child = bool(basket.get("parent"))
        is_ai_super_parent = bool(basket.get("is_ai_super_parent", False))
        is_ai_group_child = bool(basket.get("is_ai_group_child", False))
        checkbox_state_key = f"{state_prefix}_check_{basket_name}"
        is_selected = selected_basket == basket_name
        st.session_state[checkbox_state_key] = is_selected
        row_cols = st.columns([0.55, 2.75, 0.75, 0.8, 0.8, 0.8, 0.8, 0.9, 0.95, 0.95, 0.95, 1.05])
        with row_cols[0]:
            st.checkbox(
                "",
                key=checkbox_state_key,
                on_change=_handle_thematics_basket_checkbox_change,
                args=(state_prefix, basket_name),
            )
        with row_cols[1]:
            if basket.get("is_parent"):
                toggle_key = f"{state_prefix}_toggle_{re.sub(r'[^0-9A-Za-z_]+', '_', basket_name.lower()).strip('_')}"
                if is_ai_super_parent:
                    st.markdown(
                        f"""
<style>
div.st-key-{toggle_key} button {{
  background: linear-gradient(135deg, #0F2747 0%, #184E82 100%);
  color: #FFFFFF !important;
  border: 1px solid #0F2747 !important;
  font-weight: 800 !important;
}}
div.st-key-{toggle_key} button p {{
  color: #FFFFFF !important;
}}
</style>
                        """,
                        unsafe_allow_html=True,
                    )
                elif is_ai_group_child:
                    st.markdown(
                        f"""
<style>
div.st-key-{toggle_key} button {{
  background: linear-gradient(135deg, #E4F1FF 0%, #D4EAFF 100%);
  color: #0B5CAD !important;
  border: 1px solid #93C5FD !important;
  font-weight: 700 !important;
}}
div.st-key-{toggle_key} button p {{
  color: #0B5CAD !important;
}}
</style>
                        """,
                        unsafe_allow_html=True,
                    )
                toggle_label = ("▾ " if basket_name in expanded_parents else "▸ ") + basket_name
                if st.button(toggle_label, key=toggle_key, use_container_width=True):
                    if basket_name in expanded_parents:
                        expanded_parents.remove(basket_name)
                    else:
                        expanded_parents.add(basket_name)
                    st.session_state[expanded_key] = sorted(expanded_parents)
                    st.session_state[focus_key] = basket_name
                    st.rerun()
            else:
                if is_ai_super_parent:
                    font_size = "1.04rem"
                    padding_left = "0rem"
                    color = "#0F2747"
                elif is_ai_group_child:
                    font_size = "0.95rem"
                    padding_left = "1.15rem"
                    color = "#0B5CAD"
                else:
                    font_size = "0.9rem" if is_child else "0.98rem"
                    padding_left = "1.4rem" if is_child else "0rem"
                    color = "#5C708A" if is_child else "#123159"
                st.markdown(
                    f"<div style='padding:0.35rem 0 0.2rem {padding_left}; font-size:{font_size}; color:{color}; font-weight:600;'>{html_escape(basket_name)}</div>",
                    unsafe_allow_html=True,
                )
        values = [
            _format_numeric_value(row.get("beta")),
            _format_percent_value(row.get("1w_perf")),
            _format_percent_value(row.get("1m_perf")),
            _format_percent_value(row.get("3m_perf")),
            _format_percent_value(row.get("ytd_perf")),
            _render_score_with_symbol(row.get("technical_scoring"), str(row.get("technical_trend_symbol", ""))),
            _format_percent_value(row.get("rel_strength_breadth")),
            _format_percent_value(row.get("rel_volume_breadth")),
            _render_score_with_symbol(row.get("fundamental_scoring"), str(row.get("fundamental_trend_symbol", ""))),
            _render_score_with_symbol(
                row.get("fundamental_momentum_scoring"),
                str(row.get("fundamental_momentum_trend_symbol", "")),
            ),
        ]
        for col, value in zip(row_cols[2:], values):
            with col:
                if is_ai_super_parent:
                    value_color = "#0F2747"
                    value_weight = "700"
                elif is_ai_group_child:
                    value_color = "#0B5CAD"
                    value_weight = "650"
                else:
                    value_color = "#29415D"
                    value_weight = "500"
                st.markdown(
                    f"<div style='padding-top:0.35rem; text-align:center; color:{value_color}; font-weight:{value_weight}; font-size:0.9rem;'>{html_escape(str(value))}</div>",
                    unsafe_allow_html=True,
                )


def _style_positive_negative_value(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "color:#7B8BA0;"
    if float(numeric) > 0:
        return "color:#15803D; font-weight:700;"
    if float(numeric) < 0:
        return "color:#B42318; font-weight:700;"
    return "color:#334E68; font-weight:600;"


def _style_breadth_threshold_value(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return "color:#7B8BA0;"
    if float(numeric) >= 50.0:
        return "color:#15803D; font-weight:700;"
    return "color:#B42318; font-weight:700;"


def _style_sign_label_value(value: object) -> str:
    label = str(value).strip().lower()
    if label == "positive":
        return "color:#15803D; font-weight:700;"
    if label == "negative":
        return "color:#B42318; font-weight:700;"
    return "color:#7B8BA0;"


def _extract_dataframe_selection_rows(selection_event: object) -> list[int]:
    if hasattr(selection_event, "selection") and hasattr(selection_event.selection, "rows"):
        return list(selection_event.selection.rows or [])
    if isinstance(selection_event, dict):
        return list(selection_event.get("selection", {}).get("rows", []))
    return []


def _trend_icon_style(symbol: object) -> str:
    symbol_text = str(symbol or "").strip()
    if symbol_text == TREND_SYMBOL_UP:
        return (
            "background-image:url(\"data:image/svg+xml;utf8,"
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
            "<path fill='none' stroke='%2315803D' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' "
            "d='M2 11 L6.1 6.9 L8.8 9.6 L14 4.4'/>"
            "<path fill='none' stroke='%2315803D' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' "
            "d='M10.6 4.4 H14 V7.8'/>"
            "</svg>\");"
            "background-repeat:no-repeat; background-position:right 0.45rem center; "
            "background-size:0.9rem 0.9rem; padding-right:1.75rem;"
        )
    if symbol_text == TREND_SYMBOL_DOWN:
        return (
            "background-image:url(\"data:image/svg+xml;utf8,"
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
            "<path fill='none' stroke='%23B42318' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' "
            "d='M2 5 L6.1 9.1 L8.8 6.4 L14 11.6'/>"
            "<path fill='none' stroke='%23B42318' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' "
            "d='M10.6 11.6 H14 V8.2'/>"
            "</svg>\");"
            "background-repeat:no-repeat; background-position:right 0.45rem center; "
            "background-size:0.9rem 0.9rem; padding-right:1.75rem;"
        )
    return ""


def _score_trend_class(symbol: object) -> str:
    symbol_text = str(symbol or "").strip()
    if symbol_text == TREND_SYMBOL_UP:
        return "trend-up"
    if symbol_text == TREND_SYMBOL_DOWN:
        return "trend-down"
    return ""


def _apply_trend_icon_styles_for_row(
    row: pd.Series,
    meta_df: pd.DataFrame,
    trend_map: Dict[str, str],
    display_columns: Sequence[str],
) -> list[str]:
    if int(row.name) < 0 or int(row.name) >= len(meta_df):
        return ["" for _ in display_columns]
    meta = meta_df.iloc[int(row.name)]
    styles = ["" for _ in display_columns]
    for column_name, trend_key in trend_map.items():
        if column_name not in display_columns:
            continue
        symbol = meta.get(trend_key, "")
        trend_style = _trend_icon_style(symbol)
        if not trend_style:
            continue
        column_index = list(display_columns).index(column_name)
        styles[column_index] = trend_style
    return styles


def _ordered_thematics_names(catalog: dict[str, object]) -> list[str]:
    items = catalog.get("items", {})
    roots = catalog.get("roots", [])
    if not isinstance(items, dict) or not isinstance(roots, list):
        return []

    ordered_names: list[str] = []
    seen: set[str] = set()

    def visit(basket_name: str) -> None:
        if basket_name in seen or basket_name not in items:
            return
        seen.add(basket_name)
        ordered_names.append(basket_name)
        basket = items.get(basket_name, {})
        if isinstance(basket, dict):
            for child_name in basket.get("children", []):
                visit(str(child_name))

    for root_name in roots:
        visit(str(root_name))
    return ordered_names


THEMATICS_VIEW_MODES: tuple[tuple[str, str], ...] = (
    ("all", "All thematics"),
    ("ai_vs_rest", "AI vs Rest"),
    ("ai_layers_vs_rest", "AI layers vs Rest"),
    ("ai_sub_layers_vs_rest", "AI sub-layers vs Rest"),
    ("ai_layers", "AI layers"),
    ("ai_sub_layers", "AI sub-layers"),
)


def _thematics_basket_lineage(catalog: dict[str, object], basket_name: str) -> list[str]:
    items = catalog.get("items", {})
    if not isinstance(items, dict) or basket_name not in items:
        return []

    lineage: list[str] = []
    current_name = basket_name
    seen: set[str] = set()
    while current_name in items and current_name not in seen:
        seen.add(current_name)
        lineage.append(current_name)
        basket = items.get(current_name, {})
        if not isinstance(basket, dict):
            break
        parent_name = str(basket.get("parent", "") or "")
        if not parent_name:
            break
        current_name = parent_name
    return lineage


def _classify_thematics_basket_for_view(
    catalog: dict[str, object],
    basket_name: str,
    meta_row: Optional[pd.Series] = None,
) -> str:
    if basket_name == "AI":
        return "ai_super_parent"

    depth = int(meta_row.get("depth", 0) or 0) if meta_row is not None else 0
    if meta_row is not None and bool(meta_row.get("is_ai_super_parent", False)):
        return "ai_super_parent"
    if meta_row is not None and bool(meta_row.get("is_ai_group_child", False)) and depth == 1:
        return "ai_layer"

    lineage = _thematics_basket_lineage(catalog, basket_name)
    if lineage and lineage[-1] == "AI":
        if depth == 1:
            return "ai_layer"
        if depth > 1:
            return "ai_sub_layer"

    if depth == 0 and basket_name != "AI":
        return "non_ai_root"
    return "other"


def _thematics_view_mode_matches(view_mode: str, basket_class: str) -> bool:
    if view_mode == "all":
        return True
    if view_mode == "ai_vs_rest":
        return basket_class in {"ai_super_parent", "non_ai_root"}
    if view_mode == "ai_layers_vs_rest":
        return basket_class in {"ai_layer", "non_ai_root"}
    if view_mode == "ai_sub_layers_vs_rest":
        return basket_class in {"ai_sub_layer", "non_ai_root"}
    if view_mode == "ai_layers":
        return basket_class == "ai_layer"
    if view_mode == "ai_sub_layers":
        return basket_class == "ai_sub_layer"
    return True


def _filter_thematics_basket_table_for_view(
    display_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    catalog: dict[str, object],
    view_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if display_df.empty or meta_df.empty or view_mode == "all":
        return display_df, meta_df

    keep_indices: list[int] = []
    for row_index in range(len(meta_df)):
        meta_row = meta_df.iloc[row_index]
        basket_name = str(meta_row.get("basket_name", ""))
        basket_class = _classify_thematics_basket_for_view(catalog, basket_name, meta_row)
        if _thematics_view_mode_matches(view_mode, basket_class):
            keep_indices.append(row_index)

    if not keep_indices:
        return display_df.iloc[0:0].copy(), meta_df.iloc[0:0].copy()

    return display_df.iloc[keep_indices].reset_index(drop=True), meta_df.iloc[keep_indices].reset_index(drop=True)


def _normalize_thematics_selected_basket(
    selected_basket: Optional[str],
    visible_baskets: set[str],
) -> Optional[str]:
    if selected_basket and selected_basket in visible_baskets:
        return selected_basket
    return None


def _render_thematics_view_mode_controls(state_prefix: str) -> str:
    view_mode_key = f"{state_prefix}_view_mode"
    allowed_modes = {mode for mode, _ in THEMATICS_VIEW_MODES}
    current_mode = str(st.session_state.get(view_mode_key, "all") or "all")
    if current_mode not in allowed_modes:
        current_mode = "all"
        st.session_state[view_mode_key] = current_mode

    theme_styles = {
        True: ("#123159", "#FFFFFF", "#123159"),
        False: ("#F4F8FC", "#123159", "#D0DDEB"),
    }
    css_rules: list[str] = []
    button_layout = [1.0, 1.0, 1.15, 1.25, 1.0, 1.0]
    button_cols = st.columns(button_layout)
    for (mode, label), col in zip(THEMATICS_VIEW_MODES, button_cols):
        is_active = mode == current_mode
        bg_color, text_color, border_color = theme_styles[is_active]
        button_key = f"{state_prefix}_view_{mode}"
        css_rules.append(
            f"""
div.st-key-{button_key} button {{
  background-color: {bg_color} !important;
  color: {text_color} !important;
  border: 1px solid {border_color} !important;
  min-height: 38px !important;
  padding: 0.18rem 0.45rem !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
  line-height: 1.08 !important;
}}
div.st-key-{button_key} button p {{
  color: {text_color} !important;
  font-size: 0.82rem !important;
  font-weight: 700 !important;
}}
            """
        )
        with col:
            if st.button(label, key=button_key, use_container_width=True):
                st.session_state[view_mode_key] = mode
                st.rerun()
    st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)
    return current_mode


def _build_thematics_basket_table_frame(
    basket_metrics_df: pd.DataFrame,
    catalog: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = catalog.get("items", {})
    if not isinstance(items, dict) or basket_metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    metrics_by_name = basket_metrics_df.set_index("name").to_dict(orient="index")
    rows: list[dict[str, object]] = []
    meta_rows: list[dict[str, object]] = []

    for basket_name in _ordered_thematics_names(catalog):
        basket = items.get(basket_name, {})
        if not isinstance(basket, dict):
            continue
        row = metrics_by_name.get(basket_name, {})
        depth = 0
        parent_name = str(basket.get("parent", "") or "")
        while parent_name:
            depth += 1
            parent_item = items.get(parent_name, {})
            if not isinstance(parent_item, dict):
                break
            parent_name = str(parent_item.get("parent", "") or "")

        rows.append(
            {
                "Name": basket_name,
                "Beta": pd.to_numeric(row.get("beta"), errors="coerce"),
                "1W": pd.to_numeric(row.get("1w_perf"), errors="coerce"),
                "1M": pd.to_numeric(row.get("1m_perf"), errors="coerce"),
                "3M": pd.to_numeric(row.get("3m_perf"), errors="coerce"),
                "YTD": pd.to_numeric(row.get("ytd_perf"), errors="coerce"),
                "RS %": pd.to_numeric(row.get("rel_strength_breadth"), errors="coerce"),
                "Vol %": pd.to_numeric(row.get("rel_volume_breadth"), errors="coerce"),
                "TS": _render_thematics_score_with_symbol(
                    row.get("technical_scoring"),
                    str(row.get("technical_trend_symbol", "") or ""),
                ),
                "FS": _render_thematics_score_with_symbol(
                    row.get("fundamental_scoring"),
                    str(row.get("fundamental_trend_symbol", "") or ""),
                ),
                "Mom. FS": _render_thematics_score_with_symbol(
                    row.get("fundamental_momentum_scoring"),
                    str(row.get("fundamental_momentum_trend_symbol", "") or ""),
                ),
                "Growth FS": _render_thematics_score_with_symbol(
                    row.get("fundamental_growth_scoring"),
                    str(row.get("fundamental_growth_trend_symbol", "") or ""),
                ),
                "Value FS": _render_thematics_score_with_symbol(
                    row.get("fundamental_value_scoring"),
                    str(row.get("fundamental_value_trend_symbol", "") or ""),
                ),
                "Quality FS": _render_thematics_score_with_symbol(
                    row.get("fundamental_quality_scoring"),
                    str(row.get("fundamental_quality_trend_symbol", "") or ""),
                ),
                "Risk FS": _render_thematics_score_with_symbol(
                    row.get("fundamental_risk_scoring"),
                    str(row.get("fundamental_risk_trend_symbol", "") or ""),
                ),
            }
        )
        meta_rows.append(
            {
                "basket_name": basket_name,
                "depth": depth,
                "is_parent": bool(basket.get("is_parent", False)),
                "is_ai_super_parent": bool(basket.get("is_ai_super_parent", False)),
                "is_ai_group_child": bool(basket.get("is_ai_group_child", False)),
                "ts_trend": str(row.get("technical_trend_symbol", "") or ""),
                "fs_trend": str(row.get("fundamental_trend_symbol", "") or ""),
                "mom_fs_trend": str(row.get("fundamental_momentum_trend_symbol", "") or ""),
                "growth_fs_trend": str(row.get("fundamental_growth_trend_symbol", "") or ""),
                "value_fs_trend": str(row.get("fundamental_value_trend_symbol", "") or ""),
                "quality_fs_trend": str(row.get("fundamental_quality_trend_symbol", "") or ""),
                "risk_fs_trend": str(row.get("fundamental_risk_trend_symbol", "") or ""),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


def _build_thematics_lens_frame(catalog: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = catalog.get("items", {})
    if not isinstance(items, dict):
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, object]] = []
    meta_rows: list[dict[str, object]] = []
    for basket_name in _ordered_thematics_names(catalog):
        basket = items.get(basket_name, {})
        if not isinstance(basket, dict):
            continue
        depth = 0
        parent_name = str(basket.get("parent", "") or "")
        while parent_name:
            depth += 1
            parent_item = items.get(parent_name, {})
            if not isinstance(parent_item, dict):
                break
            parent_name = str(parent_item.get("parent", "") or "")

        rows.append(
            {
                "Thematic": basket_name,
                "Tier": str(basket.get("tier_label", "") or "Unspecified"),
                "Parent": str(basket.get("parent", "") or "Root basket"),
            }
        )
        meta_rows.append(
            {
                "basket_name": basket_name,
                "depth": depth,
                "is_parent": bool(basket.get("is_parent", False)),
                "is_ai_super_parent": bool(basket.get("is_ai_super_parent", False)),
                "is_ai_group_child": bool(basket.get("is_ai_group_child", False)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


def _build_thematics_hierarchy_styler(
    display_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    name_column: str,
    selected_basket: Optional[str] = None,
    value_color_rules: Optional[Dict[str, Callable[[object], str]]] = None,
    format_map: Optional[Dict[str, object]] = None,
) -> "Styler":
    styler = display_df.style.hide(axis="index")
    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#F4F8FC"),
                    ("color", "#173A5E"),
                    ("font-weight", "800"),
                    ("text-align", "center"),
                    ("border-bottom", "1px solid #D9E6F2"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid #E7EFF7"),
                    ("vertical-align", "middle"),
                ],
            },
        ],
        overwrite=False,
    )

    centered_columns = [column for column in display_df.columns if column != name_column]
    if centered_columns:
        styler = styler.set_properties(subset=centered_columns, **{"text-align": "center"})
    score_columns = [
        column
        for column in [
            "TS",
            "RSI Regime",
            "Sector Regime Fit",
            "FS",
            "Mom. FS",
            "Growth FS",
            "Value FS",
            "Quality FS",
            "Risk FS",
        ]
        if column in display_df.columns
    ]
    if score_columns:
        styler = styler.set_properties(subset=score_columns, **{"text-align": "center", "white-space": "nowrap"})

    def apply_row_background(row: pd.Series) -> list[str]:
        meta = meta_df.iloc[int(row.name)]
        basket_name = str(meta.get("basket_name", ""))
        if basket_name == selected_basket:
            background = "#FFF7E0"
        elif bool(meta.get("is_ai_super_parent", False)):
            background = "#EEF4FF"
        elif bool(meta.get("is_ai_group_child", False)):
            background = "#F7FBFF"
        else:
            background = "#FFFFFF" if int(row.name) % 2 == 0 else "#FBFDFF"
        return [f"background-color:{background};" for _ in row]

    def apply_name_style(row: pd.Series) -> list[str]:
        meta = meta_df.iloc[int(row.name)]
        depth = int(meta.get("depth", 0) or 0)
        is_parent = bool(meta.get("is_parent", False))
        is_ai_super_parent = bool(meta.get("is_ai_super_parent", False))
        is_ai_group_child = bool(meta.get("is_ai_group_child", False))
        padding_left = 0.55 + (depth * 1.2)
        color = "#123159"
        font_weight = "700" if is_parent else "600"
        extra = ""
        if is_ai_super_parent:
            color = "#123159"
            font_weight = "800"
            extra = "border-left:4px solid #184E82;"
        elif is_ai_group_child:
            color = "#0B5CAD"
            font_weight = "700"
            extra = "border-left:3px solid #93C5FD;"
        elif depth > 0:
            color = "#4F647C"
        styles = ["" for _ in row]
        name_index = list(display_df.columns).index(name_column)
        styles[name_index] = (
            f"text-align:left; padding-left:{padding_left}rem; color:{color}; "
            f"font-weight:{font_weight}; {extra}"
        )
        return styles

    styler = styler.apply(apply_row_background, axis=1)
    styler = styler.apply(apply_name_style, axis=1)
    if value_color_rules:
        for column, rule_fn in value_color_rules.items():
            if column in display_df.columns:
                styler = styler.applymap(rule_fn, subset=[column])

    if format_map:
        valid_formats = {column: formatter for column, formatter in format_map.items() if column in display_df.columns}
        if valid_formats:
            styler = styler.format(valid_formats, na_rep="N/A")

    return styler


def _build_thematics_company_styler(
    display_df: pd.DataFrame,
    trend_meta_df: Optional[pd.DataFrame] = None,
) -> "Styler":
    styler = display_df.style.hide(axis="index")
    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#F4F8FC"),
                    ("color", "#173A5E"),
                    ("font-weight", "800"),
                    ("text-align", "center"),
                    ("border-bottom", "1px solid #D9E6F2"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid #E7EFF7"),
                    ("vertical-align", "middle"),
                    ("color", "#334E68"),
                ],
            },
        ],
        overwrite=False,
    )

    left_columns = [
        "Thematic",
        "Ticker",
        "Company",
        "Sector",
        "Industry",
        "Short Term Flow",
        "Rel Strength",
        "Rel Volume",
        "AI Revenue Exposure",
        "AI Disruption Risk",
    ]
    centered_columns = [column for column in display_df.columns if column not in left_columns]
    usable_left_columns = [column for column in left_columns if column in display_df.columns]
    if usable_left_columns:
        styler = styler.set_properties(subset=usable_left_columns, **{"text-align": "left"})
    if centered_columns:
        styler = styler.set_properties(subset=centered_columns, **{"text-align": "center"})
    score_columns = [
        column
        for column in ["TS", "FS", "Mom. FS", "Growth FS", "Value FS", "Quality FS", "Risk FS"]
        if column in display_df.columns
    ]
    if score_columns:
        styler = styler.set_properties(subset=score_columns, **{"text-align": "center", "white-space": "nowrap"})

    def stripe_rows(row: pd.Series) -> list[str]:
        background = "#FFFFFF" if int(row.name) % 2 == 0 else "#FBFDFF"
        return [f"background-color:{background};" for _ in row]

    styler = styler.apply(stripe_rows, axis=1)
    usable_perf_columns = [column for column in ["1W", "1M", "3M", "YTD"] if column in display_df.columns]
    if usable_perf_columns:
        styler = styler.applymap(_style_positive_negative_value, subset=usable_perf_columns)
    usable_score_columns = [
        column
        for column in [
            "TS",
            "RSI Regime",
            "Sector Regime Fit",
            "FS",
            "Mom. FS",
            "Growth FS",
            "Value FS",
            "Quality FS",
            "Risk FS",
        ]
        if column in display_df.columns
    ]
    if usable_score_columns:
        styler = styler.applymap(_score_color_css, subset=usable_score_columns)
    usable_sign_columns = [
        column for column in ["Short Term Flow", "Rel Strength", "Rel Volume"] if column in display_df.columns
    ]
    if usable_sign_columns:
        styler = styler.applymap(_style_sign_label_value, subset=usable_sign_columns)
    usable_divergence_columns = [
        column for column in ["RSI Divergence (D)", "RSI Divergence (W)"] if column in display_df.columns
    ]
    if usable_divergence_columns:
        styler = styler.applymap(_style_sign_label_value, subset=usable_divergence_columns)
    return styler


def _render_thematics_basket_table_v2(
    basket_metrics_df: pd.DataFrame,
    catalog: dict[str, object],
    *,
    view_mode: str = "all",
) -> None:
    display_df, meta_df = _build_thematics_basket_table_frame(basket_metrics_df, catalog)
    if display_df.empty or meta_df.empty:
        st.info("No basket metrics available for the selected dates.")
        return

    display_df, meta_df = _filter_thematics_basket_table_for_view(display_df, meta_df, catalog, view_mode)
    if display_df.empty or meta_df.empty:
        st.info("No thematic baskets match the selected AI view filter.")
        return
    display_df = _apply_grid_column_layout(
        display_df,
        GRID_SURFACE_THEMATICS_BASKET,
        locked_columns=["Name"],
    )

    selected_basket = st.session_state.get("thematics_impl_selected_basket")
    visible_baskets = set(meta_df.get("basket_name", pd.Series(dtype=object)).astype(str).tolist())
    normalized_selected_basket = _normalize_thematics_selected_basket(
        str(selected_basket) if selected_basket else None,
        visible_baskets,
    )
    if normalized_selected_basket != selected_basket:
        st.session_state["thematics_impl_selected_basket"] = normalized_selected_basket
        selected_basket = normalized_selected_basket
    styler = _build_thematics_hierarchy_styler(
        display_df,
        meta_df,
        name_column="Name",
        selected_basket=str(selected_basket) if selected_basket else None,
        value_color_rules={
            "1W": _style_positive_negative_value,
            "1M": _style_positive_negative_value,
            "3M": _style_positive_negative_value,
            "YTD": _style_positive_negative_value,
            "RS %": _style_breadth_threshold_value,
            "Vol %": _style_breadth_threshold_value,
            "TS": _score_color_css,
            "FS": _score_color_css,
            "Mom. FS": _score_color_css,
            "Growth FS": _score_color_css,
            "Value FS": _score_color_css,
            "Quality FS": _score_color_css,
            "Risk FS": _score_color_css,
        },
        format_map={
            "Beta": _format_numeric_value,
            "1W": _format_percent_value,
            "1M": _format_percent_value,
            "3M": _format_percent_value,
            "YTD": _format_percent_value,
            "RS %": _format_percent_value,
            "Vol %": _format_percent_value,
        },
    )

    estimated_height = max(280, 72 + len(display_df) * 38)
    basket_table_nonce = st.session_state.setdefault("thematics_impl_basket_table_nonce", 0)
    try:
        selection_event = st.dataframe(
            styler,
            use_container_width=True,
            height=estimated_height,
            on_select="rerun",
            selection_mode="single-row",
            hide_index=True,
            key=f"thematics_impl_basket_table_{basket_table_nonce}",
        )
    except TypeError:
        st.dataframe(styler, use_container_width=True, height=estimated_height, hide_index=True)
        st.warning("Row click selection is unavailable on this Streamlit version.")
        return

    selected_rows = _extract_dataframe_selection_rows(selection_event)
    if not selected_rows:
        return
    selected_index = selected_rows[0]
    if selected_index < 0 or selected_index >= len(meta_df):
        return
    basket_name = str(meta_df.iloc[selected_index]["basket_name"])
    if st.session_state.get("thematics_impl_selected_basket") != basket_name:
        _queue_show_all_company_reset("thematics_impl")
        st.session_state["thematics_impl_selected_basket"] = basket_name
        st.rerun()


def _render_thematics_lens_tab(
    catalog: dict[str, object],
    report_df: Optional[pd.DataFrame],
) -> None:
    lens_df, meta_df = _build_thematics_lens_frame(catalog)
    if lens_df.empty or meta_df.empty:
        st.info("No thematic baskets available.")
        return

    lens_focus_key = "thematics_lens_focus_basket"
    selected_focus = st.session_state.get(lens_focus_key)
    selector_col, card_col = st.columns([1.3, 1.9])
    with selector_col:
        st.caption("Click a thematic to open its lens card.")
        styler = _build_thematics_hierarchy_styler(
            lens_df,
            meta_df,
            name_column="Thematic",
            selected_basket=str(selected_focus) if selected_focus else None,
        )
        estimated_height = max(280, 72 + len(lens_df) * 38)
        try:
            selection_event = st.dataframe(
                styler,
                use_container_width=True,
                height=estimated_height,
                on_select="rerun",
                selection_mode="single-row",
                hide_index=True,
                key="thematics_lens_selector",
            )
        except TypeError:
            st.dataframe(styler, use_container_width=True, height=estimated_height, hide_index=True)
            selection_event = None
        selected_rows = _extract_dataframe_selection_rows(selection_event)
        if selected_rows:
            selected_index = selected_rows[0]
            if 0 <= selected_index < len(meta_df):
                basket_name = str(meta_df.iloc[selected_index]["basket_name"])
                if st.session_state.get(lens_focus_key) != basket_name:
                    st.session_state[lens_focus_key] = basket_name
                    st.rerun()
    with card_col:
        _render_thematics_context_card(catalog, st.session_state.get(lens_focus_key))
        _render_thematics_constituents_card(catalog, st.session_state.get(lens_focus_key), report_df)


def render_thematics_tab(config: ReportConfig) -> None:
    render_page_intro(
        "Thematics",
        "Hierarchy-aware thematic baskets with basket metrics, narratives, and drill-down company grids.",
        "Equipilot / Thematics",
    )
    render_subtab_group_intro(
        "Thematics sections",
        "Use the implementation workspace for sortable basket and company tables, or open Thematic Lens for the narrative card view.",
    )
    thematics_config_signature = _path_cache_signature(THEMATICS_CONFIG_PATH)
    catalog = (
        build_thematics_catalog(str(THEMATICS_CONFIG_PATH), thematics_config_signature)
        if THEMATICS_CONFIG_PATH.exists()
        else {"items": {}, "roots": []}
    )
    if not THEMATICS_CONFIG_PATH.exists():
        st.error(f"Missing thematics config: {THEMATICS_CONFIG_PATH}")
        return

    implementation_tab, lens_tab = st.tabs(["thematics-implementation", "thematic-lens"])
    with implementation_tab:
        reference_date = render_report_select_date_input(
            "Reference EOD date",
            value=get_default_board_eod(config),
            key="thematics_reference_eod",
        )
        previous_eod = render_report_select_date_input(
            "Previous EOD date (for trend arrows)",
            value=get_default_previous_board_eod(reference_date),
            key="thematics_previous_eod",
        )
        show_perf = st.checkbox(
            "Show perf timings",
            value=False,
            key="thematics_perf_toggle",
        )
        warnings: list[str] = []
        timings: list[tuple[str, float]] = []

        t_start = time.perf_counter()
        current_report_df, current_report_path, current_candidates, current_error = load_report_select_for_eod(reference_date)
        _perf_mark(timings, "load current", t_start)
        if current_report_path is None:
            warnings.append(
                f"Metadata/scoring report is missing for {reference_date.isoformat()}; metadata, beta, scoring, RS, and OBVM fields are shown as N/A."
            )
            current_report_df = None
        elif current_error:
            warnings.append(
                f"Current report_select file could not be read ({current_report_path}); metadata, beta, scoring, RS, and OBVM fields are shown as N/A."
            )
            current_report_df = None

        t_start = time.perf_counter()
        previous_report_df, previous_report_path, previous_candidates, previous_error = load_report_select_for_eod(previous_eod)
        _perf_mark(timings, "load previous", t_start)
        previous_ready = True
        if previous_report_path is None:
            warnings.append(
                f"Previous report_select file is missing for {previous_eod.isoformat()}; trend arrows are hidden."
            )
            previous_report_df = None
            previous_ready = False
        elif previous_error:
            warnings.append(
                f"Previous report_select file could not be read ({previous_report_path}); trend arrows are hidden."
            )
            previous_report_df = None
            previous_ready = False

        prices_cache_file = prices_cache_path("daily", reference_date.year)
        prices_cache_signature = _path_cache_signature(prices_cache_file) if prices_cache_file.exists() else ""
        price_lookup: dict[str, dict[str, list[object]]] = {}
        if prices_cache_file.exists():
            try:
                t_start = time.perf_counter()
                price_lookup = build_price_history_lookup(str(prices_cache_file), prices_cache_signature)
                _perf_mark(timings, "load prices", t_start)
            except Exception as exc:  # pragma: no cover - UI feedback
                warnings.append(f"Daily prices cache could not be read ({prices_cache_file}): {exc}")
        else:
            warnings.append(
                f"Daily prices cache is missing for {reference_date.year}; price performance fields are shown as N/A."
            )

        t_start = time.perf_counter()
        basket_metrics_df, anchor_missing = _build_thematics_basket_metrics(
            catalog,
            current_report_df,
            previous_report_df if previous_ready else None,
            price_lookup,
            reference_date,
        )
        _perf_mark(timings, "build baskets", t_start)
        if anchor_missing:
            warnings.append(
                f"Some tickers do not have an exact daily price for {reference_date.isoformat()}; affected 1W/1M/3M/YTD values are shown as N/A."
            )
        if current_report_df is not None and "beta" not in current_report_df.columns:
            warnings.append(
                "The selected report_select file does not contain `beta`; basket and company beta values are shown as N/A until the report cache is regenerated with the updated schema."
            )

        chips = [f"Reference EOD: {reference_date.isoformat()}"]
        if current_report_path is not None:
            chips.append(f"Current report: {current_report_path.name}")
        else:
            chips.append(f"Expected report: {current_candidates[0].name}")
        if previous_ready and previous_report_path is not None:
            chips.append(f"Previous report: {previous_report_path.name}")
        else:
            chips.append(f"Previous EOD: {previous_eod.isoformat()} (trend off)")
        if prices_cache_file.exists():
            chips.append(f"Prices cache: {prices_cache_file.name}")
        render_chip_row(chips)

        for warning_message in dict.fromkeys(warnings):
            st.warning(warning_message)

        view_mode = _render_thematics_view_mode_controls("thematics_impl")
        _apply_pending_show_all_company_reset("thematics_impl")
        st.checkbox(
            "Show all companies",
            key="thematics_impl_show_all_companies",
            on_change=_handle_thematics_show_all_company_toggle,
            args=("thematics_impl",),
        )
        _render_thematics_basket_table(basket_metrics_df, catalog, view_mode=view_mode)

        show_all_companies = bool(st.session_state.get("thematics_impl_show_all_companies", False))
        selected_basket = st.session_state.get("thematics_impl_selected_basket")
        if show_all_companies or selected_basket:
            if show_all_companies:
                company_scope = "show_all"
                company_scope_name = "All thematic companies"
            else:
                company_scope = "selected"
                company_scope_name = str(selected_basket)
            t_start = time.perf_counter()
            company_universe, company_anchor_missing = _build_thematics_company_universe_for_scope_cached(
                company_scope,
                company_scope_name,
                str(THEMATICS_CONFIG_PATH),
                thematics_config_signature,
                str(current_report_path) if current_report_path is not None else "",
                _path_cache_signature(current_report_path) if current_report_path is not None else "",
                str(prices_cache_file) if prices_cache_file.exists() else "",
                prices_cache_signature,
                reference_date,
                _market_regime_company_metrics_cache_signature(reference_date),
                _company_divergence_cache_signature(),
            )
            _perf_mark(timings, "company universe", t_start)
            if company_anchor_missing:
                st.warning(
                    f"Some companies in {company_scope_name} do not have an exact anchor close for {reference_date.isoformat()}; affected rows show N/A performance."
                )
            if previous_ready and previous_report_df is not None and not company_universe.empty:
                t_start = time.perf_counter()
                company_universe = _annotate_company_score_trends(company_universe, previous_report_df, threshold=5.0)
                _perf_mark(timings, "company trend", t_start)
            filter_signature = (
                reference_date.isoformat(),
                previous_eod.isoformat(),
                company_scope,
                company_scope_name,
                "trend_on" if previous_ready and previous_report_df is not None else "trend_off",
            )
            _sync_drilldown_filter_defaults(
                "thematics",
                filter_signature,
                default_thematics=[],
                default_sectors=[],
                default_industries=[],
                default_cap_buckets=["Large", "Mega"],
                default_fund_range=(50.0, 100.0),
                default_tech_range=(60.0, 100.0),
                default_rsi_regime_range=(70.0, 100.0),
                default_sector_regime_fit_range=_default_sector_regime_fit_range_for_company_scope(company_scope),
                default_fund_momentum_range=(60.0, 100.0),
                default_tech_trend_dir="All",
                default_rel_strength="All",
                default_rel_volume="All",
                default_ai_revenue_exposure="All",
                default_ai_disruption_risk="All",
                default_beta_range=(0.0, 5.0),
            )
            st.markdown("---")
            if show_all_companies:
                st.caption("Companies across all thematic baskets")
            else:
                st.caption(f"Companies in thematic basket: {selected_basket}")
            _, regime_warning = _load_market_regime_company_metrics_for_date(reference_date)
            if regime_warning:
                st.caption(regime_warning)
            t_start = time.perf_counter()
            filtered_companies = render_company_drilldown_filters(
                company_universe,
                prefix="thematics",
                ticker_label="Ticker filter (Thematics grid)",
                include_fundamental_momentum_filter=True,
                include_technical_trend_filter=bool(previous_ready and previous_report_df is not None),
                include_thematic_filter=True,
                include_rel_strength_filter=True,
                include_rel_volume_filter=True,
                include_ai_exposure_filters=True,
                include_beta_filter=True,
            )
            _perf_mark(timings, "company filters", t_start)
            st.caption(f"Companies after filters: {len(filtered_companies)} of {len(company_universe)}")
            t_start = time.perf_counter()
            thematic_display = format_thematics_company_display(filtered_companies)
            thematic_display = _apply_grid_column_layout(
                thematic_display,
                GRID_SURFACE_THEMATICS_COMPANY,
            )
            _perf_mark(timings, "company display", t_start)
            if thematic_display.empty:
                st.info("No companies matched the current thematic filters.")
            else:
                display_height = _company_grid_height(len(thematic_display), row_height=36, min_height=260)
                use_fast_grid = _use_fast_company_grid_render(len(thematic_display))
                if use_fast_grid:
                    st.caption("Large result set: using fast grid rendering.")
                else:
                    t_start = time.perf_counter()
                    thematic_trend_meta = _build_thematics_company_trend_meta(filtered_companies)
                    thematic_styler = _build_thematics_company_styler(
                        thematic_display,
                        trend_meta_df=thematic_trend_meta,
                    )
                    _perf_mark(timings, "company styler", t_start)
                t_start = time.perf_counter()
                st.dataframe(
                    thematic_display if use_fast_grid else thematic_styler,
                    use_container_width=True,
                    height=display_height,
                    hide_index=True,
                    key="thematics_company_grid",
                )
                _perf_mark(timings, "company render", t_start)
            if st.button("Hide thematic company list", key="thematics_hide_company_list"):
                st.session_state["thematics_impl_selected_basket"] = None
                _queue_show_all_company_reset("thematics_impl")
                st.rerun()
        _render_perf_timings(show_perf, timings)
    with lens_tab:
        _render_thematics_lens_tab(catalog, current_report_df)

def render_quadrants(default_anchor: date) -> None:
    render_page_intro(
        "T vs P Quadrants",
        "Relative sector positioning across technical strength and participation.",
        "Equipilot / Quadrants",
    )
    st.caption("Quadrants are computed from existing report_select files only (no SQL).")
    if st.button("Refresh Quadrants", use_container_width=True, key="quadrants_refresh"):
        clear_quadrant_caches()
        st.rerun()

    available_quadrant_dates = get_available_report_select_dates()
    fallback_curr = default_anchor
    fallback_prev = fallback_curr - timedelta(days=30)
    default_quadrant_curr = available_quadrant_dates[-1] if available_quadrant_dates else fallback_curr
    default_quadrant_prev = (
        available_quadrant_dates[-2]
        if len(available_quadrant_dates) >= 2
        else (available_quadrant_dates[-1] if available_quadrant_dates else fallback_prev)
    )
    quadrant_date_curr = render_report_select_date_input(
        "Quadrant date (current)",
        value=default_quadrant_curr,
        key="quadrant_date_curr",
    )
    quadrant_date_prev = render_report_select_date_input(
        "Quadrant date (previous)",
        value=default_quadrant_prev,
        key="quadrant_date_prev",
    )

    if quadrant_date_prev > quadrant_date_curr:
        st.warning("Previous date is after current date; deltas may be inverted.")

    curr_path, curr_candidates = resolve_report_select_path(quadrant_date_curr)
    if curr_path is None:
        st.error(
            f"Missing report_select file for {quadrant_date_curr.isoformat()}. "
            f"Expected {curr_candidates[0]} (or {curr_candidates[1]})."
        )
        return
    prev_path, prev_candidates = resolve_report_select_path(quadrant_date_prev)
    if prev_path is None:
        st.error(
            f"Missing report_select file for {quadrant_date_prev.isoformat()}. "
            f"Expected {prev_candidates[0]} (or {prev_candidates[1]})."
        )
        return

    render_chip_row(
        [
            f"Current file: {curr_path}",
            f"Previous file: {prev_path}",
            f"Delta window: {quadrant_date_prev.isoformat()} -> {quadrant_date_curr.isoformat()}",
        ]
    )

    with st.expander("Quadrants settings", expanded=False):
        st.markdown("**Participation (P) discounts**")
        c1, c2, c3 = st.columns(3)
        with c1:
            p_strong_pos_thresh = st.number_input(
                "Strong +% threshold",
                value=5.0,
                step=0.5,
                key="quadrants_p_strong_pos_thresh",
            )
            p_weak_pos_thresh = st.number_input(
                "Weak +% threshold",
                value=0.0,
                step=0.5,
                key="quadrants_p_weak_pos_thresh",
            )
            p_weak_neg_thresh = st.number_input(
                "Weak -% threshold",
                value=-5.0,
                step=0.5,
                key="quadrants_p_weak_neg_thresh",
            )
        with c2:
            p_mult_strong_pos = st.number_input(
                "Multiplier > strong +%",
                value=1.00,
                step=0.05,
                key="quadrants_p_mult_strong_pos",
            )
            p_mult_weak_pos = st.number_input(
                "Multiplier > weak +%",
                value=0.90,
                step=0.05,
                key="quadrants_p_mult_weak_pos",
            )
        with c3:
            p_mult_weak_neg = st.number_input(
                "Multiplier > weak -%",
                value=0.80,
                step=0.05,
                key="quadrants_p_mult_weak_neg",
            )
            p_mult_strong_neg = st.number_input("Multiplier ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â°Ãƒâ€šÃ‚Â¤ weak -%", value=0.70, step=0.05)

        st.markdown("**Technical (T) adjustment**")
        t_max_tilt = st.number_input("Max tilt (ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â±)", value=0.10, step=0.01, format="%.2f")
        t_extreme_target = st.number_input(
            "Extreme coverage target",
            value=0.15,
            step=0.01,
            format="%.2f",
            help="Fraction of sector stocks at near highs/lows for full tilt.",
        )
        if not (p_strong_pos_thresh > p_weak_pos_thresh > p_weak_neg_thresh):
            st.warning("P thresholds should be descending: strong +% > weak +% > weak -%.")

    try:
        with st.spinner("Computing T/P components and quadrants..."):
            t_curr = compute_sector_T(str(curr_path), t_max_tilt, t_extreme_target)
            t_prev = compute_sector_T(str(prev_path), t_max_tilt, t_extreme_target)
            p_curr, curr_mc_var_all_negative = compute_sector_P(
                str(curr_path),
                p_strong_pos_thresh,
                p_weak_pos_thresh,
                p_weak_neg_thresh,
                p_mult_strong_pos,
                p_mult_weak_pos,
                p_mult_weak_neg,
                p_mult_strong_neg,
            )
            p_prev, prev_mc_var_all_negative = compute_sector_P(
                str(prev_path),
                p_strong_pos_thresh,
                p_weak_pos_thresh,
                p_weak_neg_thresh,
                p_mult_strong_pos,
                p_mult_weak_pos,
                p_mult_weak_neg,
                p_mult_strong_neg,
            )
    except ValueError as exc:
        st.error(str(exc))
        return

    curr_metrics = t_curr.merge(p_curr, on="sector", how="inner")
    prev_metrics = t_prev.merge(p_prev, on="sector", how="inner")
    missing_prev = sorted(set(curr_metrics["sector"]) - set(prev_metrics["sector"]))
    missing_curr = sorted(set(prev_metrics["sector"]) - set(curr_metrics["sector"]))
    if missing_prev:
        st.warning(f"Sectors missing in previous date: {', '.join(missing_prev)}")
    if missing_curr:
        st.warning(f"Sectors missing in current date: {', '.join(missing_curr)}")
    if curr_metrics.empty or prev_metrics.empty:
        st.error("No sector metrics available for the selected dates.")
        return

    quadrant_df = build_quadrant_df(prev_metrics, curr_metrics)
    if quadrant_df.empty:
        st.error("No overlapping sectors between the selected dates.")
        return

    curr_p_median = float(np.nanmedian(curr_metrics["P"]))
    curr_t_median = float(np.nanmedian(curr_metrics["T"]))

    output_path = QUADRANTS_DIR / (
        f"quadrants_{quadrant_date_prev.isoformat()}_to_{quadrant_date_curr.isoformat()}.json"
    )
    settings_payload = {
        "p_discounts": {
            "strong_pos_threshold": p_strong_pos_thresh,
            "weak_pos_threshold": p_weak_pos_thresh,
            "weak_neg_threshold": p_weak_neg_thresh,
            "mult_strong_pos": p_mult_strong_pos,
            "mult_weak_pos": p_mult_weak_pos,
            "mult_weak_neg": p_mult_weak_neg,
            "mult_strong_neg": p_mult_strong_neg,
        },
        "t_adjustment": {
            "max_tilt": t_max_tilt,
            "extreme_coverage_target": t_extreme_target,
        },
    }
    saved_path = save_quadrant_json(
        quadrant_df,
        output_path,
        quadrant_date_prev,
        quadrant_date_curr,
        curr_mc_var_all_negative,
        prev_mc_var_all_negative,
        curr_p_median,
        curr_t_median,
        settings_payload,
    )
    st.caption(f"Quadrant JSON saved: {saved_path}")

    mode = st.radio("Axes mode", ["Absolute", "Percentile"], horizontal=True, index=0, key="quadrants_axes_mode")
    fig = plot_quadrants(quadrant_df, mode)
    st.plotly_chart(fig, use_container_width=True)

    def _fmt(value: float) -> str:
        if value is None or np.isnan(value):
            return "n/a"
        return f"{value:.1f}"

    st.caption(f"P median (current): {_fmt(curr_p_median)} | T median (current): {_fmt(curr_t_median)}")
    if curr_p_median >= 50 and curr_t_median >= 50:
        st.info("Market regime: Broad risk-on (P_med >= 50 and T_med >= 50).")
    elif curr_p_median < 50:
        st.info("Market regime: Risk-off / narrow (P_med < 50).")
    if curr_mc_var_all_negative:
        st.warning("All sectors show negative 1-month market cap variation (absolute weakness).")
    st.markdown(
        f"""
**How to read the quadrants**
- A (High T + High P): confirmed leadership (dT > 0 accelerating, dT < 0 fading)
- B (High T + Low P): narrow/fragile (dT > 0 strengthening but fragile; dT < 0 breakdown risk)
- C (Low T + High P): early rotation (dT > 0 emerging; dT < 0 wait)
- D (Low T + Low P): weak (dT > 0 bottoming attempt; dT < 0 breakdown)
- Border color: green when dT > {QUADRANT_BORDER_D_T_THRESHOLD:.0f}, red when dT < -{QUADRANT_BORDER_D_T_THRESHOLD:.0f}, no border otherwise
        """
    )


def main() -> None:
    st.set_page_config(page_title="Equipilot", layout="wide", page_icon=_favicon, initial_sidebar_state="expanded")
    apply_theme_styles()
    force_sync = st.session_state.pop("force_sync", False)
    sync_editors(force=force_sync)
    ensure_api_state()

    try:
        config = load_report_config(CONFIG_PATH)
    except Exception as exc:  # pragma: no cover - UI feedback
        st.error(f"Invalid config: {exc}")
        config = ReportConfig(report_date=date.today(), eod_as_of_date=date.fromisoformat(bucharest_today_str()))

    config = render_sidebar(config)
    render_header()

    home_tab, indices_tab, market_tab, sector_tab, thematics_tab, trade_ideas_tab, quadrants_tab, api_tab = st.tabs(
        [
            "Home",
            "Indices",
            "Market",
            "Sector",
            "Thematics",
            "Trade Ideas",
            "Quadrants",
            "API",
        ]
    )
    with home_tab:
        render_home(config)
    with indices_tab:
        render_indices_tab()
    with market_tab:
        render_market_tab(config)
    with sector_tab:
        render_sector_tab(config)
    with thematics_tab:
        render_thematics_tab(config)
    with trade_ideas_tab:
        render_trade_ideas(config)
    with quadrants_tab:
        default_anchor = config.eod_as_of_date or config.report_date
        render_quadrants(default_anchor)
    with api_tab:
        render_api_tab()

    render_footer()


if __name__ == "__main__":
    main()






