"""Streamlit UI for managing Monthly Scoring Board generation and helper files."""
from __future__ import annotations

import io
import json
import logging
import base64
import re
from html import escape as html_escape
from contextlib import redirect_stdout, redirect_stderr
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

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
from equipicker_filters import accel_weak, extreme_accel
from report_select_service import generate_report_select_cache
from report_config import DEFAULT_CONFIG_PATH, ReportConfig, load_report_config, save_report_config
from weekly_scoring_board import generate_weekly_scoring_board_pdf
from weekly_scoring_board import compute_sector_overview_stats

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPT_PATH = BASE_DIR / "summary_prompt.txt"
REPORTS_DIR = BASE_DIR / "reports"
QUADRANTS_DIR = REPORTS_DIR / "quadrants"
CONFIG_PATH = DEFAULT_CONFIG_PATH
CONFIG_DIR = CONFIG_PATH.parent
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


def list_report_select_dates() -> list[date]:
    dates: list[date] = []
    for pattern in ("report_select_*.xlsx", "report_select_*.csv"):
        for path in DATA_DIR.glob(pattern):
            parsed = _parse_report_select_date(path)
            if parsed:
                dates.append(parsed)
    return sorted(set(dates))


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
def load_report_select(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


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

        rs_breadth = np.nan
        rs_mask = group["rs_daily"].notna() & group["rs_sma20"].notna()
        if rs_mask.any():
            rs_breadth = (group.loc[rs_mask, "rs_daily"] > group.loc[rs_mask, "rs_sma20"]).mean() * 100.0

        obvm_breadth = np.nan
        obvm_mask = group["obvm_daily"].notna() & group["obvm_sma20"].notna()
        if obvm_mask.any():
            obvm_breadth = (group.loc[obvm_mask, "obvm_daily"] > group.loc[obvm_mask, "obvm_sma20"]).mean() * 100.0

        raw_components = [avg_intermediate, avg_long_term, avg_momentum, rs_breadth, obvm_breadth]
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
        y_prev_col = "T_pct_prev"
        x_label = "Participation (P) percentile"
        y_label = "Technical strength (T) percentile"
    else:
        x_col = "P_curr"
        y_col = "T_curr"
        y_prev_col = "T_prev"
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
    outline_colors = []
    for _, row in df_plot.iterrows():
        d_t = row.get("dT")
        if pd.isna(d_t):
            outline_colors.append("#A0A8B5")
        elif d_t > 0:
            outline_colors.append("#0BA360")
        elif d_t < 0:
            outline_colors.append("#EB5757")
        else:
            outline_colors.append("#A0A8B5")

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
        line_color = outline_colors[idx]
        fig.add_trace(go.Scatter(
            x=[row.get(x_col)],
            y=[row.get(y_col)],
            mode="markers",
            marker=dict(
                size=28,
                color=color,
                line=dict(width=5, color=line_color),
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
    st.session_state["file_bundle"] = load_all_texts()
    sync_editors(force=True)


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
        return "JSON invalid – cannot compute counts."
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
:root {
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
</style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    now_bucharest = datetime.now(ZoneInfo("Europe/Bucharest"))
    st.markdown(
        f"""
<div class="ep-appbar">
  <div>
    <div class="ep-brand">Equipilot</div>
    <div class="ep-tagline">Professional cockpit for monthly scoring and market monitoring.</div>
  </div>
  <div class="ep-time-wrap">
    <div class="ep-time-label">Current time (Europe/Bucharest)</div>
    <div class="ep-time-value">{now_bucharest:%Y-%m-%d %H:%M:%S}</div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
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
    available_dates = list_report_select_dates()
    if config.eod_as_of_date:
        selected, _ = resolve_report_select_path(config.eod_as_of_date)
        if selected is not None:
            return config.eod_as_of_date
    if available_dates:
        return available_dates[-1]
    return date.fromisoformat(bucharest_today_str())


def get_default_previous_board_eod(current_eod: date) -> date:
    previous_dates = [entry for entry in list_report_select_dates() if entry < current_eod]
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
        loaded_df = normalize_report_columns(load_report_select(str(resolved_path)).copy())
    except Exception as exc:  # pragma: no cover - UI feedback
        return None, resolved_path, expected_candidates, str(exc)
    return loaded_df, resolved_path, expected_candidates, None


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
    st.caption("Generate it from Home tab using `Generate report_select Excel`.")


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
        return "color:#A0A8B5;"
    if numeric >= 80:
        return "color:#0BA360; font-weight:700;"
    if numeric >= 60:
        return "color:#3BAE72; font-weight:700;"
    if numeric >= 40:
        return "color:#F2994A; font-weight:700;"
    return "color:#EB5757; font-weight:700;"


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
) -> None:
    if df.empty:
        st.info("No data available for current selection.")
        return

    style_frame = df.copy()
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
    # Keep board tables fully visible by sizing the widget to fit all rows.
    estimated_height = max(220, 72 + len(style_frame) * 38)
    st.dataframe(styler, use_container_width=True, height=estimated_height)


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
            symbol = ""
            if delta_value is not None:
                if delta_value > threshold:
                    symbol = " 📈"
                elif delta_value < -threshold:
                    symbol = " 📉"
            rendered_values.append(f"{curr_value:.1f}{symbol}")
        annotated[col] = rendered_values
    return annotated


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


def build_fundamental_table(df: pd.DataFrame, selected_sector: str) -> pd.DataFrame:
    if selected_sector == "All sectors":
        stats = compute_sector_overview_stats(df)
        if stats.empty:
            return pd.DataFrame(columns=["Sector", "Total", "P1", "P2", "P3", "P4", "P5"])
        table = stats[[
            "sector",
            "avg_total_score",
            "avg_value",
            "avg_growth",
            "avg_quality",
            "avg_risk",
            "avg_momentum",
        ]].rename(columns={
            "sector": "Sector",
            "avg_total_score": "Total",
            "avg_value": "P1",
            "avg_growth": "P2",
            "avg_quality": "P3",
            "avg_risk": "P4",
            "avg_momentum": "P5",
        })
        return table.reset_index(drop=True)

    filtered = df.copy()
    filtered["sector"] = filtered["sector"].fillna("Unspecified")
    filtered["industry"] = filtered.get("industry", pd.Series(index=filtered.index)).fillna("Unspecified")
    filtered = filtered[filtered["sector"] == selected_sector]
    if filtered.empty:
        return pd.DataFrame(columns=["Industry", "Total", "P1", "P2", "P3", "P4", "P5"])

    grouped = (
        filtered.groupby("industry", dropna=False)[[
            "fundamental_total_score",
            "fundamental_value",
            "fundamental_growth",
            "fundamental_quality",
            "fundamental_risk",
            "fundamental_momentum",
        ]]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={
            "industry": "Industry",
            "fundamental_total_score": "Total",
            "fundamental_value": "P1",
            "fundamental_growth": "P2",
            "fundamental_quality": "P3",
            "fundamental_risk": "P4",
            "fundamental_momentum": "P5",
        })
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
            .rename(columns={
                "sector": "Sector",
                "general_technical_score": "Total",
                "relative_performance": "Relative Performance",
                "relative_volume": "Relative Volume",
                "momentum": "Momentum",
                "intermediate_trend": "Intermediate Trend",
                "long_term_trend": "Long-term Trend",
            })
            .sort_values("Total", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        return grouped

    filtered = work[work["sector"] == selected_sector]
    if filtered.empty:
        return pd.DataFrame(columns=[
            "Industry",
            "Total",
            "Relative Performance",
            "Relative Volume",
            "Momentum",
            "Intermediate Trend",
            "Long-term Trend",
        ])
    grouped = (
        filtered.groupby("industry", dropna=False)[score_columns]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={
            "industry": "Industry",
            "general_technical_score": "Total",
            "relative_performance": "Relative Performance",
            "relative_volume": "Relative Volume",
            "momentum": "Momentum",
            "intermediate_trend": "Intermediate Trend",
            "long_term_trend": "Long-term Trend",
        })
        .sort_values("Total", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    return grouped


def render_sector_pulse_board(config: ReportConfig) -> None:
    render_page_intro(
        "Sector Pulse",
        "Sector breadth, monthly variation, and signal parity with PDF logic.",
        "Equipilot / Sector Pulse",
    )
    selected_eod = st.date_input(
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
    selected_eod = st.date_input("EOD date", value=default_eod, key="fundamental_scoring_eod")
    previous_eod = st.date_input(
        "EOD date (previous)",
        value=get_default_previous_board_eod(selected_eod),
        key="fundamental_scoring_prev_eod",
    )
    with st.spinner("Loading fundamental scoring data..."):
        report_df, source_path, candidates, load_error = load_report_select_for_eod(selected_eod)
    if source_path is None:
        render_missing_report_select(selected_eod, candidates)
        return
    if load_error:
        st.error(f"Failed reading {source_path}: {load_error}")
        return
    previous_report_df: Optional[pd.DataFrame] = None
    previous_path: Optional[Path] = None
    previous_ready = False
    previous_df, previous_path, previous_candidates, previous_error = load_report_select_for_eod(previous_eod)
    if previous_path is None:
        render_missing_report_select(previous_eod, previous_candidates)
    elif previous_error:
        st.error(f"Failed reading {previous_path}: {previous_error}")
    else:
        previous_report_df = previous_df
        previous_ready = True

    chips = [f"Current file: {source_path}", f"EOD: {selected_eod.isoformat()}"]
    if previous_path is not None:
        chips.append(f"Previous file: {previous_path}")
    chips.append(f"EOD previous: {previous_eod.isoformat()}")
    render_chip_row(chips)
    if not validate_required_columns(
        report_df,
        {
            "sector",
            "industry",
            "1m_close",
            "eod_price_used",
            "ic_eod_price_used",
            "market_cap",
            "rs_monthly",
            "obvm_monthly",
            "fundamental_total_score",
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
    if previous_ready and previous_report_df is not None and previous_path is not None:
        previous_ready = validate_required_columns(
            previous_report_df,
            {
                "sector",
                "industry",
                "1m_close",
                "eod_price_used",
                "ic_eod_price_used",
                "market_cap",
                "rs_monthly",
                "obvm_monthly",
                "fundamental_total_score",
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
    selected_sector = st.selectbox(
        "Sector view",
        options=sector_options,
        index=0,
        key="fundamental_scoring_sector_select",
    )
    table_df = build_fundamental_table(report_df, selected_sector)
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
    if previous_ready and previous_report_df is not None:
        previous_table_df = build_fundamental_table(previous_report_df, selected_sector)
        render_df = apply_trend_symbols_to_table(table_df, previous_table_df, score_columns, threshold=5.0)
        format_map = None
    if selected_sector == "All sectors":
        render_board_title_band("Cross-Sector Fundamental Scoring")
    else:
        render_board_title_band(f"{selected_sector} - Industry Fundamental Scoring")
    render_pdf_like_table(
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
    )


def render_technical_scoring_board(config: ReportConfig) -> None:
    render_page_intro(
        "Technical Scoring",
        "Technical pillar scoring with sector view and industry drill-down by selected sector.",
        "Equipilot / Technical Scoring",
    )
    default_eod = get_default_board_eod(config)
    selected_eod = st.date_input("EOD date", value=default_eod, key="technical_scoring_eod")
    previous_eod = st.date_input(
        "EOD date (previous)",
        value=get_default_previous_board_eod(selected_eod),
        key="technical_scoring_prev_eod",
    )
    with st.spinner("Loading technical scoring data..."):
        report_df, source_path, candidates, load_error = load_report_select_for_eod(selected_eod)
    if source_path is None:
        render_missing_report_select(selected_eod, candidates)
        return
    if load_error:
        st.error(f"Failed reading {source_path}: {load_error}")
        return
    previous_report_df: Optional[pd.DataFrame] = None
    previous_path: Optional[Path] = None
    previous_ready = False
    previous_df, previous_path, previous_candidates, previous_error = load_report_select_for_eod(previous_eod)
    if previous_path is None:
        render_missing_report_select(previous_eod, previous_candidates)
    elif previous_error:
        st.error(f"Failed reading {previous_path}: {previous_error}")
    else:
        previous_report_df = previous_df
        previous_ready = True

    chips = [f"Current file: {source_path}", f"EOD: {selected_eod.isoformat()}"]
    if previous_path is not None:
        chips.append(f"Previous file: {previous_path}")
    chips.append(f"EOD previous: {previous_eod.isoformat()}")
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
    if previous_ready and previous_report_df is not None and previous_path is not None:
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
    selected_sector = st.selectbox(
        "Sector view",
        options=sector_options,
        index=0,
        key="technical_scoring_sector_select",
    )
    table_df = build_technical_table(report_df, selected_sector)
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
    if previous_ready and previous_report_df is not None:
        previous_table_df = build_technical_table(previous_report_df, selected_sector)
        render_df = apply_trend_symbols_to_table(table_df, previous_table_df, score_columns, threshold=5.0)
        format_map = None
    render_pdf_like_table(
        render_df,
        center_all_except_first=True,
        highlight_max_cols=score_columns,
        value_color_rules={column: _score_color_css for column in score_columns},
        format_map=format_map,
    )


def _run_trade_idea_filter(strategy_name: str, report_df: pd.DataFrame) -> pd.DataFrame:
    working_df = report_df.copy()
    if strategy_name == "extreme_accel":
        return extreme_accel(working_df, DATA_DIR, save_output=False)
    if strategy_name == "accel_weak":
        return accel_weak(working_df, DATA_DIR, save_output=False)
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
        "eod_price_date",
    ]
    available = [col for col in preferred_columns if col in df.columns]
    if available:
        return df.loc[:, available]
    return df


def _render_trade_idea_strategy(
    config: ReportConfig,
    *,
    strategy_name: str,
    board_title: str,
    strategy_subtitle: str,
    required_columns: set[str],
    eod_key: str,
) -> None:
    st.caption(strategy_subtitle)
    selected_eod = st.date_input(
        "EOD date",
        value=get_default_board_eod(config),
        key=eod_key,
    )
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

    export_cols = st.columns([1, 1, 4])
    csv_data = display_df.to_csv(index=False).encode("utf-8")
    with export_cols[0]:
        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"{strategy_name}_{selected_eod.isoformat()}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"{strategy_name}_{eod_key}_download_csv",
        )
    with export_cols[1]:
        st.download_button(
            "Download JSON",
            data=display_df.to_json(orient="records", force_ascii=False, indent=2),
            file_name=f"{strategy_name}_{selected_eod.isoformat()}.json",
            mime="application/json",
            use_container_width=True,
            key=f"{strategy_name}_{eod_key}_download_json",
        )

    estimated_height = max(260, 72 + len(display_df) * 38)
    st.dataframe(display_df, use_container_width=True, height=estimated_height, hide_index=True)


def render_trade_ideas(config: ReportConfig) -> None:
    render_page_intro(
        "Trade Ideas",
        "Actionable candidates from curated acceleration filters.",
        "Equipilot / Trade Ideas",
    )
    render_chip_row(
        [
            "Strategies currently active: extreme_accel, accel_weak",
            "Data source: report_select_<EOD> file",
        ]
    )
    extreme_tab, weak_tab = st.tabs(["extreme_accel", "accel_weak"])

    with extreme_tab:
        render_board_title_band("Trade Ideas - extreme_accel")
        _render_trade_idea_strategy(
            config,
            strategy_name="extreme_accel",
            board_title="Trade Ideas / extreme_accel",
            strategy_subtitle="Highest-conviction acceleration setup with strict technical and thrust constraints.",
            required_columns={
                "general_technical_score",
                "relative_performance",
                "relative_volume",
                "momentum",
                "intermediate_trend",
                "long_term_trend",
                "rs_daily",
                "rs_sma20",
                "obvm_daily",
                "obvm_sma20",
            },
            eod_key="trade_ideas_extreme_eod",
        )

    with weak_tab:
        render_board_title_band("Trade Ideas - accel_weak")
        _render_trade_idea_strategy(
            config,
            strategy_name="accel_weak",
            board_title="Trade Ideas / accel_weak",
            strategy_subtitle="Moderate acceleration profile with constructive but cooling momentum behavior.",
            required_columns={
                "general_technical_score",
                "relative_performance",
                "relative_volume",
                "momentum",
                "intermediate_trend",
                "long_term_trend",
            },
            eod_key="trade_ideas_weak_eod",
        )


def render_home(config: ReportConfig) -> None:
    render_page_intro(
        "Home Cockpit",
        "Generate report_select Excel only (no PDF) and manage EOD report cache.",
        "Equipilot / Home",
    )
    default_anchor = config.eod_as_of_date or date.fromisoformat(bucharest_today_str())
    controls_col, toggles_col = st.columns([1.15, 1])
    with controls_col:
        home_anchor_date = st.date_input(
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
    available_dates = list_report_select_dates()
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

    render_chip_row(
        [
            f"Output target: {report_cache_path(cache_date=home_anchor_date).name}",
        ]
    )
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


def render_monthly_board(config: ReportConfig) -> None:
    render_page_intro(
        "Monthly Scoring Board",
        "Configure report dates, run generation, and manage prompt/source files.",
        "Equipilot / Monthly Scoring Board",
    )
    report_date_value = st.date_input("Report date", value=config.report_date, key="monthly_report_date")
    eod_as_of_value = st.date_input(
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
        )
        copy_button(final_prompt, "monthly_final_prompt_copy")


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

    available_quadrant_dates = list_report_select_dates()
    fallback_curr = default_anchor
    fallback_prev = fallback_curr - timedelta(days=30)
    default_quadrant_curr = available_quadrant_dates[-1] if available_quadrant_dates else fallback_curr
    default_quadrant_prev = (
        available_quadrant_dates[-2]
        if len(available_quadrant_dates) >= 2
        else (available_quadrant_dates[-1] if available_quadrant_dates else fallback_prev)
    )
    quadrant_date_curr = st.date_input(
        "Quadrant date (current)",
        value=default_quadrant_curr,
        key="quadrant_date_curr",
    )
    quadrant_date_prev = st.date_input(
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
            p_strong_pos_thresh = st.number_input("Strong +% threshold", value=5.0, step=0.5)
            p_weak_pos_thresh = st.number_input("Weak +% threshold", value=0.0, step=0.5)
            p_weak_neg_thresh = st.number_input("Weak -% threshold", value=-5.0, step=0.5)
        with c2:
            p_mult_strong_pos = st.number_input("Multiplier > strong +%", value=1.00, step=0.05)
            p_mult_weak_pos = st.number_input("Multiplier > weak +%", value=0.90, step=0.05)
        with c3:
            p_mult_weak_neg = st.number_input("Multiplier > weak -%", value=0.80, step=0.05)
            p_mult_strong_neg = st.number_input("Multiplier ≤ weak -%", value=0.70, step=0.05)

        st.markdown("**Technical (T) adjustment**")
        t_max_tilt = st.number_input("Max tilt (±)", value=0.10, step=0.01, format="%.2f")
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
        """
**How to read the quadrants**
- A (High T + High P): confirmed leadership (dT > 0 accelerating, dT < 0 fading)
- B (High T + Low P): narrow/fragile (dT > 0 strengthening but fragile; dT < 0 breakdown risk)
- C (Low T + High P): early rotation (dT > 0 emerging; dT < 0 wait)
- D (Low T + Low P): weak (dT > 0 bottoming attempt; dT < 0 breakdown)
        """
    )


def main() -> None:
    st.set_page_config(page_title="Equipilot", layout="wide")
    apply_theme_styles()
    force_sync = st.session_state.pop("force_sync", False)
    sync_editors(force=force_sync)

    try:
        config = load_report_config(CONFIG_PATH)
    except Exception as exc:  # pragma: no cover - UI feedback
        st.error(f"Invalid config: {exc}")
        config = ReportConfig(report_date=date.today(), eod_as_of_date=date.fromisoformat(bucharest_today_str()))

    render_header()

    home_tab, monthly_tab, sector_pulse_tab, fundamental_tab, technical_tab, trade_ideas_tab, quadrants_tab = st.tabs(
        [
            "Home",
            "Monthly Scoring Board",
            "Sector Pulse",
            "Fundamental Scoring",
            "Technical Scoring",
            "Trade Ideas",
            "Quadrants",
        ]
    )
    with home_tab:
        render_home(config)
    with monthly_tab:
        render_monthly_board(config)
    with sector_pulse_tab:
        render_sector_pulse_board(config)
    with fundamental_tab:
        render_fundamental_scoring_board(config)
    with technical_tab:
        render_technical_scoring_board(config)
    with trade_ideas_tab:
        render_trade_ideas(config)
    with quadrants_tab:
        default_anchor = config.eod_as_of_date or config.report_date
        render_quadrants(default_anchor)

    st.markdown("---")
    st.markdown(
        "**Shortcut tip:** "
        "`C:/Users/razva/PycharmProjects/equipicker/equipicker/.venv/Scripts/python.exe -m streamlit run equipilot_app.py`"
    )


if __name__ == "__main__":
    main()
