"""Streamlit UI for managing Monthly Scoring Board generation and helper files."""
from __future__ import annotations

import io
import json
import logging
from contextlib import redirect_stdout, redirect_stderr
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def refresh_files() -> None:
    st.session_state["file_bundle"] = load_all_texts()
    sync_editors(force=True)


def append_log(message: str) -> None:
    current = st.session_state.get("logs", "")
    st.session_state["logs"] = (current + ("\n" if current else "") + message)
    placeholder = st.session_state.get("log_placeholder")
    if placeholder is not None:
        placeholder.code(st.session_state["logs"] or "(no logs yet)")


class StreamlitLogHandler(logging.Handler):
    """Logging handler that feeds messages into the Streamlit log panel."""

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI side-effects
        append_log(self.format(record))


def run_generation(
    report_date: date,
    run_sql: bool,
    eod_as_of_date: Optional[date],
    cache_date: Optional[date],
) -> None:
    output = REPORTS_DIR / f"Monthly_Scoring_Board_{report_date.isoformat()}.pdf"
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
                eod_as_of_date=eod_as_of_date,
                cache_date=cache_date,
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


st.set_page_config(page_title="Monthly Scoring Board", layout="wide")
force_sync = st.session_state.pop("force_sync", False)
sync_editors(force=force_sync)

st.title("Monthly Scoring Board")
try:
    config = load_report_config(CONFIG_PATH)
except Exception as exc:  # pragma: no cover - UI feedback
    st.error(f"Invalid config: {exc}")
    config = ReportConfig(report_date=date.today())

st.subheader("Report Config")
report_date_value = st.date_input("Report date", value=config.report_date, key="report_date")
override_eod = st.checkbox("Override EOD as-of date", value=config.eod_as_of_date is not None)
eod_as_of_value = None
if override_eod:
    eod_as_of_value = st.date_input(
        "EOD as-of date (30-day window anchor)",
        value=config.eod_as_of_date or report_date_value,
        key="eod_as_of_date",
    )
override_cache = st.checkbox("Override cache date", value=config.cache_date is not None)
cache_date_value = None
if override_cache:
    cache_date_value = st.date_input(
        "Cache date",
        value=config.cache_date or report_date_value,
        key="cache_date",
    )
available_quadrant_dates = list_report_select_dates()
fallback_curr = cache_date_value or report_date_value
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
if st.button("Save config"):
    save_report_config(ReportConfig(report_date_value, eod_as_of_value, cache_date_value), CONFIG_PATH)
    st.success(f"Saved config: {CONFIG_PATH}")
    st.rerun()

effective_cache_date = cache_date_value.isoformat() if cache_date_value else bucharest_today_str()
st.caption(f"Cache date in use: {effective_cache_date}")
st.caption(f"Report cache: {report_cache_path(cache_date=cache_date_value)}")
st.caption(f"Scoring cache: {scoring_cache_path(cache_date=cache_date_value)}")

st.caption(report_date_value.strftime("%B %d, %Y"))
output_hint = REPORTS_DIR / f"Monthly_Scoring_Board_{report_date_value.isoformat()}.pdf"
st.caption(f"Output path: {output_hint}")

controls_col, _ = st.columns([1, 3])
with controls_col:
    run_sql_toggle = st.checkbox("Run SQL (ignore cache)", value=False)
    if st.button("Generate PDF", use_container_width=True):
        run_generation(report_date_value, run_sql_toggle, eod_as_of_value, cache_date_value)
    if st.button("Refresh Files", use_container_width=True):
        refresh_files()
        st.session_state["force_sync"] = True
        st.rerun()

logs_tab, prompt_tab, sector_tab, summary_tab, quadrants_tab = st.tabs(
    ["Logs", "Summary Prompt", "Sector Data Tables", "Summary Text (JSON)", "Quadrants"]
)

with logs_tab:
    placeholder = st.empty()
    st.session_state["log_placeholder"] = placeholder
    placeholder.code(st.session_state.get("logs", "") or "(no logs yet)")
    if st.button("Clear logs"):
        st.session_state["logs"] = ""
        placeholder.code("(no logs yet)")

bundle = st.session_state["file_bundle"]

with prompt_tab:
    st.caption(f"File: {PROMPT_PATH} (last updated: {_format_ts(PROMPT_PATH)})")
    prompt_text = st.text_area(
        "Prompt contents",
        value=st.session_state.get("prompt_content", ""),
        height=300,
    )
    st.session_state["prompt_content"] = prompt_text
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save prompt"):
            write_text(PROMPT_PATH, prompt_text)
            refresh_files()
            st.session_state["force_sync"] = True
            st.rerun()
    with c2:
        copy_button(prompt_text, "prompt_copy")

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
        )
        st.session_state["sector_content"] = sector_text
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Save sector tables"):
                write_text(sector_path, sector_text)
                refresh_files()
                st.session_state["force_sync"] = True
                st.rerun()
        with c2:
            copy_button(sector_text, "sector_copy")

with summary_tab:
    st.caption(f"File: {SUMMARY_JSON_PATH} (last updated: {_format_ts(SUMMARY_JSON_PATH)})")
    summary_text = st.text_area(
        "Summary JSON",
        value=st.session_state.get("summary_content", ""),
        height=300,
    )
    st.session_state["summary_content"] = summary_text
    st.caption(f"Paragraph counts: {summary_counts(summary_text)}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save summary JSON"):
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
        copy_button(summary_text, "summary_copy")

with quadrants_tab:
    st.subheader("T vs P Quadrants")
    st.caption("Quadrants are computed from existing report_select files only (no SQL).")

    if st.button("Refresh Quadrants", use_container_width=True):
        clear_quadrant_caches()
        st.rerun()

    if quadrant_date_prev > quadrant_date_curr:
        st.warning("Previous date is after current date; deltas may be inverted.")

    curr_path, curr_candidates = resolve_report_select_path(quadrant_date_curr)
    if curr_path is None:
        st.error(
            f"Missing report_select file for {quadrant_date_curr.isoformat()}. "
            f"Expected {curr_candidates[0]} (or {curr_candidates[1]})."
        )
        st.stop()
    prev_path, prev_candidates = resolve_report_select_path(quadrant_date_prev)
    if prev_path is None:
        st.error(
            f"Missing report_select file for {quadrant_date_prev.isoformat()}. "
            f"Expected {prev_candidates[0]} (or {prev_candidates[1]})."
        )
        st.stop()

    st.caption(f"Current file: {curr_path}")
    st.caption(f"Previous file: {prev_path}")

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
        st.stop()

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
        st.stop()

    quadrant_df = build_quadrant_df(prev_metrics, curr_metrics)
    if quadrant_df.empty:
        st.error("No overlapping sectors between the selected dates.")
        st.stop()

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

    mode = st.radio("Axes mode", ["Absolute", "Percentile"], horizontal=True, index=0)
    fig = plot_quadrants(quadrant_df, mode)
    st.plotly_chart(fig, use_container_width=True)

    def _fmt(value: float) -> str:
        if value is None or np.isnan(value):
            return "n/a"
        return f"{value:.1f}"

    st.caption(f"P median (current): {_fmt(curr_p_median)} | T median (current): {_fmt(curr_t_median)}")

    if curr_p_median >= 50 and curr_t_median >= 50:
        st.info("Market regime: Broad risk-on (P_med ≥ 50 and T_med ≥ 50).")
    elif curr_p_median < 50:
        st.info("Market regime: Risk-off / narrow (P_med < 50).")

    if curr_mc_var_all_negative:
        st.warning("All sectors show negative 1‑month market cap variation (absolute weakness).")

    st.markdown(
        """
**How to read the quadrants**
- A (High T + High P): confirmed leadership (ΔT > 0 accelerating, ΔT < 0 fading)
- B (High T + Low P): narrow/fragile (ΔT > 0 strengthening but fragile; ΔT < 0 breakdown risk)
- C (Low T + High P): early rotation (ΔT > 0 emerging; ΔT < 0 wait)
- D (Low T + Low P): weak (ΔT > 0 bottoming attempt; ΔT < 0 breakdown)
        """
    )

st.markdown("---")
st.markdown(
    "**Shortcut tip:** target `C:/Users/razva/PycharmProjects/equipicker/equipicker/.venv/Scripts/python.exe -m streamlit run monthly_scoring_app.py` with start-in folder `C:/Users/razva/PycharmProjects/equipicker/equipicker`."
)
